.DEFAULT_GOAL := help

# ==============================================================================
# Змінні за замовчуванням (можна перевизначати: make dashboard PORT=8502)
# ==============================================================================
PYTHON        ?= uv run python
STREAMLIT     ?= uv run streamlit
PORT          ?= 8501
SYMBOL        ?= BTCUSDT
INTERVAL      ?= 1h
DAYS          ?= 90
STRATEGY      ?= pairs_arb
LEG1          ?= XRPUSDT
LEG2          ?= BTCUSDT
MINUTES       ?= 60
JOBS          ?= 1

# VPS налаштування для синхронізації даних
VPS_USER      ?= tradebot
VPS_HOST      ?= 46.36.216.98
VPS_DIR       ?= /home/tradebot/scalper-hft/data
VPS_PORT      ?= 22
SSH_KEY       ?=
FILE          ?=
DRY           ?= 0
MAKEFILE_DIR  := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
# --whole-file: parquet на VPS атомарно підміняється рекордером; delta-xfer
# mmap дає ENODATA (61) і rsync 23. Обгортка повторює 23/24.
RSYNC_RETRY   := bash $(MAKEFILE_DIR)scripts/rsync_retry.sh
RSYNC_FLAGS    = -avzP --whole-file $(if $(filter 1 true,$(DRY)),--dry-run,)
RSYNC_SSH      = ssh -p $(VPS_PORT)$(if $(SSH_KEY), -i $(SSH_KEY),)

# ==============================================================================
# Допомога
# ==============================================================================
.PHONY: help
help: ## Показати список доступних команд
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║                   scalper-hft — Команди управління                   ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ==============================================================================
# Дашборд
# ==============================================================================
.PHONY: dashboard dashboard-dev
dashboard: ## Запустити Streamlit дашборд (порт за замовчуванням 8501)
	$(STREAMLIT) run scalper_hft/dashboard.py --server.port $(PORT)

dashboard-dev: ## Запустити дашборд у режимі розробки з автоперезавантаженням
	$(STREAMLIT) run scalper_hft/dashboard.py --server.port $(PORT) --server.runOnSave true

# ==============================================================================
# Docker / База даних
# ==============================================================================
.PHONY: up up-postgres down ps logs-postgres logs-pgadmin
up: ## Підняти PostgreSQL + pgAdmin у фоні
	docker compose up -d

up-postgres: ## Підняти лише PostgreSQL у фоні
	docker compose up -d postgres

down: ## Зупинити всі контейнери
	docker compose down

ps: ## Перевірити статус контейнерів та healthcheck
	docker compose ps

logs-postgres: ## Переглянути логи PostgreSQL у реальному часі
	docker compose logs -f postgres

logs-pgadmin: ## Переглянути логи pgAdmin у реальному часі
	docker compose logs -f pgadmin

# ==============================================================================
# Встановлення та оточення
# ==============================================================================
.PHONY: install install-all sync
install: ## Встановити базові залежності та пакет scalper-hft
	uv sync

install-all: ## Встановити всі екстра-залежності (dashboard, optim, ml, live, dev)
	uv sync --all-extras

sync: ## Синхронізувати оточення через uv sync --all-extras
	uv sync --all-extras

# ==============================================================================
# Тестування та перевірка якості коду
# ==============================================================================
.PHONY: test test-cov lint format typecheck check spec-check spec-validate
test: ## Запустити швидкі тести pytest
	uv run pytest tests/ -q

test-cov: ## Запустити тести з покриттям коду
	uv run pytest --cov=scalper_hft tests/

lint: ## Перевірити лінтинг коду за допомогою ruff
	uv run ruff check .

format: ## Автоформатування коду через ruff
	uv run ruff format .

typecheck: ## Перевірити типи за допомогою mypy
	uv run mypy scalper_hft

# ==============================================================================
# Spec-Driven Development (SDD)
# ==============================================================================
spec-check: ## [SDD] Запустити повний spec-runner: валідація схеми + behavioral тести
	uv run pytest tests/test_strategy_specs.py -v --tb=short

spec-validate: ## [SDD] Тільки валідація YAML-схеми (без behavioral тестів)
	uv run python specs/strategies/_validator.py

check: lint typecheck test spec-check ## Повна перевірка: lint + typecheck + test + spec-check

# ==============================================================================
# Ринкові дані
# ==============================================================================
.PHONY: download record-bookticker sync-data sync-vps sync-depth5 ls-vps-data pull-data
download: ## Завантажити історичні klines (SYMBOL=BTCUSDT INTERVAL=1h DAYS=90)
	$(PYTHON) -m scalper_hft.cli download --symbol $(SYMBOL) --interval $(INTERVAL) --days $(DAYS)

record-bookticker: ## Записати WS bookTicker у Parquet (SYMBOL=BTCUSDT MINUTES=60)
	$(PYTHON) -m scalper_hft.cli record-bookticker --symbol $(SYMBOL) --minutes $(MINUTES)

sync-data: ## Підтягнути ринкові дані з VPS у локальну папку data/ (rsync)
	@mkdir -p data
	@if [ -n "$(FILE)" ]; then \
		echo "==> Синхронізація $(FILE) з $(VPS_USER)@$(VPS_HOST):$(VPS_DIR)/..."; \
		$(RSYNC_RETRY) $(RSYNC_FLAGS) -e "$(RSYNC_SSH)" $(VPS_USER)@$(VPS_HOST):$(VPS_DIR)/$(FILE) data/; \
	else \
		echo "==> Синхронізація всіх даних з $(VPS_USER)@$(VPS_HOST):$(VPS_DIR)/..."; \
		$(RSYNC_RETRY) $(RSYNC_FLAGS) -e "$(RSYNC_SSH)" $(VPS_USER)@$(VPS_HOST):$(VPS_DIR)/ data/; \
	fi

sync-depth5: ## Підтягнути лише depth5 Parquet файли з VPS у data/
	@mkdir -p data
	@echo "==> Синхронізація *_depth5.parquet з $(VPS_USER)@$(VPS_HOST):$(VPS_DIR)/..."; \
	$(RSYNC_RETRY) $(RSYNC_FLAGS) -e "$(RSYNC_SSH)" '$(VPS_USER)@$(VPS_HOST):$(VPS_DIR)/*depth5.parquet' data/

ls-vps-data: ## Показати список файлів даних на VPS
	ssh -p $(VPS_PORT)$(if $(SSH_KEY), -i $(SSH_KEY),) $(VPS_USER)@$(VPS_HOST) "ls -lh $(VPS_DIR)/"

sync-vps: sync-data ## Аліас для sync-data
pull-data: sync-data ## Аліас для sync-data

# ==============================================================================
# Бектести та дослідження
# ==============================================================================
.PHONY: backtest plot pairs pairs-portfolio overfit cscv report coint-scan
backtest: ## Запустити бектест (STRATEGY=mean_reversion SYMBOL=BTCUSDT INTERVAL=1h DAYS=90)
	$(PYTHON) -m scalper_hft.cli backtest --strategy $(STRATEGY) --symbol $(SYMBOL) --interval $(INTERVAL) --days $(DAYS)

plot: ## Згенерувати інтерактивний HTML-графік бектесту
	$(PYTHON) -m scalper_hft.cli plot --strategy $(STRATEGY) --symbol $(SYMBOL) --interval $(INTERVAL) --days $(DAYS)

pairs: ## Запустити бектест парного арбітражу (LEG1=XRPUSDT LEG2=BTCUSDT INTERVAL=1h DAYS=90)
	$(PYTHON) -m scalper_hft.cli pairs --strategy pairs_arb --leg1 $(LEG1) --leg2 $(LEG2) --interval $(INTERVAL) --days $(DAYS) --maker

pairs-portfolio: ## Бектест портфеля валідованих пар
	$(PYTHON) -m scalper_hft.cli pairs-portfolio --interval $(INTERVAL) --days $(DAYS) --maker

overfit: ## Провести повний аудит на перенавчання (STRATEGY=pairs_arb SYMBOL=BTCUSDT)
	$(PYTHON) -m scalper_hft.cli overfit --strategy $(STRATEGY) --symbol $(SYMBOL) --interval $(INTERVAL) --days $(DAYS)

cscv: ## Обчислити PBO через Combinatorial Purged CV
	$(PYTHON) -m scalper_hft.cli cscv --strategy $(STRATEGY) --symbol $(SYMBOL) --interval $(INTERVAL) --days $(DAYS)

report: ## Згенерувати повний markdown-звіт у docs/reports/
	$(PYTHON) -m scalper_hft.cli report --strategy $(STRATEGY) --symbol $(SYMBOL) --interval $(INTERVAL) --days $(DAYS)

coint-scan: ## Сканування коінтеграції між основними символами
	$(PYTHON) -m scalper_hft.cli coint-scan --symbols "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT" --interval $(INTERVAL) --days $(DAYS)

# ==============================================================================
# Черга задач та воркери (Research Jobs)
# ==============================================================================
.PHONY: worker job-list job-prune
worker: ## Запустити воркер дослідницьких задач (JOBS=1)
	$(PYTHON) -m scalper_hft.cli job worker --jobs $(JOBS)

job-list: ## Переглянути список задач у черзі
	$(PYTHON) -m scalper_hft.cli job list

job-prune: ## Очистити застарілі задачі та звільнити диск (вік >= 14 днів)
	$(PYTHON) -m scalper_hft.cli job prune --days 14

# ==============================================================================
# Paper Trading
# ==============================================================================
.PHONY: paper-run-pairs paper-audit
paper-run-pairs: ## Запустити paper pairs трейдер (LEG1=XRPUSDT LEG2=BTCUSDT)
	$(PYTHON) -m scalper_hft.cli paper-run-pairs --leg1 $(LEG1) --leg2 $(LEG2)

paper-audit: ## Порівняльний аудит paper SQLite vs бектест
	$(PYTHON) -m scalper_hft.cli paper-audit

# ==============================================================================
# Очищення
# ==============================================================================
.PHONY: clean
clean: ## Очистити тимчасові файли кешу (.pytest_cache, .ruff_cache, __pycache__)
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
