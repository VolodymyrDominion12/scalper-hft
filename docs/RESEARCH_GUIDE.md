# scalper-hft: Повний посібник з досліджень та запусків (Research Runbook)

Цей документ містить повний опис дослідницького процесу (R&D) у проекті **scalper-hft**:
від формування гіпотези та завантаження даних до матричних бектестів, аудиту на перенавчання (AFML), управління чергою фонових задач (`job worker`) та переходу до paper trading.

---

## Зміст

1. [Архітектура досліджень та філософія](#1-архітектура-досліджень-та-філософія)
2. [Швидкий старт (Cheat Sheet)](#2-швидкий-старт-cheat-sheet)
3. [Робота з даними](#3-робота-з-даними)
4. [Бектести та візуалізація](#4-бектести-та-візуалізація)
5. [Оптимізація та матричний скан (Sweep)](#5-оптимізація-та-матричний-скан-sweep)
6. [Аудит на перенавчання (Anti-Overfitting & AFML)](#6-аудит-на-перенавчання-anti-overfitting--afml)
7. [Глибока аналітика: фічі, стрес-тести, мікроструктура](#7-глибока-аналітика-фічі-стрес-тести-мікроструктура)
8. [Черга задач та фонові воркери (Research Jobs & Workers)](#8-черга-задач-та-фонові-воркери-research-jobs--workers)
9. [Інтерактивний дашборд (Streamlit)](#9-інтерактивний-дашборд-streamlit)
10. [Робота в Jupyter Notebook](#10-робота-в-jupyter-notebook)
11. [Покроковий чекліст валідації альфи](#11-покроковий-чекліст-валідації-альфи)

---

## 1. Архітектура досліджень та філософія

Система побудована за канонами книги **"Inside the Black Box" (Rishi K. Narang)** та методології **"Advances in Financial Machine Learning" (Marcos López de Prado)**:

$$\text{Alpha} \longrightarrow \text{Risk Model} \longrightarrow \text{Transaction Cost Model} \longrightarrow \text{Portfolio Construction} \longrightarrow \text{Execution}$$

### Головні непорушні принципи:

1. **Сувора заборона Lookahead-bias:**
   - Сигнали розраховуються **лише** на інформації, що доступна до закриття бару $t$.
   - Виконання ордерів завжди моделюється з відкриття наступного бару $t+1$ (рушій автоматично зсуває сигнали на 1 крок: `lag(1)`).
2. **Обов'язковий облік реальних комісій та прослизання:**
   - Комісії Binance USDT-M: **Maker 0.02%** / **Taker 0.05%** (з урахуванням BNB-дисконту).
   - Прослизання (`slippage_frac`): типово 0.01%–0.05% залежно від ліквідності та таймфрейму.
   - Безкомісійні тести у системі заборонені, оскільки у скальпінгу fee drag часто з'їдає 100% прибутку.
3. **Spec-Driven Development (SDD):**
   - Будь-яка стратегія починається зі специфікації в `specs/strategies/<name>.yaml` з чіткою економічною гіпотезою та умовами edge.
   - Валідація схеми: `uv run python specs/strategies/_validator.py`.
4. **Недоторканний холдаут (Holdout 20%):**
   - Останні 20% історичних даних ніколи не використовуються для підбору параметрів. Це фінальний сліпий тест.

---

## 2. Швидкий старт (Cheat Sheet)

```bash
# 1. Завантажити дані (1 година, 90 днів)
uv run python -m scalper_hft.cli download --symbol BTCUSDT --interval 1h --days 90

# 2. Запустити базовий бектест
uv run python -m scalper_hft.cli backtest --strategy mean_reversion --symbol BTCUSDT --interval 1h --days 90

# 3. Інтерактивний HTML графік зі свічками, сигналами та рівнями SL/TP
uv run python -m scalper_hft.cli plot --strategy mean_reversion --symbol BTCUSDT --interval 1h --days 90 --out docs/plots/backtest.html

# 4. Повний аудит на перенавчання (Walk-Forward + Deflated Sharpe + Sensitivity)
uv run python -m scalper_hft.cli overfit --strategy mean_reversion --symbol BTCUSDT --interval 1h --days 90

# 5. Перевірка PBO через Combinatorial Purged Cross-Validation (CSCV)
uv run python -m scalper_hft.cli cscv --strategy mean_reversion --symbol BTCUSDT --interval 1h --days 90 --variants 30

# 6. Запустити фоновий worker черги досліджень на 2 паралельні процеси
uv run python -m scalper_hft.cli job worker --jobs 2

# 7. Запустити Streamlit дашборд
make dashboard
```

---

## 3. Робота з даними

Модуль `scalper_hft.data` забезпечує роботу з Parquet-кешем (`data/` або шлях із `.env`), REST API Binance та історичними архівами.

### 3.1. Завантаження історичних свічок (Klines)

```bash
# Завантаження 90 днів 1-годинних свічок для BTCUSDT
uv run python -m scalper_hft.cli download --symbol BTCUSDT --interval 1h --days 90

# Завантаження з ресемплінгом зі свічок 1m (рекомендовано для швидкості й консистентності)
uv run python -m scalper_hft.cli download --symbol ETHUSDT --interval 15m --days 60 --base 1m

# Пряме завантаження з біржі без деривації
uv run python -m scalper_hft.cli download --symbol SOLUSDT --interval 5m --days 30 --no-derive

# Примусове оновлення останнього хвоста свічок
uv run python -m scalper_hft.cli download --symbol BTCUSDT --interval 1h --days 30 --force
```

### 3.2. Завантаження aggTrades (кумулятивна дельта CVD) та Funding Rates

```bash
# Свічки + aggTrades (глибина до 2 днів через обмеження Binance REST API)
uv run python -m scalper_hft.cli download --symbol BTCUSDT --interval 1m --days 10 --trades --trades-days 2

# Свічки + історія ставок фінансування (funding rates)
uv run python -m scalper_hft.cli download --symbol BTCUSDT --interval 1h --days 90 --funding

# Завантаження глибокої історії aggTrades з офіційних архівів Binance Vision
uv run python -m scalper_hft.cli download --symbol BTCUSDT --vision --vision-start 2024-01-01 --vision-freq monthly
```

### 3.3. Запис реального стакана (L2 / bookTicker / depth5) через WebSocket

Для аналізу мікроструктури, дисбалансу стакана (Order Book Imbalance) та спреду:

```bash
# Запис найкращого біду/аску (bookTicker) протягом 60 хвилин у Parquet
uv run python -m scalper_hft.cli record-bookticker --symbol BTCUSDT --minutes 60

# Запис 5 рівнів глибини стакана (depth5)
uv run python -m scalper_hft.cli record-bookticker --symbol BTCUSDT --minutes 30 --depth

# Запис декількох символів одночасно
uv run python -m scalper_hft.cli record-bookticker --symbol "BTCUSDT,ETHUSDT,SOLUSDT" --minutes 120
```

### 3.4. Міграція даних з PostgreSQL у Parquet

Якщо дані зберігалися у PostgreSQL, їх можна легко експортувати у швидкий локальний Parquet-кеш:

```bash
uv run python -m scalper_hft.cli migrate-to-parquet --symbol BTCUSDT,ETHUSDT
```

---

## 4. Бектести та візуалізація

### 4.1. Одиночні стратегії (`backtest`)

```bash
# Базовий бектест із параметрами за замовчуванням
uv run python -m scalper_hft.cli backtest --strategy mean_reversion --symbol BTCUSDT --interval 1h --days 90

# Бектест із передачею кастомних параметрів (-p key=value)
uv run python -m scalper_hft.cli backtest --strategy mean_reversion --symbol BTCUSDT --interval 1h --days 90 \
    -p rsi_period=14 -p oversold=30 -p overbought=70

# Breakeven gate: блокувати угоди, де очікуваний рух (ATR) менший за round-trip комісії
uv run python -m scalper_hft.cli backtest --strategy cvd_momentum --symbol BTCUSDT --interval 5m --days 30 --breakeven-gate

# Використання нечасових барів (доларові або об'ємні бари за Marcos López de Prado)
uv run python -m scalper_hft.cli backtest --strategy mean_reversion --symbol BTCUSDT --days 5 \
    --bar-type dollar --bar-threshold 1000000
```

### 4.2. Парний статистичний арбітраж (`pairs`)

```bash
# Бектест пари з Maker-виконанням (post-only)
uv run python -m scalper_hft.cli pairs --strategy pairs_arb --leg1 XRPUSDT --leg2 BTCUSDT --interval 1h --days 90 --maker

# Парний арбітраж із динамічним розрахунком коефіцієнта хеджування через фільтр Калмана (Kalman Filter)
uv run python -m scalper_hft.cli pairs --strategy pairs_arb --leg1 ETHUSDT --leg2 BTCUSDT --interval 1h --days 90 --use-kalman

# Walk-Forward валідація пари (In-Sample 1500 барів, Out-of-Sample 500 барів)
uv run python -m scalper_hft.cli pairs --strategy pairs_arb --leg1 SOLUSDT --leg2 BTCUSDT --interval 1h --days 120 --walkforward --train 1500 --test 500
```

### 4.3. Портфель валідованих пар (`pairs-portfolio`)

Бектест кошика некорельованих пар з алокацією за рівними вагами або **Equal Risk Contribution (ERC)**:

```bash
# Портфель із рівними вагами
uv run python -m scalper_hft.cli pairs-portfolio --interval 1h --days 90 --maker --method equal

# Портфель із ризик-паритетом (Equal Risk Contribution за Narang гл. 6)
uv run python -m scalper_hft.cli pairs-portfolio --interval 1h --days 90 --maker --method erc --turnover-rate 0.0005
```

### 4.4. Дельта-нейтральний арбітраж фінансування (`arb`)

```bash
uv run python -m scalper_hft.cli arb --symbol BTCUSDT --interval 1h --days 90 --maker --position-pct 0.1
```

### 4.5. Інтерактивний HTML-графік (`plot`)

Створює візуалізацію угод, індикаторів, рівнів Stop-Loss / Take-Profit та кривої капіталу:

```bash
uv run python -m scalper_hft.cli plot --strategy mean_reversion --symbol BTCUSDT --interval 1h --days 90 \
    --out docs/plots/btc_mean_reversion.html --bars 5000
```
Відкрийте створений файл у браузері або через IDE для детального перегляду окремих трейдів.

---

## 5. Оптимізація та матричний скан (Sweep)

### 5.1. Оптимізація параметрів через Optuna (`optimize`)

Пошук оптимальних параметрів алгоритмом TPE (Tree-structured Parzen Estimator) із Purged Cross-Validation:

```bash
# Оптимізація стратегії на 60 спроб із 4 сплітами
uv run python -m scalper_hft.cli optimize --strategy mean_reversion --symbol BTCUSDT --interval 1h --days 90 --trials 60 --splits 4

# Оптимізація гіперпараметрів ML-моделі (AFML)
uv run python -m scalper_hft.cli ml-opt --symbol BTCUSDT --interval 5m --days 30 --trials 40 --splits 5 --embargo 0.01
```

### 5.2. Матричний прогін (Sweep)

Матричний запуск: комбінація **Стратегії × Символи × Таймфрейми**. Базові дані (1m) завантажуються один раз на символ, після чого ресемпляться на інші таймфрейми без повторних запитів:

```bash
# Швидкий скан усіх стратегій на основних монетах
uv run python -m scalper_hft.cli sweep \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT \
    --intervals 15m,1h,4h \
    --days 60 \
    --workers 4 \
    --out results/sweep.csv

# Матричний прогін у режимі Walk-Forward (OOS метрики замість простих бектестів)
uv run python -m scalper_hft.cli sweep \
    --strategies mean_reversion,supertrend,pairs_arb \
    --symbols BTCUSDT,ETHUSDT \
    --intervals 1h \
    --days 90 \
    --mode walkforward \
    --workers 2
```

---

## 6. Аудит на перенавчання (Anti-Overfitting & AFML)

Згідно зі скілом `.agents/skills/overfitting-audit/SKILL.md`, жодна стратегія не допускається до реальної торгівлі без статистичного аудиту.

### 6.1. Повний аудит (`overfit`)

Запускає Walk-Forward аналіз, рахує Deflated Sharpe Ratio (DSR) та будує профіль чутливості параметрів:

```bash
uv run python -m scalper_hft.cli overfit --strategy mean_reversion --symbol BTCUSDT --interval 1h --days 90 --trials 50
```

**Критерії проходження:**
- `avg_oos_sharpe` > **0.3** (середній коефіцієнт Шарпа на OOS вікнах)
- `pct_oos_positive` $\ge$ **50%** (відсоток прибуткових OOS вікон)
- `DSR` (Deflated Sharpe) > **0.95** (ймовірність того, що edge статистично значущий, а не результат багаторазових тестів $N$)
- `smoothness` > **0.30** (плато стійкості навколо знайдених параметрів)

### 6.2. Combinatorial Purged Cross-Validation (`cscv`)

Обчислює ймовірність перенавчання бектесту (**PBO** — Probability of Backtest Overfitting) за методом Бейлі та Лопеса де Прадо:

```bash
uv run python -m scalper_hft.cli cscv --strategy mean_reversion --symbol BTCUSDT --interval 1h --days 90 \
    --variants 30 --blocks 8 --max-combos 200
```
- **PBO < 0.50** (ідеально < 0.25): In-Sample переможці залишаються прибутковими Out-of-Sample.

### 6.3. Генерація підсумкового Markdown-звіту (`report`)

Створює комплексний звіт у `docs/reports/report_<strategy>_<symbol>_<timestamp>.md`:

```bash
uv run python -m scalper_hft.cli report --strategy mean_reversion --symbol BTCUSDT --interval 1h --days 90
```

---

## 7. Глибока аналітика: фічі, стрес-тести, мікроструктура

### 7.1. Важливість фіч та фільтрація (Feature Importance & Lift)

```bash
# MDI / MDA / SFI важливість фіч для Machine Learning (AFML Ch. 8)
uv run python -m scalper_hft.cli featimp --symbol BTCUSDT --interval 5m --days 30 --splits 4

# Кластеризована важливість фіч (Clustered Feature Importance, CFI)
uv run python -m scalper_hft.cli cfi --symbol BTCUSDT --interval 5m --days 30 --max-clusters 6

# Децильний Lift-аналіз (чи розділяє фіча прибуткові трейди від збиткових)
uv run python -m scalper_hft.cli lift --strategy mean_reversion --symbol BTCUSDT --interval 1h --days 90 --bins 10 --top 10
```

### 7.2. Стрес-тести та ємність стратегії (Capacity & Stress)

```bash
# Стрес-тест на екстремальних ринкових подіях (крах, шок волатильності, шок фандінгу)
uv run python -m scalper_hft.cli stress --strategy mean_reversion --symbol BTCUSDT --interval 1h --days 180 \
    --scenarios crash,liquidity,vol_spike,funding_shock

# Тест ємності (Capacity): деградація Sharpe при збільшенні капіталу в 1, 2, 5, 10, 20 разів
uv run python -m scalper_hft.cli capacity --strategy mean_reversion --symbol BTCUSDT --interval 1h --days 90 \
    --scales 1,2,5,10,20 --maker

# Time-Decay аналіз: чутливість прибутку до затримки виконання входу на 1, 2, 3 бари
uv run python -m scalper_hft.cli time-decay --strategy mean_reversion --symbol BTCUSDT --interval 1h --days 90 --max-lag 3

# Аналіз виживання угод (Kaplan–Meier Survival Analysis): час утримання позицій
uv run python -m scalper_hft.cli survival --strategy mean_reversion --symbol BTCUSDT --interval 1h --days 90
```

### 7.3. Коінтеграція та аналіз пар

```bash
# Скан матриці коінтеграції (Johansen / Engle-Granger) між символами
uv run python -m scalper_hft.cli coint-scan --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT --interval 1h --days 90

# Квінтильний аналіз z-score спреду пари
uv run python -m scalper_hft.cli quintile --leg1 XRPUSDT --leg2 BTCUSDT --interval 1h --days 90

# Порівняння динамічного OLS/Johansen коефіцієнта хеджування проти 1:1
uv run python -m scalper_hft.cli hedge-ratio --leg1 XRPUSDT --leg2 BTCUSDT --interval 1h --days 90
```

---

## 8. Черга задач та фонові воркери (Research Jobs & Workers)

Для довгих досліджень (матричні sweep, Optuna, важкі бектести) scalper-hft має вбудовану чергу задач на **SQLite** (`results/jobs.sqlite`).

### 8.1. Принцип роботи черги

```
[CLI / Dashboard / Script]
         │
         ▼ (submit або --enqueue)
  results/jobs.sqlite  ◄─────── SHA-256 fingerprint (дедуплікація)
         │
         ▼ (claim job)
[scalper-hft job worker] ───► Окремий дочірній процес (multiprocessing)
         │                    ├── Heartbeat кожні 15 сек
         │                    └── results/jobs/<id>/job.log
         ▼
  results/jobs/<id>/  (артефакти: metrics.json, equity.csv, sweep.csv)
```

### 8.2. Як запускати воркери

Воркер — це довгоживучий процес, який періодично перевіряє `results/jobs.sqlite`, бере задачу в статус `running`, виконує її в ізольованому дочірньому процесі та записує результати.

```bash
# Запуск 1 воркера у поточному терміналі
uv run python -m scalper_hft.cli job worker

# Запуск пулу з 4 паралельних воркерів
uv run python -m scalper_hft.cli job worker --jobs 4

# Запуск воркера у фоновому режимі (nohup)
nohup uv run python -m scalper_hft.cli job worker --jobs 2 > results/worker.log 2>&1 &
```

> [!TIP]
> При запуску Streamlit-дашборду (`uv run python -m scalper_hft.cli dashboard`) фоновий воркер запускається **автоматично** як супутній процес, якщо не вказано прапорець `--no-worker`.

### 8.3. Як ставити задачі в чергу

1. **Через прапорець `--enqueue` у звичайних CLI-командах:**
   ```bash
   # Поставити бектест у чергу (команда завершується миттєво, друкуючи job id)
   uv run python -m scalper_hft.cli backtest --strategy mean_reversion --symbol BTCUSDT --interval 1h --days 90 --enqueue

   # Поставити перевірку парного арбітражу
   uv run python -m scalper_hft.cli pairs --strategy pairs_arb --leg1 XRPUSDT --leg2 BTCUSDT --interval 1h --days 90 --maker --enqueue

   # Поставити важкий матричний скан у чергу
   uv run python -m scalper_hft.cli sweep --symbols BTCUSDT,ETHUSDT --intervals 15m,1h --days 90 --enqueue

   # Поставити аудит на перенавчання
   uv run python -m scalper_hft.cli overfit --strategy mean_reversion --symbol BTCUSDT --interval 1h --days 90 --enqueue
   ```

2. **Через прямий виклик `job submit` (JSON-параметри):**
   ```bash
   uv run python -m scalper_hft.cli job submit backtest --params '{"strategy": "mean_reversion", "symbol": "BTCUSDT", "interval": "1h", "days": 90}'
   ```

### 8.4. Управління чергою задач

```bash
# Переглянути список задач у черзі та перевірити статус воркера
uv run python -m scalper_hft.cli job list

# Переглянути детальну інформацію та хвіст логів задачі за її ID
uv run python -m scalper_hft.cli job status 1

# Скасувати задачу, що очікує або виконується
uv run python -m scalper_hft.cli job cancel 1

# Перезапустити задачу заново (скидає артефакти й повертає в чергу)
uv run python -m scalper_hft.cli job rerun 1

# Очистити застарілі завершені задачі та їхні артефакти на диску (віком > 14 днів)
uv run python -m scalper_hft.cli job prune --days 14

# Тестове очищення (dry-run, лише підрахунок звільненого місця)
uv run python -m scalper_hft.cli job prune --days 7 --dry-run
```

---

## 9. Інтерактивний дашборд (Streamlit)

Для візуального аналізу, моніторингу черги та запуску досліджень доступний повнофункціональний вебінтерфейс:

```bash
# Запуск дашборду (порт 8501)
make dashboard

# Або пряма команда:
uv run python -m scalper_hft.cli dashboard --port 8501 --address 127.0.0.1
```

### Основні розділи дашборду:
- **Research Hub & Queue:** Моніторинг задач, запуск нових бектестів/sweep, перегляд прогресу та логів у реальному часі.
- **Strategy Book:** Каталог зареєстрованих стратегій, параметрів та режимів ринку.
- **Backtest Analysis:** Інтерактивні графіки кривої капіталу, підводні камені просадок (drawdowns), розподіл угод.
- **Overfitting & Validation:** Перегляд результатів Walk-Forward, таблиці DSR та метрик чутливості.
- **Pairs & Portfolio:** Матриця коінтеграції, статистика спреду, ваги ERC.
- **Paper / Live Monitor:** Стан балансів, активні позиції, звірка (reconciliation).

---

## 10. Робота в Jupyter Notebook

Інтерактивний дослідницький ноутбук розташований за адресою:
`notebooks/research.ipynb`

### Як запустити:
```bash
# В активному оточенні uv:
uv run jupyter lab
# Або з PyCharm / VS Code: оберіть інтерпретатор .venv/bin/python
```

Усі ключові функції Python API доступні безпосередньо в коді:
```python
from scalper_hft.config import get_settings
from scalper_hft.data.access import ensure_klines
from scalper_hft.strategies import get_strategy
from scalper_hft.backtest.engine import run_backtest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.validation.walk_forward import run_walk_forward
from scalper_hft.validation.deflated_sharpe import deflated_sharpe_ratio, estimate_n_trials
from scalper_hft.research.jobs import JobStore

settings = get_settings()
df = ensure_klines("BTCUSDT", "1h", days=90)
strat = get_strategy("mean_reversion")
cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
res = run_backtest(df, strat, cost=cost)
print(res.summary())
```

---

## 11. Покроковий чекліст валідації альфи

Перед тим як запускати будь-яку стратегію на реальні кошти або в paper trading, пройдіть цей чекліст:

1. [ ] **Специфікація (SDD):** Створено та перевірено `specs/strategies/<name>.yaml` (`uv run python specs/strategies/_validator.py`).
2. [ ] **Комісії та прослизання:** Бектест запущено з реальними `maker_fee` / `taker_fee` та `slippage_frac`.
3. [ ] **Статистична вибірка:** Загальна кількість угод $\ge 100$ для частоти 1m/5m, або $\ge 40$ для 1h.
4. [ ] **Walk-Forward:** Середній OOS Sharpe $> 0.3$, понад 50% вікон показують позитивний результат.
5. [ ] **Deflated Sharpe (DSR):** Значення DSR $> 0.95$ після врахування реальної кількості протестованих конфігурацій (`n_trials`).
6. [ ] **CSCV / PBO:** Ймовірність перенавчання (PBO) $< 0.50$ (ідеально $< 0.25$).
7. [ ] **Аналіз чутливості (Sensitivity):** Відсутні "гострі шпилі" у просторі параметрів (гладкість $> 0.30$).
8. [ ] **Time-Decay:** Стратегія зберігає прибутковість при затримці виконання входу на 1 бар.
9. [ ] **Стрес-тест:** Стратегія витримує сценарії шоку волатильності та просадки ліквідності без ліквідації рахунку.
10. [ ] **Paper-валідація:** Мінімум 2 тижні успішної роботи на демо-рахунку або в режимі `paper-run-pairs` без суттєвого tracking error до бектесту.
