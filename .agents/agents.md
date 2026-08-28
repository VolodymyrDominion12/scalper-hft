# Агенти у scalper-hft

Правила для AI-агентів (співпрацюють з AGENTS.md у корені проєкту).

## Ролі

- **researcher** — web-дослідження ринків/стратегій → `docs/RESEARCH.md`.
- **strategist** — реалізація нових альфа-моделей за `skills/strategy-development.md`.
- **auditor** — аудит перенавчання за `skills/overfitting-audit.md`; блокує стратегії
  з OOS Sharpe < 0.3 або DSR < 0.9.
- **data-eng** — підтримка завантажувачів/кешу (`scalper_hft/data/`).

## Процес (цикл інвестігейт → реалізація → тест → аудит → покращення)

1. **Investigate**: researcher оновлює `docs/RESEARCH.md` (нові сигнали, комісії, ризики).
2. **Implement**: strategist додає стратегію + тест (`pytest` зелений).
3. **Test**: `python -m scalper_hft.cli backtest` на свіжих даних.
4. **Audit**: auditor запускає `python -m scalper_hft.cli overfit`; вердикт у `docs/reports/`.
5. **Improve**: лише якщо аудит пройдено — оптимізація (Optuna) та paper/live крок.
6. Повтор: кожні N днів оновлювати дані та повторювати аудит (ринок нестабільний).

## Критерії блокування
- OOS Sharpe < 0.3 або частка позитивних вікон < 50%;
- DSR < 0.9; smoothness < 0.3; n_trades < 100;
- середній трейд менший за `CostModel.round_trip_taker`.
