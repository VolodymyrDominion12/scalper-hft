# Протокол досліджень стратегій (7 етапів)

Кожна гіпотеза проходить **послідовно**. Наступний етап — лише після PASS попереднього.

| Етап | Назва | Критерій PASS | CLI / модуль |
|------|-------|---------------|--------------|
| 0 | Economic plausibility | round-trip cost < expected move | `CostModel.breakeven_move_pct` |
| 1 | Signal quality | quintile ρ≥0.7, monotonic | `report`, `quintile` |
| 2 | Architecture WF | fixed params, ≥30 trades, multi-symbol | `overfit --audit-mode exploratory` |
| 3 | Sensitivity | smoothness≥0.3, плато | `overfit`, `sensitivity` |
| 4 | Regime attribution | per-regime SR, ≥7/10 symbols | `experiments/iter7_*` |
| 5 | Anti-snooping | DSR>0.95, PBO<0.5, OOS burn | `overfit --audit-mode final` |
| 6 | Blind holdout | holdout Sharpe > 0, один раз | `HOLDOUT_PCT=20`, `final` |
| 7 | Paper | tracking error ≥8 weeks | `paper-audit` |

## Правила

1. **Spec before code** — `.agents/skills/spec-driven-strategy/SKILL.md`
2. **Pre-registration** — `docs/reports/hypothesis_<name>.md` до прогону
3. **Мінімум 30 угод** за 3y на клітинку — інакше `degenerate`
4. **Capability gate** — `strategy.requires` визначає TF/дані
5. **Trial ledger** — `TRIAL_LEDGER_PATH` для чесного DSR
6. **Не змішувати mode** — ML internal WF vs in-sample backtest

## Шаблон

Скопіюй `experiments/_template_experiment.py` → `experiments/iterN_<name>.py`.

## Артефакти

- Результати: `results/`
- OOS burn: `docs/reports/oos_usage.md`
- Вердикти: `results/verdicts.jsonl`
