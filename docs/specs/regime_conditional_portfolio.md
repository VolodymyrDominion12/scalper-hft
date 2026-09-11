# Spec: Regime-Conditional Portfolio (iter8)

Дата: 2026-09-11 · Статус: **DRAFT** (після iter7, до fresh-data validation)

## Гіпотеза

Одна універсальна стратегія не існує. Портфель:
- **Core**: `pairs_arb` LINK/BTC 1h maker + `regime_scale=0.25` (market-neutral, T0 validated)
- **Bull sleeve**: momentum (`supertrend` або `smc_fvg`) — лише в `trend_up`
- **Flat/bear sleeve**: `funding_carry` — у `range` та `trend_down`

## Джерело карти

Empirical OOS з iter7 (`docs/reports/strategy_rating_regime.md`):
- `trend_up`: supertrend +3.96 SR (10/10 symbols)
- `range`: funding_carry +1.42 SR (9/10)
- `trend_down`: funding_carry +2.78 SR (9/10)

## Реалізація

- `validation/regime_map.py` → `RegimeStrategyMap` у `data/regime_strategy_map.json`
- `regime_supervisor` / новий selector читає JSON замість taxonomy `preferred_regimes`
- Лаг 1 бар на перемикання (без lookahead)
- Порожній `preferred_regimes` ≠ вага 1.0 у всіх режимах

## Критерії PASS (iter8)

1. Sensitivity порогів детектора — плато (не пік)
2. Fresh symbols (ATOMUSDT, forward period) — DSR portfolio > 0.95
3. `audit_cell --audit-mode final` з holdout PASS
4. Paper tracking error vs BT

## Не в scope

- Live (`DRY_RUN=false`)
- Оптимізація на спаленому iter7 OOS
