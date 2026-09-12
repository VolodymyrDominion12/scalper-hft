# Лідерборд стратегій scalper-hft

Оновлено: автоматично з `results/` · Топ-30 комірок

## Пороги гейту (directional)

- OOS Sharpe > 0.3
- OOS позитивних вікон ≥ 50%
- DSR > 0.95
- PBO < 0.5

## Тієри

- **✅ Validated (paper)**: 2 комірок
- **🟢 Paper-ready**: 1 комірок
- **🟡 Monitoring**: 11 комірок
- **🔵 Candidate**: 2 комірок
- **⛔ Rejected**: 60 комірок

## Топ комірок

|   # | Тіер                 | Стратегія      | Інструмент                  | ТФ   | OOS SR   | OOS+   | Port SR   | t_NW   | Вердикт    | Нотатки                                               |
|----:|:---------------------|:---------------|:----------------------------|:-----|:---------|:-------|:----------|:-------|:-----------|:------------------------------------------------------|
|   1 | ✅ Validated (paper) | pairs_arb      | LINK/BTC                    | 1h   | +0.006   | 65%    | —         | —      | PASS       | VALIDATED_PAIRS; regime_scale=0.25; paper-v0.2.0 gate |
|   2 | ✅ Validated (paper) | pairs_arb      | LINKUSDT/BTCUSDT            | 1h   | —        | —      | —         | —      | PASS       | audit; DSR>0.95, OOS>0.3                              |
|   3 | 🟢 Paper-ready       | pairs_arb      | LINKUSDT                    | 1h   | —        | —      | —         | —      | PASS       | audit; DSR>0.95, OOS>0.3                              |
|   4 | 🟡 Monitoring        | ts_momentum    | PORTFOLIO_smooth3_long_lb20 | 1d   | +0.164   | 87%    | +1.36     | +2.67  | monitoring | smooth=3, short=False, n_sym=15                       |
|   5 | 🟡 Monitoring        | ts_momentum    | PORTFOLIO_long_only_lb20    | 1d   | +0.126   | 73%    | +1.36     | +2.61  | monitoring | smooth=1, short=False, n_sym=15                       |
|   6 | 🟡 Monitoring        | ts_momentum    | PORTFOLIO_smooth3_top10     | 1d   | +0.276   | 90%    | +1.10     | +2.49  | monitoring | smooth=3, short=True, n_sym=10                        |
|   7 | 🟡 Monitoring        | ts_momentum    | PORTFOLIO_top10_lb20        | 1d   | +0.212   | 90%    | +1.01     | +2.30  | monitoring | smooth=1, short=True, n_sym=10                        |
|   8 | 🟡 Monitoring        | ts_momentum    | PORTFOLIO_smooth3_lb20      | 1d   | +0.195   | 80%    | +0.82     | +2.01  | monitoring | smooth=3, short=True, n_sym=15                        |
|   9 | 🟡 Monitoring        | ts_momentum    | PORTFOLIO_baseline_lb20     | 1d   | +0.137   | 80%    | +0.76     | +1.87  | monitoring | smooth=1, short=True, n_sym=15                        |
|  10 | 🟡 Monitoring        | ts_momentum    | PORTFOLIO_15                | 1d   | —        | —      | +0.76     | +1.87  | MONITORING | портфель 15 символів; variant=lb20                    |
|  11 | 🟡 Monitoring        | ts_momentum    | PORTFOLIO_15                | 1d   | —        | —      | +0.76     | +1.87  | MONITORING | портфель 15 символів; variant=vw120                   |
|  12 | 🟡 Monitoring        | ts_momentum    | PORTFOLIO_15                | 1d   | —        | —      | +0.76     | +1.87  | MONITORING | портфель 15 символів; variant=vw20                    |
|  13 | 🟡 Monitoring        | ts_momentum    | PORTFOLIO_15                | 1d   | —        | —      | +0.76     | +1.87  | MONITORING | портфель 15 символів; variant=vw40                    |
|  14 | 🟡 Monitoring        | ts_momentum    | PORTFOLIO_15                | 1d   | —        | —      | +0.76     | +1.87  | MONITORING | портфель 15 символів; variant=vw90                    |
|  15 | 🔵 Candidate         | ts_momentum    | ADAUSDT                     | 1d   | +0.256   | 50%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  16 | 🔵 Candidate         | cross_momentum | ADAUSDT                     | 1d   | +0.256   | 50%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  17 | ⛔ Rejected          | ts_momentum    | PORTFOLIO_15                | 1d   | —        | —      | +0.56     | +1.45  | CANDIDATE  | портфель 15 символів; variant=lb30                    |
|  18 | ⛔ Rejected          | ts_momentum    | PORTFOLIO_15                | 1d   | —        | —      | +0.55     | +1.50  | CANDIDATE  | портфель 15 символів; variant=lb40                    |
|  19 | ⛔ Rejected          | ts_momentum    | PORTFOLIO_15                | 1d   | —        | —      | +0.51     | +1.32  | CANDIDATE  | портфель 15 символів; variant=lb60                    |
|  20 | ⛔ Rejected          | supertrend     | ETHUSDT                     | 1h   | +0.076   | 52%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  21 | ⛔ Rejected          | supertrend     | BNBUSDT                     | 4h   | +0.075   | 43%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  22 | ⛔ Rejected          | cross_momentum | AVAXUSDT                    | 1d   | +0.059   | 50%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  23 | ⛔ Rejected          | ts_momentum    | DOTUSDT                     | 1d   | +0.065   | 75%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  24 | ⛔ Rejected          | cross_momentum | DOTUSDT                     | 1d   | +0.065   | 75%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  25 | ⛔ Rejected          | cross_momentum | DOGEUSDT                    | 1d   | +0.035   | 38%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  26 | ⛔ Rejected          | supertrend     | BTCUSDT                     | 4h   | +0.119   | 44%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  27 | ⛔ Rejected          | cross_momentum | ADAUSDT                     | 4h   | -0.027   | 59%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  28 | ⛔ Rejected          | ts_momentum    | AVAXUSDT                    | 1d   | +0.059   | 50%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  29 | ⛔ Rejected          | cross_momentum | SOLUSDT                     | 1d   | -0.055   | 50%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  30 | ⛔ Rejected          | supertrend     | LINKUSDT                    | 4h   | +0.062   | 37%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |

## Рекомендації для paper

1. **pairs_arb LINK/BTC 1h** — єдиний validated; paper на `paper-v0.2.0`.
2. **ts_momentum 1d портфель** — monitoring (Sharpe +0.76, vol-target +1.09); потрібна pre-registration перед live.
3. Directional single-symbol комірки **не проходять** гейт 0.3 — диверсифікація обов'язкова.
