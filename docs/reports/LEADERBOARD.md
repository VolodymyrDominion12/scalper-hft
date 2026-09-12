# Лідерборд стратегій scalper-hft

Оновлено: автоматично з `results/` · Топ-30 комірок

## Пороги гейту (directional)

- OOS Sharpe > 0.3
- OOS позитивних вікон ≥ 50%
- DSR > 0.95
- PBO < 0.5

## Тієри

- **✅ Validated (paper)**: 1 комірок
- **🟢 Paper-ready**: 0 комірок
- **🟡 Monitoring**: 3 комірок
- **🔵 Candidate**: 10 комірок
- **⛔ Rejected**: 59 комірок

## Топ комірок

|   # | Тіер                 | Стратегія      | Інструмент                  | ТФ   | OOS SR   | OOS+   | Port SR   | t_NW   | Вердикт    | Нотатки                                               |
|----:|:---------------------|:---------------|:----------------------------|:-----|:---------|:-------|:----------|:-------|:-----------|:------------------------------------------------------|
|   1 | ✅ Validated (paper) | pairs_arb      | LINK/BTC                    | 1h   | +0.006   | 65%    | —         | —      | PASS       | VALIDATED_PAIRS; regime_scale=0.25; paper-v0.2.0 gate |
|   2 | 🟡 Monitoring        | ts_momentum    | PORTFOLIO_long_only_lb20    | 1d   | +0.126   | 73%    | +1.36     | +2.61  | MONITORING | smooth=1, short=False, n_sym=15                       |
|   3 | 🟡 Monitoring        | ts_momentum    | PORTFOLIO_core15_long_1d    | 1d   | +0.126   | 73%    | +1.36     | +2.61  | MONITORING | H0_replication; short=False; n=15/15 1d               |
|   4 | 🟡 Monitoring        | ts_momentum    | PORTFOLIO_core15_long_4h    | 4h   | +0.116   | 73%    | +1.01     | +2.49  | MONITORING | H3; short=False; n=15/15 4h                           |
|   5 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_smooth3_long_lb20 | 1d   | +0.164   | 87%    | +1.36     | +2.67  | CANDIDATE  | smooth=3, short=False, n_sym=15                       |
|   6 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_smooth3_top10     | 1d   | +0.276   | 90%    | +1.10     | +2.49  | CANDIDATE  | smooth=3, short=True, n_sym=10                        |
|   7 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_full45_long_1d    | 1d   | +0.037   | 60%    | +1.14     | +2.28  | CANDIDATE  | H2; short=False; n=42/45 1d                           |
|   8 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_top10_lb20        | 1d   | +0.212   | 90%    | +1.01     | +2.30  | CANDIDATE  | smooth=1, short=True, n_sym=10                        |
|   9 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_smooth3_lb20      | 1d   | +0.195   | 80%    | +0.82     | +2.01  | CANDIDATE  | smooth=3, short=True, n_sym=15                        |
|  10 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_baseline_lb20     | 1d   | +0.137   | 80%    | +0.76     | +1.87  | CANDIDATE  | smooth=1, short=True, n_sym=15                        |
|  11 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_new30_long_1d     | 1d   | -0.012   | 52%    | +0.84     | +1.70  | CANDIDATE  | H1; short=False; n=27/30 1d                           |
|  12 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_15                | 1d   | —        | —      | +0.76     | +1.87  | CANDIDATE  | портфель 15 символів; variant=lb20                    |
|  13 | 🔵 Candidate         | ts_momentum    | ADAUSDT                     | 1d   | +0.256   | 50%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  14 | 🔵 Candidate         | cross_momentum | ADAUSDT                     | 1d   | +0.256   | 50%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  15 | ⛔ Rejected          | ts_momentum    | PORTFOLIO_core15_ls_4h      | 4h   | +0.019   | 60%    | +0.32     | +0.88  | REJECTED   | H3_control; short=True; n=15/15 4h                    |
|  16 | ⛔ Rejected          | ts_momentum    | PORTFOLIO_new30_ls_1d       | 1d   | -0.038   | 37%    | +0.18     | +0.46  | REJECTED   | H1_control; short=True; n=27/30 1d                    |
|  17 | ⛔ Rejected          | supertrend     | ETHUSDT                     | 1h   | +0.076   | 52%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  18 | ⛔ Rejected          | supertrend     | BNBUSDT                     | 4h   | +0.075   | 43%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  19 | ⛔ Rejected          | cross_momentum | AVAXUSDT                    | 1d   | +0.059   | 50%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  20 | ⛔ Rejected          | ts_momentum    | DOTUSDT                     | 1d   | +0.065   | 75%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  21 | ⛔ Rejected          | cross_momentum | DOTUSDT                     | 1d   | +0.065   | 75%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  22 | ⛔ Rejected          | cross_momentum | DOGEUSDT                    | 1d   | +0.035   | 38%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  23 | ⛔ Rejected          | supertrend     | BTCUSDT                     | 4h   | +0.119   | 44%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  24 | ⛔ Rejected          | cross_momentum | ADAUSDT                     | 4h   | -0.027   | 59%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  25 | ⛔ Rejected          | ts_momentum    | AVAXUSDT                    | 1d   | +0.059   | 50%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  26 | ⛔ Rejected          | cross_momentum | SOLUSDT                     | 1d   | -0.055   | 50%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  27 | ⛔ Rejected          | supertrend     | LINKUSDT                    | 4h   | +0.062   | 37%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  28 | ⛔ Rejected          | supertrend     | BNBUSDT                     | 1h   | +0.029   | 50%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  29 | ⛔ Rejected          | supertrend     | SOLUSDT                     | 4h   | +0.041   | 40%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |
|  30 | ⛔ Rejected          | ts_momentum    | DOGEUSDT                    | 1d   | +0.035   | 38%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                              |

## Рекомендації для paper

1. **pairs_arb LINK/BTC 1h** — єдиний validated; paper на `paper-v0.2.0`.
2. **ts_momentum 1d PORTFOLIO_long_only_lb20** — monitoring (Port Sharpe +1.36, t_NW +2.61); paper-моніторинг поруч із pairs, не live.
3. **ts_momentum 1d PORTFOLIO_core15_long_1d** — monitoring (Port Sharpe +1.36, t_NW +2.61); paper-моніторинг поруч із pairs, не live.
4. **ts_momentum 4h PORTFOLIO_core15_long_4h** — monitoring (Port Sharpe +1.01, t_NW +2.49); paper-моніторинг поруч із pairs, не live.
5. Directional single-symbol комірки **не проходять** гейт 0.3 — диверсифікація обов'язкова.
