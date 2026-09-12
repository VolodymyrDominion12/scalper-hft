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
- **🔵 Candidate**: 17 комірок
- **⛔ Rejected**: 61 комірок

## Топ комірок

|   # | Тіер                 | Стратегія      | Інструмент                   | ТФ    | OOS SR   | OOS+   | Port SR   | t_NW   | Вердикт    | Нотатки                                                      |
|----:|:---------------------|:---------------|:-----------------------------|:------|:---------|:-------|:----------|:-------|:-----------|:-------------------------------------------------------------|
|   1 | ✅ Validated (paper) | pairs_arb      | LINK/BTC                     | 1h    | +0.006   | 65%    | —         | —      | PASS       | VALIDATED_PAIRS; regime_scale=0.25; paper-v0.2.0 gate        |
|   2 | 🟡 Monitoring        | ts_momentum    | PORTFOLIO_long_only_lb20     | 1d    | +0.126   | 73%    | +1.36     | +2.61  | MONITORING | smooth=1, short=False, n_sym=15                              |
|   3 | 🟡 Monitoring        | ts_momentum    | PORTFOLIO_core15_long_1d     | 1d    | +0.126   | 73%    | +1.36     | +2.61  | MONITORING | H0_replication; short=False; n=15/15 1d                      |
|   4 | 🟡 Monitoring        | ts_momentum    | PORTFOLIO_core15_long_4h     | 4h    | +0.116   | 73%    | +1.01     | +2.49  | MONITORING | H3; short=False; n=15/15 4h                                  |
|   5 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_smooth3_long_lb20  | 1d    | +0.164   | 87%    | +1.36     | +2.67  | CANDIDATE  | smooth=3, short=False, n_sym=15                              |
|   6 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_combined_1w1d_lb13 | 1w+1d | +0.000   | —      | +1.51     | +2.61  | CANDIDATE  | H13-B value-added 50/50 1w⊕1d; corr +0.09; portfolio-constru |
|   7 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_smooth3_top10      | 1d    | +0.276   | 90%    | +1.10     | +2.49  | CANDIDATE  | smooth=3, short=True, n_sym=10                               |
|   8 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_full45_long_1d     | 1d    | +0.037   | 60%    | +1.14     | +2.28  | CANDIDATE  | H2; short=False; n=42/45 1d                                  |
|   9 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_top10_lb20         | 1d    | +0.212   | 90%    | +1.01     | +2.30  | CANDIDATE  | smooth=1, short=True, n_sym=10                               |
|  10 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_weekly_long_lb8    | 1w    | +0.000   | 87%    | +0.98     | +1.81  | CANDIDATE  | iter13 long-only; lookback=8w; n=15/15 1w                    |
|  11 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_weekly_long_lb13   | 1w    | +0.000   | 93%    | +0.96     | +1.85  | CANDIDATE  | iter13 long-only; lookback=13w; n=15/15 1w                   |
|  12 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_smooth3_lb20       | 1d    | +0.195   | 80%    | +0.82     | +2.01  | CANDIDATE  | smooth=3, short=True, n_sym=15                               |
|  13 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_weekly_long_lb4    | 1w    | +0.000   | 80%    | +0.93     | +1.73  | CANDIDATE  | iter13 long-only; lookback=4w; n=15/15 1w                    |
|  14 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_baseline_lb20      | 1d    | +0.137   | 80%    | +0.76     | +1.87  | CANDIDATE  | smooth=1, short=True, n_sym=15                               |
|  15 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_new30_long_1d      | 1d    | -0.012   | 52%    | +0.84     | +1.70  | CANDIDATE  | H1; short=False; n=27/30 1d                                  |
|  16 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_weekly_ls_lb4      | 1w    | +0.000   | 80%    | +0.75     | +1.52  | CANDIDATE  | iter13 long-short контроль; lookback=4w; n=15/15 1w          |
|  17 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_weekly_long_lb26   | 1w    | +0.000   | 60%    | +0.75     | +1.53  | CANDIDATE  | iter13 long-only; lookback=26w; n=15/15 1w                   |
|  18 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_15                 | 1d    | —        | —      | +0.76     | +1.87  | CANDIDATE  | портфель 15 символів; variant=lb20                           |
|  19 | 🔵 Candidate         | ts_momentum    | PORTFOLIO_weekly_ls_lb8      | 1w    | +0.000   | 80%    | +0.67     | +1.55  | CANDIDATE  | iter13 long-short контроль; lookback=8w; n=15/15 1w          |
|  20 | 🔵 Candidate         | ts_momentum    | ADAUSDT                      | 1d    | +0.256   | 50%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                                     |
|  21 | 🔵 Candidate         | cross_momentum | ADAUSDT                      | 1d    | +0.256   | 50%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                                     |
|  22 | ⛔ Rejected          | ts_momentum    | PORTFOLIO_weekly_ls_lb13     | 1w    | +0.000   | 80%    | +0.61     | +1.32  | REJECTED   | iter13 long-short контроль; lookback=13w; n=15/15 1w         |
|  23 | ⛔ Rejected          | ts_momentum    | PORTFOLIO_weekly_ls_lb26     | 1w    | +0.000   | 67%    | +0.38     | +0.78  | REJECTED   | iter13 long-short контроль; lookback=26w; n=15/15 1w         |
|  24 | ⛔ Rejected          | ts_momentum    | PORTFOLIO_core15_ls_4h       | 4h    | +0.019   | 60%    | +0.32     | +0.88  | REJECTED   | H3_control; short=True; n=15/15 4h                           |
|  25 | ⛔ Rejected          | ts_momentum    | PORTFOLIO_new30_ls_1d        | 1d    | -0.038   | 37%    | +0.18     | +0.46  | REJECTED   | H1_control; short=True; n=27/30 1d                           |
|  26 | ⛔ Rejected          | supertrend     | ETHUSDT                      | 1h    | +0.076   | 52%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                                     |
|  27 | ⛔ Rejected          | supertrend     | BNBUSDT                      | 4h    | +0.075   | 43%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                                     |
|  28 | ⛔ Rejected          | cross_momentum | AVAXUSDT                     | 1d    | +0.059   | 50%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                                     |
|  29 | ⛔ Rejected          | ts_momentum    | DOTUSDT                      | 1d    | +0.065   | 75%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                                     |
|  30 | ⛔ Rejected          | cross_momentum | DOTUSDT                      | 1d    | +0.065   | 75%    | —         | —      | FAIL       | audit; DSR>0.95, OOS>0.3                                     |

## Рекомендації для paper

1. **pairs_arb LINK/BTC 1h** — єдиний validated; paper на `paper-v0.2.0`.
2. **ts_momentum 1d PORTFOLIO_long_only_lb20** — monitoring (Port Sharpe +1.36, t_NW +2.61); paper-моніторинг поруч із pairs, не live.
3. **ts_momentum 1d PORTFOLIO_core15_long_1d** — monitoring (Port Sharpe +1.36, t_NW +2.61); paper-моніторинг поруч із pairs, не live.
4. **ts_momentum 4h PORTFOLIO_core15_long_4h** — monitoring (Port Sharpe +1.01, t_NW +2.49); paper-моніторинг поруч із pairs, не live.
5. Directional single-symbol комірки **не проходять** гейт 0.3 — диверсифікація обов'язкова.
