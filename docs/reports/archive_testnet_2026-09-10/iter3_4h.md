# Iteration 3 — WF 4h maker (3y база)

- days=1000, train=2000, test=500, symbols=['AAVEUSDT', 'LINKUSDT', 'XRPUSDT', 'SOLUSDT', 'AVAXUSDT', 'UNIUSDT', 'DOGEUSDT', 'ETHUSDT']

| variant | avg OOS SR | pos frac | trades | per-symbol OOS |
|---|---|---|---|---|
| single:cross_momentum | -1.4777 | 20% | 3095 | AAVEUSDT:-1.00 | AVAXUSDT:-1.66 | DOGEUSDT:-2.03 | ETHUSDT:-0.43 | LINKUSDT:-1.77 | SOLUSDT:-1.88 | UNIUSDT:-1.10 | XRPUSDT:-1.94 |
| single:smc_fvg | -0.3937 | 38% | 1282 | AAVEUSDT:-0.49 | AVAXUSDT:-1.00 | DOGEUSDT:-0.28 | ETHUSDT:-0.06 | LINKUSDT:-0.51 | SOLUSDT:+0.01 | UNIUSDT:-0.48 | XRPUSDT:-0.35 |
| single:stoch_rsi | -0.2359 | 38% | 330 | AAVEUSDT:-0.49 | AVAXUSDT:-0.75 | DOGEUSDT:-0.28 | ETHUSDT:+0.06 | LINKUSDT:-0.24 | SOLUSDT:+0.06 | UNIUSDT:+0.18 | XRPUSDT:-0.43 |
| single:supertrend:lo | -0.2439 | 36% | 241 | AAVEUSDT:-0.61 | AVAXUSDT:-0.35 | DOGEUSDT:-0.02 | ETHUSDT:+0.07 | LINKUSDT:-0.09 | SOLUSDT:+0.21 | UNIUSDT:-0.80 | XRPUSDT:-0.35 |
| single:supertrend:ls | -0.5498 | 31% | 670 | AAVEUSDT:-0.82 | AVAXUSDT:-1.18 | DOGEUSDT:-0.22 | ETHUSDT:+0.08 | LINKUSDT:-0.66 | SOLUSDT:-0.19 | UNIUSDT:-1.15 | XRPUSDT:-0.27 |
| sup:split:best_prior | -0.3565 | 31% | 533 | AAVEUSDT:-0.66 | AVAXUSDT:-0.62 | DOGEUSDT:-0.33 | ETHUSDT:-0.31 | LINKUSDT:-0.55 | SOLUSDT:+0.08 | UNIUSDT:-0.29 | XRPUSDT:-0.17 |
| sup:split:regime_soft | -1.2469 | 20% | 5246 | AAVEUSDT:-1.10 | AVAXUSDT:-1.62 | DOGEUSDT:-1.77 | ETHUSDT:-0.23 | LINKUSDT:-1.64 | SOLUSDT:-1.20 | UNIUSDT:-0.94 | XRPUSDT:-1.48 |
