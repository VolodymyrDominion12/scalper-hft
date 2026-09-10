# Iteration 3 — WF 1d maker (3y база)

- days=1000, train=600, test=120, symbols=['AAVEUSDT', 'LINKUSDT', 'XRPUSDT', 'SOLUSDT', 'AVAXUSDT', 'UNIUSDT', 'DOGEUSDT', 'ETHUSDT']

| variant | avg OOS SR | pos frac | trades | per-symbol OOS |
|---|---|---|---|---|
| single:cross_momentum | +0.1906 | 58% | 264 | AAVEUSDT:-0.59 | AVAXUSDT:+0.71 | DOGEUSDT:+0.41 | ETHUSDT:+0.52 | LINKUSDT:+0.35 | SOLUSDT:+0.75 | UNIUSDT:-0.05 | XRPUSDT:-0.57 |
| single:smc_fvg | -0.8102 | 25% | 116 | AAVEUSDT:-1.14 | AVAXUSDT:-0.43 | DOGEUSDT:-1.63 | ETHUSDT:-0.22 | LINKUSDT:-0.87 | SOLUSDT:-0.63 | UNIUSDT:-0.51 | XRPUSDT:-1.04 |
| single:stoch_rsi | -0.6312 | 4% | 19 | AAVEUSDT:-0.37 | AVAXUSDT:-0.52 | DOGEUSDT:-0.88 | ETHUSDT:-0.36 | LINKUSDT:-0.88 | SOLUSDT:-0.57 | UNIUSDT:-1.15 | XRPUSDT:-0.31 |
| single:supertrend:lo | -0.4064 | 12% | 16 | AAVEUSDT:+0.05 | AVAXUSDT:-0.29 | DOGEUSDT:-0.28 | ETHUSDT:-0.29 | LINKUSDT:-0.41 | SOLUSDT:-0.26 | UNIUSDT:-1.08 | XRPUSDT:-0.68 |
| single:supertrend:ls | +0.4971 | 71% | 51 | AAVEUSDT:+0.64 | AVAXUSDT:+0.90 | DOGEUSDT:+0.70 | ETHUSDT:+0.81 | LINKUSDT:+0.59 | SOLUSDT:+0.19 | UNIUSDT:-0.30 | XRPUSDT:+0.45 |
| sup:split:best_prior | -0.6338 | 17% | 31 | AAVEUSDT:-0.37 | AVAXUSDT:-0.03 | DOGEUSDT:-1.14 | ETHUSDT:-0.27 | LINKUSDT:-0.56 | SOLUSDT:-1.03 | UNIUSDT:-1.19 | XRPUSDT:-0.48 |
| sup:split:regime_soft | +0.0225 | 54% | 383 | AAVEUSDT:-0.48 | AVAXUSDT:+0.83 | DOGEUSDT:-0.45 | ETHUSDT:+0.37 | LINKUSDT:+0.18 | SOLUSDT:+0.54 | UNIUSDT:-0.44 | XRPUSDT:-0.37 |
