# Iteration 4 — 1d trend 5y WF (maker)

- days=1900, train=700, test=150, symbols=['AAVEUSDT', 'AVAXUSDT', 'DOGEUSDT', 'ETHUSDT', 'LINKUSDT', 'SOLUSDT', 'UNIUSDT', 'XRPUSDT']

| variant | avg OOS SR | pos frac | trades | per-symbol OOS |
|---|---|---|---|---|
| single:cross_momentum | -0.7343 | 28% | 1015 | AAVEUSDT:-1.15 | AVAXUSDT:-0.77 | DOGEUSDT:-0.65 | ETHUSDT:-0.28 | LINKUSDT:-1.04 | SOLUSDT:-0.36 | UNIUSDT:-0.91 | XRPUSDT:-0.72 |
| single:supertrend:lo | -0.1262 | 19% | 40 | AAVEUSDT:-0.39 | AVAXUSDT:-0.36 | DOGEUSDT:-0.03 | ETHUSDT:+0.17 | LINKUSDT:-0.05 | SOLUSDT:-0.24 | UNIUSDT:-0.16 | XRPUSDT:+0.04 |
| single:supertrend:ls | -0.1510 | 28% | 141 | AAVEUSDT:-0.68 | AVAXUSDT:-0.02 | DOGEUSDT:-0.14 | ETHUSDT:+0.10 | LINKUSDT:-0.01 | SOLUSDT:-0.24 | UNIUSDT:-0.37 | XRPUSDT:+0.16 |
| single:supertrend:ls:mult2 | -0.2923 | 31% | 242 | AAVEUSDT:-0.90 | AVAXUSDT:-0.40 | DOGEUSDT:+0.15 | ETHUSDT:-0.34 | LINKUSDT:+0.01 | SOLUSDT:-0.28 | UNIUSDT:-0.71 | XRPUSDT:+0.14 |
