# Iteration 1 — WF-порівняння regime-сутності vs singles (1h)

- days=365, train=2500, test=500, symbols=['XRPUSDT', 'AVAXUSDT', 'UNIUSDT', 'LINKUSDT', 'AAVEUSDT', 'ADAUSDT']
- children sets: official=mean_reversion,supertrend,hmm_reversion; split=supertrend,cross_momentum,stoch_rsi; is_maker=True, без overlay

| variant | avg OOS SR | pos win frac | n_trades OOS | per-symbol OOS SR |
|---|---|---|---|---|
| single:stoch_rsi | -0.2853 | 38% | 645 | AAVEUSDT:-0.033 | ADAUSDT:-0.357 | AVAXUSDT:-0.327 | LINKUSDT:-0.319 | UNIUSDT:-0.022 | XRPUSDT:-0.653 |
| single:supertrend | -0.0270 | 53% | 311 | AAVEUSDT:+0.141 | ADAUSDT:-0.319 | AVAXUSDT:-0.267 | LINKUSDT:+0.141 | UNIUSDT:+0.051 | XRPUSDT:+0.091 |
| sup:official:best_prior:d0 | -0.2403 | 44% | 489 | AAVEUSDT:-0.467 | ADAUSDT:-0.322 | AVAXUSDT:-0.365 | LINKUSDT:-0.081 | UNIUSDT:+0.162 | XRPUSDT:-0.368 |
| sup:official:best_prior:d3 | -0.2046 | 39% | 455 | AAVEUSDT:-0.525 | ADAUSDT:-0.175 | AVAXUSDT:-0.087 | LINKUSDT:-0.009 | UNIUSDT:-0.045 | XRPUSDT:-0.386 |
| sup:official:contextual_hedge:d0 | -0.2599 | 43% | 6251 | AAVEUSDT:-0.215 | ADAUSDT:-0.549 | AVAXUSDT:+0.042 | LINKUSDT:-0.322 | UNIUSDT:-0.325 | XRPUSDT:-0.192 |
| sup:official:contextual_hedge:d3 | -0.3574 | 40% | 6247 | AAVEUSDT:-0.521 | ADAUSDT:-0.315 | AVAXUSDT:-0.393 | LINKUSDT:-0.339 | UNIUSDT:-0.233 | XRPUSDT:-0.344 |
| sup:official:regime_soft:d0 | -0.2298 | 44% | 940 | AAVEUSDT:-0.363 | ADAUSDT:-0.375 | AVAXUSDT:-0.350 | LINKUSDT:-0.062 | UNIUSDT:+0.076 | XRPUSDT:-0.305 |
| sup:official:regime_soft:d3 | -0.1876 | 44% | 855 | AAVEUSDT:-0.382 | ADAUSDT:-0.243 | AVAXUSDT:-0.132 | LINKUSDT:+0.032 | UNIUSDT:-0.080 | XRPUSDT:-0.320 |
| sup:split:best_prior:d0 | -0.3545 | 32% | 728 | AAVEUSDT:-0.554 | ADAUSDT:-0.653 | AVAXUSDT:-0.281 | LINKUSDT:-0.185 | UNIUSDT:+0.080 | XRPUSDT:-0.532 |
| sup:split:best_prior:d3 | -0.3100 | 39% | 682 | AAVEUSDT:-0.540 | ADAUSDT:-0.438 | AVAXUSDT:-0.165 | LINKUSDT:-0.047 | UNIUSDT:-0.074 | XRPUSDT:-0.597 |
| sup:split:contextual_hedge:d0 | -0.4582 | 32% | 23184 | AAVEUSDT:-0.552 | ADAUSDT:-0.558 | AVAXUSDT:-0.260 | LINKUSDT:-0.427 | UNIUSDT:-0.198 | XRPUSDT:-0.754 |
| sup:split:contextual_hedge:d3 | -0.2513 | 36% | 23190 | AAVEUSDT:-0.207 | ADAUSDT:-0.462 | AVAXUSDT:-0.026 | LINKUSDT:-0.253 | UNIUSDT:-0.150 | XRPUSDT:-0.409 |
| sup:split:regime_soft:d0 | -0.1985 | 46% | 6407 | AAVEUSDT:-0.171 | ADAUSDT:-0.604 | AVAXUSDT:+0.002 | LINKUSDT:-0.142 | UNIUSDT:+0.035 | XRPUSDT:-0.312 |
| sup:split:regime_soft:d3 | -0.1744 | 43% | 6251 | AAVEUSDT:-0.143 | ADAUSDT:-0.445 | AVAXUSDT:+0.046 | LINKUSDT:-0.158 | UNIUSDT:-0.054 | XRPUSDT:-0.293 |
