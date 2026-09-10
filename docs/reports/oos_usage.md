# Спалений OOS

Реєстр діапазонів, уже використаних для відбору параметрів (Narang гл. 9).
Не крутити Optuna / sensitivity повторно на тих самих вікнах.

| strategy | symbol | start | end | purpose |
|---|---|---|---|---|
| pairs_arb | XRPUSDT/BTCUSDT | 2025-01-01 | 2026-08-28 | walk-forward + DSR validation |
| pairs_arb | LINKUSDT/BTCUSDT | 2025-01-01 | 2026-08-28 | walk-forward + DSR validation |
| pairs_arb | BTCUSDT/ETHUSDT | 2025-01-01 | 2026-08-28 | walk-forward + DSR validation |
| mean_reversion | BTCUSDT | 2024-01-01 | 2024-01-05 | sweep/wf:1m |
| mean_reversion | BTCUSDT | 2025-01-01 | 2025-01-02 | sweep/wf:1m |
| mean_reversion | BTCUSDT | 2024-03-26 | 2024-06-01 | sweep/wf:4h |
| ensemble | BTCUSDT | 2024-09-09 | 2026-09-09 | audit_cell/cscv |
| mean_reversion | MOCKUSDT | 2025-01-01 | 2025-01-05 | audit_cell |
| ensemble | SOLUSDT | 2024-09-09 | 2026-09-09 | audit_cell/cscv |
| ensemble | ETHUSDT | 2024-09-09 | 2026-09-09 | audit_cell/cscv |
| ensemble | LINKUSDT | 2024-09-09 | 2026-09-09 | audit_cell/cscv |
| ensemble | LTCUSDT | 2024-09-09 | 2026-09-09 | audit_cell/cscv |
| ml_strategy | SOLUSDT | 2024-09-09 | 2026-09-09 | audit_cell/cscv |
| regime_supervisor | SOLUSDT | 2023-09-10 | 2026-09-09 | audit_cell/cscv |
| cvd_momentum | AAVEUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| basis_reversion | AAVEUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| ensemble | AAVEUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| funding_carry | AAVEUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| hmm_reversion | AAVEUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| market_maker | AAVEUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| mean_reversion | AAVEUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| ml_strategy | AAVEUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| ob_imbalance | AAVEUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| regime_supervisor | AAVEUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| smc_fvg | AAVEUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| stoch_rsi | AAVEUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| supertrend | AAVEUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| basis_reversion | ADAUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| cvd_momentum | ADAUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| ensemble | ADAUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| funding_carry | ADAUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| hmm_reversion | ADAUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| market_maker | ADAUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| mean_reversion | ADAUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| ml_strategy | ADAUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| ob_imbalance | ADAUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| regime_supervisor | ADAUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| smc_fvg | ADAUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| stoch_rsi | ADAUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| supertrend | ADAUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:15m |
| mean_reversion | ATOMUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:1h |
| ml_strategy | ATOMUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:1h |
| ob_imbalance | ATOMUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:1h |
| regime_supervisor | ATOMUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:1h |
| smc_fvg | ATOMUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:1h |
| stoch_rsi | ATOMUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:1h |
| supertrend | ATOMUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:1h |
| basis_reversion | ATOMUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:30m |
| cvd_momentum | ATOMUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:30m |
| ensemble | ATOMUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:30m |
| funding_carry | ATOMUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:30m |
| hmm_reversion | ATOMUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:30m |
| market_maker | ATOMUSDT | 2023-09-09 | 2026-09-08 | sweep/wf:30m |
