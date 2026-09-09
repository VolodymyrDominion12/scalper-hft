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
