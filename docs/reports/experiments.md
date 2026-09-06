# Каталог експериментів (val → OOS)

OOS запускається лише якщо val = IMPROVED і trades ≥ 20. Після OOS вікно
записується в [oos_usage.md](oos_usage.md). Одна зміна на рядок.

| id | strategy | hypothesis | val_start | val_end | test_start | test_end | baseline |
|---|---|---|---|---|---|---|---|
| sparse-basket-v1 | sparse_basket | Lasso/PCA кошик 5+ монет vs pairs | 2025-01-01 | 2025-12-31 | 2026-01-01 | 2026-06-30 | - |
| ml-strategy-v1 | ml_strategy | LightGBM meta-label + OOD veto | 2025-01-01 | 2025-12-31 | 2026-01-01 | 2026-06-30 | - |
