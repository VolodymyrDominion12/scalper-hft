# iter12 — pre-registration: two-stage vol-target overlay + value-added

Дата: 2026-09-12 · Цикл: improvement-loop (після iter11).

## Мотивація

iter11 виявив, що `ts_momentum` 1d long-only CORE_15 — monitoring (Port Sharpe
+1.36, t_NW +2.61, CI [+0.49, +2.25]). Vol-target overlay (σ-вікно=20) підняв
портфель до +1.03…+1.09 з t_NW 2.05…2.54, **але позначено пост-хок** (overlay
обрано після перегляду метрик) → не monitoring. Також не тестовано, чи додає
`ts_momentum` вартості поверх валідованого ядра `pairs_arb LINK/BTC 1h` (W1-VA).

## Дані

Перевикористовується вже спалений OOS `ts_momentum PORTFOLIO_CORE15 1d`
(2019-11-08 → 2026-09-12, iter11). Нового OOS-снупінгу немає — це two-stage
protocol на наявному OOS, де validation-половина була невидима під час selection.

## Гіпотези (зафіксовано ДО прогону)

### H12-A: two-stage vol-target overlay (de-post-hoc-ифікація)

- **Сітка σ-вікон**: `{20, 40, 60}` днів. Три значення, не підганяємо.
- **Selection half**: 2019-01-01 → 2022-12-31. Обираю вікно з максимальним
  Sharpe(vol-targeted portfolio) на цій половині.
- **Validation half**: 2023-01-01 → 2026-12-31. Застосовую обране вікно.
- **Рішення (заздалегідь)**: overlay → **monitoring** IFF на validation-half
  `Sharpe(vt) > Sharpe(plain)` **AND** `t_NW(vt) ≥ 2.0`. Інакше → overlay лишається
  пост-хок/відхиляється. Жодного іншого порогу не підбираю після.
- **Параметри альфи заморожені**: lookback=20, smooth=1, long-only, maker
  (ти самі, що в iter11).

### H12-B: value-added (W1-VA) — pairs_arb ⊕ ts_momentum

- `pairs_arb LINK/BTC 1h maker` (regime_scale=0.25, validated) → equity →
  daily returns.
- `ts_momentum 1d CORE_15 long-only` → portfolio daily returns.
- Комбінація 50/50 risk-equal (vol-normalized) на спільному span.
- **Метрики**: combined Sharpe, maxDD, Calmar vs standalone pairs та standalone ts.
- **Рішення**: якщо combined Sharpe > max(standalone) AND combined maxDD <
  standalone maxDD → фіксуємо диверсифікацію у звіті (не paper-gate; це
  portfolio-construction evidence, не нова альфа).

### H12-C: per-symbol attribution

- Для `ts_momentum 1d CORE_15`: contribution = mean(symbol_oos) / port_std,
  ранг символів, найгірший/найкращий. Діагностика, не гейт.

## Що НЕ роблю

- Не підбираю lookback/smooth/імена після метрик.
- Не додаю нові символи в paper (H1 iter11 FAIL).
- Не котирую single-symbol directional cells (всі <0.3 gate).
- Не запускаю live.
