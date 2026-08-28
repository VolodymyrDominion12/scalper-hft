# TODO / Backlog (стан після ітерації 4)

## ✅ Зроблено в ітерації 4
- [x] **Paper-replay** (`paper-replay`): відтворення історії через risk-трейдера
      з funding-платежами та mark-to-market; звіт про риск-блокування
- [x] **Виправлено баг risk-шару**: ліміт позицій блокував ЗАКРИТТЯ (тепер закриття
      ніколи не блокується); пауза після серії збитків скидається щодня
- [x] **⚠ Виправлено баг обліку funding у бектесті**: ставка нараховувалася кожен
      бар замість 1 разу/8h (завищення ~480×) → funding_carry ПЕРЕАУДИТОВАНО і
      ВІДХИЛЕНО (деталі: docs/audit_findings.md)
- [x] **WeightedDepthImbalance** фіча (зважена глибина: harmonic/equal/linear + EMA):
      |imb|>0.5 знизився з 83% (top-of-book) до 46% (depth5)
- [x] **Deploy безперервного запису стакана**: scripts/record_loop.sh +
      deploy/scalper-record.service (systemd)
- [x] **GitHub Actions**: .github/workflows/weekly-audit.yml (щотижневий аудит +
      Telegram-звіт)
- [x] **Тести**: +2 (funding один раз на блок; daily reset паузи) — всього 15

## Дані (data)
- [ ] Запустити systemd-юніт scalper-record (накопичення bookTicker/depth5, дні/тижні)
- [ ] 1s/5s свічки для реалістичних maker-філів

## Стратегії (після чесного переаудиту)
- [ ] **Delta-neutral funding arb** (перп + спот, потребує спотового API) — єдиний
      кандидат, де фандінг реально покриває витрати (великий ноціонал, без цінового ризику)
- [ ] depth-weighted OB-стратегія після накопичення depth5 (WeightedDepthImbalance + EMA)
- [ ] funding_carry: переробити на delta-neutral або з ціновим фільтром (не шортити ралі)
- [ ] Maker з L2: калібрування adverse_sel_haircut на depth5

## Live
- [ ] Live-режим з валідними Binance-ключами (поточні ключі невалідні)
- [ ] Persistence угод у SQLite

## Інфраструктура
- [ ] Підключити репозиторій до GitHub і перевірити weekly-audit workflow
