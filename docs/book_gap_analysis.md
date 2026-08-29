# Конспект слабких/відсутніх шарів за книгою «Inside the Black Box» (R. Narang, 2-ге вид.)

Аналіз: шар → що в книзі → що в проєкті → план імплементації (формули/кроки) → складність → пріоритет.

---

## 1. Transaction Cost Modeling (гл. 5, с. 79–91)

**Книга.** Витрати = комісії + slippage + market impact. Slippage — функція латенсі, волатильності та тренду (momentum страждає більше, mean-reversion може отримувати *негативний* slippage). Impact — функція розміру ордера відносно ліквідності. Чотири типи моделей: flat, linear, piecewise-linear, quadratic. Моделі мають бути **емпіричними й per-instrument**, оновлюватись із власних виконань; мета — не мінімізувати витрати, а **інформувати** portfolio construction про поріг, який має перекрити альфа (недооцінка → «кровотеча» від надмірного обороту; стратегія з win 55% і прибутком $0.01/угода гине при витратах $0.01).

**Проєкт.** `CostModel` (backtest/execution.py) — flat: константні maker 0.02% / taker 0.05%, константний slippage 2 bps, `impact_frac = 0.0` (жорстко зашито «для retail на Binance вплив нехтовний»). Слабко:
- **немає market impact** взагалі (0.0) — навіть для пар/дельта-нейтральних, де ноціонал більший;
- slippage не залежить від волатильності/спреду/латенсі — порушує головний висновок гл. 5;
- немає **timing risk / opportunity cost** (ціна рухається, поки ордер «працюється»);
- немає емпіричної калібровки з реальних філів (fills.py пише логи, але вони не зворотно-зв'язані в модель).

**План (складність M, пріоритет ВИСОКИЙ — скальпінг живе/вмирає на витратах):**
1. **Slippage як функція волатильності:** `slippage_t = k_spread · spread_t/2 + k_vol · σ_t · √τ`, де `σ_t` — реалізована волатильність бару (з `features/indicators`), `τ` — очікувана затримка виконання (для maker ≈ 0). Калібрувати `k_*` регресією на фактичних філлах з paper_runner (філл vs mid у момент рішення).
2. **Market impact (квадратний корінь):** `impact = k_imp · σ_t · √(Q / ADV_daily)` (Square-Root Law) або Almgren-Chriss `impact = η · |Q|^{3/5}` + `spread/2`. Для Binance оцінювати `k_imp` на **bookTicker/L2-даних** (`record-bookticker` вже є!): для maker-ордера impact ≈ 0, для taker = частка від з'їдання глибини. Мінімально: `impact_frac(Q) = a·Q^b`, параметри `a,b` підганяються під кумулятивну глибину стакана.
3. **Opportunity/timing risk:** до round-trip додавати `σ_t·√T_hold` як ціну зволікання (потрібно для рішення «слайсити чи ні»).
4. **Форма моделі:** перейти від flat до **piecewise-linear** (або quadratic при великих Q): `cost(Q) = fee + slippage + (Q < Q_break ? m1·Q : m1·Q_break + m2·(Q−Q_break))`.
5. **Per-symbol словник** параметрів (`BTCUSDT`, `ETHUSDT`, `SOLUSDT`, дрібні альти) + періодичний перерахунок зі свіжих даних (книга: «empirical models evolve»).
6. Використати `breakeven_move_pct` як **поріг входу** в стратегіях (зараз це лише property — не входить у генерацію сигналів): сигнал = 0, якщо `|expected_move| < breakeven`.

---

## 2. Portfolio Construction (гл. 6, с. 93–115; blending у гл. 3, с. 56–62)

**Книга.** Дві родини: rule-based (equal position, equal risk, alpha-driven, decision-tree) та оптимізатори (mean-variance, constrained, Black-Litterman, Grinold–Kahn факторні портфелі, Resampled Efficiency). Входи: очікувана дохідність (з альфи), волатильність (GARCH/історична), кореляційна матриця (нестабільна в часі: S&P/Nikkei 0.01→0.66). Уроки: equal-weight контролює ризик хвостових/помилкових сигналів (adverse selection у mean-reversion — «nickels in front of a steamroller»); alpha-driven дає найбільші позиції на піку тренду; **turnover tax** — витрати входять у цільовий портфель, ребаланс лише інкрементальний («trade the gap»); substitution effect — дорогий актив замінюється корельованим дешевим. Grinold–Kahn: портфелі факторів (кожен = 1 альфа) потім оптимізуються (10–20 «інструментів»). Blending (гл. 3): linear/equal-risk/умовний/ML; «змішувати портфелі, не сигнали».

**Проєкт.** `pairs_portfolio.py` — тільки **equal weighting** пар; решта стратегій — одиночні позиції незалежно; `ensemble.py` — equal-weight mean/vote. Слабко:
- **немає equal-risk / risk-parity** алокації (гл. 6: вага ∝ 1/σ);
- **немає кореляційного аналізу стратегій** (гл. 9: «Relationship with Other Strategies» — value-added тест «з і без нової ідеї»);
- немає turnover tax / порогу ребалансу — портфель не перераховується «на різницю»;
- немає оптимізатора (навіть простого constrained mean-variance; Грінольд-Кана — ідеальний для 12 стратегій);
- blended алокація не враховує, що mean_reversion + cvd_momentum + ob_imbalance на одному символі сильно корельовані.

**План (складність M, пріоритет ВИСОКИЙ):**
1. **Матриця кореляцій прибутковостей стратегій** (годинні/денні повернення equity з бектестів; `pairs_portfolio` вже обчислює спільний індекс — узагальнити на всі стратегії). Звіт: теплова карта + середня попарна кореляція (ціль < 0.3).
2. **Equal Risk Contribution (ERC):** ваги `w_i ∝ 1/σ_i` з ітеративною нормалізацією (або risk-parity через оптимізацію `min Σ_i Σ_j w_i w_j Σ_ij − λ Σ ln w_i`). Це крок 1 до risk parity без повного оптимізатора.
3. **Turnover tax:** при побудові портфеля враховувати `cost_model.cost(Q) × |Δw|`; ребалансувати лише коли `Σ|Δw_i| > threshold` (напр. 0.15) — книга: «trade the gap».
4. **Grinold–Kahn факторний портфель:** кожна стратегія = «фактор», робимо її часовий ряд повернень, потім constrained mean-variance на 12 «інструментах» (scipy.optimize, обмеження: ваги ≥ 0, Σ=1, позиційний кап ліміт з risk-моделі). Це і є «як об'єднати поточні стратегії».
5. **Умовний ансамбль (гл. 3):** замість mean/vote — regime-gated: у високій волатильності вага momentum зростає, у флеті — mean-reversion (вже є `features/regimes.py`, бракує зв'язку з вагами).
6. **Value-added тест** нової стратегії: Sharpe(портфель+нова) vs Sharpe(портфель) — не просто кореляція (гл. 9).

---

## 3. Risk Management (гл. 4, с. 67–78; гл. 10, с. 175–195)

**Книга.** Ризик — *не* уникнення збитків, а **усвідомлений вибір і розмір експозицій**. Обмеження розміру: hard constraints vs **penalty-функції** (винятки для сильних сигналів). Вимірювання: волатильність + **dispersion/кореляція**. Рівні: позиція → група → **леверидж портфеля** (VaR; критика: «замало ризику в нормі, забагато в бурю»; альтернатива — **Kelly criterion**, на практиці half-Kelly через серійну кореляцію). Обмеження типів: theory-driven (market/sector/size — «beta-подібні» експозиції, за які не платять) та empirical (PCA; ризик spurious факторів, відставання при зміні режиму). Гл. 10: **model risk**, **regime change risk** (структурні злами відносин — SCHW/MER), **exogenous shock**, **contagion/crowding** (серпень 2007). Моніторинг (с. 193–195): exposure, P&L (не тільки скільки, а *як* заробляє), execution (fill rate, slippage), systems.

**Проєкт.** `trader.py` risk_check: position_pct, daily_loss_limit, max_consecutive_losses, max_open_positions — hard constraints на рівні стратегії/символу; `regimes.py` — волатильність/тренд/фандінг-фільтри. Слабко:
- **немає портфельного рівня**: ризик-бюджет між стратегіями, сумарна експозиція, агрегований VaR;
- немає penalty-функцій замість жорстких лімітів (сильний сигнал не може перевищити ліміт);
- немає **stress-testing / сценаріїв** (крипто-краш −30%, ліквідність-криза, фандінг-шок, BTC-домінанс-зсув);
- немає dispersion/PCA-аналізу експозицій (усі стратегії на 3 символах — фактично 1–2 фактори);
- немає live-моніторингу у сенсі гл. 10 (є Telegram-сповіщення, але не «exposure/P&L dashboard» з аномаліями);
- немає Kelly-оцінки розміру (у metrics.py є `risk_of_ruin`, але не використовується для розміру позиції).

**План (складність M, пріоритет ВИСОКИЙ):**
1. **Risk budget поверх стратегій:** сумарна волатильність портфеля = ціль (напр. 20% річних); кожна стратегія отримує capital slice = цільова волатильність/σ_i (vol targeting — «constant risk level» з гл. 4, але без леверидж-пастки: cap на леверидж).
2. **Penalty-функції:** замінити `if size > limit: block` на `size = limit · (1 − exp(−k·|signal|/limit))` — книга прямо описує цю форму.
3. **Stress-test модуль** (новий `scalper_hft/validation/stress.py`): реплей бектесту на модифікованих рядах — (a) краш: `close *= exp(−0.30·ramp)` на 2 дні з відновленням; (b) ліквідність-криза: спред ×10, slippage ×5, impact увімкнено; (c) фандінг-шок: funding_rate = −0.005..+0.005 (для carry/arb); (d) кореляційний шок: всі стратегії одночасно (contagion). Виходити: maxDD, втрати, чи спрацювали ліміти вчасно.
4. **PCA експозицій:** на поверненнях стратегій знайти 1–2 головні компоненти (усі 3 символи BTC/ETH/SOL — майже 1 фактор) і обмежити сумарне навантаження на PC1 (empirical risk model з гл. 4).
5. **Kelly для розміру:** `f = (p·b − q)/b` по історичних угодах, використовувати half-Kelly як верхню межу position_pct; пов'язати з наявним `risk_of_ruin`.
6. **Live-моніторинг:** скрипт `monitor.py` — годинні звіти: поточна експозиція/вага на символ, PnL день, hit-rate vs очікуваний, fill rate maker-ордерів, slippage реалізований vs модель — з алертами при відхиленні (гл. 10: «pattern in performance can alert to problems»).

---

## 4. Execution algorithms (гл. 7, с. 117–131; HFT: гл. 13–15)

**Книга.** Дві цілі: **completeness** і **cheapness**. Агресивність = спектр: market → lift offer → join → improve; вибір залежить від типу альфи (momentum → агресивно; mean-reversion → пасивно), сили сигналу, впевненості, micro-price (дисбаланс стакана). Пасивні ордери: adverse selection, rebates, queue position. **Великі ордери:** слайсинг 100k → 100×1k; розмір шматка визначається t-cost моделлю. **Вимірювання якості:** mid-market, VWAP (обережно: власний обсяг зміщує бенчмарк), IS («рішення → фактичне виконання»). Micro-price: bid 10k vs ask 1 → fair ближче до bid. Iceberg/hidden — приховати footprint.

**Проєкт.** `_submit_order`: бінарний maker (limit+postOnly) або market; у бектесті — векторизоване виконання з бару t+1. Слабко:
- **немає слайсингу** (VWAP/TWAP/IS/POV) — для пар і дельта-нейтральних, де ноціонал 30–60% капіталу, taker-виконання на 1 бар може з'їсти спред+impact;
- немає **micro-price** для maker-котирувань (є ob_imbalance-фіча, але не в execution);
- немає вимірювання якості виконання (slippage vs mid/бенчмарк) у live;
- maker fill-wait `MAKER_FILL_WAIT_BARS=1` — немає моделі ймовірності філу.

**План (складність S–L, пріоритет СЕРЕДНІЙ):**
1. **TWAP-слайсер** (S): для ордерів > X% ADV розбити на N шматків по T барів; виконувати через наявний maker/market шлях. Достатньо для пар.
2. **VWAP/POV** (M): розподіл обсягу ∝ профілю volume з історичних klines; POV: шматок = цільова частка виконаного ринкового обсягу.
3. **Micro-price для maker** (M): fair = mid + (bid_size−ask_size)/(bid_size+ask_size)·spread/2; зсувати котирування до fair. Дані L2 вже записуються (`record-bookticker`).
4. **Implementation Shortfall** (S): у live логувати `price_vs_mid_at_decision` — щоденний звіт slippage maker/taker окремо, порівняння з CostModel (зворотний зв'язок у п.1 розділу 1).
5. Для чистого скальпінгу 1-контрактними ордерами слайсинг не потрібен (гл. 7: «якщо частота низька — інфраструктура overkill») — лише для portfolio/pairs-виконань.

---

## 5. Alpha-таксономія (гл. 3, с. 23–65)

**Книга.** Шість класів: trend/momentum, mean reversion, technical sentiment, value/yield, growth, quality (+ data-driven/ML). Розрізнення імплементації: forecast target, time horizon (короткий → більша диференціація), bet structure (directional vs relative), universe, run frequency. Mean-reversion: «надає ліквідність, платять за ризик adverse selection». Value/yield у ф'ючерсах: **roll yield** (backwardation/contango — прямо переноситься на крипто-базис). Carry = margin of safety. Technical sentiment: форма стакана, обсяг, open interest. Data-driven: HFT віддає перевагу емпіриці на коротких горизонтах (більше даних → більше статистичної сили), але ризик spurious patterns.

**Проєкт.** Покрито: mean_reversion, cvd_momentum, ob_imbalance (sentiment/стакан), funding_carry (yield), funding_arb (delta-neutral carry), basis_reversion (roll-yield-подібний), pairs_arb (relative), market_maker, ml_strategy, ensemble. Слабко/відсутньо:
- **cross-sectional momentum/ranking** (momentum зараз — один символ, «time-series»; немає ранжування BTC vs ETH vs SOL і довгих/коротких хвостів — аналог QLS з гл. 3);
- **seasonality/time-of-day** як standalone альфа (є `session_filter`, але як фільтр, не як сигнал: годинні/день-тижня ефекти);
- **technical sentiment через volume profile / put-call аналог** (у крипті: funding rate як «sentiment» вже є; можна додати open interest delta);
- **довгостроковий trend** на вищих ТФ (5m/1h) — зараз усе 1m-орієнтоване, а гл. 3: trend на довших, reversion на коротших горизонтах співіснують;
- **value через базис/фандінг-спред між символами** (basis_reversion є, але «relative value» між перпами й ф'ючерсами різних експірацій можна розширити).

**План (складність S–M, пріоритет СЕРЕДНІЙ):**
1. **Cross-sectional momentum** (S): на кожному барі ранжувати N символів за momentum (ret_5m/ret_1h), позиції: top-квантиль long, bottom — short (уже є інфраструктура декількох символів). Класичний доповнювач до time-series momentum.
2. **Time-of-day/seasonality сигнал** (S): середній годинний прибуток по символу за історію → сигнал на найбільш/найменш сприятливі години; перетнути з наявним session_filter.
3. **Open Interest / ліквідації** (M): Binance надає openInterest/liquidation дані — sentiment-альфа «crowding» (contagion-ризик гл. 10 як сигнал).
4. **Data-driven на тіках** (M): ml_strategy на klines; розширити фічами з aggTrades (buy/sell volume imbalance, крупні угоди) — книга прямо радить емпірику на коротких горизонтах.

---

## 6. Research-процес, пастки бектесту, моніторинг (гл. 9, с. 147–171; гл. 10)

**Книга (гл. 9).** Науковий метод + **falsification**. Джерела ідей: спостереження, академлітература, міграція, дискреційні трейдери. Міри моделі: cumulative profits, avg return, variability/lumpiness, **max drawdown + час відновлення**, R² (хороший OOS ≈ 0.02–0.05; >0.15 = помилка), **quintile/monotonicity** (повернення монотонно зростають із сигналом), win rate, Sharpe/IR/Sterling/Calmar/Omega, **value-added vs портфель** (не просто кореляція), **time decay** (затримка входу 1–5 днів), **sensitivity** (плато, не пік — «Choose B, not D»). Пастки: **overfitting** (парсимонія/Occam), рідкі угоди, **look-ahead / burning data** (OOS стає IS після повторного використання), припущення про **витрати** (надто низькі → ілюзія; win 55/45 з $0.01 гине), hard-to-borrow аналог. **Гл. 10**: моніторинг exposure/P&L/execution/systems.

**Проєкт.** Сильний: walk-forward, purged CV, CPCV, Deflated Sharpe, sensitivity, Optuna, тест no-lookahead, комісії обов'язкові, docs/reports/ markdown-звіти. Слабко:
- **немає quintile/monotonicity study** (гл. 9 — ключовий тест «чи не випадково»);
- **немає time-decay тесту** (затримка входу на 1–N барів — критично для 1m-скальпінгу: чи вмирає альфа за 1 бар?);
- **немає value-added тесту** нової стратегії проти портфеля (тільки індивідуальні звіти);
- немає **stress/regime-change** перевірки на бектесті (див. п.3 розділу 3);
- live-моніторинг — лише Telegram; немає систематичного exposure/P&L/execution монітора з алертами на відхилення;
- звіти — «на вимогу», не щоденні (гл. 10: intraday pattern — рання ознака проблеми);
- немає фіксації «burned» OOS даних (прозорість: які сегменти вже використані для відбору параметрів).

**План (складність S–M, пріоритет ВИСОКИЙ для 1–2, СЕРЕДНІЙ для решти):**
1. **Quintile study** (S): розбити сигнали на 5 квінтилів, порівняти середні forward-повернення; вимагати монотонність (Spearman ρ). Додати у `cmd_report`.
2. **Time-decay** (S): прибуток при вході з лагом 0,1,2,3 бари — якщо падає швидко, стратегія залежить від швидкості виконання (переоцінка у векторизованому бектесті!).
3. **Value-added тест** (M): портфель усіх стратегій ± нова; ΔSharpe, ΔmaxDD.
4. **Щоденний `monitor.py`** (M): equity стратегій, експозиція, hit rate, fill rate, slippage vs модель; алерти при |z|>3 (див. п.6 розділу 3).
5. **Реєстр OOS** (S): `docs/reports/oos_usage.md` — які діапазони вже «спалені» відбором параметрів; заборона повторного використання.

---

## Зведена таблиця

| Шар/Підхід | Розділ | Стан у проєкті | Складність | Пріоритет |
|---|---|---|---|---|
| Slippage як функція вол-ті/спреду | гл. 5 | слабко (flat, 2 bps const) | S–M | Високий |
| Market impact (√-law / Almgren-Chriss, калібровка з L2) | гл. 5 | немає (impact=0) | M | Високий |
| Timing/opportunity cost | гл. 5 | немає | S | Середній |
| Piecewise-linear/quadratic t-cost, per-symbol | гл. 5 | немає (flat) | S | Середній |
| Breakeven як поріг входу в сигнал | гл. 5 | слабко (property лише) | S | Високий |
| Equal-risk / ERC алокація стратегій | гл. 6 | слабко (тільки equal-weight пар) | M | Високий |
| Кореляція стратегій + value-added тест | гл. 6, 9 | немає | S–M | Високий |
| Turnover tax / поріг ребалансу | гл. 6 | немає | S | Середній |
| Grinold–Kahn факторний оптимізатор | гл. 6 | немає | M–L | Середній |
| Умовний/regime-gated ансамбль | гл. 3 | слабко (mean/vote) | S | Середній |
| Risk budget / vol-targeting на портфелі | гл. 4 | немає | M | Високий |
| Penalty-функції замість hard limits | гл. 4 | немає (hard only) | S | Середній |
| Stress-test: краш, ліквідність-криза, фандінг-шок, contagion | гл. 4, 10 | немає | M | Високий |
| PCA-експозиції / dispersion | гл. 4 | немає | M | Середній |
| Kelly / half-Kelly розмір ставки | гл. 4 | слабко (risk_of_ruin є) | S | Середній |
| TWAP/VWAP/POV-слайсинг для великих ордерів | гл. 7 | немає | S–M | Середній |
| Micro-price для maker-котирувань | гл. 7, 15 | немає | M | Середній |
| IS-бенчмарк виконання (реалізований slippage) | гл. 7 | немає | S | Високий |
| Cross-sectional momentum (ранжування символів) | гл. 3 | немає | S | Середній |
| Time-of-day/seasonality альфа | гл. 3 | слабко (фільтр, не сигнал) | S | Низький |
| Open Interest / ліквідації як sentiment | гл. 3, 10 | немає | M | Низький |
| Quintile/monotonicity тест | гл. 9 | немає | S | Високий |
| Time-decay тест (лаг входу) | гл. 9 | немає | S | Високий |
| Щоденний моніторинг P&L/експозиції/виконання | гл. 10 | слабко (Telegram тільки) | M | Високий |
| Реєстр «спаленого» OOS (burning data) | гл. 9 | немає | S | Середній |

**Підсумок.** Архітектура за книгою правильна і покриває всі 5 шарів, але реалізація кожного — «перша версія»: CostModel flat, Portfolio Construction = equal-weight, Risk = hard limits, Execution = бінарний maker/market. Найбільший вплив на скальпінг дадуть: (1) емпіричний t-cost з impact і breakeven-порогом у сигналах; (2) ERC/кореляційна алокація стратегій з turnover tax; (3) stress-test модуль і портфельний risk budget; (4) IS-вимірювання виконання зі зворотним зв'язком у CostModel. Усі — середньої складності, без зміни ядра архітектури.
