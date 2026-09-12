# Regime supervisor: як зробити перемикання за режимом робочим — огляд літератури та практики

Дата: 2026-09 · Контекст: crypto futures (Binance USDT-M), 15 символів, 4h/1d · Обсяг: академія + practitioner-джерела, застосовані до результатів `iter7` цього проєкту.

## 0. Короткий висновок (TL;DR)

1. Перемикання стратегій за режимом **не є edge саме по собі** — це форма таймінгу, і кожен перемикач платить за (а) помилкові сигнали, (б) лаг детекції, (в) комісії за turnover, (г) скорочення вибірки на клітинку. Після виправлення 1-барного lookahead у цьому проєкті перемикач 1h/10 символів дав чесний net Sharpe ≈ **+0.37**, **DSR ≈ 0**, **PBO ≈ 0.5** — тобто класичний результат «працює in-sample, зникає OOS».
2. Що реально підтверджується даними: (а) **зниження експозиції за волатильністю** (vol targeting) дає стабільний приріст Sharpe без жодної детекції режимів; (б) **персистентність** важливіша за точність детекції — jump-моделі з штрафом за перехід обганяють HMM саме завдяки персистентності ([Shu, Yu, Mulvey 2024](https://arxiv.org/abs/2402.05272), [Nystrup et al. 2017](https://backend.orbit.dtu.dk/ws/files/139272081/Dynamic_Portfolio_Optimization_Across_Hidden_Market_Regimes_ACCEPTED.pdf)); (в) **бінарний risk-on/risk-off гейт з гістерезисом і асиметричними порогами** працює краще за багатоклітинковий argmax; (г) **meta-labeling на рівні окремої угоди** обходить потребу в стабільній мітці режиму взагалі.
3. Головний вбивця regime switching — не «поганий режимний індикатор», а **кількість вільних ступенів волі**: 3 режими × 5 стратегій × пороги детектора = десятки-сотні конфігурацій, а [Bailey et al.](https://www.carmamaths.org/resources/jon/backtest.pdf) показують, що на 5 роках даних безпечно пробувати **не більше ~45 незалежних конфігурацій**, на 2 роках — **~7**.
4. Рекомендація для 4h/1d на 15 символах: **не перемикати стратегії**, а масштабувати експозицію (vol targeting) + бінарний ризик-гейт + meta-labeling як фільтр на валідованих рукавах (`pairs_arb` LINK/BTC maker, `ts_momentum` daily long-only). Деталі — розділ 6.

---

## 1. Методи детекції режимів

Для кожного методу нижче: **що міряє · каузальність · типовий лаг · failure modes**. Формули — у псевдокоді, без реалізації.

### 1.1 Rule-based trend/vol (те, що вже є в проєкті)

- **Що міряє.** Структуру (`range|trend_up|trend_down`) через нормовану відстань EMA та vol-стан (`low|normal|high`) через процентиль реалізованої волатильності — див. `scalper_hft/features/regimes.py` (`market_structure`, `volatility_regime`, `htf_market_structure`).
- **Каузальність.** Повністю онлайн: EMA і rolling percentile на барі `t` використовують лише `≤ t`. Це єдина перевага, яка завжди переважує «розумніші» моделі в production.
- **Лаг.** Структурна частина — це фактично EMA(9/50) на робочому ТФ: лаг ≈ `ema_slow/2` барів, тобто ~25 барів для 1h і ~25 діб для 1d. Коментар у `htf_market_structure` про «структуру на робочому ТФ, що запізнюється» — це той самий lag-феномен, лише перенесений на старший ТФ.
- **Failure modes.** (а) **Порогове перенавчання**: `trend_threshold=0.35` виглядає як «нуль параметрів», але це 1 ступінь вільності × кількість комбінацій вікон vol/EMA; у [практичному огляді regime-фільтрів](https://www.alphanume.com/blog/what-is-a-market-regime-filter) прямо названо, що «choice of threshold is the area where overfitting most commonly enters». (б) **Whipsaw** у невизначених ринках: перемикання range↔trend на шумі. (в) **Проциклічність**: `trend_up` визначається після того, як рух уже стався — гейт починає дозволяти лонги ближче до локального максимуму.

### 1.2 HMM: **filtered vs smoothed** — головна пастка

- **Що міряє.** Латентний стан `S_t` з Марківською динамікою; кожен стан має власні mean/vol. Для фінансів це найпопулярніший метод, бо дає ймовірності, а не жорсткі мітки.
- **Каузальність — розрізняти три режими інференсу:**
  - **filtered**: `P(S_t | x_1..x_t)` — дані лише до `t`. Формально: «estimate of the probability at time *t* based on data up to and including time *t* (but excluding time *t+1,...,T*)» — [statsmodels, Markov switching](https://www.statsmodels.org/stable/examples/notebooks/generated/markov_autoregression.html).
  - **smoothed**: `P(S_t | x_1..x_T)` — «estimate of the probability at time *t* using all the data in the sample» ([там само](https://www.statsmodels.org/stable/examples/notebooks/generated/markov_autoregression.html)). У production це **lookahead**: мітка бару `t` знає майбутнє.
  - **Viterbi decode**: «finds the most likely sequence of states, given all [observations]» ([hmmlearn API](https://hmmlearn.readthedocs.io/en/latest/api.html)) — глобально оптимальний шлях, тобто **офлайн**-алгоритм. Використовувати його для торгової мітки без re-run у кожен бар = той самий lookahead, що знайшов аудит K1 в `iter7`.
- **Лаг.** Навіть коректний filtered-інференс має лаг = 1/hazard: чим вища ймовірність переходу в матриці переходів, тим пізніше (і тим частіше хибно) детектується зміна. Практика [Shu et al.](https://arxiv.org/abs/2402.05272) — **одноденний** лаг між рішенням і виконанням (сигнал дня `t` діє з `t+2`), і вони окремо тестують лаги 1/5/10 днів.
- **Failure modes.** (а) Оцінювання параметрів на вибірці, що включає тест → **in-sample мітки**; (б) виродження станів (один стан на 2 барах); (в) нестабільність transition matrix між вікнами; (г) «smoothing bias» у звітах — саме так з'явився фальшивий Sharpe +2.64 у `iter7`.

### 1.3 Change-point detection: CUSUM, BOCPD, ruptures

- **Що міряє.** Момент **структурного зламу** параметрів (найчастіше рівня/волатильності), а не «стан». CUSUM накопичує нормовані відхилення і сигналить, коли сума перевищує поріг `h` ([CUSUM, Wikipedia](https://en.wikipedia.org/wiki/CUSUM)); BOCPD — байєсівська онлайн-модель run-length з hazard-функцією ([Bayesian online changepoint detection](https://en.wikipedia.org/wiki/Bayesian_online_changepoint_detection)); бібліотека `ruptures` дає і офлайн (Pelt/BinSeg), і віконні/онлайн варіанти ([ruptures docs](https://centre-borelli.github.io/ruptures-docs/)).
- **Каузальність.** CUSUM і BOCPD — онлайн за конструкцією (дивляться лише назад). Офлайн-алгоритми `ruptures` (Pelt, Dynp) — **не каузальні**: вони оптимально розбивають весь ряд.
- **Лаг.** CUSUM: лаг `≈ h / (Δμ/σ)` барів — прямо пропорційний порогу й обернено пропорційний величині зламу. Для 4h-барів і зламу 1σ при `h=5` це ~5 барів ≈ 20 годин.
- **Failure modes.** (а) Детекція підтверджується лише **після** зламу; (б) CUSUM чутливий до тренду в даних (false alarms на дрейфі волатильності); (в) для торгівлі важливо не «коли зламалось», а «скільки ще триватиме» — change point цього не дає.

### 1.4 Volatility / turbulence / absorption ratio

- **Kritzman turbulence.** Узагальнення Mahalanobis-відстані: `d_t = (r_t − μ̂)·Σ̂⁻¹·(r_t − μ̂)ᵀ` для вектора доходностей усього універсу. Практична реалізація з rolling-коваріацією описана в [R-bloggers, FAJ turbulence](https://www.r-bloggers.com/2011/04/great-faj-article-on-statistical-measure-of-financial-turbulence/) та [частина 2](https://www.r-bloggers.com/2011/04/great-faj-article-on-statistical-measure-of-financial-turbulence-part-2/); там же прямо сказано, що повновибіркова коваріація — це «hindsight», і для реального часу треба rolling-оцінку.
- **Absorption ratio (AR).** Частка сукупної дисперсії, пояснена фіксованим числом власних векторів: «absorption ratio, which equals the fraction of the total variance of a set of asset returns explained or "absorbed" by a fixed number of eigenvectors» ([Kritzman, Li, Page, Rigobon](http://boston.qwafafew.org/wp-content/uploads/sites/3/2017/01/revere-1.pdf)). У роботі: коваріація на **500-денному** вікні, кількість власних векторів ≈ **1/5** числа активів; стандартизований зсув `ΔAR` = зміна AR / σ(AR за рік), плюс `AR15day` vs `AR1year`. Медіанний зсув починає зростати **~за 40 днів** до початку турбулентного епізоду.
- **Каузальність.** Турбулентність і AR каузальні за умови rolling-вікон і каузальної нормалізації (процентиль/σ **лише з минулого**). Порушення: `μ̂, Σ̂` з усього семплу або процентиль, порахований на повному ряді, — типова помилка.
- **Failure modes.** (а) AR і турбулентність — **співпадаючі** (coincident) з турбулентністю, не випереджальні для торгівлі по ціні; (б) на 15 крипто-символах перша власна компонента ≈ «крипто-бета» — AR міряє майже те саме, що BTC-vol; (в) розмірність: при `N=15` і вікні 500 оцінка Σ̂ шумна, `Σ̂⁻¹` нестабільна → регуляризація (Ledoit-Wolf/shrinkage) обов'язкова; (г) квадратична форма чутлива до важких хвостів — у крипті це норма, тому краще ранк/процентиль, ніж абсолютний рівень.

### 1.5 Correlation / dispersion режими

- **Що міряє.** Середню попарну кореляцію або частку дисперсії в PC1 (по суті AR з `k=1`). Високий AR / висока середня кореляція = «один фактор домінує» → диверсифікація ламається, alpha-стратегії з різних ринків рухаються разом.
- **Каузальність.** Онлайн за умови rolling-оцінки; каузальний перерахунок PC на кожному барі дорогий — на практиці беруть PCA раз на `k` барів.
- **Лаг.** Кореляція — **найповільніша** з ознак: на 4h-барах вікно 30 днів = 180 барів, лаг ≈ половина вікна. Для перемикання стратегій це майже завжди запізно.
- **Failure modes.** Кореляція нестаціонарна і «стрибає» саме в стресі (те, що потрібно вловити); оцінка на малому вікні волатильна; високий AR може тривати місяцями, тому гейт тримає позицію вимкненою довго і з'їдає весь upside.

### 1.6 Markov-switching моделі та statistical jump models

- **MarkovRegression / MarkovAutoregression** ([statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_regression.MarkovRegression.html)) — регресія зі зміною параметрів за станами; дає `filtered_*` та `smoothed_*` ймовірності, тобто обидва варіанти інференсу в одному API.
- **Statistical jump model (JM)** — k-means з **штрафом за кожен перехід** (`λ`), який задає пріор персистентності: «the jump penalty moderates the frequency of state transitions, and more intrinsically relates to the tradeoff between accuracy and latency» ([Shu et al.](https://arxiv.org/abs/2402.05272)). Це найважливіша ідея для практики: **точність і лаг — це один і той самий регулятор**, і його треба підбирати під торгову задачу, а не під статистичний критерій. У роботі `λ=50`, тренувальне вікно 3000 днів, переоцінка раз на 6 місяців, онлайн-інференс через DP на trailing-вікні 3000 днів (щоб стабілізувати мітки).

**Таблиця зведення**

| Метод | Що міряє | Каузальність | Лаг (4h / 1d) | Головний failure mode |
|---|---|---|---|---|
| Rule-based EMA+vol | локальна структура, vol-перцентиль | повністю онлайн | ~25 барів (EMA50/2) | порогове перенавчання, whipsaw |
| HMM filtered | латентний стан, ймовірності | онлайн (`predict_proba`/filtered) | 1–10 барів + hazard | нестабільні transition-ймовірності |
| HMM smoothed / Viterbi | те саме, глобально оптимально | **офлайн = lookahead** | — | фальшивий in-sample Sharpe |
| CUSUM | момент зламу рівня/vol | онлайн | `h/(Δμ/σ)` барів | підтверджує постфактум |
| BOCPD | run-length, злам | онлайн | залежить від hazard | чутливість до пріора |
| Turbulence | Mahalanobis-дистанція універсу | онлайн при rolling Σ̂ | 0 (співпадаючий) | Σ̂⁻¹ нестабільна при N≈15 |
| Absorption ratio | частка дисперсії в PC1..k | онлайн при rolling | повільний (вікно 500 днів) | випереджає епізод на ~40 днів, але не дає точки входу |
| Jump model | стан + штраф за перехід | онлайн (DP на trailing-вікні) | регулюється `λ` | потрібен вибір `λ` під задачу |

*(синтез)* Практична ієрархія для 4h/1d crypto: rule-based vol-режим + hysteresis як **робочий baseline**, jump model — якщо хочеться статистичної моделі, і **ніколи** smoothed/Viterbi-мітки без re-run на кожному барі.

---

## 2. Чому перемикання за режимом падає out-of-sample

### 2.1 Хибні сигнали та whipsaw

Regime-фільтри «change frequently in indecisive markets; trading on each classification change generates transaction costs without benefit» ([Alphanume](https://www.alphanume.com/blog/what-is-a-market-regime-filter)). У цьому проєкті це видно буквально в цифрах: `meta:sup_ctx_hedge` і `meta:ens_hedge` дали **84 995** і **80 570** угод за 3 роки на 10 символах через щобарні дробові ваги — комісійний бюджет з'їдає сигнал.

### 2.2 Лаг детекції

Два різні лаги, які зазвичай плутають:
1. **Лаг оцінки** — режим визначається після руху (усі методи вище, окрім, частково, AR/турбулентності).
2. **Лаг реалізації** — від рішення до виконання. У [Shu et al.](https://arxiv.org/abs/2402.05272) це 1 день: сигнал дня `t` → позиція з `t+2`; рух далі тестують на лагах 5 і 10 днів і отримують **монотонну деградацію**. Будь-який бектест, де мітка бару `t` вибирає дохідність бару `t`, — це нульовий лаг, і результат такий самий недійсний, як у `iter7` до виправлення K1.

### 2.3 Нестабільність параметрів (parameter instability)

Параметри режимних моделей (transition matrix, means, vols) оцінюються на rolling-вікні; між вікнами вони «пливуть». У [Shu et al.](https://arxiv.org/abs/2402.05272) прямо названо причини: «limited sample sizes, unbalanced data, or high state persistence... non-normal and time-varying distributions». Наслідок: карта «режим → стратегія», обрана на фолді `k`, на фолді `k+1` може бути іншою. У `iter7` карта збіглася з найкращою у 2–3 з 3 режимів у 9 з 10 символів на H1→H2, але після фіксу лагу це дало лише +0.36.

### 2.4 Малі вибірки на клітинку та multiple testing

Це головна структурна проблема. При 3 режимах × 5 стратегій = 15 клітинок, кожна спостерігається лише в тій частці часу, коли режим активний (типово 20–40%). На 3 роках 1h-даних «ведмежий» режим може дати 2–4 тисячі барів → Sharpe кожної клітинки оцінений з похибкою, що перевищує сам ефект. Формальні запобіжники:

- **MinBTL** ([Bailey, Borwein, López de Prado, Zhu, «Pseudo-Mathematics and Financial Charlatanism»](https://www.carmamaths.org/resources/jon/backtest.pdf)): `MinBTL < 2·ln[N] / E[max_N]²` років. Наслідок у тексті: «if only 5 years of data are available, no more than 45 independent model configurations should be tried» (і «after trying only 7 independent strategy configurations, the expected maximum SR IS is 1 for a 2-year long backtest, while the expected SR OOS is 0»).
- **PBO/CSCV** ([Bailey, Borwein, López de Prado, Zhu](https://www.carmamaths.org/resources/jon/backtest2.pdf)) — ймовірність того, що конфігурація, оптимальна in-sample, виявиться нижчою за медіану OOS. У проєкті: **PBO 0.004 до фіксу** (артефакт — у пул кандидатів додавався сам lookahead-селектор) → **0.66 після фіксу**.
- **Deflated Sharpe (DSR)** ([Bailey & López de Prado](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)) — штраф за selection bias, число спроб і ненормальність. У проєкті DSR = **0.000** після фіксу.

*(синтез)* Практичний висновок: будь-яка схема, де число конфігурацій `≥ ~20` і горизонт даних `≤ 3 роки`, апріорі знаходиться в зоні, де очікуваний OOS Sharpe ≈ 0 навіть за нульового справжнього edge.

### 2.5 Вартість перемикання

Документовані цифри:
- [Nystrup et al.](https://backend.orbit.dtu.dk/ws/files/139272081/Dynamic_Portfolio_Optimization_Across_Hidden_Market_Regimes_ACCEPTED.pdf): MPC з 10 bps на транзакцію і **одноденним лагом** — Sharpe 0.56 при річному turnover **4.16**; додавання штрафу за торгівлю `ρ=0.02` знижує turnover до **1.17** і **підвищує** Sharpe до **0.63**. Тобто штраф за turnover не «зменшує прибуток», а покращує risk-adjusted результат. Там же: «the presence of transaction costs leads to a no-trade zone... the higher the transaction costs, the larger the no-trade zone».
- [Shu et al.](https://arxiv.org/abs/2402.05272): бінарна 0/1 стратегія на S&P з JM дає turnover лише **44%** на рік (купівля+продаж = 88%) при 10 bps one-way — «a relatively mild figure despite the large portfolio rebalancing that occurs with each regime shift».
- Binance USDT-M для нашого проєкту: maker 2 bps / taker 5 bps + slippage. Один round-trip перемикання на 15 символах = 2×(2..5) bps + slippage ≈ 5–15 bps на символ-перемикання.

*(синтез)* Оскільки crypto-перемикачі на 4h-барах схильні фліпати (через шум у vol-перцентилі), очікувана кількість перемикань на рік × вартість перемикання — це **перше число**, яке треба порахувати, а не Sharpe.

### 2.6 Мітки, підігнані на тих самих даних, і lookahead у побудові мітки

Три конкретні механізми:
1. **Оцінка моделі на всьому ряді** — усі параметри HMM/JM оцінені на даних, що включають тестовий період; мітка стає «передбаченням заднім числом».
2. **Smoothed/Viterbi-мітка на барі `t`** — мітка знає `t+1..T` ([statsmodels](https://www.statsmodels.org/stable/examples/notebooks/generated/markov_autoregression.html), [hmmlearn](https://hmmlearn.readthedocs.io/en/latest/api.html)).
3. **Пороги, пораховані на повному ряді** — навіть «статичний» квантиль (наприклад, 75-й процентиль bandwidth) стає lookahead-фічею; у [практичному розборі meta-labeling](https://www.mql5.com/en/articles/22755) це окремо виділено: «the bandwidth quantile threshold used in `bb_bw_regime` must be computed on the training window alone; computing it on the full dataset makes it a look-ahead feature, even though it appears static».

### 2.7 Циркулярність: мітка режиму містить PnL самої стратегії

Найковарніший механізм. Якщо режим визначається не зовнішніми даними, а через продуктивність стратегії (або якщо карта «режим → стратегія» вибирається на даних, де ці ж стратегії оцінювались), то «режим» стає переупаковкою PnL. [Alphanume](https://www.alphanume.com/blog/what-is-a-market-regime-filter) називає це прямо: «Some regime filters implicitly condition on the strategy's own historical performance, which is circular».

*(синтез)* Три інженерні тести на цю хворобу:
1. **Тест незалежності мітки**: побудувати мітку режиму **без** будь-яких доходностей стратегій (лише ціни/vol/кореляції/фандинг) і перевірити, чи карта «режим → стратегія» взагалі відтворюється.
2. **Тест на «перестановку стратегій»**: підмінити сигнали стратегій випадковими з тим самим turnover/експозицією — якщо селектор і далі «перемагає», він вимірює не те, що заявлено.
3. **Тест на виродження карти**: якщо в карті ≥50% капіталу в одному режимі йде одній стратегії, «режим» — просто dummy для «тримай X». У `iter7` саме це й сталося: `best_prior` вироджувався в «завжди перша універсальна стратегія».

### 2.8 Дрейф самих режимів (non-stationarity of the regime process)

Режими — не фізичні стани, а статистичні кластери, які перенумеровуються при кожному перенавчанні. У [Shu et al.](https://arxiv.org/abs/2402.05272) це показано як залежність оцінених mean/vol станів від вікна: «Figure 2/3 illustrates the evolution over time of the estimated conditional returns and volatilities for each regime from the rolling fit». *(синтез)* Наслідок: `state 0` у 2023 і `state 0` у 2026 — різні економічні явища; жорстка карта «номер стану → стратегія» ламається. Правильна практика — **упорядковувати стани за змістовною ознакою** (наприклад, за реалізованою vol), а не за індексом, і перебудовувати карту лише за змістовними ознаками.

---

## 3. Дизайн-принципи, які за даними допомагають

### 3.1 Гістерезис / min-dwell / confirmation delay

Механізм: новий режим приймається лише після `N` послідовних підтверджень (у проєкті вже є `apply_min_dwell`). Це прямо торгує точність на лаг — і саме цей trade-off описаний як центральний у [Shu et al.](https://arxiv.org/abs/2402.05272) («accuracy and latency»).
Емпірика: у [HMM_TR_Alg](https://raw.githubusercontent.com/Krishhiv/HMM_TR_Alg/main/README.md) min-dwell 2 дні «cutting HMM churn 68% and roughly halving the raw drawdown» — тобто безкоштовне зменшення просідання при майже незмінній дохідності. Параметр рівно один.

### 3.2 Dead-band / мінімальний розрив продуктивності перед перемиканням

Замість «перемикатись, коли A>B», перемикатись лише коли `SR_A − SR_B > δ` (δ — запас на похибку + вартість). Формальний аналог — **no-trade zone** з [Nystrup et al.](https://backend.orbit.dtu.dk/ws/files/139272081/Dynamic_Portfolio_Optimization_Across_Hidden_Market_Regimes_ACCEPTED.pdf), де оптимальна зона розширюється зі зростанням транзакційних витрат. *(синтез)* Економічна умова перемикання:

```
перемикатись, тільки якщо  E[ΔSharpe/період] × E[dwell] > round_trip_cost
```

Це єдиний «захист», який не потребує нових даних і не додає параметрів до торгової логіки (δ виводиться з cost, а не підбирається).

### 3.3 Soft / probability-weighted allocation замість argmax

- [Shu et al.](https://arxiv.org/abs/2402.05272) цитують результат Nystrup et al. (2016): «gradually adjusting the weight on the equity index as a linear function of the forecasted probability delivers similar out-of-sample performance compared to switching allocations between 100% and 0%». Тобто **плавне зважування не гірше за жорстке перемикання, але має менший turnover**.
- [Micro Alphas](https://microalphas.com/glossary/regime-detection/): «Signals conditioned on smoothed probabilities degrade more gracefully when regime detection is uncertain».
- ⚠️ Важливе застереження: у нашому `iter7` «soft» реалізації (`regime_soft`, `ens_mean`) провалились — але **не через soft-логіку**, а через хибні taxonomy-пріори + відсутність carry-рукава. М'яке зважування варто перевіряти з емпіричною картою, а не з ручними тегами.

### 3.4 Shrinkage оцінок «стратегія в режимі» до безумовного середнього

Проблема: Sharpe у режимі оцінений на малій вибірці; argmax по 15 клітинках майже завжди вибирає шум. Рішення — James–Stein / Bayes–Stein shrinkage: `μ̂ = α·μ̄ + β·μ_target`, де ціль — безумовне середнє або zero (див. [skfolio `ShrunkMu`](https://skfolio.org/_modules/skfolio/moments/expected_returns/_shrunk_mu.html), де реалізовано обидва методи і наведено обґрунтування через Stein 1955: «it's possible to find an estimator with reduced total error using shrinkage by trading a small bias against high variance»).
*(синтез)* Для режимної карти: `SR_shrunk(regime, strat) = (n/(n+k))·SR_obs + (k/(n+k))·SR_uncond`, де `k` — фіксований пріор (наприклад, `k = 200` барів). Це один параметр, який не оптимізується, а задається з міркувань похибки оцінки, і який **автоматично** дає «всередню стратегію» в рідкісних режимах.

### 3.5 Бінарний risk-on/risk-off гейт замість багатьох клітинок

- [Syntax Data](https://www.syntaxdata.com/research/why-binary-allocation-and-asymmetric-signals-matter): two-state framework «moves fully between equities and short-term U.S. Treasuries», з **асиметричними** порогами — довгостроковий тренд для виходу, середньостроковий для повернення. Обґрунтування: «avoids the complexity and potential instability associated with frequent incremental allocation changes».
- [Alphanume](https://www.alphanume.com/blog/what-is-a-market-regime-filter) серед випадків, коли фільтри **справді** додають: (1) структурна regime-залежність самої аномалії; (2) **risk-management overlay** («reducing leverage or exposure during high-volatility regimes is a risk-management decision that doesn't require timing alpha»); (3) capital allocation між рукавами.
- [Shu et al.](https://arxiv.org/abs/2402.05272) тестують саме бінарну 0/1-стратегію і показують, що вона покращує vol/MDD/ES і Sharpe.

*(синтез)* Ключове: бінарний гейт має **2 ступені вільності (два пороги + dwell)**, а не 15. Це і є основна економія на multiple testing.

### 3.6 Scaling експозиції (vol targeting) замість перемикання стратегій

Найсильніший доказовий блок:
- [Moreira & Muir, «Volatility-Managed Portfolios»](https://www.nber.org/system/files/working_papers/w22208/w22208.pdf): vol-timing дає «an alpha of 4.9%, an Appraisal ratio of 0.33, and an overall **25% increase** in the buy-and-hold Sharpe ratio» для ринкового портфеля; ефект стійкий до включення momentum-фактора.
- [Daniel & Moskowitz, «Momentum Crashes»](https://www.nber.org/system/files/working_papers/w20439/w20439.pdf): у ведмежих ринках momentum «behaves like written call options on the market» (бети минулих лузерів різко зростають), і «an implementable dynamic momentum strategy based on forecasts of momentum's mean and variance approximately **doubles** the alpha and Sharpe Ratio of a static momentum strategy». Важливо: **constant-vol** momentum гірший за динамічний, а хеджування ринкової дисперсії **не** відновлює прибутковість momentum у ведмежих ринках.
- [Shu et al.](https://arxiv.org/abs/2402.05272): JM-guided 0/1 дає зниження vol з 18.2% до 13.1% і кращий Calmar/Sharpe.

*(синтез)* Для нашого проєкту це найдешевший хід: vol targeting додається до вже валідованих рукавів (`pairs_arb`, `ts_momentum`) і не вимагає жодної мітки режиму — отже, не має multiple-testing-боргу.

### 3.7 Switch-cost-aware objective (оптимізувати net-of-cost, штрафувати turnover)

[Nystrup et al.](https://backend.orbit.dtu.dk/ws/files/139272081/Dynamic_Portfolio_Optimization_Across_Hidden_Market_Regimes_ACCEPTED.pdf) включають trading penalty прямо в цільову функцію і отримують turnover 4.16 → 1.17 при зростанні Sharpe. У цьому проєкті аналогічний параметр уже є (`turnover_penalty` у `ContextualHedgeBlend`/Exp3), але в `iter7` він не застосовувався до `regime_soft`/`best_prior` — це прямий незакритий важіль.

### 3.8 Обмеження складності політики

Правило: політика описується ≤ 3 параметрами і ≤ 2 станами. [Bailey et al.](https://www.carmamaths.org/resources/jon/backtest.pdf) дають кількісну стелю через MinBTL; [arXiv «From Hypotheses to Factors»](https://arxiv.org/abs/2604.26747) демонструє інженерний підхід — детермінований engine, який фіксує спліти, selection gates і транзакційні витрати, щоб «both successful and failed hypotheses» були audit-able.

### 3.9 Оцінювання OOS, де лаг застосований до **рішення**, а не до дохідності

Це виправлення K1 у проєкті. Правильна схема:

```
# НЕПРАВИЛЬНО (знайдено в iter7):
ret_used[t] = returns[ map(regime[t-1]) ][t-1]   # мітка і дохідність того самого бару

# ПРАВИЛЬНО:
sel[t] = map(regime[t-1]) if regime[t-1] stable else sel[t-1]
ret_used[t] = returns[ sel[t] ][t]               # рішення з t-1, дохідність t
```

Плюс явна модель реалізації: затримка детектора (`min_dwell`) **і** час виконання (t+1) — як у [Shu et al.](https://arxiv.org/abs/2402.05272) (сигнал `t` → позиція `t+2`) і [Nystrup et al.](https://backend.orbit.dtu.dk/ws/files/139272081/Dynamic_Portfolio_Optimization_Across_Hidden_Market_Regimes_ACCEPTED.pdf) (one-day delay).

### 3.10 Purged / embargoed CV, CPCV

Перекриття міток — мовчазний leak: «A label such as "return over the next twenty bars" overlaps the labels of its neighbours, so a shuffled split puts near-duplicates of a test point into training and the answer leaks» ([purgedcv paper](https://raw.githubusercontent.com/eslazarev/purged-cross-validation/refs/heads/main/paper/paper.md)). Там же: purging, embargo і CPCV походять з глав 7 і 12 AFML. Для нашого 4h/1d випадку: при triple-barrier з вертикальним бар'єром `H` барів — purge вікно `H`, embargo `1–2%` семплу. *(`purgedcv` як reference implementation, не рекомендація залежності.)*

### 3.11 Deflated Sharpe з чесним числом спроб + PBO/CSCV

- DSR ([Bailey & López de Prado](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)) вимагає **чесного `N`**. У `iter7` розрив між per-symbol (`N=27`) і портфельним (`N=270`) рахунком давав DSR 0.82 → 0.67; при `N=2700` — 0.44. Це і є «ціна брехні про N».
- PBO/CSCV ([Bailey et al.](https://www.carmamaths.org/resources/jon/backtest2.pdf)) — обов'язковий тест, бо «standard statistical techniques designed to prevent regression overfitting, such as hold-out, tend to be unreliable and inaccurate in the context of investment backtests».
*(синтез)* Практичне правило для проєкту: `N` = (кількість варіантів політики) × (кількість порогів) × (кількість рукавів), і цей `N` має бути зареєстрований **до** прогону.

---

## 4. Meta-labeling як альтернатива regime switching

### 4.1 Механізм

Двоетапна схема з AFML гл. 3: **primary model** визначає **сторону** (long/short), **secondary model** вирішує **чи брати ставку** і яким розміром. Формулювання з практичного розбору: «It separates two questions that the primary model conflates: in which direction should we trade, and should we trade at all? The primary model answers the first question. A secondary binary classifier answers the second» ([MQL5, Meta-Labeling the Classics](https://www.mql5.com/en/articles/22755)).

Псевдокод:

```
1. primary → events (entries) + side ∈ {−1,+1}
2. triple-barrier на кожній події: [profit_take, stop_loss, vertical_barrier] → label ∈ {−1,0,+1}
3. secondary labels = 1, якщо угода прибуткова (side-agnostic), інакше 0
4. secondary model: P(y=1 | x), x = features, доступні на закритті бару t−1
5. bet size = 2·Φ(Φ⁻¹(P)) − 1, дискретизований кроком; P < threshold → 0
```

### 4.2 Чому це часто обходить regime switching

*(синтез, що спирається на цитовані джерела)*

1. **Немає потреби в стабільній мітці режиму.** Модель вчиться на рівні **події** (entry), а не на рівні календарного стану. Їй байдуже, чи існує «режим» як персистентний об'єкт — важливо лише, чи конкретна подія схожа на прибуткову.
2. **Режимні фічі стають звичайними фічами.** Vol, trend strength, funding, breadth — входять у `x` і використовуються **лише там, де вони релевантні**, без окремої політики перемикання. У згаданому розборі `bb_bw_regime` (індикатор "trending vs ranging") виявився найважливішою фічею — тобто режимна інформація **все одно використовується**, але не як дискретний перемикач.
3. **Менша дисперсія рішення.** Замість «перемкнути весь капітал у режимі» — «пропустити 40% угод і зменшити розмір решти». Це і є soft weighting, але на правильному рівні гранулярності.
4. **Економія на multiple testing.** Замість 15 клітинок «режим × стратегія» — одна модель із набором фіч; число спроб падає на порядок.
5. **Прямий контроль витрат.** Bet sizing природно зменшує експозицію там, де впевненість низька, і тим самим скорочує turnover у поганих режимах.

### 4.3 Валідація без leakage (чекліст)

1. **Features `.shift()`.** «Features computed at bar t would include price and volatility information from bar t itself... Shifting by one bar enforces this constraint» ([MQL5](https://www.mql5.com/en/articles/22755)). Тест: перевірити кореляцію feature[t] з return[t].
2. **Пороги/квантилі — лише з train-вікна** (те саме джерело).
3. **Purged K-fold + embargo** для triple-barrier міток ([purgedcv](https://raw.githubusercontent.com/eslazarev/purged-cross-validation/refs/heads/main/paper/paper.md)).
4. **Calibration обов'язкова перед bet sizing**: «An uncalibrated GBM tends to produce overconfident probabilities clustered near 0 and 1, which leads to maximum sizing on most signals» — тобто без calibration bet sizing вироджується в on/off ([MQL5](https://www.mql5.com/en/articles/22755)).
5. **Метрики — precision/recall/f1, не accuracy**, бо «the downstream bet sizing amplifies the effect of both false positives... and false negatives» ([там само](https://www.mql5.com/en/articles/22755)).
6. **Concurrency correction**: одночасні події (кілька символів відкриваються на одному барі) → cap сумарної експозиції.
7. **Критерій прийняття**: secondary має перевершувати random на прибутковій половині подій primary; у наведеному прикладі (синтетика) trade count 300 → 140 (−53%) при зростанні win rate 50.2% → 56.8%.

### 4.4 Набір фіч, який працює (crypto 4h/1d, 15 символів)

| Група | Фічі | Джерело/обґрунтування |
|---|---|---|
| Volatility | realized vol (EWMA, кілька горизонтів), vol-percentile, downside deviation | [Shu et al.](https://arxiv.org/abs/2402.05272) — risk/return міри з return series; [Moreira & Muir](https://www.nber.org/system/files/working_papers/w22208/w22208.pdf) |
| Trend strength | нормована відстань EMA(9/50), ADX-подібна міра, нахил | [MQL5](https://www.mql5.com/en/articles/22755): ADX+bandwidth разом розділяють «справжній тренд» vs «vol-сплеск у флеті» |
| Volatility expansion | bandwidth / range-ratio та його percent | [MQL5](https://www.mql5.com/en/articles/22755): `bb_bw_regime` — найсильніша фіча |
| Funding | рівень funding (не z-score!) та його знак, distance-to-settlement | [funding-rate-alpha](https://raw.githubusercontent.com/OctopusTakopi/funding-rate-alpha/main/README.md): «funding does predict returns, in the level and not the z-score, at eight hours and shorter»; [Fundamentals of Perpetual Futures](https://arxiv.org/abs/2212.06888) |
| Breadth | частка символів вище власної MA, частка з позитивним 24h-моментом | режимна ознака, що не залежить від однієї ціни (синтез) |
| Correlation / dispersion | середня попарна кореляція 15 символів, частка дисперсії в PC1 | [absorption ratio](http://boston.qwafafew.org/wp-content/uploads/sites/3/2017/01/revere-1.pdf) |
| Відносні (не абсолютні) значення | усі фічі — у ранках/процентилях **минулого** вікна | захист від дрейфу рівнів між 2023 і 2026 |

⚠️ **Що НЕ ставити у фічі**: минуле PnL primary-моделі *без* concurrency-контролю (легко перетворюється на самопідтвердження); майбутні фандинг-ставки (у них є lag на публікацію); будь-які ознаки, обчислені з повного ряду.

### 4.5 Crypto-специфічні застереження

- [funding-rate-alpha](https://raw.githubusercontent.com/OctopusTakopi/funding-rate-alpha/main/README.md) (survivorship-free тест на 6.1M settlements): funding — це насамперед **carry**, а не ціновий сигнал; price-коефіцієнт **змінює знак** між «regime 2020-21» і «regime 2023-26», а net-прибуток 8.9 bps/день при 4 bps one-way на дві третини припадає на 2020-21; управління ризиком обмежене capacity ~$17M/leg. Тобто фандинг як фіча має проходити перевірку **по підперіодах**, а не в пулі.
- Той самий документ: жодна з популярних prescriptions (open-interest conditioning, composite z-score, cross-venue index) не вижила контролю — приклад того, як «розумна» фіча зникає під чесною перевіркою.

---

## 5. Емпірика: TSMOM / trend following + regime overlay

### 5.1 Що документовано реплікується

- **TSMOM як премія.** [Moskowitz, Ooi, Pedersen](https://pages.stern.nyu.edu/~lpederse/papers/TimeSeriesMomentum.pdf): 58 ф'ючерсів, 1985–2009; «a diversified portfolio of time series momentum across all assets is remarkably stable and robust, yielding a Sharpe ratio greater than one on an annual basis, or roughly 2.5 times the Sharpe ratio for the equity market»; альфа не пояснюється SMB/HML/UMD (intercept 1.58%/міс, t=7.99); прибутковість **найбільша саме в екстремальних ринках** (тобто TSMOM — це «long straddle»-подібний профіль, і саме тому regime-фільтр «вимкни в стресі» його вбиває).
- **Довгострокова стабільність.** [Two centuries of trend following](https://arxiv.org/abs/1404.3274): t-stat ≈5 з 1960 і ≈10 з 1800; «no sign of a statistical degradation of long trends, whereas **shorter trends have significantly withered**». Прямий висновок для нас: денний/тижневий тренд — реплікується; 1h-тренд — ні.
- **Vol scaling.** [Moreira & Muir](https://www.nber.org/system/files/working_papers/w22208/w22208.pdf) (+25% Sharpe, alpha 4.9%, appraisal 0.33); [Daniel & Moskowitz](https://www.nber.org/system/files/working_papers/w20439/w20439.pdf) (динамічне масштабування ~×2 Sharpe; constant-vol гірше).
- **Персистентність > точність.** [Shu et al.](https://arxiv.org/abs/2402.05272): JM-guided > HMM-guided > buy&hold на S&P/DAX/Nikkei 1990–2023 з 10 bps і 1-денним лагом; покращення річної дохідності ~1–4%; turnover JM на S&P лише 44%/рік.
- **Публічний crypto-кейс.** [AdaptiveTrend](https://arxiv.org/abs/2602.11708): 6h trend following + місячна адаптивна конструкція портфеля + **асиметричний 70/30 long-short** аллокація «grounded in the empirical positive drift of crypto markets»; OOS 2022–2024, 150+ пар, Sharpe 2.41, MDD −12.7%. [HMM_TR_Alg](https://raw.githubusercontent.com/Krishhiv/HMM_TR_Alg/main/README.md): walk-forward 3-стан HMM (мітка дня `T` декодується з даних до `T−1`, перенавчання кожні 30 днів на 730-денному вікні) + momentum breakout + min-dwell; OOS Sharpe 1.46 проти 1.18/1.13 у рукавів; block-bootstrap 5-й перцентиль Sharpe 0.79. **Обмеження, названі автором**: один цикл ~4.5 роки, обидва рукави — крипто-бета, 2026 YTD −2.3%.
- **Режими в крипті існують статистично.** [Koki et al.](https://arxiv.org/abs/2011.03741): 4-стан NHHM найкраще прогнозує one-step-ahead для BTC/ETH/XRP; стани розділяють bull/bear/calm.

### 5.2 Що НЕ реплікується

- **Дрібнотаймфреймове перемикання.** Наш власний результат: 1h, 10 символів, 3 роки — після lookahead-фіксу switch Sharpe +0.36 (mean), DSR 0.000, PBO 0.664. Це узгоджується з [Two centuries](https://arxiv.org/abs/1404.3274) («shorter trends have withered») і з тим, що коротьші горизонти найбільше страждають від витрат і лагу.
- **HMM-guided vs простіший baseline.** [Shu et al.](https://arxiv.org/abs/2402.05272) показують, що HMM-guided бінарна стратегія **програє** jump-model-guided. Тобто «складніша модель» ≠ краще, а «персистентніша модель» = краще.
- **Regime-фільтри в загальному випадку.** [Alphanume](https://www.alphanume.com/blog/what-is-a-market-regime-filter): «many regime filters look effective in backtests due to overfitting and produce less benefit in live use than expected». Поміч лише у трьох випадках (структурна залежність, risk overlay, capital allocation).
- **Funding-сигнали після контролів.** [funding-rate-alpha](https://raw.githubusercontent.com/OctopusTakopi/funding-rate-alpha/main/README.md): pooled-результат «collapses from a pooled +34 against +6 bps per day to +7.0 at t = 0.35 once cohorts are compared inside the same day».
- **Симетричність long/short.** Крипто має позитивний дрейф: [AdaptiveTrend](https://arxiv.org/abs/2602.11708) свідомо робить 70/30 замість 50/50. Наш єдиний вцілілий напрямковий рукав — `ts_momentum` **long-only** — узгоджується з цим. Regime-перемикач, який симетрично шортить у «ведмежому» режимі, бореться проти дрейфу.
- **«Crisis alpha» через overlay.** [Moskowitz et al.](https://pages.stern.nyu.edu/~lpederse/papers/TimeSeriesMomentum.pdf) і [Daniel & Moskowitz](https://www.nber.org/system/files/working_papers/w20439/w20439.pdf): TSMOM заробляє найбільше **саме** в екстремальних рухах; [Daniel & Moskowitz](https://www.nber.org/system/files/working_papers/w20439/w20439.pdf) додатково показують, що хеджування ринкової дисперсії не відновлює momentum у ведмежих ринках. Отже «вимкнути тренд у high-vol» часто вирізає найкращі хвости.

---

## 6. Конкретні рецепти для 4h/1d crypto (15 символів), за спаданням очікуваної робастності

Умова входу для всіх: реалізація без lookahead (лаг на **рішенні**), комісії maker/taker + slippage з `.env`, оцінка — walk-forward + DSR + PBO/CSCV, OOS-вікна реєструються як спалені.

### R1. Vol-targeting експозиції на валідованих рукавах (без жодного regime-перемикання)

- **Механізм.** Експозиція кожного рукава `w_t = min(cap, σ_target / σ̂_t)`, де `σ̂_t` — EWMA/HAR-волатильність з даних `≤ t`. Для `pairs_arb` — scaling обох ніг однаково; для `ts_momentum` — scaling лонг-портфеля.
- **Параметри (2–3).** `σ_target` (річна), вікно EWMA (`half-life` 5–20 днів), `cap` (наприклад 2.0–3.0).
- **Чому перший.** Документований найбільш стабільний ефект ([Moreira & Muir](https://www.nber.org/system/files/working_papers/w22208/w22208.pdf): +25% Sharpe; [Daniel & Moskowitz](https://www.nber.org/system/files/working_papers/w20439/w20439.pdf): ~×2 Sharpe), нуль regime-міток → нуль multiple-testing боргу.
- **Falsification.** Порівняти net Sharpe/Calmar/CVaR зі сталою вагою на **тих самих** сигналах. Якщо scaling не перевершує constant-weight на ≥1 з 3 метрик і DSR < 0.95 — вбити.

### R2. Meta-labeling як фільтр/сайзер на `ts_momentum` (long-only, daily)

- **Механізм.** Primary = TSMOM (сторона long на 15 символах), secondary = LightGBM на triple-barrier мітках (вертикальний бар'єр 5–10 днів, бар'єри в одиницях EWMA-vol), bet size з каліброваної ймовірності.
- **Параметри (2–3).** множники бар'єрів (наприклад 1×/2× vol), поріг прийняття `P>0.55`, кількість фіч-груп (фіксована).
- **Чому другий.** Знімає потребу в стабільній мітці режиму; режимні фічі (vol, trend strength, funding level, breadth, PC1-share) входять як фічі.
- **Falsification.** Purged CV (embargo ≥ горизонту мітки) + перевірка на `feature[t]` vs `return[t]` кореляцію. Якщо precision uplift не перекриває зменшення кількості угод після чесних витрат (net Sharpe ≤ unfiltered, або DSR < 0.95) — вбити. Додатковий тест: **calibration curve** — без неї bet sizing вироджується.

### R3. Бінарний risk-off гейт з гістерезисом і асиметричними порогами

- **Механізм.** Один біт: `risk_off`, якщо vol-перцентиль (з минулого) > `p_off`; повернення в `risk_on` лише коли < `p_on < p_off` **і** пройшло `min_dwell` барів. У `risk_off` — експозиція множиться на `0` (або `0.25`).
- **Параметри (3).** `p_off`, `p_on` (=асиметрія, задається логікою: вихід швидший за вхід), `min_dwell`.
- **Чому третій.** Документовано як два найкорисніші застосування regime-фільтрів — risk overlay і capital allocation ([Alphanume](https://www.alphanume.com/blog/what-is-a-market-regime-filter)); бінарні асиметричні схеми описані як інституційний стандарт ([Syntax Data](https://www.syntaxdata.com/research/why-binary-allocation-and-asymmetric-signals-matter)); min-dwell дає −68% churn ([HMM_TR_Alg](https://raw.githubusercontent.com/Krishhiv/HMM_TR_Alg/main/README.md)).
- **Falsification.** Порахувати **flip rate** (перемикань/рік) і net-of-cost результат проти no-gate. Вбити, якщо (а) flip rate > 6/рік на 4h, або (б) зменшення MDD < 20% при втраті CAGR > 15%, або (в) гейт виграє лише завдяки 1–2 хвостовим подіям (перевірити sub-period stability).

### R4. М'яке зважування з shrinkage (замість argmax по режимних клітинках)

- **Механізм.** `w(regime, strat) ∝ exp( λ · SR_shrunk )`, де `SR_shrunk = (n/(n+k))·SR_obs + (k/(n+k))·SR_uncond` — James–Stein/Bayes–Stein shrinkage ([skfolio `ShrunkMu`](https://skfolio.org/_modules/skfolio/moments/expected_returns/_shrunk_mu.html)). Ніякого argmax, ніяких taxonomy-пріорів.
- **Параметри (2).** `k` (сила shrinkage, задається зі статистики, не оптимізується), `λ` (температура, 1 значення).
- **Чому четвертий.** Soft weighting не гірший за hard switching за персистентності, але дешевший ([Shu et al.](https://arxiv.org/abs/2402.05272) з посиланням на Nystrup 2016); shrinkage прямо лікує малі вибірки на клітинку.
- **Falsification.** Обов'язково **спершу** перевірити, що карта не вироджується в константу (усі ваги в межах ±10% від рівноважних) і що мітка режиму не містить PnL стратегій (§2.7). Якщо результат ≈ рівноважному бленду — вбити (немає чого масштабувати).

### R5. Change-point (BOCPD/CUSUM) для vol-режиму замість HMM

- **Механізм.** Онлайн-детекція зламу рівня волатильності; торговий ефект — тільки один: посилення/послаблення vol-таргету. `ruptures`-офлайн варіанти не використовувати ([ruptures docs](https://centre-borelli.github.io/ruptures-docs/)).
- **Параметри (2).** поріг `h` (CUSUM) або hazard rate (BOCPD); мінімальний очікуваний dwell для прийняття.
- **Чому п'ятий.** Change point каузальний і не потребує мітки «стану» — але дає менше, ніж R1 (вона й так міряє vol).
- **Falsification.** Явно виміряти **lag у барах** = середнє(детекція − справжній злам) на історичних зламах. Вбити, якщо lag ≥ 50% медіанної тривалості vol-епізоду; також вбити, якщо гейт не покращує нічого понад R1 (порівняння з R1 як baseline).

### R6. Switch-cost-aware політика (оптимізація net-of-cost + dead-band)

- **Механізм.** Будь-яке перемикання дозволене лише за умови `E[ΔSR/період]·E[dwell] > round_trip_cost`; turnover-штраф входить у цільову функцію, як у [Nystrup et al.](https://backend.orbit.dtu.dk/ws/files/139272081/Dynamic_Portfolio_Optimization_Across_Hidden_Market_Regimes_ACCEPTED.pdf) (turnover 4.16 → 1.17, Sharpe 0.56 → 0.63).
- **Параметри (2).** очікуваний dwell (з історії режиму), cost per switch (з `.env`: maker/taker + slippage).
- **Чому шостий.** Це не окрема стратегія, а **обов'язковий фільтр** для R3/R4/R5; без нього будь-яке перемикання буде had been overfit до нульових витрат.
- **Falsification.** Побудувати криву «net Sharpe vs припущення про витрати» (як sensitivity до 2/5/10/20 bps). Вбити політику, якщо breakeven-cost < фактичних витрат (у [funding-rate-alpha](https://raw.githubusercontent.com/OctopusTakopi/funding-rate-alpha/main/README.md) саме цей тест показав break-even на 23 bps — і це врятувало від висновку «сигнал працює»).

### R7. Кореляційний / dispersion overlay (AR або PC1-share) — лише як ризик-модифікатор

- **Механізм.** Scaling сукупної експозиції за `PC1_share` або середньою попарною кореляцією 15 символів: `gross_t = f(percentile(AR_t))`, де AR рахується на rolling-вікні з каузальним процентилем ([absorption ratio](http://boston.qwafafew.org/wp-content/uploads/sites/3/2017/01/revere-1.pdf): вікно 500 днів, `k ≈ N/5`).
- **Параметри (2).** вікно (для 4h: 300–500 барів ≈ 50–80 днів), поріг процентиля.
- **Чому сьомий.** AR — повільна ознака (лаг ≈ половина вікна) і на крипто-універсумі майже колінеарна з BTC-vol; тому тільки як **ризик**-модифікатор, ніколи як перемикач стратегій.
- **Falsification.** Порівняти з R1 (vol targeting) на тих самих даних: якщо AR-overlay не покращує Calmar/CVaR понад R1 — вбити (ймовірний результат: R1 поглинає AR).

### R8. Deflated-trial дисципліна як «рецепт» (нуль параметрів, pre-registration)

- **Механізм.** До прогону зареєструвати: `N` конфігурацій, MinBTL-перевірку, схему CPCV, дизайн DSR. Після прогону — DSR з чесним `N` + PBO/CSCV.
- **Параметри.** Немає (це процес).
- **Правило.** `MinBTL < 2·ln[N]/E[max]²` років ([Bailey et al.](https://www.carmamaths.org/resources/jon/backtest.pdf)): для 3 років даних безпечно `N ≲ 15–20` конфігурацій. Вбити результат, якщо DSR < 0.95 **або** PBO > 0.5 **або** `N` невідомий/не зареєстрований.

**Ранжування за очікуваною робастністю:** R1 > R2 > R3 > R8(процес) > R6(фільтр) > R4 > R5 > R7. Логіка: чим менше міток режиму — тим менше multiple-testing-боргу; чим більше доказової бази в літературі — тим менше шансів, що «працює випадково».

---

## 7. Що НЕ робити

| Анти-патерн | Причина |
|---|---|
| **Hard argmax «режим → одна стратегія»** | Максимізує дисперсію рішення по малих клітинках; у `iter7` taxonomy-`best_prior` вироджувався в «завжди перша стратегія» (порожній `preferred_regimes` = вага 1.0 усюди). |
| **Smoothed/Viterbi мітки для торгових рішень** | Мітка бару `t` знає `t+1..T` ([statsmodels](https://www.statsmodels.org/stable/examples/notebooks/generated/markov_autoregression.html), [hmmlearn](https://hmmlearn.readthedocs.io/en/latest/api.html)) — це і був наш lookahead K1: Sharpe +2.64 → +0.36. |
| **Оцінка режимної моделі на всьому ряді** | Параметри, оцінені з тестом у вибірці, дають in-sample мітки; класична причина «OOS-колапсу» ([Shu et al.](https://arxiv.org/abs/2402.05272): mis-estimation при малих вибірках і високій персистентності). |
| **Багато клітинок (3+ режими × 5+ стратегій) на 2–3 роках даних** | MinBTL: 5 років → ≤45 незалежних конфігурацій, 2 роки → ≤7 ([Bailey et al.](https://www.carmamaths.org/resources/jon/backtest.pdf)); інакше expected OOS SR = 0. |
| **Режим, визначений через перформанс стратегії** | Циркулярність ([Alphanume](https://www.alphanume.com/blog/what-is-a-market-regime-filter): «conditioning on the strategy's own returns... is circular»); селектор просто переупаковує PnL. |
| **Перемикання без штрафу за turnover** | У проєкті `meta:ctx_hedge`/`ens_hedge` дали 80–85 тис. угод; [Nystrup et al.](https://backend.orbit.dtu.dk/ws/files/139272081/Dynamic_Portfolio_Optimization_Across_Hidden_Market_Regimes_ACCEPTED.pdf) показують, що штраф за торгівлю **покращує** Sharpe (0.56 → 0.63) при скороченні turnover у 3.5×. |
| **Режимний фільтр, що вимикає тренд у high-vol** | TSMOM заробляє найбільше в екстремальних рухах ([Moskowitz et al.](https://pages.stern.nyu.edu/~lpederse/papers/TimeSeriesMomentum.pdf)); хеджування дисперсії не відновлює momentum у ведмежих ринках ([Daniel & Moskowitz](https://www.nber.org/system/files/working_papers/w20439/w20439.pdf)). |
| **Симетричний long/short там, де є позитивний дрейф** | Крипто має позитивний дрейф: [AdaptiveTrend](https://arxiv.org/abs/2602.11708) свідомо 70/30; наш єдиний вцілілий напрямковий рукав — long-only. |
| **Regime-перемикання на 1h/15m** | «Shorter trends have significantly withered» ([Two centuries](https://arxiv.org/abs/1404.3274)); витрати на перемикання домінують на коротких горизонтах. |
| **Trust у «розумні» conditioning-фічі без контролів** | [funding-rate-alpha](https://raw.githubusercontent.com/OctopusTakopi/funding-rate-alpha/main/README.md): OI-conditioning «collapses from +34 to +7.0 bps/day at t=0.35 once cohorts are compared inside the same day». |
| **DSR/PBO з прихованим або заниженим `N`** | У `iter7` DSR 0.82 (N=27) → 0.67 (N=270) → 0.44 (N=2700) ([Bailey & López de Prado](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)); PBO 0.004 був артефактом (у пул додали сам lookahead-селектор). |
| **Порівняння in-sample і walk-forward клітинок в одній таблиці** | Різні `mode` дають різні умови; це зафіксовано і в проєкті (haircut групує за `mode`), і в літературі (PBO-фреймворк вимагає однорідного семплу). |
| **Оптимізація порогів детектора по Sharpe** | «Threshold overfitting... produces optimistic results» ([Alphanume](https://www.alphanume.com/blog/what-is-a-market-regime-filter)); пороги треба виводити з логіки (витрати, очікуваний dwell), а не з бектесту. |

---

## 8. Джерела

**Академія / preprints**

- Shu, Y., Yu, C., Mulvey, J. — *Downside Risk Reduction Using Regime-Switching Signals: A Statistical Jump Model Approach* (arXiv:2402.05272) — [abs](https://arxiv.org/abs/2402.05272), [повний текст](https://arxiv.org/html/2402.05272v3)
- Nystrup, P., Madsen, H., Lindström, E. — *Dynamic portfolio optimization across hidden market regimes* — [PDF (DTU Orbit)](https://backend.orbit.dtu.dk/ws/files/139272081/Dynamic_Portfolio_Optimization_Across_Hidden_Market_Regimes_ACCEPTED.pdf)
- Bailey, D., Borwein, J., López de Prado, M., Zhu, Q. J. — *The Probability of Backtest Overfitting* (CSCV) — [PDF](https://www.carmamaths.org/resources/jon/backtest2.pdf)
- Bailey, D., Borwein, J., López de Prado, M., Zhu, Q. J. — *Pseudo-Mathematics and Financial Charlatanism* (MinBTL) — [PDF](https://www.carmamaths.org/resources/jon/backtest.pdf)
- Bailey, D., López de Prado, M. — *The Deflated Sharpe Ratio* — [PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- Moreira, A., Muir, T. — *Volatility-Managed Portfolios* (NBER WP 22208) — [PDF](https://www.nber.org/system/files/working_papers/w22208/w22208.pdf)
- Daniel, K., Moskowitz, T. — *Momentum Crashes* (NBER WP 20439) — [PDF](https://www.nber.org/system/files/working_papers/w20439/w20439.pdf)
- Moskowitz, T., Ooi, Y. H., Pedersen, L. H. — *Time Series Momentum* — [PDF](https://pages.stern.nyu.edu/~lpederse/papers/TimeSeriesMomentum.pdf)
- Lempérière, Y., Deremble, C., Seager, P., Potters, M., Bouchaud, J.-P. — *Two centuries of trend following* (arXiv:1404.3274) — [abs](https://arxiv.org/abs/1404.3274)
- Bui, D., Nguyen, T. — *Systematic Trend-Following with Adaptive Portfolio Construction...* (crypto, arXiv:2602.11708) — [abs](https://arxiv.org/abs/2602.11708)
- Koki, C., Leonardos, S., Piliouras, G. — *Exploring the Predictability of Cryptocurrencies via Bayesian Hidden Markov Models* (arXiv:2011.03741) — [abs](https://arxiv.org/abs/2011.03741)
- He, S., Manela, A., Ross, O., von Wachter, V. — *Fundamentals of Perpetual Futures* (arXiv:2212.06888) — [abs](https://arxiv.org/abs/2212.06888)
- Huang, Y. et al. — *From Hypotheses to Factors: Constrained LLM Agents in Cryptocurrency Markets* (arXiv:2604.26747) — [abs](https://arxiv.org/abs/2604.26747)
- Kritzman, M., Li, Y., Page, S., Rigobon, R. — *Principal Components as a Measure of Systemic Risk* (absorption ratio) — [PDF](http://boston.qwafafew.org/wp-content/uploads/sites/3/2017/01/revere-1.pdf)
- *purgedcv: purged and combinatorial cross-validation* (purge/embargo/CPCV, з AFML гл. 7 і 12) — [paper.md](https://raw.githubusercontent.com/eslazarev/purged-cross-validation/refs/heads/main/paper/paper.md)

**Документація бібліотек**

- statsmodels — *Markov switching models* (filtered vs smoothed ймовірності) — [приклад](https://www.statsmodels.org/stable/examples/notebooks/generated/markov_autoregression.html), [MarkovRegression API](https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_regression.MarkovRegression.html)
- hmmlearn — *API* (`decode`/Viterbi vs posterior probabilities) — [docs](https://hmmlearn.readthedocs.io/en/latest/api.html)
- ruptures — *документація* (офлайн/віконні/онлайн алгоритми) — [docs](https://centre-borelli.github.io/ruptures-docs/)
- skfolio — *ShrunkMu: James–Stein / Bayes–Stein shrinkage* — [source](https://skfolio.org/_modules/skfolio/moments/expected_returns/_shrunk_mu.html)

**Practitioner / блоги / репозиторії**

- *Meta-Labeling the Classics (Part 3): Filtering and Sizing Bollinger Band Trades* — triple-barrier, `.shift()`, квантилі з train-вікна, calibration, purged K-fold — [MQL5](https://www.mql5.com/en/articles/22755)
- *Funding Rates as a Cross-Sectional Factor in Perpetual Futures* — survivorship-free тест на 6.1M settlements; carry vs price; breakeven-cost 23 bps; capacity — [README](https://raw.githubusercontent.com/OctopusTakopi/funding-rate-alpha/main/README.md)
- *HMM-TR3: Regime-Filtered Crypto Momentum Portfolio* — walk-forward HMM, min-dwell (−68% churn), vol targeting, robustness suite — [README](https://raw.githubusercontent.com/Krishhiv/HMM_TR_Alg/main/README.md)
- Alphanume — *What Is a Market Regime Filter?* — threshold overfitting, whipsaw, lag, циркулярність, «коли фільтри допомагають» — [стаття](https://www.alphanume.com/blog/what-is-a-market-regime-filter)
- Syntax Data — *Why Binary Allocation and Asymmetric Signals Matter* — [стаття](https://www.syntaxdata.com/research/why-binary-allocation-and-asymmetric-signals-matter)
- Micro Alphas — *Regime Detection* (glossary; filtered/smoothed, estimation uncertainty) — [сторінка](https://microalphas.com/glossary/regime-detection/)
- Robot Wealth / Edge Alchemy (K. Longmore) — *To Trend or Not To Trend?* — pattern vs reason; чому тренд у крипті правдоподібний — [стаття](https://edgealchemy.robotwealth.com/p/to-trend-or-not-to-trend)
- R-bloggers — реалізація turbulence (Mahalanobis) — [частина 1](https://www.r-bloggers.com/2011/04/great-faj-article-on-statistical-measure-of-financial-turbulence/), [частина 2](https://www.r-bloggers.com/2011/04/great-faj-article-on-statistical-measure-of-financial-turbulence-part-2/)
- Wikipedia — [CUSUM](https://en.wikipedia.org/wiki/CUSUM), [Bayesian online changepoint detection](https://en.wikipedia.org/wiki/Bayesian_online_changepoint_detection), [Hysteresis](https://en.wikipedia.org/wiki/Hysteresis)

**Внутрішні джерела проєкту (контекст)**

- `docs/reports/strategy_rating_regime.md` — iter7: switch Sharpe +2.64 → **+0.36** після фіксу lookahead K1, DSR 0.821 → **0.000**, PBO 0.004 → **0.664**
- `scalper_hft/features/regimes.py` — каузальні `market_structure`/`volatility_regime`/`apply_min_dwell`/`apply_regime_gates`
- `scalper_hft/strategies/regime_supervisor.py` — blend_mode (`regime_soft`/`best_prior`/`contextual_hedge`/`exp3`), `regime_map_path`, `min_dwell_bars`, `turnover_penalty`

*(синтез)* — позначено там, де твердження є висновком автора цього звіту, а не прямою цитатою джерела.
