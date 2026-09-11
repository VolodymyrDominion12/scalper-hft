# Аудит обчислень: бектест, walk-forward, валідація, дані

**Дата:** 2026-09-11
**Обсяг:** `scalper_hft/backtest/*`, `scalper_hft/validation/*` (33 модулі), `scalper_hft/data/*`,
`scalper_hft/features/*`, `scalper_hft/strategies/*`, `experiments/iter7_*`, `experiments/iter8_*`,
артефакти `results/*`.
**Стан репо:** коміт `2c4258e`, робоче дерево чисте. Тести: **1428 passed / 9 skipped**
(єдина помилка — `test_engine_perf.py::test_pairs_step_latency_smoke`, флейкі-таймінг під
навантаженням; ізольовано проходить).

---

## 0. Вердикт

Обчислювальне ядро **арифметично справне**: я перерахував усі 200 клітинок `iter7` з
parquet-артефактів — розбіжностей із `results/iter7_shard_*.csv` **нуль**, агрегати сходяться
до 1e-16. Лаг виконання в рушії (`signals.shift(1)`) правильний, комісії та funding
нараховуються коректно, `bfill`/`interpolate` в даних і фічах немає, ресемплінг коректний.

Але **висновки, які зараз лежать у `docs/reports/`, переважно не витримують перевірки**, і
причина не в арифметиці, а в чотирьох речах: (1) 1-барний lookahead у тесті regime-switching,
(2) ~28% клітинок — «мертві», але позначені `status="ok"`, (3) результати `results/iter7_*`
застарілі й не відтворюються поточним кодом, (4) анти-снупінг протокол (holdout + OOS burn)
вимкнений у `.env`.

**Головне число, яке змінюється:** заявлений у `strategy_rating_regime.md` edge
regime-selector **Sharpe +2.64 → +0.36** (H1→H2) / **+2.58 → +0.41** (rolling), DSR
**0.821 → 0.00**, PBO **0.004 → 0.66**. Тобто статус T1 («Пріоритет №1 для розвитку»,
«PASS») необґрунтований.

**Форвард-тесту (paper) як вимірювання не існує взагалі.** Paper-демони запитують свічки з
`since_ms=0`, а ccxt перетворює це на `startTime=0` → Binance віддає **найстаріші** свічки
(2019 р.), а не «останні N»; тому `paper-run`/`paper-run-pairs` стоять на місці, а артефакт
`results/paper_pairs.sqlite`, на який спирається тижневий гейт «≥8 тижнів», — це 90-денний
**історичний replay на тому ж кеші**, на якому робилися дослідження.

---

## 1. Критичні знахідки

### K1. Lookahead на 1 бар у тесті regime-switching — головний висновок iter7 хибний

**Файл:** `experiments/iter7_regime_analysis.py:350-356` (`switch_test`), `:213-224`
(`rolling_switch.apply_map`).

```python
# switch_test — БЕЗ лага взагалі:
switch_ret.iloc[np.flatnonzero(mask)] = h2.loc[mask, c].to_numpy()

# rolling_switch.apply_map — лаг є, але не той:
out.iloc[pos] = te.loc[reg_te == key, c].to_numpy()
return out.shift(1).fillna(0.0)   # ← «shift(1)» на СЕРІЇ ДОХІДНОСТЕЙ
```

**Чому це lookahead.** Індекс klines — open time (перевірено: `resample.py:126`
`label="left", closed="left"`). Тому `ret[t]` — це дохідність інтервалу `[t, t+1h)`, а
`structure[t]` рахується з `close[t]`, тобто з ціни на **кінець** того ж інтервалу. Обидві
величини стають відомі в один момент, і `structure[t]` фізично не може вибирати `ret[t]`.

`shift(1)`, застосований до *серії дохідностей*, проблеми не усуває: він дає
`out[t-1] = te[m(reg[t-1]), t-1]` — мітка й дохідність усе одно з одного бару. Правильно
лагувати **вибір**, а не дохідність: `sel[t] = te[m(reg[t-1]), t]`.

**Виміряно (я перерахував із тих самих `results/iter7_oos/*.parquet`):**

| Варіант | mean Sharpe (H1→H2) | mean Sharpe (rolling, 30 спостережень) |
|---|---|---|
| як опубліковано (`shift(1)`) | **+2.64** | **+2.58**, 30/30 > 0 |
| без лага взагалі | +2.64 | +2.59 |
| **коректний лаг вибору (1 бар)** | **+0.36** | **+0.41**, 19/30 > 0 |

`shift(1)` і «без лага» дають те саме число (2.64 vs 2.587) — це прямий доказ, що заявлений
«lag-1» у `strategy_rating_regime.md` (рядок 208: «WF + lag-1 + PBO 0.004») не робить нічого.

**Наслідки для анти-снупінгу (та сама таблиця, розділ 5.3 звіту):** DSR рахувався на
lookahead-серії (`iter7_regime_analysis.py:302`) → 0.821; на коректній серії **0.00 для всіх
10 символів**. PBO рахувався на матриці, у яку **додано сам селектор** як кандидата
(`:306`: `np.vstack([sub.to_numpy().T, series["empirical"][None, :]])`) — тобто lookahead-ряд
змагався з реальними стратегіями й тому вигравав у кожному спліті → PBO 0.004. На 9 реальних
singles без селектора **PBO = 0.66** (перенавчання ймовірне за критерієм <0.50).

> Правило: серія, яку оцінюють, не має бути кандидатом у власному PBO.

**Що НЕ зачеплено:** сам продакшн-`RegimeSupervisor` (`strategies/regime_supervisor.py:250`)
використовує мітку того ж бару, але його сигнал далі проходить `signals.shift(1)` у рушії —
це коректно. Баг живе **тільки в офлайн-аналізі селектора**, тому він не «зламав» бектест,
але повністю зіпсував висновок про наявність edge.

---

### K2. 28% клітинок «мертві» (нуль угод), але всі 200 мають `status="ok"`

`results/iter7_shard_*.csv`: **57 із 200 клітинок мають `n_trades_oos == 0`**, і всі позначені
`ok`. Їхні OOS-дохідності — **рівно нулі** в кожному барі (перевірено по parquet):

| variant | клітинок із 0 угод | `mean_oos_sharpe` у рейтингу |
|---|---|---|
| `meta:ens_vote` | 10/10 | 0.0000 |
| `single:basis_reversion` | 10/10 | 0.0000 |
| `single:cvd_momentum` | 10/10 | 0.0000 |
| `single:ob_imbalance` | 10/10 | 0.0000 |
| `single:hmm_reversion` | 9/10 | −0.0026 |
| `single:mean_reversion` | 8/10 | +0.0006 (2 угоди за 3 роки) |

`results/iter7_rating.csv` подає ці нулі як виміряні Sharpe поряд із реальними. Це порушує
власне правило проєкту (`AGENTS.md`: «sweep маркує `status="degenerate"`»;
`experiments/README.md`: «Мінімум 30 угод за 3y на клітинку — інакше `degenerate`»).
Механізм `flag_degenerate_row` (`validation/sweep.py:100-138`) існує й працює — але
`experiments/iter7_regime_rating.py` його **не викликає**, тому дегенерація не фіксується.

**Корінь для 2 із 5 варіантів — тихе неспівпадіння даних:**

```
data/BTCUSDT_aggTrades.parquet:  2026-09-08 15:34 → 2026-09-11 09:23   (~2.7 доби)
walk-forward OOS:                2023-12-24       → 2026-08-29         (~3 роки)
```

`run_walk_forward` ріже trades **за часом** для кожного вікна (`walk_forward.py:175-179`), тому
`trades_te` порожній у **всіх 47 вікнах**. Але `validate_inputs` викликається на **нерозрізаному**
фреймі (`walk_forward.py:43-46`) — перевірка проходить, і стратегія тихо повертає нулі.
Єдиний слід — `pooled_oos_sharpe = 0.0`.

**Фальсифікаційний тест (я виконав):** на 74 барах, що перекривають наявні aggTrades,
`ob_imbalance` дає 25/74 ненульових сигналів, `cvd_momentum` — 32/74. Тобто стратегії
працюють; мертві вони виключно через покриття даних. Це 20 клітинок (10% матриці) із
фальшивим «нульовим» результатом.

**Окремо:** `single:mean_reversion` і `single:hmm_reversion` дають 1–2 угоди за 3 роки на
10 символах при `MIN_TRADES["1h"] = 30` — вони теж мусять бути `degenerate`, а не «нульовим edge».
І `meta:ens_vote` структурно не може торгувати: один із дітей (`mean_reversion`) не торгує,
тож голосування не спрацьовує ніколи.

---

### K3. `results/iter7_*` не відтворюються поточним кодом (і не відтворюються взагалі)

`results/iter7_shard_*.csv` записані **12:22–12:28** 2026-09-11. Після цього змінилися:

| артефакт / код | mtime |
|---|---|
| `results/iter7_shard_*.csv`, `iter7_rating.csv`, `iter7_switch_*`, `iter7_dsr_pbo` | 12:11–12:28 |
| `scalper_hft/strategies/supertrend.py`, `funding_carry.py`, `regime_supervisor.py` | 13:11 |
| `scalper_hft/validation/regime_fit.py` | 13:15 |
| `scalper_hft/backtest/execution.py` | 14:55 |
| `scalper_hft/backtest/engine.py` | 15:18 |
| `data/BTCUSDT_1m_klines.parquet` (самі ринкові дані) | 15:33 |

Я перезапустив одну клітинку end-to-end (`BTCUSDT`, 1h, days=1095, train=2500/test=500,
maker, фіксовані дефолтні параметри) і порівняв зі шардом:

| метрика | shard (12:22) | свіжий прогін |
|---|---|---|
| `single:supertrend` avg_oos_sharpe | −0.0758 | −0.0818 |
| `single:supertrend` n_trades | 237 | 236 |
| `single:supertrend` pooled_oos_sharpe | −0.2140 | −0.1914 |
| `single:supertrend` oos_positive_frac | 0.4468 | 0.4681 |
| `single:funding_carry` avg_oos_sharpe | −0.0698 | −0.0825 |
| `single:funding_carry` n_trades | 367 | 366 |

Розбіжності малі, але системні: **жоден висновок, зроблений на цих CSV, не є відтворюваним
твердженням про поточний код і поточні дані.** Це та сама пастка, про яку попереджає
`AGENTS.md` («довгоживучий worker тримає старий код»), тільки на рівні ручних прогонів.
`research/code_version.py` існує, але `code_version` не пишеться в CSV експериментів — тому
застарілість непомітна.

**Додатково, `iter8` змішує дві епохи коду:** `iter8_fresh_validation.csv` (13:04) — до зміни
стратегій о 13:11; `iter8_validation.csv` (13:23) і `iter8_regime_map.json` (13:17) — після.
Порівнювати iter7 та iter8 між собою зараз некоректно.

---

### K4. Анти-снупінг протокол фактично вимкнений

`.env`:
```
HOLDOUT_PCT=0
OOS_ENFORCE_BURN=false
```

- `validation/holdout.py:45-53` при `HOLDOUT_PCT=0` повертає порожній holdout →
  `sensitivity._research_slice` (`sensitivity.py:45-49`) стає no-op → **сітка параметрів
  будується на всій 3-річній вибірці, включно з хвостом, який звіти називають холдаутом.**
- `cell_audit.py:386-404` дозволяє `mode="final"` при `explicit_holdout=True` навіть коли
  `HOLDOUT_PCT=0`; тоді `ho_slice` — останні 20% `df_full`, які **вже** брали участь у
  `run_walk_forward(df, ...)` (`:477`), у сітці sensitivity (`:494`), в `wf.oos_returns` для
  DSR (`:534-575`) і у `run_strategy_backtest(df, ...)` (`:521`). `cell_verdict` перевіряє лише
  `holdout_sharpe > 0` (`:324-329`) → `record_verdict("PASS")` → `live/audit_gate._verdict_ok`
  перевіряє лише label + вік ≤30 діб. **Тобто єдиний label, що відкриває paper/live, може бути
  виданий без жодного неторканого бару.**
- `OOS_ENFORCE_BURN=false` + `validation/experiments.py:74-75` (`symbol = experiment.id`)
  → burn-реєстр fail-open: вікно, спалене audit-ом для `BTCUSDT`, не виявляється, бо
  `is_burned` порівнює `symbol` з id експерименту (`sparse-basket-v1`).
- `oos_registry.py:98-117` записує весь діапазон датасету, а не реальні OOS-вікна
  (доказ у `docs/reports/oos_usage.md`: `basis_reversion | AAVEUSDT | 2023-09-11 | 2026-09-10`),
  тож burn надто широкий там, де він усе-таки працює.

---

### K5. `aggTrades`-кеш втратив ~60% угод (дедуп за мілісекундним індексом)

**`data/downloader.py:790`** і **`data/binance_vision.py:197,200`**

```python
merged = pd.concat(all_f).sort_index()
return merged[~merged.index.duplicated(keep="last")]
```

Індекс — це `transact_time` у мс, а унікальний ключ угоди — `agg_trade_id`. У реальному потоці
кілька aggTrades регулярно ділять одну мс, і дедуп знищує всі, крім однієї.

**Виміряно на `data/` (я перевірив незалежно):**

| символ | рядків | `id_span` | покриття `trade_id` | `trade_id` унікальні |
|---|---|---|---|---|
| BTCUSDT | 1 173 552 | 2 983 494 | **39.3%** | так |
| ETHUSDT | 1 328 930 | 2 668 618 | **49.8%** | так |
| DOGEUSDT | 478 148 | 502 487 | **95.2%** | так |

Сигнатура саме дедупу за мс: `index.duplicated().sum() == 0` і `index.is_monotonic_increasing`,
при цьому 252 156 випадків, де `Δ trade_id > 1` **у межах ≤1 мс** (медіанний стрибок = 5).
Чим активніший інструмент, тим більша втрата — тобто викривлення структурне, а не масштабне.

**Наслідок:** `cvd_from_trades`/`buy_ratio`, `vpin`, `kyle_t`, `signed_flow_ac`, стратегії
`cvd_momentum`/`ob_imbalance` рахуються на ~40% реального потоку BTC. `validate_trades` цього
не ловить, бо після дедупу дублікатів індексу немає.

---

### K6. Paper/live тягнуть свічки з 2019 року: `since_ms=0` → `startTime=0`

**`live/paper_runner.py:54-57`**, **`live/pairs_runner.py:198`**, **`live/pairs_live.py:164`**

```python
def _fetch_recent(symbol: str, interval: str, limit: int = _RECENT_BARS) -> pd.DataFrame:
    """Останні N свічок напряму через REST (швидко, без повного кешу)."""
    client = ExchangeClient()
    batch = client.fetch_klines(symbol, interval, since_ms=0, limit=limit)
```

`ExchangeClient.fetch_klines` (`data/client.py:181-183`) — прямий passthrough у ccxt.
У ccxt (`ccxt/binance.py:4989-4990`): `if since is not None: request['startTime'] = since`.
`0 is not None` → `startTime=0` → **найстаріші** свічки лістингу (BTCUSDT perp — 2019-09),
а не «останні N», як обіцяє docstring.

Наслідки:
- `PaperRunner.step()` рахує сигнал на 600 барах 2019 року й «виконує» його за ціною 2019 року;
  вікно щоразу те саме → сигнал і mark-ціна не змінюються → equity стоїть. Саме тому
  `results/paper_equity_BTCUSDT.csv` містить **один рядок `10000.0`** (fee не списалась →
  позиція не забукалась) з wall-clock ts `2026-09-06 20:23:19`, що збігається з
  `pd.Timestamp.utcnow()` у `paper_runner.py:108` (тобто це `run(iterations=1)`).
- `PairsPaperRunner.step` (`pairs_runner.py:441-449`): `ts = common.index[-1]` назавжди той самий
  → після першого кроку `return "hold:same_bar"` **назавжди**. Саме цей шлях запускає
  задеплоєний гейт (`deploy/scalper-paper-pairs.service:27` → `paper-run-pairs --portfolio --daemon`).
- `pairs_live._fetch_last_price` (`pairs_live.py:159-174`): `fetch_klines(symbol, "1m", since_ms=0, limit=1)`
  → close **першої 1m-свічки в історії**; ним гідратуються реальні біржові позиції
  (`pairs_live.py:145-148`). Для BTCUSDT це ~10 300 USDT замість ~77 000 → entry_price, UPNL,
  equity і всі ризик-гейти/сайзинг (`size_pct*equity/price`) у live рахуються від фіктивної ціни.
  `reconcile_positions` порівнює лише side/size, тому цього не спіймає.
- Додатково `ExchangeClient()` має дефолт `exchange_id="binance"` (`data/client.py:31`), тобто
  paper-шляхи оминають `DATA_EXCHANGE`/`require_live_data_exchange` (`config.py:266-283`).

---

### K7. Артефакт форвард-тесту `results/paper_pairs.sqlite` — не forward test

Файл читає `cmd_paper_audit` (`cli/paper.py:207-212`, дефолт `--db results/paper_pairs.sqlite`),
і на нього посилається критерій гейта `scripts/start_paper_gate.sh:11`
(«≥8 тижнів без tracking error vs BT»). Факти з файлу:

- `equity`: 2160 рядків, `2026-06-08 22:00 → 2026-09-06 21:00`, крок строго 1 год,
  **нуль розривів** = рівно 90 діб (це `--days 90`) і жодного рестарту.
- `orders` (44 рядки): ліміт-ціни збігаються з close **кешованих** 1h-барів на тих самих ts
  (LINK 2026-06-23 07:00 limit `7.664` vs кеш `7.657`; BTC 2026-07-14 13:00 limit `63802.8`
  = кеш close `63802.8`) → це `replay_pairs` (історичний replay по кешу), а не демон.
  Демон за поточним кодом (K6) далі першого бару не йде.
- `orders.mid` і `decision_mid` = **NULL у всіх 44 рядках** → `records_from_orders` відкидає все
  → IS-звіт порожній (див. Major нижче).
- `positions`: 456 рядків синтетичних **тестових фікстур** (`BTCUSDT long 0.5 @60000/mark 60600`)
  з wall-clock ts `2026-09-06 … 2026-09-11` — юніт-тести пишуть у **продакшн-БД**, бо
  `PaperStore()` за замовчуванням = `results/paper_pairs.sqlite` (`store.py:35`).
- Внутрішня арифметика при цьому зістикована: `equity 10189.52 = cash 10326.03 + UPNL (−136.51)`,
  `realized 327.27 = trades 310.46 + funding 16.81`; остання позиція відкрита на кінець вікна.

---

## 2. Major

**M1. Бенчмарк Buy&Hold ануалізується по 8760 барів/рік незалежно від ТФ, і це гейт.**
`validation/benchmark.py:15` (`periods_per_year=8760` за замовчуванням) викликається з
`audit_extensions.py:213` → `cell_audit.py:319-322` робить `bt_sharpe <= benchmark` FAIL-умовою.
Натомість `bt_sharpe` береться з `metrics.py:100-112`, де `bars_per_year` визначається з індексу
equity. Оскільки Sharpe масштабується як √ppy, бенчмарк занижено: 1m ×7.75, 5m ×3.46, 15m ×2.0,
30m ×1.42 (для 4h/1d навпаки завищено).
*Приклад із `results/audit_verdicts.jsonl`:* BTCUSDT 1h, `bt_sharpe=-1.461 ≤ benchmark=1.006` —
на 1h множник ≈1, тому там усе гаразд; але той самий код на 1m порівнює різні шкали.

**M2. HMM режимного детектора навчається на даних, які потім оцінюються.**
`features/regime_detector.py:288-289` (`self.fit(close)`) + `:232` (`n_fit = min(hmm_fit_bars=2000, len(obs))`).
`filtered_proba` сам по собі каузальний (`hmm_regime.py:198-212` — тільки forward,
перевірено порядково), але **параметри** π, A, μ, Σ оцінені на перших 2000 барах тієї ж серії,
яку бектестять. Дефолтні вікна WF для 1h — `train=500, test=200` (`cell_audit.py:30`), тобто
для типової 90-денної клітинки (2161 бар) зона fit-у покриває ~93% барів, включно з усіма
OOS-вікнами. Заміряно: при усіченні серії до 1200 барів `hmm_p0` змінюється у 1009 із 1200
значень → значення в `t` залежить від барів > `t`.
**Область дії:** не зачіпає iter7 (там `train=2500`, і fit-зона 0..2000 лежить цілком у першому
train-вікні), але зачіпає **всі** sweep/audit-клітинки з дефолтними вікнами, тобто весь
`regime_supervisor` і `hmm_reversion` у матрицях.
Наявний `tests/test_lookahead_fixes.py:159-161` цього не ловить — він свідомо виключає зону
до кінця `fit_window`.

**M3. `is_maker` не передається в «свіжу» валідацію iter8 — інший базис витрат, ніж iter7.**
`experiments/iter8_fresh_validation.py:36-45` не передає `is_maker`, тому працює дефолт
`run_walk_forward(..., is_maker=False)` → **тейкер**. iter7 передає `is_maker=True` → maker.
Виміряно мною на ATOMUSDT/NEARUSDT (`funding_carry`, 500/200, 400 днів):

| символ | taker (як в iter8) | maker (як в iter7) |
|---|---|---|
| ATOMUSDT | avg_oos −0.218 | −0.169 |
| NEARUSDT | avg_oos −0.133 | −0.086 |

Різниця ≈0.05 Sharpe — не драматична, але порівнювати iter7 і iter8 в одній таблиці не можна.
Окремо: iter7 конструює `CostModel(...)` вручну (`iter7_regime_rating.py:250-254`), тому
`vol_ref=0` → vol-aware slippage вимкнено; iter8 використовує `CostModel.from_settings(df=df)`,
де `vol_ref` калібрується. Це **дві різні cost-моделі** для однієї й тієї ж стратегії
(те саме в `validation/sweep.py:516` і `validation/regime_fit.py:134` проти `cell_audit.py:415`).

**M4. Той самий поріг `OOS_SHARPE_MIN = 0.3` означає різний річний Sharpe на різних ТФ.**
`cell_audit.py:43`, застосовується до `avg_oos_sharpe`, який є **per-window t-stat шкалою**
(`walk_forward.py:100-111`: `mean/std·√n_барів`), а не ануалізованим Sharpe. Перевід до
річного = √(ppy/test_bars): 1m/2000 → ×16.2 (0.3 ⇔ річний 4.86), 5m/1000 → ×10.25 (3.08),
15m/30m/500/250 → ×8.37 (2.51), 1h/200 → ×6.62 (1.99), 4h/100 → ×4.68 (1.40).
Розкид 3.5× в економічному сенсі одного числа; у skill-таблиці та в CSV воно подане як
«Sharpe > 0.3» без маркера шкали. `walk_forward.py` чесно попереджає про це в docstring —
`cell_audit`, `sweep` і звіти — ні. Те саме в `sweep.py:712-768`: у колонку `sharpe` пишуться
то ануалізований (mode=backtest), то t-stat (mode=wf).

**M5. `iter8_fresh_validation.py:91` подає *значення Sharpe* у функцію, що очікує *ряд дохідностей*.**
```python
dsr = deflated_sharpe_ratio(np.array(oos_returns, dtype=float), n_trials=len(rows))
```
`deflated_sharpe_ratio` вважає вхід рядом періодичних дохідностей: рахує `sr = mean/std` від
9 річних Sharpe і `n = 9`. Тобто «Sharpe від Sharpe-ів». На наявних даних (усі від'ємні)
випадково виходить 0.0 і виглядає як FAIL; але на додатних клітинках це **генератор фальшивого
PASS**: я перевірив — вектор із 9 значень ~+0.3…+0.4 дає **DSR = 0.999997**. Для цього випадку
треба `deflated_sharpe_ratio_from_sharpe(...)`, а не `deflated_sharpe_ratio(...)`.

**M6. `regime_map.py:82-99` — та сама помилка вибору, що в K1, і «мертва» комірка = Sharpe 0.0.**
`labels = regime_df["label"].reindex(ret.index)` зіставляє мітку бару `t` з дохідністю бару `t`.
Додатково `volatility_regime` (`features/regimes.py:33-40`) включає `log(close/close.shift(1))`
у rolling std, тож мітка «high» на `t` механічно визначається тим самим `r_t`. Ефект менший,
ніж у K1 (режими персистентні), але клас помилки той самий. Плюс `regime_map.py:94`
(`sharpe = 0.0 if sd <= 1e-12`) дає «мертвій» комірці Sharpe рівно 0.0, а `hard_off_sharpe=0.0`
пропускає її фільтром — тобто стратегія, що в цьому режимі взагалі не торгувала, може стати
`best_prior` з вагою 1.0, витіснивши реальну стратегію з −0.01.
Docstring `regime_map.py:14-15` і назви `*_oos_*` обіцяють WF-OOS-матрицю, а `fit_regime_map`
(`regime_fit.py:175-198`) подає **один in-sample** бектест fit-вікна; max-over-strategies
робиться без корекції на множинне тестування.

**M7. `run_pairs_walk_forward` ковтає винятки в нейтральний нуль і не сходиться по знаменнику.**
`backtest/pairs.py:350-361`:
```python
try: ... oos.append(r_oos.metrics.sharpe_hourly)
except Exception: oos.append(0.0)      # ← провал виглядає як «нульовий Sharpe»
```
`is_s` при цьому **не** дописується, тож `avg_is_sharpe` і `avg_oos_sharpe` рахуються по різних
наборах вікон. Додатково `sharpe_hourly` (`metrics.py:153-167`) — це per-bar Sharpe, тобто
**третя шкала**, не сумісна ні з `avg_oos_sharpe` (t-stat), ні з `bt_sharpe` (ануалізований).
Заявлене в `strategy_rating_regime.md` «WF>0 у 65% вікон» для `pairs_arb` — саме ця шкала.
Окремо `pairs.py:207-218`: якщо в даних немає high/low, позиції симулюються як тейкерські
(гарантований філ), але комісія стягується як maker (`:263-269`) — оптимістично з обох боків.

**M8. Повносемплова статистика задає per-bar фічу (VPIN).**
`features/microstructure.py:284-286`: `avg_vol = out["volume"].rolling(100).mean().median()`
— медіана по **всій** завантаженій серії визначає розмір об'ємного бару, а отже всю серію `vpin`,
яка йде у `ml/features.py:102`. Кожен walk-forward фолд бачить константу, виведену з даних усіх
фолдів. Додатково `... or out["volume"].median()` хибно спрацьовує на легітимному `0.0`.

**M9. Фантомний перший бар у похідних ТФ.** `data/access.py:87` обрізає за `cutoff` по індексу
**похідного** ряду, а `resample_klines` виконується на всьому базовому кеші; `dropna(subset=["open"])`
(`resample.py:127`) прибирає лише повністю порожні біни. Заміряно: перший 1d-бар має 741/1440 хв
(BTC), 709 (ETH), 598 (BNB) — «добовий» бар із половиною дня й volume ~48% медіани. Це не
lookahead, а misalignment; але перший бар клітинки фіктивний, і `htf_market_structure(htf="1d")`
бере з нього EMA.

**M10. `funding_skew` у режимному детекторі фактично мертвий.** `features/regime_detector.py:341`
`reindex` без `method` — а реальні `fundingTime` мають мс-зсув (у кеші лише 2204/3288 = 67%
стоять рівно на годині), тож 33% ставок тихо стають `0.0` (заміряно 7.15% ненульових барів
проти теоретичних 12.5%). `window=720` — це 720 funding-спостережень ≈ 240 днів, хоч docstring
каже «~30 днів»; при `len(f) < 720` функція повертає суцільні нулі (`:57`), тому для стандартної
90/180-денної клітинки тег `f` у `joint_label` **завжди** `f0` і фандинг-фактор у роутингу зникає
без попередження.

**M11. Holdout не зафіксований у часі.** `holdout.py:45-53` ріже `df.iloc[-n_hold:]` від
ковзного `ensure_klines(days)`: при дозавантаженні нових барів межа їде, і перший місяць
попереднього holdout потрапляє в research. Твердження `sweep.py:314-316` («щоб матриця не
бачила holdout жодного разу») для повторних прогонів хибне; межа ніде не персиститься.

**M12. `paper_replay` має +1 бар лагу проти бектесту.** `live/paper_replay.py:105-107`:
`price = close[i]`, `signal = signals[i-1]`, тоді як рушій входить за `close[t]` при сигналі
`signals[t]` і заробляє дохідність бару `t+1` (`engine.py:511-514,587`). Тобто replay входить
на бар пізніше й **втрачає дохідність першого бару утримання**. Чисельно на синтетиці
(+5% на барі після сигнального): backtest `+0.00049`, replay `−0.00001`. Для 1-барного
momentum-скальпера це весь edge, а docstring (`:13-14`) стверджує «точно як у бектест-рушії».

**M13. `paper_replay` завжди звітує 0 угод.** `live/paper_replay.py:161`:
`compute_metrics(equity, exposure=0.0, turnover=0.0)` — `trades` не передається. На прогоні з
28 угодами: `metrics.n_trades = 0`, `win_rate = 0.0`, `profit_factor = 0.0`. Тобто саме ті
цифри, за якими читач вирішує, чи risk-шар торгував, — фабриковані нулі.

**M14. Paper Gate не порівнює paper з бектестом, якщо не передати `--bt-equity`, а
задокументований тижневий виклик його не передає.** `paper_audit.py:126-130` ставить
`dd_ok = None`, якщо `bt_dd is None`; `cli/main.py:343` має `--bt-equity default=None`;
`scripts/start_paper_gate.sh:11` його не передає. Фактичний вивід: `tracking-error=— maxDD
paper=2.85% gate=—`, і **жодного консюмера**, який би читав `dd_gate_ok/tracking_error/fill_gap`
(окрім самого `paper_audit.py`). Відсутність порівняння читається як відсутність розбіжності.

**M15. `[ok]` для IS vs CostModel — тавтологія, коли даних немає.** `live/is_report.py:44-48`
повертає `0.0` для порожнього списку; `paper_audit.py:160-161` кличе
`calibrate_from_is(..., fallback=0.0002)`, а `0.0002` чисельно дорівнює `SLIPPAGE_BPS/1e4`
(`config.py:100`) → `delta = 0` **за побудовою**. Реальний вивід: `IS records extracted: 0`,
`Realized=0.0 bps ↓ Model=2.0 bps`, і одразу `IS vs CostModel: maker=2.00 bps … [ok]`.
Плюс у тому ж звіті два різні fill-rate: `paper=100%` (з `fill_stats`) і `0%` (з IS) —
різні знаменники.

**M16. Paper-maker філи резолвляться по тому ж бару, чий close задав ліміт.**
`live/trader_loop.py:169-176` ставить ордер і **одразу** резолвить його по тому ж `closed`-бару;
`trader.py:224-226`: `limit = close[t]`, а тест дотику — `low[t] <= limit` / `high[t] >= limit`.
Оскільки `low ≤ close ≤ high` виконується за побудовою, «не торкнулось» перестає бути фільтром.
Заміряно на 26 306 кешованих 1h-барах BTCUSDT: `low < close` у 99.5%, `high > close` у 99.4%,
медіана `P(fill|touch) = 1.0`. У бектесті (`engine.py:205-216`) ліміт = `close[i-1]`, а тест —
по бару `i`, тобто хоча б наступний бар. Наявний `tests/test_paper_maker_parity.py:63-69`
**навмисно** передає H/L наступного бару й абсолютний `bar_idx`, тому прод-порядок
(«той самий closed») не покритий. Наслідок: `fill-rate paper = 100%` — властивість моделі, не ринку.

**M17. Funding у single-symbol paper/live не нараховується взагалі.** `apply_funding`
викликається лише в `paper_replay.py:119`, `pairs_engine.py:631,637`. `trader.py:287-290` передає
funding лише в `generate_signals`, тобто впливає на сигнал, але **не створює грошового потоку**.
Для `funding_carry` (edge якого і є funding) paper-рахунок за період утримання платить 0, тоді як
бектест (`engine.py:589-604`) має цей потік.

**M18. Funding у paper-шляхах лягає на 1 бар пізніше, ніж у бектесті.** `pairs_runner.py:185-193` і
`paper_replay.py:116` беруть інтервал `(prev, ts]`, а рушій — `searchsorted(rates.index, side="right") - 1`
(бар, що **містить** момент ставки). Індекс кешованого funding має суб-секундний джитер
(`LINKUSDT_funding.parquet`: `2026-06-20 00:00:00.001`), тому подія `16:00:00.001` для бектесту —
бар `16:00`, для paper — `17:00`. Підтверджено артефактом: у БД BTCUSDT funding-рядки лежать на
годинах 01/09/17, LINKUSDT — на 00/08/16.

**M19. Ковтання помилок у live-шляху з підстановкою «значення за замовчуванням як факт».**
`pending_orders.py:87-96` — `cancel()` ловить будь-який виняток і **безумовно** прибирає ордер із
журналу: якщо скасування на біржі не пройшло, бот «забуває» живий resting-ордер без retry, а
`reconcile_exchange_state` порівнює лише позиції. `pending_orders.py:195-197` — при WS-події FILLED
без `cumulative_filled_qty` букається **повний** розмір за ціною ліміту (невідоме = факт).
`pending_orders.py:233-234` — `_book_pending_fill` для `kind="close"` без гілки `else`: філ без
позиції тихо не букається, а виклик усе одно pop-ає ордер і звітує `filled`; `_normalize_pending_parts`
ще й переписує `close_pending` → `closed`. Тобто дія звітує «закрито» при незміненому стані.

**M20. `paper_audit` `fill_rate` не враховує `pending`.** `paper_audit.py:78-84` + `store.py:313-321`:
`fill_rate = filled/(filled+unfilled)`, а статус `pending` (22 рядки в реальній БД) не входить ні
в чисельник, ні в знаменник, хоча кожне котирування логують двічі. Тому 44 рядки orders дають
«fill-rate 100%», і жодне незавершене котирування не погіршує метрику.

---

## 3. Minor (стисло)

- `sensitivity.py:99-105`: `smooth = 1.0` за замовчуванням, якщо в сітці ≤2 робочих точок або
  `metric.std()==0`. Зламані точки глушаться (`except Exception: continue`, `:91-92`), тож комірка,
  де параметр не впливає на сигнал, отримує `smoothness=1.0` і проходить гейт `> 0.30`.
  Сітка будується лише по `next(iter(param_space))` (`cell_audit.py:505`) — 1 з до 6 параметрів.
- `cell_audit.py:494-496`: `if len(df_oos) < 200: df_oos = df` тихо повертає IS-забруднену
  повну вибірку в sensitivity без позначки у виводі.
- `cell_audit.py:592-594`: `except Exception: pbo = None`, а `cell_verdict` перевіряє PBO лише
  коли не `None` (`:301`) → **збій CSCV прибирає гейт** замість FAIL. Стратегії без `param_space`
  (`ensemble`) проходять `final` без PBO взагалі.
- `sweep.py:661`: `n_trials = len(cand)` — кількість кандидатів в одній таблиці, а не реальне
  число спроб (стратегії × символи × інтервали ≈ 1080) і не вміст `trial_ledger`. `z(180)=2.73`
  проти `z(1080)=3.27` → `sr0` занижено ~17%, `deflated_sharpe` завищено на ~0.54·sd(SR).
- `sweep.py:124,132`: перевірка `|Sharpe| > 20` лише в `mode == "backtest"`; WF-рядок з
  абсурдною шкалою не флагується.
- `optimize.py:85`: `purge`/`embargo` впливають лише на train-індекси, які `_default_objective`
  не використовує → весь апарат `_effective_purge` (`:52-64`) на результат не впливає.
- `verdict_store.py:48-56`: вердикт не містить параметрів аудиту (days, вікно, holdout_pct,
  train/test, n_trials), тому `live/audit_gate.py:61-74` може перевірити лише label + вік ≤30 діб.
  PASS, отриманий на 14 днях із забрудненим holdout, авторизує live для будь-якого вікна.
- `audit_extensions.py:106,112,123,180,192`: на помилці пишуться нейтральні `0.0`
  (`quintile_spearman=0.0` виглядає як виміряне ρ=0).
- `metrics.py:112`: при `std == 0` Sharpe повертається як рівно `0.0` — «мертва» серія
  не відрізняється від стратегії з нульовим edge. Це корінь K2 на рівні метрик.
- `metrics.py:370-374` (`_extract_trades`): для останньої незакритої позиції exit-комісія не
  нараховується → per-trade ret останньої угоди завищений (equity не страждає).
- `deflated_sharpe.py:142` використовує `ddof=1`, а `metrics.py:111-112` — `ddof=0`; на великих
  `n` несуттєво, але це два різні Sharpe в одному звіті.
- `walk_forward.py:232`: `degradation = (avg_is - avg_oos)/abs(avg_is)` — при від'ємному `avg_is`
  знак і зміст метрики інвертуються.
- `walk_forward.py:222`: перша дохідність кожного OOS-вікна примусово `0.0`
  (`pct_change().fillna(0.0)`), тож у конкатенованій серії 47 штучних нулів.
- `optimize.py:131,355`, `engine.py:484`: `cost or CostModel()` = клас-дефолти, не `from_settings`.
  За поточного `.env` числа збігаються — ризик латентний.
- `bars.py:41,120`: індекс = час **останньої** угоди (close-like), тоді як klines/resample/htf/engine
  скрізь трактують індекс як open time → зсув на бар при змішуванні з klines-логікою.
- `storage.py:146`: `validate_bars(df)` без `interval=` → `n_gaps` не рахується взагалі.
- `features/microstructure.py` F5 і `data/storage.py:80-82` (`_safe_load` ковтає помилки → `None`).
- `results/paper_equity_BTCUSDT.csv` містить **один рядок** (`2026-09-06 20:23:19, 10000.0`) —
  equity рівно = `initial_capital`, тобто жодної угоди; закомічений у git і малюється в дашборді
  як «Paper-run» (`app_pages/overview.py:574-581`). Причина — K6.
- `live/trader.py:228-241`: `chase_limit` приймається, документований як «= close попереднього
  бару» (parity з `_simulate_maker_fills`), але в тілі використовується лише `live.price` —
  модель «chase» у paper не реалізована. RNG-паритет maker-філів теж не виконується:
  `trader_loop.py:87` передає віконний індекс (`len(closed)-1`), а `trader.py:219-222` трактує
  його як абсолютний.
- `live/trader.py:128-132`: `LiveTrader.cost` створюється і не використовується — комісії беруться
  лише з `PaperAccount`. У taker-режимі (`MAKER_EXECUTION=false`) paper списує taker 5 bp/сторону,
  а бектест — `taker_fee + slippage_frac` = 7 bp/сторону, тобто **4 bp/round-trip оптимістично**.
- `live/account.py:237`: funding-ноціонал = `entry_price × size`, тоді як біржа бере mark-ноціонал,
  а бектест — `pos × equity`; для довгого утримання (LINK 7.664 → 12.3) funding недозараховано.
- `live/is_report.py:107-127`: «реалізований IS» = `limit_px` проти `mid` **бару філу**
  (`pairs_engine.py:336-337`), а не проти `decision_mid`; `_FILLED` містить загальний токен `"ok"`
  (`:13`). Тобто метрика міряє дрейф ринку за бар, і саме вона йде в `calibrate_from_is` →
  `CostModel.slippage_frac` (`execution.py:138-160`).
- Різний `initial_capital` у двох paper-шляхах: `PaperRunner` — 10 000 (`paper_runner.py:81-85`),
  `LiveTrader` — `position_pct*100_000` = **1000** (`trader.py:120-124`); абсолютні цифри незіставні.
- `paper_runner.py:104-108`: будь-який виняток кроку стає `action = f"error:{exc}"`, але equity
  все одно записується як точка вимірювання і CSV зберігається.
- `pairs_engine.py:219-236`: рядок `months` пишеться лише при переході місяця, тому поточний
  місяць у розкладі PnL відсутній (у БД вересневий funding −0.02 USDT не залоговано).
- `.env` містить 20 продубльованих ключів (`MAKER_FEE`, `TAKER_FEE`, `SLIPPAGE_BPS`,
  `POSITION_PCT`, `MAX_CONSECUTIVE_LOSSES`, … у двох блоках). Я перевірив — усі значення
  ідентичні, конфлікту немає, але це пастка на майбутнє.
- `results/iter8_validation.csv`: рядки `map.v2` і `map.v2:bp` **побайтово ідентичні**
  (ті самі 173 угоди, −0.5533, −0.0067), і обидва враховані в `iter8_validation_summary.csv`
  як окремі варіанти — подвійний облік одного результату.

---

## 4. Що підтверджено як коректне

1. **Рушій: лаг виконання правильний.** `engine.py:514` `target_pos = signals.shift(1)`,
   `:587` `strat_ret = pos * bar_ret - fees`, `:606` `equity = cumprod(1 + strat_ret)`.
   Позиція бару `t` вирішена на закритті `t-1` — без lookahead.
2. **Комісії:** `turnover = |Δpos|`, `fees = turnover × fee_rate` — одна сторона на кожну зміну
   позиції, без подвійного обліку. Maker-гілка (`:532-549`) справді симулює пропущені філи.
3. **Funding:** `engine.py:597` `searchsorted(rates.index, side="right") - 1` мапить ставку в бар,
   що її містить; мс-зсув `fundingTime` переноситься коректно (заміряно **0/3288** місмапів на
   сітках 1h/4h/8h/1d); кратні ставки сумуються рівно раз; знак `−pos × rate` правильний;
   `funding_carry` бере попередню ставку (`shift(1)`) — консервативно.
4. **Ресемплінг:** `label="left", closed="left"` — open-time конвенція, бар 17:00 містить
   [17:00, 18:00); `drop_incomplete` відкидає неповний останній бін. Жодного `bfill`/`interpolate`
   в `data/` і `features/` (лише `ffill`).
5. **HMM `filtered_proba` каузальний** (`hmm_regime.py:198-212`: `alpha[t] = (alpha[t-1] @ A) * B[t]`);
   згладжені `predict_proba`/Viterbi/post eriors існують, але жоден production-шлях їх не кличе.
   (Проблема — in-sample **fit**, див. M2, а не inference.)
6. **Rule-based фічі режиму каузальні:** `volatility_regime`, `trend_strength`, `apply_min_dwell`,
   `_liquidity_label_from_volume`, `_funding_skew_series` — усі rolling/минуле.
7. **`htf_market_structure`** (`regimes.py:131-137`) мапить htf-мітки лише через **закриті**
   вікна (`searchsorted(..., side="right") - 1`).
8. **Walk-forward нарізка без lookahead:** train/test — позиційні зрізи; `trades`/`funding`
   ріжуться **за часом** (`walk_forward.py:175-179`), тобто старий баг `trades.iloc[...]` виправлено.
   Сигнали для не-event стратегій рахуються один раз на повній історії (`:160-164`) — для
   ковзних індикаторів це каузально; `ml_strategy` має власний внутрішній walk-forward
   (`ml_strategy.py:190-212`), тому «модель бачить повну історію» там коректно.
9. **`hedge_blend_signals`** (`strategies/blend.py`) каузальний: `p = np.vstack([1/n, p[:-1]])`
   зсуває ваги правильно, expert-return = `sig[t-1]·ret[t]`, як у рушії. (Адаптивний `eta`
   рахується з повної `loss`-матриці — скалярна калібровка, дрібний lookahead.)
10. **CSCV/PBO** (`cscv.py`) реалізований правильно: ω — зростаючий ранг OOS Sharpe IS-кращого
    серед усіх варіантів того ж спліту, λ = logit(ω), PBO = частка λ<0. Претензія не до формули,
    а до складу матриці в `iter7_regime_analysis.py:306` (див. K1).
11. **`deflated_sharpe_ratio`** сам по собі коректний: пер-барний `sr = mean/std`, `sr0 = √V·z(N)`,
    `kurt = pandas.kurt() + 3` (`:146-148`) — конверсія excess → raw зроблена. Проблема лише
    в тому, **що** йому подають (M5) і на якій серії (K1).
12. **Дегенеративні клітинки реально виключаються** в `sweep.py` (`flag_degenerate_row:100-138`,
    NaN-safe) і в `sweep_winners_haircut`/`save_sweep_report`/`SweepStore.summary`; resume
    перевіряє `code_hash` (`already_done`) — пастка «воркер тримає старий код» закрита.
13. **`cv.py` відповідає AFML:** `purged_kfold_indices` прибирає `purge` перед test і `embargo`
    після; `PurgedKFold.split` додатково викидає train-зразки з `t1 > t_test_start`.
14. **`audit_extensions.py` quintile/time-decay без lookahead:** зсунуто **дохідність**
    (`close.pct_change().shift(-1)`, `fwd = -ratio.diff().shift(-1)`), а не рішення — правильний напрямок.
15. **`trial_ledger` / `oos_registry` / `verdict_store`** коректні як append-only журнали;
    `Path("") == Path(".")`-пастка закрита в `_enabled_path`/`_env_path`.
16. **Дані:** 1m-кеш без дірок (BTC 1 578 295 барів, усі кроки рівно 60 с); `readonly=True`
    справді не торкається мережі; `_fetch_klines_windows` відкидає `ts >= end_ms` (без overlap);
    fail-fast `_require_funding_coverage` (поріг 0.90) працює; аудит порівнює з клієнтом,
    побудованим з `DATA_EXCHANGE` під `require_live_data_exchange` (`config.py:266-283`).
17. **Арифметична цілісність артефактів:** 200/200 клітинок `iter7` перераховано з parquet —
    `pooled_oos_sharpe`, `oos_total_return`, `oos_max_dd` збігаються з шардами (0 розбіжностей);
    `iter7_rating.csv` сходиться з шардами до 1e-16. Тобто проблема **методологічна, а не в
    арифметиці агрегації**.
18. **Лаг single-symbol paper/live відповідає бектесту.** `trader_bars.closed_klines:22-43`
    відкидає формуючу свічку; `execute_signal` виконує за `close` того самого бару
    (`trader_loop.py:103-110`) → позиція діє з наступного бару, як `signals.shift(1)` у рушії.
    Той самий лаг і в pairs (`PairsEngine._quote` ставить ліміт = `close[t]`, `_resolve_pending`
    спрацьовує на `t+1`; `replay_pairs` не додає зайвого зсуву).
19. **Облік комісій у `PaperAccount` не подвоюється.** Вхід: `_cash -= fee` (`account.py:142-143`);
    вихід: `_cash += price_pnl - fee` (`:194`), а `_realized_pnl` отримує
    `price_pnl - exit_fee - entry_fee` (`:191-195`) — журнал у cash удруге не додається; exit-комісія
    присутня завжди, часткові закриття розподіляють `entry_fee` пропорційно (`:190-191,215-220`).
20. **Funding у pairs/replay: знак і каденція правильні.** `pnl = -side × rate × notional`
    (`account.py:239`) — лонг платить додатній funding, шорт отримує (у БД: short LINK +16.95 при
    переважно додатніх ставках, long BTC −0.14); `_funding_between` підсумовує ставки інтервалу й
    подає їх один раз на бар; `if pos is None: return 0.0` — хибні напрямки не створюють платежів.
21. **DRY_RUN/live-гейти на paper-шляхах виконуються:** `PaperRunner.__init__` кидає `RuntimeError`
    при `dry_run=False` (`paper_runner.py:72-77`), `cmd_paper`/`cmd_paper_run` — `SystemExit`
    (`cli/paper.py:46-49,71-75`), `PairsPaperRunner.__init__` — `RuntimeError`
    (`pairs_runner.py:339-344`); `_submit_order` у dry_run не торкається біржі (`trader.py:594-615`);
    `reconcile_exchange_state`/`halt_if_drift` у live без `fetch_positions` — fail-closed
    (`reconcile.py:97-110`); live-`KillSwitch` не ковтається (`pairs_runner.py:838-843,868-871`);
    `control.py` fail-closed при битому JSON. `paper-replay*` ордерів не ставлять узагалі.
22. **`DrawdownBreaker`** монотонний (latch до `reset`) і не дає «відліку» відскоком
    (`risk_gate.py:74-100`); закриття ніколи не блокується (`trader.py:436-438`,
    `trader_loop.py:120-135`), daily/weekly/cooldown/attrition блокують лише входи.
23. **`money.py`**: `Decimal` через `str`, 8 знаків, `ROUND_HALF_UP` — грошові поля не проходять
    через float-дрифт (`account.py:69-72,140-144`). `orders.py::is_transient_exchange_error`
    коректно розрізняє `ccxt.NetworkError` (ковтається) від решти (піднімається).
24. **Внутрішня арифметика `paper_pairs.sqlite` зістикована:** `10189.52 = 10326.03 + (−136.51)`,
    `327.27 = 310.46 + 16.81`, 2160 барів без дір і дублікатів ts, `months` 06+07+08 = 327.29
    (розбіжність 0.02 = незалогований вересневий funding).

---

## 5. Що це означає для наявних висновків

| Твердження в `docs/reports/` | Статус після аудиту |
|---|---|
| T1 «Regime-conditional selector», switch SR +2.64, 30/30 rolling > 0, «PASS / Пріоритет №1» | **Спростовано.** SR +0.36/+0.41 на коректному лагу, DSR 0.00, PBO 0.66. Edge як гіпотеза лишається, статус — «не перевірено» |
| Розділ 5.3 «Анти-перенавчання: DSR 0.821, PBO 0.004» | **Не valid.** DSR на lookahead-серії; PBO рахувався разом із цією ж серією як кандидатом |
| T3 «mean_reversion, hmm_reversion ≈0 (2 і 1 угода)» | Числа правильні, але це не «edge ≈0», а **degenerate** (нижче `MIN_TRADES=30`); плюс для `cvd_momentum`/`ob_imbalance` «0» — артефакт покриття aggTrades |
| `iter8`: «карта не переноситься на свіжі символи» | Висновок правдоподібний, але базис витрат інший (тейкер замість maker) і код іншої ревізії → порівняння з iter7 некоректне |
| `pairs_arb` як «єдине валідоване ядро» (PBO 0.000, WF>0 у 65% вікон) | Не перевірявся в цьому аудиті глибоко; відомо, що його WF-шкала (`sharpe_hourly`) не сумісна з рештою, а провали вікон глушаться в 0.0 (M7). Потребує окремого прогону |
| «Paper ≥8 тижнів» / тижневий paper-гейт | **Недійсний.** Артефакт — 90-денний in-sample replay по тому ж кешу (K7), демони стоять на 2019 році (K6), порівняння з бектестом не виконується без `--bt-equity` (M14), `[ok]` для cost-hint — тавтологія (M15) |

---

## 6. Пріоритет виправлень

**P0 (без цього решта досліджень не має сенсу):**
1. `iter7_regime_analysis.py`: лагувати **вибір** (`reg.map.shift(1)` → дохідність бару `t`), а не
   серію дохідностей; у `switch_test` додати лаг узагалі. Перерахувати `iter7_switch_*`,
   `iter7_rolling_switch`, `iter7_dsr_pbo`.
2. Прибрати селектор із власної PBO-матриці (`:306`) — PBO рахувати лише по реальних кандидатах.
3. `data/downloader.py:790` + `binance_vision.py:197,200`: дедуп і курсор пагінації за `trade_id`,
   не за мілісекундним індексом (`:827` — `fromId`/`trade_id+1`). Потім перекачати aggTrades.
4. Вмикати дегенерацію в експериментах: викликати `flag_degenerate_row` (або еквівалент) в
   `iter7_regime_rating.run_cell` — 0 угод / `< MIN_TRADES` → `status="degenerate"`, і не
   включати в рейтинг.
5. Валідувати trades/funding **на кожному розрізаному вікні**, а не на повному фреймі:
   `_generate_signals(..., trades=trades_tr)` замість повного `trades`
   (`walk_forward.py:164` vs `:175-179`).
6. **`since_ms=0` → реальний `since`** у `paper_runner._fetch_recent`, `pairs_runner._fetch_recent`,
   `pairs_live._fetch_last_price` (і `limit=1` замінити на ticker/останній бар). Це блокує
   будь-який сенс форвард-тесту.
7. Очистити `results/paper_pairs.sqlite` від синтетичних тестових фікстур (тести мають писати в
   `tmp_path`), а тижневий гейт перевести на справжній forward-артефакт (демон + `--bt-equity`).

**P1:**
8. `HOLDOUT_PCT` > 0 і `OOS_ENFORCE_BURN=true`; заборонити `explicit_holdout` при `holdout_pct<=0`;
   писати параметри аудиту (вікно, holdout_pct, n_trials) у `verdict_store`.
9. Записувати `code_version`/хэш даних у кожен CSV експерименту — щоб застарілість була видима.
10. `RegimeDetector`: фітити HMM лише на даних строго до зони оцінювання (або відкидати перші
    `hmm_fit_bars` барів у WF).
11. Уніфікувати cost-модель: скрізь `CostModel.from_settings(settings, df=df)` і явний `is_maker`
    з одного джерела; прибрати `is_maker=False` за замовчуванням у валідаційних шляхах.
12. `sweep.py:661`: `n_trials` брати з реального обсягу перебору (стратегії × символи × інтервали
    + `trial_ledger`), а не з розміру однієї таблиці.
13. `paper_audit`: зробити `--bt-equity` обов'язковим (або друкувати `gate=NOT_EVALUATED` з
    ненульовим exit code); не друкувати `[ok]` для cost-hint при нулі IS-записів.
14. `paper_replay`: прибрати «+1 бар» і передати `trades` у `compute_metrics`.
15. Резолвити maker-pending лише по **наступному** бару (як у `test_paper_maker_parity`) з
    абсолютним `bar_idx`; додати funding у `LiveTrader`/`trader_loop` з тією ж каденцією, що в
    `pairs_engine`, і вирівняти межу funding-бару з `searchsorted(side="right")-1`.
16. `pending_orders.cancel` — не видаляти ордер при невдалому скасуванні (retry + алерт);
    у WS-філах не підставляти `po.size`/`po.price` як факт.

**P2:**
17. Ввести явну шкалу Sharpe в усі таблиці (`sharpe_ann` / `sharpe_window_t` / `sharpe_bar`) і
    маркувати нею колонки `sweep`-звіту.
18. `buy_and_hold_sharpe(df, periods_per_year=...)` — передавати ppy, визначений з індексу.
19. `regime_map`: `labels.shift(1)`, і не давати «мертвій» комірці Sharpe рівно 0.0, що проходить
    `hard_off_sharpe=0.0`.
20. `pairs.py:350-361`: не ковтати виняток у `0.0` — писати `status="error"` і виключати вікно з
    обох середніх.
21. `microstructure.py:284-286`: барний розмір VPIN — з rolling-вікна з `shift(1)`, не з медіани
    всієї серії.
22. Обрізати базовий кеш до `days` **до** ресемплу (M9) або відкидати перший бар, чий бін
    покритий джерелом не повністю.
23. `sensitivity.py`: `smooth` = `nan` (а не 1.0), якщо робочих точок ≤2 або `std==0`;
    вважати провали точок явними, а не мовчазними.
24. `cell_audit.py:592-594`: збій CSCV → FAIL, не «гейт зникає».
25. `paper_audit.fill_rate`: включати `pending` у знаменник; `LiveTrader` має читати `self.cost`
    (зараз taker-режим на 4 bp/round-trip оптимістичніший за бектест).

---

## 7. Метод аудиту (для відтворення)

- Перерахунок усіх 200 клітинок `iter7` із `results/iter7_oos/*.parquet` і звірка з
  `iter7_shard_*.csv` / `iter7_rating.csv` (0 розбіжностей).
- Реконструкція `switch_test` і `rolling_switch` із тих самих parquet у 4 варіантах лага
  (без лага / `shift(1)` на дохідності / лаг вибору 1 бар / лаг вибору 2 бари).
- Перерахунок DSR і PBO з матрицею кандидатів у двох складах (з селектором і без нього).
- End-to-end перепрогін однієї WF-клітинки (`BTCUSDT`, 1h, 1095 днів, 2500/500, maker) і
  порівняння з шардом.
- Порівняння taker/maker на ATOMUSDT/NEARUSDT (`funding_carry`, 500/200, 400 днів).
- Фальсифікаційний тест flow-стратегій на 74 барах, що перекривають наявні aggTrades.
- Перевірка покриття `trade_id` в aggTrades-кеші (BTC/ETH/DOGE).
- `pytest tests/ -q` — 1428 passed, 9 skipped, 1 флейкі-таймінг.
- Читацький аудит `validation/*` (33 модулі), `data/*`, `features/*`, `live/paper*` —
  окремими проходами, без змін файлів і без мережі.
- Аудит форвард-тесту: читання `results/paper_pairs.sqlite` (копія в `/tmp`, щоб не писати в
  репо-БД), звірка `orders.price` із кешованими close, перевірка каденції й дір в `equity`,
  перевірка `months`/`realized`/`UPNL` на зістикованість, аналіз `pending_orders`/`fills`/
  `is_report`/`paper_audit` і перевірка обробки `since` у ccxt
  (`ccxt/binance.py:4989-4990`: `if since is not None: request['startTime'] = since`).

Скрипти аудиту: `/tmp/audit/sw_check.py`, `/tmp/audit/roll_check.py`,
`/tmp/audit/pbo_dsr_check.py`, `/tmp/audit/csv_vs_parquet.py`, `/tmp/audit/repro_cell.py`.

**Обмеження аудиту.** Семантика `startTime=0` на боці Binance узята з контракту REST API та
сорсу ccxt; мережевих запитів не робилося (локальні наслідки — 2019-вікно, `hold:same_bar`
назавжди, `_fetch_last_price` з першої свічки — виводяться з коду й узгоджуються з
1-рядковим CSV). Те, який саме скрипт записав `results/paper_pairs.sqlite`, прямо не доведено:
висновок «це replay, а не демон» спирається на бар-вирівняні ts, рівно 90 діб без розривів,
збіг ліміт-цін із кешованими close і на те, що поточний демон-шлях (K6) далі першого бару не
просувається. `pairs_arb` як «єдине валідоване ядро» окремо не перепрогонювався — це
наступний за пріоритетом аудит.
