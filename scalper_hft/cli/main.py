"""Argparse-складальник CLI (точка входу `python -m scalper_hft.cli`)."""

from __future__ import annotations

import argparse

from scalper_hft.cli._common import _add_overlay_flag, _parse_param_dict


def main(argv: list[str] | None = None) -> None:
    import scalper_hft.cli as _cli_pkg  # call-time резолюція (тести патчать пакет)

    parser = argparse.ArgumentParser(
        prog="scalper-hft", description="Високочастотна скальпінг-система (Binance USDT-M)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--exchange", default=None, help="Біржа, e.g. binance, bybit (за замовч. з .env)")
        p.add_argument("--symbol", default=None, help="Символ, e.g. BTCUSDT (за замовч. з .env)")
        p.add_argument("--interval", default=None, help="Таймфрейм: 1s/5s/1m/5m/15m/1h")
        p.add_argument(
            "--strategy",
            default="mean_reversion",
            help="Стратегія з реєстру: mean_reversion, cvd_momentum, pairs_arb, ...",
        )
        p.add_argument("--days", type=int, default=60, help="Глибина історії, днів")
        p.add_argument(
            "--base",
            default="1m",
            help="Базовий таймфрейм для ресемплінгу (за замовч. 1m)",
        )
        p.add_argument(
            "--derive",
            dest="derive",
            action="store_true",
            default=True,
            help="Ресемплити старші таймфрейми з --base (за замовч.)",
        )
        p.add_argument(
            "--no-derive",
            dest="derive",
            action="store_false",
            help="Качати цільовий інтервал з Binance замість ресемплінгу з --base",
        )
        p.add_argument("-p", "--param", action="append", default=[], help="Параметр стратегії: key=value")

    p = sub.add_parser("download", help="Завантажити klines/aggTrades/funding")
    add_common(p)
    p.add_argument("--trades", action="store_true", help="Також aggTrades")
    p.add_argument("--trades-days", type=int, default=None, help="Глибина aggTrades (Binance обмежує 2 доби)")
    p.add_argument("--funding", action="store_true", help="Також funding")
    p.add_argument("--force", action="store_true", help="Оновити хвіст навіть якщо кеш уже покриває період")
    p.add_argument("--retries", type=int, default=None, help="Кількість спроб при помилках/лімітах (за замовч. 8)")
    p.add_argument("--delay", type=float, default=None, help="Затримка між батчами у сек (за замовч. 0.15)")
    p.add_argument(
        "--checkpoint-batches", type=int, default=None, help="Періодичність збереження чекпоінтів (за замовч. 50)"
    )
    p.add_argument("--vision", action="store_true", help="Історичні aggTrades з data.binance.vision")
    p.add_argument("--vision-start", default=None, help="YYYY-MM-DD початок Vision-дампів")
    p.add_argument("--vision-freq", default="daily", choices=["daily", "monthly"])
    p.set_defaults(func=_cli_pkg.cmd_download)

    p = sub.add_parser("backtest", help="Запустити бектест")
    add_common(p)
    _add_overlay_flag(p)
    p.add_argument(
        "--breakeven-gate", action="store_true", help="Вимикати сигнали, де очікуваний рух (ATR) < round-trip витрат"
    )
    p.add_argument(
        "--bar-type",
        default="time",
        choices=["time", "dollar", "volume"],
        help="Тип барів для бектесту (time, dollar, volume). Для не-time використовується aggTrades",
    )
    p.add_argument("--bar-threshold", type=float, default=100000.0, help="Поріг для об'ємних або доларових барів")
    p.add_argument(
        "--intrabar",
        action="store_true",
        help="Внутрішньобарові SL/TP: вихід за ціною рівня на барі дотику (песимістично SL при SL+TP разом)",
    )
    p.add_argument(
        "--queue-model",
        action="store_true",
        dest="queue_model",
        help="Maker-філи через QueuePositionModel (черга/VPIN-проксі): touch-through не гарантує філ",
    )
    p.add_argument(
        "--spread-bps", type=float, default=2.0, dest="spread_bps", help="Оцінка спреду (bps) для queue-моделі"
    )
    p.add_argument(
        "--vol-ref",
        type=float,
        default=0.0,
        dest="vol_ref",
        help="Референс-волатильність для vol-aware slippage (0 = вимкнено)",
    )
    p.add_argument("--enqueue", action="store_true", help="Поставити в чергу jobs.sqlite і вийти (не рахувати тут)")
    p.set_defaults(func=_cli_pkg.cmd_backtest)

    p = sub.add_parser("plot", help="Інтерактивний HTML-графік бектесту (свічки+індикатори+угоди+SL/TP)")
    add_common(p)
    _add_overlay_flag(p)
    p.add_argument("--out", default="docs/plots/backtest.html", help="Шлях до HTML-файлу")
    p.add_argument(
        "--bars", type=int, default=20_000, help="Максимум барів на графіку (даунсемплінг; бари угод зберігаються)"
    )
    p.add_argument("--start", default=None, help="Початок вікна, ISO: YYYY-MM-DD[ HH:MM]")
    p.add_argument("--end", default=None, help="Кінець вікна, ISO: YYYY-MM-DD[ HH:MM]")
    p.add_argument("--no-trades", action="store_true", help="Не малювати точки входу/виходу")
    p.add_argument("--no-sl-tp", action="store_true", help="Не малювати рівні SL/TP")
    p.set_defaults(func=_cli_pkg.cmd_plot)

    p = sub.add_parser("walkforward", help="Walk-forward аналіз")
    add_common(p)
    _add_overlay_flag(p)
    p.add_argument("--train", type=int, default=None, help="Барів у train (IS). Пропуск = per-TF дефолт")
    p.add_argument("--test", type=int, default=None, help="Барів у test (OOS). Пропуск = per-TF дефолт")
    p.add_argument("--use-kalman", action="store_true", help="PairsArb: динамічний Kalman hedge ratio")
    p.set_defaults(func=_cli_pkg.cmd_walkforward)

    p = sub.add_parser("optimize", help="Оптимізація параметрів (Optuna)")
    add_common(p)
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--splits", type=int, default=4)
    p.set_defaults(func=_cli_pkg.cmd_optimize)

    p = sub.add_parser("ml-opt", help="Оптимізація параметрів ML-моделі (AFML)")
    add_common(p)
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--embargo", type=float, default=0.01)
    p.add_argument("--scoring", default="neg_log_loss")
    p.add_argument("--decay", type=float, default=0.9)
    p.add_argument("--frac-d", type=float, default=0.4)
    p.add_argument("--no-frac-diff", action="store_true")
    p.add_argument("--trades", action="store_true", help="Використовувати aggTrades")
    p.add_argument("--sampler", default="tpe", choices=["tpe", "random"])
    p.set_defaults(func=_cli_pkg.cmd_ml_opt)

    p = sub.add_parser("overfit", help="Повний аудит на перенавчання")
    add_common(p)
    p.add_argument("--train", type=int, default=2000)
    p.add_argument("--test", type=int, default=500)
    p.add_argument(
        "--purge-bars",
        type=int,
        default=None,
        help="AFML прогін train→test (барів). За замовч. max(1, 1%% test-вікна); 0 = вимкнути",
    )
    p.add_argument(
        "--embargo-bars",
        type=int,
        default=None,
        help="AFML ембарго між OOS-вікнами (барів). За замовч. max(1, 1%% test-вікна); 0 = вимкнути",
    )
    p.add_argument("--trials", type=int, default=50, help="Оцінка кількості спроб для DSR")
    p.add_argument("--use-kalman", action="store_true", help="PairsArb: динамічний Kalman hedge ratio")
    p.add_argument("--leg1", default=None, help="Pairs-стратегії: перша нога (для авто pair-вердикту)")
    p.add_argument("--leg2", default=None, help="Pairs-стратегії: друга нога (для авто pair-вердикту)")
    p.add_argument("--enqueue", action="store_true", help="Поставити в чергу jobs.sqlite і вийти (не рахувати тут)")
    p.set_defaults(func=_cli_pkg.cmd_overfit)

    p = sub.add_parser("cscv", help="PBO через Combinatorial Purged CV")
    add_common(p)
    p.add_argument("--variants", type=int, default=30, help="Кількість випадкових варіантів параметрів")
    p.add_argument("--blocks", type=int, default=8, help="Кількість блоків для розбиття")
    p.add_argument("--max-combos", type=int, default=200, help="Обмеження комбінацій")
    p.add_argument(
        "--purge-bars",
        type=int,
        default=None,
        help="AFML purge навколо test-блоків. За замовч. max(1, 1%% ряду); 0 = вимкнути",
    )
    p.add_argument(
        "--embargo-bars",
        type=int,
        default=None,
        help="AFML embargo після test-блоків. За замовч. max(1, 1%% ряду); 0 = вимкнути",
    )
    p.set_defaults(func=_cli_pkg.cmd_cscv)

    p = sub.add_parser(
        "regime-backtest",
        help="Порівняльний бектест: базові стратегії vs RegimeSupervisor",
    )
    p.add_argument(
        "--strategies",
        default="mean_reversion,supertrend,hmm_reversion",
        help="Стратегії через кому (default: mean_reversion,supertrend,hmm_reversion)",
    )
    p.add_argument("--symbol", default="BTCUSDT", help="Символ")
    p.add_argument("--interval", default="1h", help="Таймфрейм")
    p.add_argument("--days", type=int, default=180, help="Кількість днів")
    p.add_argument(
        "--blend-mode",
        default="all",
        choices=["all", "regime_soft", "contextual_hedge", "exp3"],
        help="Режим зважування supervisor-а (default: all — запускає всі три)",
    )
    p.add_argument("--n-hmm-states", type=int, default=3, help="Кількість HMM-станів")
    p.add_argument("--hmm-fit-bars", type=int, default=2000, help="Бари для навчання HMM")
    p.add_argument("--no-regime-table", action="store_true", help="Не виводити таблицю по режимах")
    p.add_argument("--save", action="store_true", help="Зберегти результати у results/")
    p.add_argument("--base", default=None, help="Базовий інтервал для деривації (напр. 1m)")
    p.add_argument("--no-derive", dest="derive", action="store_false", help="Не деривувати ТФ")
    p.set_defaults(func=_cli_pkg.cmd_regime_backtest, derive=True)

    p = sub.add_parser("record-bookticker", help="Запис bookTicker/depth5 (WS) у parquet")
    p.add_argument("--symbol", default="BTCUSDT", help="Символ(и) через кому")
    p.add_argument("--minutes", type=int, default=60, help="Тривалість запису, хв")
    p.add_argument("--depth", action="store_true", help="Записувати depth5 (5 рівнів стакана) замість bookTicker")
    p.set_defaults(func=_cli_pkg.cmd_record_bookticker)

    p = sub.add_parser("macro-recorder", help="Запис ліквідацій (forceOrder WS) та OI у parquet")
    p.add_argument("--symbol", default="BTCUSDT", help="Символ(и) через кому")
    p.add_argument("--minutes", type=int, default=60, help="Тривалість запису, хв")
    p.set_defaults(func=_cli_pkg.cmd_macro_recorder)

    p = sub.add_parser("download-liquidations", help="Завантажити історичні ліквідації з Binance Vision")
    p.add_argument("--symbol", default="BTCUSDT", help="Символ(и) через кому")
    p.add_argument("--days", type=int, default=30, help="Глибина історії, днів")
    p.set_defaults(func=_cli_pkg.cmd_download_liquidations)

    p = sub.add_parser("download-oi", help="Завантажити історичний Open Interest (REST API)")
    p.add_argument("--symbol", default="BTCUSDT", help="Символ(и) через кому")
    p.add_argument("--days", type=int, default=30, help="Глибина історії, днів")
    p.set_defaults(func=_cli_pkg.cmd_download_oi)
    p = sub.add_parser("arb", help="Delta-neutral funding arb (перп+спот)")
    add_common(p)
    p.add_argument("--position-pct", type=float, default=None, help="Ноціонал кожної ноги (за замовч. 0.1)")
    p.add_argument("--maker", action="store_true", help="Комісії maker (post-only) на обох ногах")
    p.add_argument("--walkforward", action="store_true", help="Додатково walk-forward")
    p.add_argument("--train", type=int, default=20000)
    p.add_argument("--test", type=int, default=5000)
    p.set_defaults(func=_cli_pkg.cmd_arb)

    p = sub.add_parser("pairs", help="Статистичний арбітраж пар перпів (BTC/ETH/SOL)")
    add_common(p)
    p.add_argument("--leg1", default="BTCUSDT", help="Перша нога")
    p.add_argument("--leg2", default="ETHUSDT", help="Друга нога")
    p.add_argument("--position-pct", type=float, default=None, help="Ноціонал кожної ноги")
    p.add_argument("--maker", action="store_true", help="Комісії maker (post-only)")
    p.add_argument("--walkforward", action="store_true", help="Додатково walk-forward")
    p.add_argument("--train", type=int, default=1500)
    p.add_argument("--test", type=int, default=500)
    p.add_argument(
        "--use-kalman", action="store_true", help="Динамічний Kalman hedge ratio (дефолт off до OOS bake-off)"
    )
    p.add_argument("--enqueue", action="store_true", help="Поставити в чергу jobs.sqlite і вийти")
    p.set_defaults(func=_cli_pkg.cmd_pairs)

    p = sub.add_parser("run", help="Dry-init перевірка YAML конфіга supervisor (торгівлю НЕ запускає)")
    p.add_argument("--config", required=True, help="Шлях до YAML файлу (напр. configs/strategies/example.yaml)")
    p.set_defaults(func=_cli_pkg.cmd_run)

    p = sub.add_parser("pairs-portfolio", help="Бектест портфеля валідованих пар")
    add_common(p)
    p.add_argument("--position-pct", type=float, default=0.3)
    p.add_argument(
        "--method",
        default="equal",
        choices=["equal", "erc"],
        help="Алокація: рівні ваги або Equal Risk Contribution (Narang гл. 6)",
    )
    p.add_argument(
        "--turnover-rate", type=float, default=0.0, help="Штраф за зміну ваг при місячному ребалансі (частка капіталу)"
    )
    p.add_argument("--no-rebalance", action="store_true", help="Без ребалансу ваг")
    p.set_defaults(func=_cli_pkg.cmd_pairs_portfolio, interval="1h")

    p = sub.add_parser("paper-run-pairs", help="Paper pairs (maker, 2 ноги) або --portfolio")
    add_common(p)
    p.add_argument("--leg1", default="XRPUSDT")
    p.add_argument("--leg2", default="BTCUSDT")
    p.add_argument("--portfolio", action="store_true", help="Три валідовані пари на спільному рахунку")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--sleep", type=int, default=300, help="Пауза між кроками, сек (для 1h — 300+)")
    p.add_argument("--notify", action="store_true")
    p.add_argument(
        "--daemon",
        action="store_true",
        help="Нескінченний цикл до SIGTERM; ігнорує --iterations (для systemd)",
    )
    p.add_argument(
        "--control",
        default="results/control.json",
        help="Шлях до control.json (pause / no_new_entries / flatten)",
    )
    p.set_defaults(func=_cli_pkg.cmd_paper_run_pairs, strategy="pairs_arb", interval="1h")

    p = sub.add_parser("paper-replay-pairs", help="Історичний paper pairs з моделлю unfilled")
    add_common(p)
    p.add_argument("--leg1", default="XRPUSDT")
    p.add_argument("--leg2", default="BTCUSDT")
    p.add_argument("--notify", action="store_true")
    p.set_defaults(func=_cli_pkg.cmd_paper_replay_pairs, strategy="pairs_arb", interval="1h")

    p = sub.add_parser("paper-audit", help="Tracking error paper SQLite vs бектест + MAE/MFE forensics")
    p.add_argument("--db", default="results/paper_pairs.sqlite", help="Шлях до paper SQLite")
    p.add_argument("--bt-equity", default=None, help="CSV ts,equity бектесту за той самий період")
    p.add_argument("--bt-fill-rate", type=float, default=None, dest="bt_fill_rate")
    p.add_argument("--dd-mult", type=float, default=1.5, dest="dd_mult", help="Paper maxDD ≤ BT×mult")
    p.set_defaults(func=_cli_pkg.cmd_paper_audit)

    p = sub.add_parser("experiments", help="Каталог val→OOS експериментів (sparse_basket / ml_strategy)")
    p.add_argument("--catalog", default=None, help="Markdown каталог (за замовч. docs/reports/experiments.md)")
    p.set_defaults(func=_cli_pkg.cmd_experiments)

    p = sub.add_parser("ml", help="Walk-forward ML-класифікатор: Triple-Barrier + LightGBM + AFML")
    add_common(p)
    p.add_argument(
        "--mode",
        default="triple_barrier",
        choices=["triple_barrier", "horizon"],
        help="Режим лейблінгу: triple_barrier (AFML, default) або horizon",
    )
    p.add_argument("--pt", type=float, default=1.0, help="Profit-take множник (× ATR)")
    p.add_argument("--sl", type=float, default=1.0, help="Stop-loss множник (× ATR)")
    p.add_argument("--holding", type=int, default=10, help="Вертикальний бар'єр (барів)")
    p.add_argument("--decay", type=float, default=0.9, help="Time-decay для sample weights")
    p.add_argument("--frac-d", type=float, default=0.4, dest="frac_d", help="Ступінь fractional differencing")
    p.add_argument("--no-frac-diff", action="store_true", dest="no_frac_diff", help="Вимкнути frac_diff фічі")
    p.add_argument("--train", type=int, default=2000)
    p.add_argument("--test", type=int, default=500)
    p.add_argument("--trades", action="store_true", help="Використати aggTrades (CVD + micro фічі)")
    p.add_argument("--hmm", action="store_true", help="Додати HMM-режими (каузальні, без lookahead)")
    p.add_argument("--garch", action="store_true", help="Додати GARCH σ_{t+1} (без lookahead)")
    p.add_argument("--hmm-states", type=int, default=3, dest="hmm_states")
    p.set_defaults(func=_cli_pkg.cmd_ml)

    p = sub.add_parser("paper", help="Один крок paper trading на останньому барі")
    add_common(p)
    p.set_defaults(func=_cli_pkg.cmd_paper)

    p = sub.add_parser("paper-run", help="Циклічний paper-прогін (кілька кроків)")
    add_common(p)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--sleep", type=int, default=60, help="Пауза між кроками, сек")
    p.add_argument("--notify", action="store_true", help="Telegram-сповіщення після прогіну")
    p.set_defaults(func=_cli_pkg.cmd_paper_run)

    p = sub.add_parser("paper-replay", help="Відтворення історії через risk-трейдера")
    add_common(p)
    p.add_argument("--position-pct", type=float, default=None, help="Частка капіталу на позицію")
    p.add_argument("--notify", action="store_true", help="Telegram-сповіщення результату")
    p.set_defaults(func=_cli_pkg.cmd_paper_replay)

    p = sub.add_parser("report", help="Markdown-звіт у docs/reports/")
    add_common(p)
    p.add_argument("--train", type=int, default=2000)
    p.add_argument("--test", type=int, default=500)
    p.add_argument("--trials", type=int, default=50)
    p.set_defaults(func=_cli_pkg.cmd_report)

    p = sub.add_parser("cohort", help="Cohort analysis: деградація edge за когортами угод")
    add_common(p)
    p.add_argument("--freq", default="ME", help="Частота когорт: ME (місяць), W (тиждень), D")
    p.set_defaults(func=_cli_pkg.cmd_cohort)

    p = sub.add_parser("lift", help="Децильний lift-аналіз фіч (відбір/фільтри входу)")
    add_common(p)
    p.add_argument("--bins", type=int, default=10, help="Кількість бінів (децилів)")
    p.add_argument("--top", type=int, default=10, help="Скільки топ-фіч показати")
    p.add_argument("--detail", action="store_true", help="Показати повні таблиці lift по фічах")
    p.set_defaults(func=_cli_pkg.cmd_lift)

    p = sub.add_parser("featimp", help="MDI/MDA/SFI feature importance (AFML Ch.8)")
    add_common(p)
    p.add_argument("--pt", type=float, default=1.0)
    p.add_argument("--sl", type=float, default=1.0)
    p.add_argument("--holding", type=int, default=10)
    p.add_argument("--decay", type=float, default=0.9)
    p.add_argument("--frac-d", type=float, default=0.4, dest="frac_d")
    p.add_argument("--no-frac-diff", action="store_true", dest="no_frac_diff")
    p.add_argument("--splits", type=int, default=4)
    p.add_argument("--embargo", type=float, default=0.01)
    p.add_argument("--trades", action="store_true", help="Використати aggTrades (CVD фічі)")
    p.set_defaults(func=_cli_pkg.cmd_featimp)

    p = sub.add_parser("cfi", help="Clustered Feature Importance (AFML Ch.8.5)")
    add_common(p)
    p.add_argument("--pt", type=float, default=1.0)
    p.add_argument("--sl", type=float, default=1.0)
    p.add_argument("--holding", type=int, default=10)
    p.add_argument("--decay", type=float, default=0.9)
    p.add_argument("--frac-d", type=float, default=0.4, dest="frac_d")
    p.add_argument("--no-frac-diff", action="store_true", dest="no_frac_diff")
    p.add_argument("--splits", type=int, default=4)
    p.add_argument("--embargo", type=float, default=0.01)
    p.add_argument("--max-clusters", type=int, default=6, dest="max_clusters")
    p.add_argument("--trades", action="store_true", help="Використати aggTrades (CVD фічі)")
    p.set_defaults(func=_cli_pkg.cmd_cfi)

    p = sub.add_parser("stress", help="Стрес-тест: crash / liquidity / vol_spike / funding_shock")

    add_common(p)
    p.add_argument("--scenarios", default=None, help="Через кому: crash,liquidity,vol_spike,funding_shock")
    p.set_defaults(func=_cli_pkg.cmd_stress)

    p = sub.add_parser("capacity", help="Capacity-тест: Sharpe при масштабуванні позицій")
    add_common(p)
    p.add_argument("--scales", default=None, help="Масштаби через кому (напр. 1,2,5,10,20)")
    p.add_argument("--maker", action="store_true", help="Комісії maker")
    p.set_defaults(func=_cli_pkg.cmd_capacity)

    p = sub.add_parser("survival", help="Survival analysis: час утримання позиції (Kaplan–Meier)")
    add_common(p)
    p.add_argument("--max-time", type=int, default=None, help="Обмежити горизонт (барів)")
    p.add_argument("--top", type=int, default=10, help="Скільки рядків кривої показати")
    p.add_argument("--feature", default=None, help="Фіча для survival_by_feature (напр. atr_14)")
    p.add_argument("--bins", type=int, default=3)
    p.set_defaults(func=_cli_pkg.cmd_survival)

    p = sub.add_parser("time-decay", help="Time-decay: Sharpe при лагу входу 0..N барів")
    add_common(p)
    p.add_argument("--max-lag", type=int, default=3)
    p.set_defaults(func=_cli_pkg.cmd_time_decay)

    p = sub.add_parser("quintile", help="Квінтилі z-score спреду (пари)")
    add_common(p)
    p.add_argument("--leg1", default="BTCUSDT")
    p.add_argument("--leg2", default="ETHUSDT")
    p.add_argument("--lookback", type=int, default=240)
    p.set_defaults(func=_cli_pkg.cmd_quintile, interval="1h")

    p = sub.add_parser("coint-scan", help="Скан коінтеграції символів")
    add_common(p)
    p.add_argument("--symbols", default=None, help="Через кому; інакше DEFAULT_SYMBOLS")
    p.set_defaults(func=_cli_pkg.cmd_coint_scan, interval="1h")

    p = sub.add_parser("hedge-ratio", help="OOS порівняння 1:1 vs OLS/Johansen hedge")
    add_common(p)
    p.add_argument("--leg1", default="BTCUSDT")
    p.add_argument("--leg2", default="ETHUSDT")
    p.add_argument("--lookback", type=int, default=240)
    p.set_defaults(func=_cli_pkg.cmd_hedge_ratio, interval="1h")

    p = sub.add_parser(
        "migrate-to-parquet",
        help="Мігрувати всі дані з PostgreSQL у Parquet-файли (data/) і вимкнути postgres-бекенд",
    )
    p.add_argument(
        "--symbol",
        default=None,
        help="Символи через кому (за замовч. — всі з Postgres). Приклад: BTCUSDT,ETHUSDT",
    )
    p.add_argument("--overwrite", action="store_true", help="Перезаписати вже наявні Parquet-файли")
    p.add_argument(
        "--skip-trades", action="store_true", dest="skip_trades", help="Не мігрувати aggTrades (великі таблиці)"
    )
    p.add_argument("--skip-funding", action="store_true", dest="skip_funding", help="Не мігрувати funding rates")
    p.add_argument(
        "--data-dir", default=None, dest="data_dir", help="Директорія для Parquet (за замовч. DATA_DIR з .env)"
    )
    p.add_argument("--dry-run", action="store_true", dest="dry_run", help="Показати план міграції без запису файлів")
    p.set_defaults(func=_cli_pkg.cmd_migrate_to_parquet)

    p = sub.add_parser("mcp", help="Запуск MCP-сервера для трейдінгу (stdio)")
    p.set_defaults(func=_cli_pkg.cmd_mcp)

    p = sub.add_parser("dashboard", help="Запуск Streamlit-дашборду (інтерпретатор цього venv)")
    p.add_argument("--port", type=int, default=None, help="Порт Streamlit (за замовч. 8501)")
    p.add_argument(
        "--address",
        default=None,
        help="Bind address Streamlit (за замовч. DASHBOARD_HOST або 127.0.0.1)",
    )
    p.add_argument("--no-worker", action="store_true", help="Не піднімати research worker разом із дашбордом")
    p.set_defaults(func=_cli_pkg.cmd_dashboard)

    p = sub.add_parser("dashboard-hash", help="Згенерувати хеш паролю для DASHBOARD_PASSWORD_HASH")
    p.set_defaults(func=_cli_pkg.cmd_dashboard_hash)

    job_p = sub.add_parser("job", help="Черга дослідницьких задач (SQLite + worker-процеси)")
    job_sub = job_p.add_subparsers(dest="job_cmd", required=True)
    jw = job_sub.add_parser("worker", help="Довгий процес: claim і виконання job")
    jw.add_argument("--jobs", type=int, default=1, help="Кількість worker-процесів")
    jl = job_sub.add_parser("list", help="Список задач")
    jl.add_argument("--limit", type=int, default=50)
    js = job_sub.add_parser("submit", help="Поставити job з JSON params")
    js.add_argument("kind", choices=["backtest", "pairs", "sweep", "overfit", "capacity"])
    js.add_argument("--params", default="{}", help="JSON-об'єкт параметрів")
    js.add_argument("--force", action="store_true", help="Перезапустити навіть succeeded")
    jst = job_sub.add_parser("status", help="Статус і хвіст логу")
    jst.add_argument("id", type=int)
    jc = job_sub.add_parser("cancel", help="Скасувати queued/running")
    jc.add_argument("id", type=int)
    jr = job_sub.add_parser("rerun", help="Інвалідувати артефакти і поставити в чергу знову")
    jr.add_argument("id", type=int)
    jp = job_sub.add_parser("prune", help="Видалити застарілі завершені jobs та їхні артефакти на диску")
    jp.add_argument("--days", type=int, default=14, help="Вік задач у днях (за замовчуванням: 14)")
    jp.add_argument(
        "--status",
        choices=["all", "succeeded", "failed", "cancelled"],
        default="all",
        help="Фільтр статусів для видалення (за замовчуванням: all)",
    )
    jp.add_argument(
        "--keep-records",
        action="store_true",
        help="Зберегти записи в SQLite, видалити лише важкі артефакти з диска",
    )
    jp.add_argument(
        "--dry-run",
        action="store_true",
        help="Показати, що буде видалено, без фактичного видалення",
    )
    jd = job_sub.add_parser(
        "delete",
        aliases=["rm"],
        help="Видалити завершені/скасовані задачі та їхні папки артефактів з диска за ID",
    )
    jd.add_argument("ids", type=int, nargs="+", help="Один або кілька ID задач для видалення")
    job_p.set_defaults(func=_cli_pkg.cmd_job)

    p = sub.add_parser(
        "sweep",
        help="Матричний прогон: всі стратегії × символи × таймфрейми (база качається раз, решта — ресемплінг)",
    )
    p.add_argument("--strategies", default=None, help="Через кому; за замовч. — всі single-symbol стратегії")
    p.add_argument("--symbols", default=None, help="Через кому; за замовч. — DEFAULT_SYMBOLS з .env")
    p.add_argument("--intervals", default=None, help="Через кому; за замовч. — 1m,5m,15m,30m,1h,4h")
    p.add_argument("--days", type=int, default=60, help="Глибина історії, днів")
    p.add_argument("--base", default=None, help="Базовий таймфрейм для ресемплінгу (за замовч. 1m)")
    p.add_argument("--mode", default="backtest", choices=["backtest", "walkforward"], help="Режим кожної клітинки")
    p.add_argument(
        "--train",
        type=int,
        default=None,
        help="Барів train для walkforward. Пропуск = per-TF (1h=500/200, 4h=200/100, 1m=4000/2000)",
    )
    p.add_argument("--test", type=int, default=None, help="Барів test (OOS). Пропуск = per-TF дефолт")
    _add_overlay_flag(p)
    p.add_argument("--workers", type=int, default=1, help="Паралельних клітинок (0/1 = послідовно)")
    p.add_argument("--all", action="store_true", help="Включити ML-стратегію та ensemble (повільно)")
    p.add_argument("--out", default=None, help="Шлях CSV-результату (за замовч. results/sweep.csv)")
    p.add_argument("--notify", action="store_true", help="Telegram-сповіщення топ-результату")
    p.add_argument("--resume", dest="resume", action="store_true", default=True, help="Пропускати вже ok клітинки")
    p.add_argument("--no-resume", dest="resume", action="store_false", help="Перерахувати всі клітинки")
    p.add_argument("--enqueue", action="store_true", help="Поставити sweep у чергу jobs.sqlite і вийти")
    p.set_defaults(func=_cli_pkg.cmd_sweep)

    # ─── telegram-bot ─────────────────────────────────────────────────────────
    tg_p = sub.add_parser(
        "telegram-bot",
        help="Інтерактивний Telegram Bot (команди + push-сповіщення)",
    )
    tg_sub = tg_p.add_subparsers(dest="tg_action", required=True)
    tg_start = tg_sub.add_parser("start", help="Запустити бота (blocking)")
    tg_start.add_argument(
        "--store",
        default=None,
        help="Шлях до SQLite store (за замовч. results/paper_pairs.sqlite)",
    )
    tg_start.add_argument(
        "--control",
        default=None,
        help="Шлях до control.json (за замовч. results/control.json)",
    )
    tg_p.set_defaults(func=_cli_pkg.cmd_telegram_bot)

    # API
    api_p = sub.add_parser("api", help="FastAPI Server")
    api_sub = api_p.add_subparsers(dest="api_action", required=True)
    api_start = api_sub.add_parser("start", help="Запустити FastAPI сервер (uvicorn)")
    api_start.add_argument(
        "--host",
        default=None,
        help="Хост (за замовч. API_HOST з .env, 0.0.0.0)",
    )
    api_start.add_argument(
        "--port",
        type=int,
        default=None,
        help="Порт (за замовч. API_PORT з .env, 8000)",
    )
    api_token = api_sub.add_parser("token", help="Згенерувати JWT токен для API")
    api_token.add_argument("--hours", type=int, default=24, help="Термін дії токена (годин)")
    api_p.set_defaults(func=_cli_pkg.cmd_api)

    args = parser.parse_args(argv)
    import dataclasses
    import os

    from scalper_hft.config import get_settings, set_settings

    settings = get_settings()
    if hasattr(args, "exchange") and args.exchange:
        os.environ["EXCHANGE"] = str(args.exchange)
        settings = dataclasses.replace(settings, exchange=str(args.exchange))
        set_settings(settings)
    args.param_dict = _parse_param_dict(getattr(args, "param", []))
    args.func(args)
