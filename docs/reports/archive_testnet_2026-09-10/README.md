# Архів результатів на testnet-даних (невалідні)

**Дата архівації:** 2026-09-10
**Статус:** ❌ НЕ ВИКОРИСТОВУВАТИ для жодних висновків про edge.

## Чому

Уся історія в `data/*_1m_klines.parquet` (і похідні ТФ, aggTrades, funding) була
скачана з **Binance Futures testnet** (`EXCHANGE=binance-testnet` +
`ExchangeClient.set_sandbox_mode(True)`). Testnet віддає синтетичну історію:

| перевірка | ETHUSDT (був чистий) | BNBUSDT | SOLUSDT |
|---|---|---|---|
| розбіжність close з LIVE (1m, вибірка) | 0.00% | 4.8% | ~2% |
| барів з \|r\| > 5% за 1m | 7 із 1.59M | 86 633 (5.46%) | 40 767 (2.57%) |
| найдовша «плита» O=H=L=C, volume=0 | 21 бар | 1 544 бари | 1 086 барів |
| покриття funding за 1095 днів | 100% | 38% | 38% |

Приклади синтетики: `BNBUSDT 2025-01-01 21:35` у кеші — `628.001, volume=0`
(реальний ринок — `706.71, volume=151`), далі стрибок `797.994`.

## Наслідки для чисел у цій теці

- `ml_strategy` BNBUSDT 5m у sweep: `total_return=+9478%`, `Sharpe=56.9`,
  `win_rate=90.4%`, `PF=168.8` — модель запам'ятала синтетичні стрибки.
- `market_maker`: 56 клітинок з `total_return ≤ −100%` (до −3415%).
- Bailey–LdP selection haircut на цих даних упевнено обрав переможцем
  `ml_strategy` (deflated Sharpe 4.4–34.4) — тобто захисні механізми працюють,
  але на отруєному вході дають упевнений хибний вердикт.

## Що зроблено в коді, щоб це не повторилось

- `Settings.data_exchange` (дефолт `binanceusdm`) окремо від торгового `exchange`;
  `require_live_data_exchange()` блокує testnet для ринкових даних.
- `validate_bars`: детекція неринкових рухів (MAD-поріг) і «плит» → fail-closed
  відмова писати parquet.
- `funding_coverage_ratio` + fail-closed перевірка покриття funding.
- CLI `data-audit` — звірка кешу з LIVE-біржою (exit code 1 при провалі).
- `flag_degenerate_row` у sweep: 0 угод / знищений капітал / inf PF /
  \|Sharpe\| > 20 → `status="degenerate"` (не потрапляє в рейтинги й haircut).
