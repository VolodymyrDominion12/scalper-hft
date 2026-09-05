#!/usr/bin/env bash
# Ланцюг walk-forward sweep-прогонів: по одному на таймфрейм (train/test підібрані
# під глибину 180 днів кожного ТФ). Пише у той самий results/sweep.db (mode=walkforward).
set -u
STRATEGIES="mean_reversion,market_maker,funding_carry,basis_reversion,hmm_reversion,cross_momentum,supertrend,stoch_rsi,smc_fvg"
SYMBOLS="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,LINKUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,NEARUSDT,DOTUSDT,ATOMUSDT,UNIUSDT,LTCUSDT,AAVEUSDT"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/results/full_matrix"

# train test  interval
declare -a ROWS=(
  "4000 2000 1m"
  "2000 1000 5m"
  "1000 500 15m"
  "500 250 30m"
  "500 200 1h"
  "200 100 4h"
)

for row in "${ROWS[@]}"; do
  read -r TRAIN TEST IV <<<"$row"
  echo "=== WF sweep $IV train=$TRAIN test=$TEST ==="
  EXCHANGE=binance "$ROOT/.venv/bin/python" -m scalper_hft.cli sweep \
    --strategies "$STRATEGIES" --symbols "$SYMBOLS" --intervals "$IV" \
    --days 180 --mode walkforward --train "$TRAIN" --test "$TEST" --workers 8 \
    --out "$ROOT/results/full_matrix/sweep_wf_${IV}_180d.csv" \
    > "$ROOT/results/full_matrix/wf_${IV}.log" 2>&1
  echo "exit=$? for $IV (tail):"
  tail -3 "$ROOT/results/full_matrix/wf_${IV}.log"
done
echo "ALL WF SWEEPS DONE"
