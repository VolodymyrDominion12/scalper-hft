import logging, sys
import pandas as pd
from scalper_hft.data.access import klines_from_store
from scalper_hft.strategies.ml_strategy import MLStrategy

bars = klines_from_store('BTCUSDT', '1h', days=90)
strat = MLStrategy(params={'symbol': 'BTCUSDT'})
# let's just hijack the generate_signals and print locals
def generate_signals_hacked(self, df: pd.DataFrame) -> pd.Series:
    from scalper_hft.ml.features import build_labeled_dataset
    from scalper_hft.ml.trainer import train_walk_forward_meta
    from scalper_hft.ml.windows import resolve_ml_windows
    
    pt, sl, holding_bars, decay, frac_d = 1.0, 1.0, 10, 0.9, 0.4
    
    X, y, w, t1 = build_labeled_dataset(
        df=df, mode="triple_barrier", pt=pt, sl=sl, holding_bars=holding_bars,
        decay=decay, frac_d=frac_d, add_frac_diff=True, add_hmm=False,
        add_garch=False, hmm_states=3, return_t1=True, depth=None
    )
    
    train_bars, test_bars = resolve_ml_windows(len(X), len(df), "1h", None, None)
    
    side, p_meta = train_walk_forward_meta(
        X=X, y=y, train_size=train_bars, test_size=test_bars,
        sample_weights=w, t1=t1, backend="xgboost"
    )
    
    print("side distribution:", side.value_counts().to_dict())
    print("p_meta distribution:", p_meta.value_counts().to_dict() if len(p_meta.unique()) < 10 else f"Unique count: {len(p_meta.unique())}")
    print("p_meta min:", p_meta.min(), "max:", p_meta.max())
    
    from scalper_hft.ml.bet_sizing import meta_size
    size = meta_size(p_meta.values) * 1.0
    print("size sum:", size.sum(), "max:", size.max())
    
    return pd.Series(0)

MLStrategy.generate_signals = generate_signals_hacked
signals = strat.generate_signals(bars)
