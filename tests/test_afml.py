"""Тести AFML-модулів: Triple Barrier, Sample Weights, Fractional Diff."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_close(n: int = 500, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    return pd.Series(prices, index=idx, name="close")


def _make_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    close = _make_close(n, seed)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * (1 + rng.uniform(0, 0.001, n)),
            "low": close * (1 - rng.uniform(0, 0.001, n)),
            "close": close,
            "volume": rng.uniform(10, 100, n),
        }
    )


# ── Fractional Differentiation ────────────────────────────────────────────────

class TestFracDiff:
    def test_ffd_returns_same_length(self):
        from scalper_hft.ml.frac_diff import frac_diff_ffd

        s = _make_close()
        out = frac_diff_ffd(s, d=0.4)
        assert len(out) == len(s)

    def test_ffd_no_nan_after_warmup(self):
        """Після warmup-рядків не повинно бути NaN."""
        from scalper_hft.ml.frac_diff import frac_diff_ffd, _get_weights_ffd

        s = _make_close(200)
        d = 0.4
        w = _get_weights_ffd(d)
        warmup = len(w) - 1
        out = frac_diff_ffd(s, d=d)
        valid = out.iloc[warmup:]
        assert valid.isna().sum() == 0, "Є NaN після warmup"

    def test_ffd_d1_approx_diff(self):
        """При d=1.0 FFD ≈ першому диференціюванню (з урахуванням обрізання ваг)."""
        from scalper_hft.ml.frac_diff import frac_diff_ffd

        s = _make_close(300)
        fd = frac_diff_ffd(s, d=1.0, threshold=1e-10).dropna()
        d1 = s.diff().dropna()
        common = fd.index.intersection(d1.index)
        corr = np.corrcoef(fd[common].values, d1[common].values)[0, 1]
        assert corr > 0.99, f"При d=1.0 кореляція з diff() = {corr:.4f} (очікується > 0.99)"

    def test_ffd_no_lookahead(self):
        """Значення fd[t] залежить лише від close[t] та минулого."""
        from scalper_hft.ml.frac_diff import frac_diff_ffd

        s = _make_close(500)
        fd_full = frac_diff_ffd(s, d=0.4)
        # обрізаємо майбутнє
        s_partial = s.iloc[:400]
        fd_partial = frac_diff_ffd(s_partial, d=0.4)
        # перші 400 значень мають збігатися (де обидва не NaN)
        common = fd_full.iloc[:400].dropna().index.intersection(fd_partial.dropna().index)
        assert len(common) > 5, f"Лише {len(common)} спільних non-NaN значень"
        np.testing.assert_allclose(
            fd_full.loc[common].values,
            fd_partial.loc[common].values,
            rtol=1e-10,
            err_msg="frac_diff_ffd містить lookahead!",
        )

    def test_add_frac_diff_creates_column(self):
        from scalper_hft.ml.frac_diff import add_frac_diff

        df = _make_df(500)  # needs more bars for warmup at threshold=1e-4
        out = add_frac_diff(df, d=0.4)
        assert "fd_close" in out.columns
        assert out["fd_close"].notna().sum() > 10

    def test_frac_diff_features_multi_col(self):
        from scalper_hft.ml.frac_diff import frac_diff_features

        df = _make_df(300)
        out = frac_diff_features(df, cols=["close", "volume"], d=0.3)
        assert "fd_close" in out.columns
        assert "fd_volume" in out.columns

    def test_find_min_d_returns_valid(self):
        """find_min_d має повернути значення в [0, 1] або -1 для стаціонарних рядів."""
        from scalper_hft.ml.frac_diff import find_min_d

        s = np.log(_make_close(500))
        d = find_min_d(s, step=0.1)
        assert d == -1.0 or (0.0 <= d <= 1.0), f"Некоректний d={d}"


# ── Triple Barrier Labeling ───────────────────────────────────────────────────

class TestTripleBarrier:
    def test_label_from_ohlcv_returns_dataframe(self):
        from scalper_hft.ml.labeling import label_from_ohlcv

        df = _make_df(300)
        events = label_from_ohlcv(df, pt=1.0, sl=1.0, holding_bars=5)
        assert isinstance(events, pd.DataFrame)
        assert "label" in events.columns
        assert "t1" in events.columns

    def test_labels_are_valid_values(self):
        from scalper_hft.ml.labeling import label_from_ohlcv

        df = _make_df(300)
        events = label_from_ohlcv(df, pt=1.0, sl=1.0, holding_bars=5)
        valid = events["label"].dropna()
        assert set(valid.unique()).issubset({-1, 0, 1}), f"Некоректні лейбли: {valid.unique()}"

    def test_t1_no_lookahead(self):
        """t1 завжди > t0."""
        from scalper_hft.ml.labeling import label_from_ohlcv

        df = _make_df(200)
        events = label_from_ohlcv(df, pt=1.0, sl=1.0, holding_bars=5)
        valid_t1 = events.dropna(subset=["t1"])
        bad = valid_t1[valid_t1["t1"] <= valid_t1.index]
        assert len(bad) == 0, f"t1 ≤ t0 у {len(bad)} рядках (lookahead!)"

    def test_get_labels_sign_matches_ret(self):
        """Лейбл має збігатися зі знаком фактичного ret."""
        from scalper_hft.ml.labeling import label_from_ohlcv

        df = _make_df(300)
        events = label_from_ohlcv(df, pt=0.5, sl=0.5, holding_bars=10)
        # для non-zero label: знак ret має збігатися зі знаком label
        non_zero = events[(events["label"] != 0) & events["ret"].notna()]
        sign_match = (np.sign(non_zero["ret"]) == non_zero["label"]).mean()
        assert sign_match >= 0.90, f"Знак label та ret збігається лише у {sign_match:.2%}"

    def test_add_vertical_barrier_ordering(self):
        from scalper_hft.ml.labeling import add_vertical_barrier

        close = _make_close(200)
        t_events = close.index[:50]
        t1 = add_vertical_barrier(t_events, close, num_days=1)
        assert (t1.values > t1.index.values).all(), "Вертикальний бар'єр до t0!"


# ── Sample Weights ────────────────────────────────────────────────────────────

class TestSampleWeights:
    def test_compute_sample_weights_shape(self):
        from scalper_hft.ml.labeling import label_from_ohlcv
        from scalper_hft.ml.sample_weights import compute_sample_weights

        df = _make_df(300)
        close = df["close"]
        events = label_from_ohlcv(df, pt=1.0, sl=1.0, holding_bars=5)
        w = compute_sample_weights(events, close, decay=0.9)
        assert len(w) == len(events)

    def test_weights_non_negative(self):
        from scalper_hft.ml.labeling import label_from_ohlcv
        from scalper_hft.ml.sample_weights import compute_sample_weights

        df = _make_df(300)
        events = label_from_ohlcv(df, pt=1.0, sl=1.0, holding_bars=5)
        w = compute_sample_weights(events, df["close"])
        assert (w >= 0).all(), "Є від'ємні ваги!"

    def test_get_ind_matrix_shape(self):
        from scalper_hft.ml.labeling import label_from_ohlcv
        from scalper_hft.ml.sample_weights import get_ind_matrix

        df = _make_df(200)
        events = label_from_ohlcv(df, pt=1.0, sl=1.0, holding_bars=5)
        t1 = events["t1"].dropna()
        ind_m = get_ind_matrix(df.index, t1)
        assert ind_m.shape[1] == len(t1)
        assert ind_m.shape[0] == len(df)

    def test_ind_matrix_values_binary(self):
        from scalper_hft.ml.labeling import label_from_ohlcv
        from scalper_hft.ml.sample_weights import get_ind_matrix

        df = _make_df(200)
        events = label_from_ohlcv(df, pt=1.0, sl=1.0, holding_bars=5)
        t1 = events["t1"].dropna()
        ind_m = get_ind_matrix(df.index, t1)
        assert set(ind_m.values.flatten().tolist()).issubset({0, 1, 0.0, 1.0})

    def test_seq_bootstrap_returns_correct_length(self):
        from scalper_hft.ml.labeling import label_from_ohlcv
        from scalper_hft.ml.sample_weights import get_ind_matrix, seq_bootstrap

        df = _make_df(150)
        events = label_from_ohlcv(df, pt=1.0, sl=1.0, holding_bars=3)
        t1 = events["t1"].dropna().head(20)  # менша вибірка для швидкості
        if len(t1) < 5:
            pytest.skip("Занадто мало подій для тесту")
        ind_m = get_ind_matrix(df.index, t1)
        phi = seq_bootstrap(ind_m, s_length=10)
        assert len(phi) == 10

    def test_no_decay_weights_sum_to_n(self):
        """Ваги нормовані: sum ≈ 1.0 (probability weights після decay)."""
        from scalper_hft.ml.labeling import label_from_ohlcv
        from scalper_hft.ml.sample_weights import compute_sample_weights

        df = _make_df(300)
        events = label_from_ohlcv(df, pt=1.0, sl=1.0, holding_bars=5)
        w = compute_sample_weights(events, df["close"], decay=1.0)
        w_nonzero = w[w > 0]
        # get_time_decay_weights нормує до суми=1
        assert abs(w_nonzero.sum() - 1.0) < 0.01, \
            f"Сума ваг {w_nonzero.sum():.4f} != 1.0"
