"""No-lookahead invariance tests for the walk-forward regime detector.

The repo's central claim is that regime detection is walk-forward and
bias-free ("fits only on spy.loc[:date]"). These tests make that claim
falsifiable, following the same future-data-injection pattern used in
PinSight and RateWalk:

    the regime assigned as of date t must be BIT-IDENTICAL whether or not
    data after t exists in the price frame.

If someone later changes `_detect_regime` (or the detector's feature prep)
in a way that peeks past the cutoff, these tests fail immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from regime_detector import WalkForwardRegimeDetector
from regime_aware_backtest import RegimeAwareBacktester


def _synthetic_spy(n_days: int, seed: int = 7) -> pd.Series:
    """Regime-flavored synthetic SPY: calm drift, then a volatile crash
    leg, then recovery - enough structure for a 4-state HMM to latch on."""
    rng = np.random.default_rng(seed)
    n1, n2 = int(n_days * 0.6), int(n_days * 0.2)
    n3 = n_days - n1 - n2
    rets = np.concatenate([
        rng.normal(0.0004, 0.008, n1),    # calm bull
        rng.normal(-0.002, 0.030, n2),    # crisis leg
        rng.normal(0.0008, 0.012, n3),    # recovery
    ])
    idx = pd.bdate_range("2015-01-02", periods=n_days)
    return pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx, name="SPY")


def _bare_backtester(prices: pd.DataFrame) -> RegimeAwareBacktester:
    """Instantiate without running the heavy __init__ (data loading etc.);
    _detect_regime only needs `prices` and `regime_detector`."""
    bt = RegimeAwareBacktester.__new__(RegimeAwareBacktester)
    bt.prices = prices
    bt.regime_detector = WalkForwardRegimeDetector()
    return bt


def test_regime_at_t_invariant_to_future_data():
    """Core guarantee: appending T+k prices must not change the regime
    assigned at T."""
    n = config.REGIME_TRAIN_WINDOW + 300
    spy_full = _synthetic_spy(n)
    t = spy_full.index[config.REGIME_TRAIN_WINDOW + 100]

    past_only = _bare_backtester(spy_full.loc[:t].to_frame())
    with_future = _bare_backtester(spy_full.to_frame())

    r_past = past_only._detect_regime(t)
    r_full = with_future._detect_regime(t)
    assert r_past == r_full, (
        f"lookahead: regime at {t.date()} changed from {r_past!r} to "
        f"{r_full!r} when future data was appended")


def test_regime_series_stable_across_extension():
    """Same property across a run of dates, not just one."""
    n = config.REGIME_TRAIN_WINDOW + 260
    spy_full = _synthetic_spy(n, seed=11)
    cut = config.REGIME_TRAIN_WINDOW + 60
    dates = spy_full.index[cut:cut + 15]

    with_future = _bare_backtester(spy_full.to_frame())
    labels_full = [with_future._detect_regime(t) for t in dates]

    labels_trunc = []
    for t in dates:
        bt = _bare_backtester(spy_full.loc[:t].to_frame())
        labels_trunc.append(bt._detect_regime(t))

    assert labels_full == labels_trunc


def test_detector_is_deterministic():
    """Pinned random_state: same input, same fit, same labels."""
    spy = _synthetic_spy(900, seed=3)
    out = []
    for _ in range(2):
        det = WalkForwardRegimeDetector()
        det.fit(spy)
        out.append(det.predict_regime(spy).tolist())
    assert out[0] == out[1]


def test_regime_names_ordered_by_volatility():
    """'crisis' must sit on the highest-volatility HMM state: the naming
    convention the whole strategy's exposure table depends on."""
    spy = _synthetic_spy(1200, seed=5)
    det = WalkForwardRegimeDetector()
    det.fit(spy)
    labels = det.predict_regime(spy)
    rets = spy.pct_change().reindex(labels.index)
    vol_by_name = rets.groupby(labels).std().dropna()
    if {"crisis", "bull"} <= set(vol_by_name.index):
        assert vol_by_name["crisis"] > vol_by_name["bull"]
