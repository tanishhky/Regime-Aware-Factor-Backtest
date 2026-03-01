"""
REGIME DETECTOR - Walk-Forward HMM
====================================
Uses only past data to detect regimes, avoiding survivorship bias.
Adapted from regime_system_modules.py.

After fitting, regime integer labels are sorted by volatility:
  Highest vol → 'crisis'
  Next        → 'bear'
  Next        → 'normal'
  Lowest vol  → 'bull'

Confirmed bias-free: fits only on spy_past = spy.loc[:date] and
predicts on trailing indicators. No future leakage.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

import config


class WalkForwardRegimeDetector:
    """
    Detects market regimes using HMM on SPY features.
    Walk-forward: only uses data up to the current date for fitting.
    """

    def __init__(self):
        self.n_regimes = config.N_REGIMES
        self.short_window = config.SHORT_TERM_TREND
        self.long_window = config.LONG_TERM_TREND
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.regime_name_map = {}

    def _prepare_features(self, spy_series: pd.Series) -> pd.DataFrame:
        """Build feature matrix from SPY prices (returns, vol, momentum, trend)."""
        returns = spy_series.pct_change()
        features = pd.DataFrame({
            'returns': returns,
            'volatility_st': returns.rolling(self.short_window).std(),
            'trend_st': spy_series.pct_change(self.short_window),
            'trend_lt': spy_series.pct_change(self.long_window),
        }, index=spy_series.index)
        features = features.dropna()
        features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return features

    def fit(self, spy_series: pd.Series):
        """Fit HMM on the provided SPY price series (must end BEFORE current date)."""
        features = self._prepare_features(spy_series)
        X = features.values

        self.scaler_mean = X.mean(axis=0)
        self.scaler_std = X.std(axis=0)
        self.scaler_std[self.scaler_std == 0] = 1.0
        X_scaled = (X - self.scaler_mean) / self.scaler_std

        for cov_type in ['diag', 'spherical']:
            try:
                self.model = hmm.GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type=cov_type,
                    n_iter=100,
                    random_state=42,
                    init_params='stmc',
                    params='stmc',
                )
                self.model.fit(X_scaled)
                break
            except (ValueError, np.linalg.LinAlgError):
                if cov_type == 'spherical':
                    self.model = KMeans(
                        n_clusters=self.n_regimes, random_state=42, n_init=10
                    )
                    self.model.fit(X_scaled)

        # Map regimes by volatility
        regimes_train = self.model.predict(X_scaled)
        self._build_regime_mapping(features, regimes_train)
        return self

    def _build_regime_mapping(self, features_df: pd.DataFrame, raw_regimes: np.ndarray):
        """Sort HMM states by avg volatility → crisis (highest) to bull (lowest)."""
        vol_by_regime = {}
        for r in range(self.n_regimes):
            mask = (raw_regimes == r)
            vol_by_regime[r] = features_df['volatility_st'].values[mask].mean() if mask.sum() > 0 else 0.0

        sorted_regimes = sorted(vol_by_regime.keys(), key=lambda r: vol_by_regime[r], reverse=True)
        regime_names = ['crisis', 'bear', 'normal', 'bull'][:self.n_regimes]

        self.regime_name_map = {}
        for i, hmm_state in enumerate(sorted_regimes):
            self.regime_name_map[hmm_state] = regime_names[i]

        config.REGIME_LABEL_MAP = dict(self.regime_name_map)

    def predict_regime(self, spy_series: pd.Series) -> pd.Series:
        """Predict regime for each date in spy_series."""
        features = self._prepare_features(spy_series)
        X = features.values
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        raw_regimes = self.model.predict(X_scaled)
        named = [self.regime_name_map.get(r, 'normal') for r in raw_regimes]
        return pd.Series(named, index=features.index, name='regime')

    def predict_single(self, spy_series: pd.Series) -> str:
        """Predict regime for the LAST date in the series."""
        regimes = self.predict_regime(spy_series)
        return regimes.iloc[-1] if len(regimes) > 0 else 'normal'
