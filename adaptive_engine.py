"""
ADAPTIVE ENGINE — Walk-Forward Optimization & Drawdown Management
==================================================================
Replaces static hardcoded parameters with rolling adaptive values.
Adds systematic drawdown controls to reduce exposure during losses.

Components:
  1. DrawdownManager      — Exposure scalar based on distance from HWM
  2. AlphaFadeManager     — Blend toward passive when alpha disappears
  3. ParameterMonitor     — Track parameter staleness, trigger early refit
  4. WalkForwardOptimizer — Re-optimize scoring weights on rolling basis

All components are behind config flags for A/B testing.

Author: Tanishk
Date: March 2026
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# REGIME-SPECIFIC DRAWDOWN PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

REGIME_DD_PARAMS = {
    'bull':   {'dd_start': -0.08, 'dd_full': -0.25, 'min_scalar': 0.35},
    'normal': {'dd_start': -0.05, 'dd_full': -0.20, 'min_scalar': 0.30},
    'bear':   {'dd_start': -0.03, 'dd_full': -0.15, 'min_scalar': 0.25},
    'crisis': {'dd_start': -0.02, 'dd_full': -0.10, 'min_scalar': 0.20},
}

# Regime-adaptive liquidation thresholds
REGIME_LIQ_THRESHOLDS = {
    'crisis': {'half': 12, 'full': 15},
    'bear':   {'half': 12, 'full': 15},
    'normal': {'half': 15, 'full': 18},
    'bull':   {'half': 18, 'full': 22},
}


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT 1: DRAWDOWN MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class DrawdownManager:
    """
    Scales portfolio exposure based on current drawdown depth.

    At HWM (no drawdown): full exposure (scalar = 1.0)
    At dd_start:          begin reducing
    At dd_full:           maximum reduction (scalar = min_scalar)

    Scales DOWN immediately (risk management), scales UP slowly
    (recovery_speed blending) to avoid whipsawing.
    """

    def __init__(self,
                 dd_start: float = -0.05,
                 dd_full: float = -0.20,
                 min_scalar: float = 0.25,
                 recovery_speed: float = 0.5,
                 use_regime_params: bool = True):
        self.dd_start = dd_start
        self.dd_full = dd_full
        self.min_scalar = min_scalar
        self.recovery_speed = recovery_speed
        self.use_regime_params = use_regime_params

        self.hwm = 0.0
        self.prev_scalar = 1.0
        self.history: List[dict] = []

    def update(self, portfolio_value: float,
               regime: str = 'normal',
               date: Optional[pd.Timestamp] = None) -> float:
        """
        Call on each trading day. Returns exposure scalar ∈ [min_scalar, 1.0].
        """
        self.hwm = max(self.hwm, portfolio_value)
        current_dd = (portfolio_value / self.hwm) - 1.0 if self.hwm > 0 else 0.0

        # Use regime-specific thresholds if enabled
        if self.use_regime_params and regime in REGIME_DD_PARAMS:
            params = REGIME_DD_PARAMS[regime]
            dd_start = params['dd_start']
            dd_full = params['dd_full']
            min_scalar = params['min_scalar']
        else:
            dd_start = self.dd_start
            dd_full = self.dd_full
            min_scalar = self.min_scalar

        # Compute target scalar
        if current_dd >= dd_start:
            target_scalar = 1.0
        elif current_dd <= dd_full:
            target_scalar = min_scalar
        else:
            # Linear interpolation between start and full thresholds
            frac = (current_dd - dd_start) / (dd_full - dd_start)
            target_scalar = 1.0 - frac * (1.0 - min_scalar)

        # Asymmetric smoothing: cut fast, recover slowly
        if target_scalar < self.prev_scalar:
            # Cutting exposure: move immediately
            self.prev_scalar = target_scalar
        else:
            # Recovering: blend slowly
            self.prev_scalar += ((target_scalar - self.prev_scalar)
                                 * self.recovery_speed)

        # Clamp
        self.prev_scalar = max(min_scalar, min(1.0, self.prev_scalar))

        # Log
        if date is not None:
            self.history.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'hwm': self.hwm,
                'drawdown': current_dd,
                'dd_scalar': self.prev_scalar,
                'regime': regime,
                'dd_start_threshold': dd_start,
                'dd_full_threshold': dd_full,
            })

        return self.prev_scalar

    def get_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT 2: ALPHA FADE MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class AlphaFadeManager:
    """
    If rolling 12-month alpha turns persistently negative, blend the
    portfolio toward passive SPY allocation.

    This prevents the strategy from destroying capital when the factor
    premium has genuinely disappeared (as observed in 2022-2025).

    Blend: strategy_weight = max(min_weight, 1.0 - (months_neg - 2) * 0.10)
    After 7 months negative alpha → 30% strategy, 70% SPY-tracking.
    Recovery is 2x faster: 2 months positive clears 1 month negative.
    """

    def __init__(self,
                 alpha_window: int = 252,
                 trigger_months: int = 3,
                 min_strategy_weight: float = 0.30):
        self.alpha_window = alpha_window
        self.trigger_months = trigger_months
        self.min_weight = min_strategy_weight
        self.consecutive_negative_months = 0
        self.last_check_month = None
        self.history: List[dict] = []

    def update(self, strategy_returns: pd.Series,
               benchmark_returns: pd.Series,
               date: Optional[pd.Timestamp] = None) -> float:
        """
        Call monthly (or daily — it self-throttles to monthly checks).
        Returns strategy_weight ∈ [min_weight, 1.0].
        """
        if len(strategy_returns) < self.alpha_window:
            return 1.0

        # Throttle to monthly checks
        if date is not None:
            current_month = (date.year, date.month)
            if self.last_check_month == current_month:
                # Return last computed weight
                return self._current_weight()
            self.last_check_month = current_month

        # Compute rolling alpha (annualized)
        strat_window = strategy_returns.iloc[-self.alpha_window:]
        bench_window = benchmark_returns.iloc[-self.alpha_window:]

        # Align
        common = strat_window.index.intersection(bench_window.index)
        if len(common) < self.alpha_window // 2:
            return 1.0

        strat_mean = strat_window.loc[common].mean()
        bench_mean = bench_window.loc[common].mean()
        rolling_alpha = (strat_mean - bench_mean) * 252

        if rolling_alpha < 0:
            self.consecutive_negative_months += 1
        else:
            # Recovery is 2x faster
            self.consecutive_negative_months = max(
                0, self.consecutive_negative_months - 2
            )

        weight = self._current_weight()

        if date is not None:
            self.history.append({
                'date': date,
                'rolling_alpha_annual': rolling_alpha,
                'consecutive_negative_months': self.consecutive_negative_months,
                'strategy_weight': weight,
            })

        return weight

    def _current_weight(self) -> float:
        if self.consecutive_negative_months >= self.trigger_months:
            fade = self.consecutive_negative_months - (self.trigger_months - 1)
            return max(self.min_weight, 1.0 - fade * 0.10)
        return 1.0

    def get_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT 3: PARAMETER MONITOR
# ═══════════════════════════════════════════════════════════════════════════

class ParameterMonitor:
    """
    Tracks rolling strategy performance to detect parameter staleness.
    Triggers early refit if alpha drops below threshold.
    """

    def __init__(self, lookback_days: int = 126):
        self.lookback_days = lookback_days
        self.history: List[dict] = []

    def check_staleness(self,
                        strategy_returns: pd.Series,
                        benchmark_returns: pd.Series,
                        date: Optional[pd.Timestamp] = None) -> dict:
        """
        Returns staleness diagnostics. If trigger_early_refit is True,
        the optimizer should re-optimize parameters immediately.
        """
        if len(strategy_returns) < self.lookback_days:
            return {'trigger_early_refit': False, 'rolling_alpha': 0.0,
                    'rolling_sharpe': 0.0}

        recent_strat = strategy_returns.iloc[-self.lookback_days:]
        recent_bench = benchmark_returns.iloc[-self.lookback_days:]

        common = recent_strat.index.intersection(recent_bench.index)
        if len(common) < self.lookback_days // 2:
            return {'trigger_early_refit': False, 'rolling_alpha': 0.0,
                    'rolling_sharpe': 0.0}

        strat_r = recent_strat.loc[common]
        bench_r = recent_bench.loc[common]

        rolling_alpha = (strat_r.mean() - bench_r.mean()) * 252
        strat_std = strat_r.std()
        rolling_sharpe = (strat_r.mean() / strat_std * np.sqrt(252)
                          if strat_std > 0 else 0.0)

        result = {
            'rolling_alpha': rolling_alpha,
            'rolling_sharpe': rolling_sharpe,
            'alpha_negative_6m': rolling_alpha < 0,
            'trigger_early_refit': rolling_alpha < -0.02,
        }

        if date is not None:
            self.history.append({**result, 'date': date})

        return result

    def get_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT 4: WALK-FORWARD OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════

class WalkForwardOptimizer:
    """
    Re-optimizes scoring weights on a rolling basis using only
    backward-looking data. Strictly walk-forward: no lookahead.

    At each refit date:
      1. Take [refit_date - lookback, refit_date - 1] as training window
      2. For each candidate weight vector, recompute rankings and form
         hypothetical top-N portfolio
      3. Maximize Sharpe ratio of that portfolio over the training window
      4. Apply smoothing: new weights can't differ from old by >0.10 per factor

    Uses scipy.optimize.minimize with SLSQP.
    """

    def __init__(self,
                 refit_interval: int = 252,
                 lookback: int = 756,
                 weight_bounds: Tuple[float, float] = (0.10, 0.40),
                 max_weight_change: float = 0.10,
                 top_n: int = 10):
        self.refit_interval = refit_interval
        self.lookback = lookback
        self.weight_bounds = weight_bounds
        self.max_weight_change = max_weight_change
        self.top_n = top_n

        self.current_weights = {
            'roic': 0.30, 'fcf_margin': 0.25,
            'de_inverse': 0.25, 'rev_growth': 0.20,
        }
        self.last_refit_idx = -self.refit_interval  # Force first refit
        self.refit_log: List[dict] = []

    def should_refit(self, day_index: int, early_trigger: bool = False) -> bool:
        """Check if it's time to re-optimize."""
        if early_trigger:
            return True
        return (day_index - self.last_refit_idx) >= self.refit_interval

    def refit(self, prices: pd.DataFrame,
              fundamental_data: pd.DataFrame,
              refit_date: pd.Timestamp,
              day_index: int) -> dict:
        """
        Re-optimize scoring weights using data strictly before refit_date.

        Parameters
        ----------
        prices : DataFrame with daily closes for all tickers + SPY
        fundamental_data : DataFrame with ticker, period_end, available_date,
                          roic, fcf_margin, d_e, rev_growth columns
        refit_date : The date of the refit (only use data BEFORE this)
        day_index : Current index in the trading day loop

        Returns
        -------
        dict with optimized weights
        """
        from scipy.optimize import minimize

        # Training window: [refit_date - lookback, refit_date - 1 day]
        train_end = refit_date - pd.Timedelta(days=1)
        train_start = refit_date - pd.Timedelta(days=self.lookback)

        # Filter prices to training window
        train_prices = prices.loc[train_start:train_end].copy()
        if len(train_prices) < 126:  # Need at least 6 months
            logger.warning(f"Insufficient price data for refit at {refit_date}")
            return self.current_weights

        # Filter fundamentals to those available before refit_date
        pit_data = fundamental_data[
            fundamental_data['available_date'] < refit_date
        ].copy()

        if pit_data.empty:
            return self.current_weights

        # Get tickers with both price and fundamental data
        price_tickers = set(train_prices.columns) - {'SPY'}
        fund_tickers = set(pit_data['ticker'].unique())
        common_tickers = price_tickers & fund_tickers

        if len(common_tickers) < self.top_n * 2:
            return self.current_weights

        previous_weights = self.current_weights.copy()

        def neg_sharpe(w):
            weights = {
                'roic': w[0], 'fcf_margin': w[1],
                'de_inverse': w[2], 'rev_growth': w[3],
            }
            return self._evaluate_weights(
                weights, pit_data, train_prices, common_tickers
            )

        constraints = [
            {'type': 'eq', 'fun': lambda w: sum(w) - 1.0},
        ]
        bounds = [self.weight_bounds] * 4

        x0 = [self.current_weights['roic'], self.current_weights['fcf_margin'],
              self.current_weights['de_inverse'], self.current_weights['rev_growth']]

        try:
            result = minimize(
                neg_sharpe, x0=x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-6},
            )

            if result.success:
                new_weights = {
                    'roic': result.x[0], 'fcf_margin': result.x[1],
                    'de_inverse': result.x[2], 'rev_growth': result.x[3],
                }

                # Apply smoothing constraint
                new_weights = self._smooth_weights(previous_weights, new_weights)
                opt_sharpe = -result.fun
            else:
                new_weights = previous_weights
                opt_sharpe = 0.0
        except Exception as e:
            logger.warning(f"Optimization failed at {refit_date}: {e}")
            new_weights = previous_weights
            opt_sharpe = 0.0

        self.current_weights = new_weights
        self.last_refit_idx = day_index

        self.refit_log.append({
            'refit_date': refit_date,
            'lookback_start': train_start,
            'lookback_end': train_end,
            'old_roic': previous_weights['roic'],
            'old_fcf': previous_weights['fcf_margin'],
            'old_de': previous_weights['de_inverse'],
            'old_rev': previous_weights['rev_growth'],
            'new_roic': new_weights['roic'],
            'new_fcf': new_weights['fcf_margin'],
            'new_de': new_weights['de_inverse'],
            'new_rev': new_weights['rev_growth'],
            'in_sample_sharpe': opt_sharpe,
            'trigger': 'scheduled',
        })

        return new_weights

    def _evaluate_weights(self, weights: dict, pit_data: pd.DataFrame,
                          prices: pd.DataFrame,
                          tickers: set) -> float:
        """
        Compute negative Sharpe of a hypothetical portfolio formed
        using the given scoring weights. Used as objective function.
        """
        # Score each ticker using latest available fundamental data
        scores = {}
        for ticker in tickers:
            tk_data = pit_data[pit_data['ticker'] == ticker]
            if tk_data.empty:
                continue

            latest = tk_data.sort_values('period_end').iloc[-1]

            # EWMA approximation: use latest values
            # (full EWMA would be expensive in the optimizer loop)
            roic_val = latest.get('roic', 0)
            fcf_val = latest.get('fcf_margin', 0)
            de_val = latest.get('d_e', 0)
            rev_val = latest.get('rev_growth', 0)

            if pd.isna(roic_val) or pd.isna(fcf_val):
                continue

            de_clamped = max(de_val, 0) if not pd.isna(de_val) else 0
            rev_val = rev_val if not pd.isna(rev_val) else 0

            score = (weights['roic'] * roic_val +
                     weights['fcf_margin'] * fcf_val +
                     weights['de_inverse'] * (1 / (de_clamped + 1)) +
                     weights['rev_growth'] * rev_val)
            scores[ticker] = score

        if len(scores) < self.top_n:
            return 0.0  # Can't form portfolio

        # Select top-N
        sorted_tickers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_tickers = [t for t, _ in sorted_tickers[:self.top_n]]

        # Compute equal-weight portfolio returns
        available_cols = [t for t in top_tickers if t in prices.columns]
        if len(available_cols) < 3:
            return 0.0

        port_returns = prices[available_cols].pct_change().dropna().mean(axis=1)

        if len(port_returns) < 63 or port_returns.std() == 0:
            return 0.0

        sharpe = port_returns.mean() / port_returns.std() * np.sqrt(252)
        return -sharpe  # Minimize negative Sharpe

    def _smooth_weights(self, old: dict, new: dict) -> dict:
        """Apply smoothing: cap weight changes at max_weight_change per factor."""
        smoothed = {}
        for key in old:
            diff = new[key] - old[key]
            capped_diff = max(-self.max_weight_change,
                              min(self.max_weight_change, diff))
            smoothed[key] = old[key] + capped_diff

        # Re-normalize to sum to 1.0
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v / total for k, v in smoothed.items()}

        return smoothed

    def get_refit_log_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.refit_log)

    def get_current_config_weights(self) -> Tuple[float, float, float, float]:
        """Return weights in config.py order: ROIC, FCF, D/E inverse, Rev Growth."""
        w = self.current_weights
        return (w['roic'], w['fcf_margin'], w['de_inverse'], w['rev_growth'])
