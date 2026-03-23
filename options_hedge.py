"""
OPTIONS HEDGE MODULE - Regime-Conditional Tail Protection (Synthetic BS)
========================================================================
Prices SPY options synthetically using VIX-based Black-Scholes with a
moneyness-dependent skew adjustment. No full OptionMetrics subscription needed.

Core Logic (Buy Cheap, Monetize Expensive):
  - Normal/Bull regimes: Accumulate rolling protective put spreads (vol is cheap)
  - Regime transition:   Existing puts appreciate on delta + vega expansion
  - Crisis/Bear regimes: Monetize (close) existing hedges at inflated vol;
                         do NOT initiate new hedges at elevated IV.

Pricing Methodology:
  1. ATM IV sourced from VIX (VIX IS SPY ATM IV by construction)
  2. OTM skew: IV(K) = VIX/100 * (1 + skew_slope * moneyness)
     - skew_slope ≈ 2.5 (Bollen & Whaley 2004 literature default)
     - Optionally calibrated from WRDS OptionMetrics sample
  3. Synthetic bid/ask: half_spread = mid * (base + vix_scaling * max(0, VIX-15))
  4. Settlement: Cash-settled at BS intrinsic on expiry

Integration Point:
  Called as Step E in RegimeBacktester.run() AFTER Steps A-D.

Author: Tanishk
Date: March 2026
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HedgeConfig:
    """
    All tunable hedge parameters.

    Design Choices:
      - Put SPREADS (not naked puts) to halve premium cost.
      - Hedge only in calm regimes where IV is low.
      - Monetize in stressed regimes where IV is high.
      - VIX-based synthetic BS pricing (no full OptionMetrics needed).
    """

    # ── Master Switch ─────────────────────────────────────────────────
    ENABLE_OPTIONS_HEDGE: bool = True

    # ── Hedge Sizing ──────────────────────────────────────────────────
    HEDGE_NOTIONAL_FRACTION: float = 1.0
    MARKET_BETA: float = 0.77

    # ── Strike Selection ──────────────────────────────────────────────
    LONG_PUT_OTM_PCT: float = 0.05            # 5% OTM long put
    SHORT_PUT_OTM_PCT: float = 0.15           # 15% OTM short put

    # ── Tenor & Roll ──────────────────────────────────────────────────
    TARGET_DTE: int = 30
    MIN_DTE_TO_HOLD: int = 5

    # ── Regime Rules ──────────────────────────────────────────────────
    HEDGE_ENTRY_REGIMES: tuple = ('normal', 'bull')
    HEDGE_EXIT_REGIMES: tuple = ('crisis', 'bear')

    # ── HMM Transition Signal ─────────────────────────────────────────
    TRANSITION_PROB_THRESHOLD: float = 0.30
    TRANSITION_SCALE_UP: float = 1.5

    # ── Synthetic Pricing Model ───────────────────────────────────────
    SKEW_SLOPE: float = 2.5           # IV(K) = VIX/100 * (1 + slope * |moneyness|)
    SPREAD_BASE_PCT: float = 0.03     # Base half-spread as % of mid
    SPREAD_VIX_SCALING: float = 0.002 # Extra spread per VIX point above 15

    # ── Execution ─────────────────────────────────────────────────────
    SPREAD_PENALTY_NORMAL: float = 0.10
    SPREAD_PENALTY_CRISIS: float = 0.30
    CONTRACT_MULTIPLIER: int = 100


# ═══════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES PRICING
# ═══════════════════════════════════════════════════════════════════════════

def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price."""
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        return max(K * np.exp(-r * T) - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put delta."""
    if T <= 0:
        return -1.0 if S < K else 0.0
    if sigma <= 0:
        return -1.0 if S < K * np.exp(-r * T) else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1.0


def skew_adjusted_iv(vix: float, spy_price: float, strike: float,
                     skew_slope: float = 2.5) -> float:
    """
    Compute skew-adjusted implied volatility.
    VIX = ATM IV for SPY. OTM puts have higher IV due to put skew.
    IV(K) = VIX/100 * (1 + skew_slope * moneyness)
    """
    atm_vol = vix / 100.0
    moneyness = max(0.0, (spy_price - strike) / spy_price)
    return atm_vol * (1.0 + skew_slope * moneyness)


def synthetic_bid_ask(mid_price: float, vix: float,
                      base_pct: float = 0.03,
                      vix_scaling: float = 0.002) -> Tuple[float, float]:
    """
    Generate synthetic bid/ask from mid price.
    Spread widens with VIX (Cao et al. 2020).
    """
    half_spread = mid_price * (base_pct + vix_scaling * max(0.0, vix - 15.0))
    bid = max(0.0, mid_price - half_spread)
    ask = mid_price + half_spread
    return bid, ask


# ═══════════════════════════════════════════════════════════════════════════
# DATA LAYER: Synthetic Options via VIX + Black-Scholes
# ═══════════════════════════════════════════════════════════════════════════

class SyntheticOptionsData:
    """
    Prices SPY options synthetically using VIX as IV proxy + skew model.
    Compatible interface with OptionsHedgeEngine.

    Data Requirements (auto-downloaded and cached):
      - VIX daily close (yfinance ^VIX)
      - SPY daily close (yfinance SPY)
      - Risk-free rate (from Fama-French cache or constant fallback)
    """

    def __init__(self, cfg: HedgeConfig = None):
        self.cfg = cfg or HedgeConfig()
        self.vix_data: Optional[pd.Series] = None
        self.spy_data: Optional[pd.Series] = None
        self.rf_data: Optional[pd.Series] = None

    def load(self, cache_path: str = None,
             start_date: str = '2005-01-01',
             end_date: str = '2026-03-01') -> None:
        """Load VIX, SPY, and RF data. Uses cache if available."""
        if cache_path and self._try_load_cache(cache_path):
            return
        self._download_data(start_date, end_date)
        if cache_path:
            self._save_cache(cache_path)

    def get_put_chain(self, date: pd.Timestamp,
                      target_dte: int = 30,
                      dte_tolerance: int = 7) -> Optional[pd.DataFrame]:
        """
        Generate a synthetic put chain for a given date.
        Returns DataFrame with: strike, best_bid, best_offer, mid_price,
        impl_volatility, delta, dte, exdate, spread
        """
        spy = self._get_spy(date)
        vix = self._get_vix(date)
        rf = self._get_rf(date)
        if spy is None or vix is None:
            return None

        T = target_dte / 365.0
        exdate = date + timedelta(days=target_dte)

        # Generate strikes: 25% OTM to ATM in $1 steps
        strike_low = int(spy * 0.75)
        strike_high = int(spy * 1.01)
        rows = []
        for K in range(strike_low, strike_high + 1):
            iv = skew_adjusted_iv(vix, spy, K, self.cfg.SKEW_SLOPE)
            mid = bs_put_price(spy, K, T, rf, iv)
            if mid < 0.01:
                continue
            bid, ask = synthetic_bid_ask(mid, vix,
                                         self.cfg.SPREAD_BASE_PCT,
                                         self.cfg.SPREAD_VIX_SCALING)
            delta = bs_put_delta(spy, K, T, rf, iv)
            rows.append({
                'strike': float(K), 'best_bid': bid, 'best_offer': ask,
                'mid_price': mid, 'impl_volatility': iv, 'delta': delta,
                'dte': target_dte, 'exdate': exdate, 'spread': ask - bid,
            })
        if not rows:
            return None
        return pd.DataFrame(rows).sort_values('strike').reset_index(drop=True)

    def get_option_price_on_date(self, date: pd.Timestamp,
                                  strike: float,
                                  exdate: pd.Timestamp) -> Optional[dict]:
        """Get synthetic bid/ask/mid for a specific strike and expiry on date."""
        spy = self._get_spy(date)
        vix = self._get_vix(date)
        rf = self._get_rf(date)
        if spy is None or vix is None:
            return None

        dte = max((exdate - date).days, 0)
        T = dte / 365.0
        iv = skew_adjusted_iv(vix, spy, strike, self.cfg.SKEW_SLOPE)
        mid = bs_put_price(spy, strike, T, rf, iv)
        bid, ask = synthetic_bid_ask(mid, vix,
                                      self.cfg.SPREAD_BASE_PCT,
                                      self.cfg.SPREAD_VIX_SCALING)
        return {
            'best_bid': bid, 'best_offer': ask, 'mid_price': mid,
            'impl_volatility': iv, 'delta': bs_put_delta(spy, strike, T, rf, iv),
            'spread': ask - bid,
        }

    def price_put(self, date: pd.Timestamp, spy_price: float,
                  strike: float, dte: int) -> dict:
        """Direct put pricing for sanity checks (uses provided spy_price)."""
        vix = self._get_vix(date)
        rf = self._get_rf(date)
        if vix is None:
            return {'mid_price': 0.0, 'impl_volatility': 0.0}
        T = dte / 365.0
        iv = skew_adjusted_iv(vix, spy_price, strike, self.cfg.SKEW_SLOPE)
        mid = bs_put_price(spy_price, strike, T, rf, iv)
        bid, ask = synthetic_bid_ask(mid, vix,
                                      self.cfg.SPREAD_BASE_PCT,
                                      self.cfg.SPREAD_VIX_SCALING)
        return {
            'mid_price': mid, 'best_bid': bid, 'best_offer': ask,
            'impl_volatility': iv,
            'delta': bs_put_delta(spy_price, strike, T, rf, iv),
            'spread': ask - bid, 'vix': vix,
        }

    # ─────────────────────────────────────────────────────────────────
    # Internal: Data Access
    # ─────────────────────────────────────────────────────────────────

    def _get_vix(self, date: pd.Timestamp) -> Optional[float]:
        if self.vix_data is None:
            return None
        if date in self.vix_data.index:
            v = self.vix_data.loc[date]
            if not pd.isna(v):
                return float(v)
        past = self.vix_data.loc[:date].dropna()
        return float(past.iloc[-1]) if len(past) > 0 else None

    def _get_spy(self, date: pd.Timestamp) -> Optional[float]:
        if self.spy_data is None:
            return None
        if date in self.spy_data.index:
            v = self.spy_data.loc[date]
            if not pd.isna(v):
                return float(v)
        past = self.spy_data.loc[:date].dropna()
        return float(past.iloc[-1]) if len(past) > 0 else None

    def _get_rf(self, date: pd.Timestamp) -> float:
        if self.rf_data is None:
            return 0.02
        if date in self.rf_data.index:
            v = self.rf_data.loc[date]
            if not pd.isna(v):
                return float(v)
        past = self.rf_data.loc[:date].dropna()
        return float(past.iloc[-1]) if len(past) > 0 else 0.02

    # ─────────────────────────────────────────────────────────────────
    # Data Loading & Caching
    # ─────────────────────────────────────────────────────────────────

    def _download_data(self, start_date: str, end_date: str) -> None:
        import yfinance as yf

        print("  Downloading VIX & SPY data for synthetic option pricing...")

        vix_raw = yf.download('^VIX', start=start_date, end=end_date, progress=False)
        if isinstance(vix_raw.columns, pd.MultiIndex):
            self.vix_data = vix_raw['Close'].squeeze()
        else:
            self.vix_data = vix_raw['Close']
        self.vix_data.index = pd.to_datetime(self.vix_data.index)

        spy_raw = yf.download('SPY', start=start_date, end=end_date, progress=False)
        if isinstance(spy_raw.columns, pd.MultiIndex):
            self.spy_data = spy_raw['Close'].squeeze()
        else:
            self.spy_data = spy_raw['Close']
        self.spy_data.index = pd.to_datetime(self.spy_data.index)

        self.rf_data = self._load_rf()

        print(f"  ✓ VIX: {len(self.vix_data)} days "
              f"({self.vix_data.index[0].date()} → {self.vix_data.index[-1].date()})")
        print(f"  ✓ SPY: {len(self.spy_data)} days")

    def _load_rf(self) -> pd.Series:
        """Load risk-free rate from FF5 cache or constant fallback."""
        ff5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ff5_data.parquet')
        try:
            if os.path.exists(ff5_path):
                ff5 = pd.read_parquet(ff5_path)
                if 'RF' in ff5.columns:
                    rf = ff5['RF'] / 100.0  # Daily pct → decimal
                    rf_annual = rf * 252     # Annualize for BS
                    rf_annual.index = pd.to_datetime(rf_annual.index)
                    print("  ✓ Risk-free rate from FF5 cache")
                    return rf_annual
        except Exception:
            pass
        print("  ✓ Risk-free rate: constant 2% fallback")
        if self.vix_data is not None:
            return pd.Series(0.02, index=self.vix_data.index)
        return pd.Series(dtype=float)

    def _try_load_cache(self, cache_path: str) -> bool:
        if os.path.exists(cache_path):
            print(f"  Loading cached VIX/SPY/RF data from {cache_path}...")
            df = pd.read_parquet(cache_path)
            df.index = pd.to_datetime(df.index)
            self.vix_data = df['vix'] if 'vix' in df.columns else None
            self.spy_data = df['spy'] if 'spy' in df.columns else None
            self.rf_data = df['rf'] if 'rf' in df.columns else None
            if self.vix_data is not None:
                print(f"  ✓ {len(self.vix_data)} days loaded from cache")
            return self.vix_data is not None
        return False

    def _save_cache(self, cache_path: str) -> None:
        df = pd.DataFrame({
            'vix': self.vix_data, 'spy': self.spy_data, 'rf': self.rf_data,
        })
        df.to_parquet(cache_path)
        print(f"  ✓ Cached VIX/SPY/RF data → {cache_path}")


# ═══════════════════════════════════════════════════════════════════════════
# OPTIONAL: WRDS Sample Calibration
# ═══════════════════════════════════════════════════════════════════════════

class WRDSSampleCalibrator:
    """
    Calibrate skew/spread parameters from WRDS OptionMetrics sample data.
    Uses optionmsamp_us.opprcd (the sample/trial database).

    This is OPTIONAL — the synthetic pricer works with literature defaults.
    Calibrating from real data strengthens the thesis argument.
    """

    def __init__(self, wrds_username: str = None):
        self.wrds_username = wrds_username

    def calibrate(self) -> dict:
        """Pull option data from WRDS sample and fit spread parameters."""
        try:
            import wrds
        except ImportError:
            logger.warning("wrds package not installed — using defaults")
            return {}

        try:
            db = wrds.Connection(wrds_username=self.wrds_username)
        except Exception as e:
            logger.warning(f"WRDS connection failed: {e}")
            return {}

        query = """
            SELECT date, exdate, strike_price / 1000.0 AS strike,
                   best_bid, best_offer, impl_volatility
            FROM optionmsamp_us.opprcd
            WHERE cp_flag = 'P'
              AND best_bid > 0
              AND best_offer > 0
              AND impl_volatility > 0
            LIMIT 100000
        """

        try:
            df = db.raw_sql(query)
        except Exception as e:
            logger.warning(f"WRDS sample query failed: {e}")
            db.close()
            return {}

        db.close()

        if df.empty or len(df) < 100:
            logger.warning("Insufficient WRDS sample data for calibration")
            return {}

        df['mid'] = (df['best_bid'] + df['best_offer']) / 2
        df['spread_pct'] = (df['best_offer'] - df['best_bid']) / df['mid']
        valid = df[df['spread_pct'].between(0, 1)]
        spread_base = float(valid['spread_pct'].median() / 2)

        result = {
            'spread_base_pct': spread_base,
            'skew_slope': 2.5,  # Conservative default
            'n_records': len(df),
            'calibration_source': 'optionmsamp_us',
        }
        print(f"  ✓ WRDS calibration: {len(df)} records, "
              f"spread_base_pct={spread_base:.4f}")
        return result


# ═══════════════════════════════════════════════════════════════════════════
# HEDGE POSITION TRACKING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PutSpreadPosition:
    """Represents one active put spread (long put + short put)."""
    entry_date: pd.Timestamp
    expiry: pd.Timestamp
    long_strike: float
    short_strike: float
    contracts: int
    net_premium_paid: float
    entry_regime: str
    long_entry_price: float
    short_entry_price: float

    current_long_mid: float = 0.0
    current_short_mid: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def max_payoff(self) -> float:
        return (self.long_strike - self.short_strike) * 100

    def intrinsic_value(self, spy_price: float) -> float:
        """Per-contract intrinsic value given SPY spot."""
        long_iv = max(0.0, self.long_strike - spy_price)
        short_iv = max(0.0, self.short_strike - spy_price)
        return (long_iv - short_iv) * 100

    def mark_to_market(self, long_mid: float, short_mid: float) -> float:
        self.current_long_mid = long_mid
        self.current_short_mid = short_mid
        current_value = (long_mid - short_mid) * self.contracts * 100
        self.unrealized_pnl = current_value - self.net_premium_paid
        return self.unrealized_pnl


# ═══════════════════════════════════════════════════════════════════════════
# HEDGE ENGINE (core logic)
# ═══════════════════════════════════════════════════════════════════════════

class OptionsHedgeEngine:
    """
    Manages the regime-conditional options hedge overlay.
    Works with SyntheticOptionsData (BS + VIX) for pricing.
    """

    def __init__(self, options_data: SyntheticOptionsData,
                 hedge_config: HedgeConfig = None):
        self.data = options_data
        self.cfg = hedge_config or HedgeConfig()

        self.active_positions: List[PutSpreadPosition] = []
        self.hedge_log: List[dict] = []
        self.daily_hedge_value: List[dict] = []

        self.total_premium_spent: float = 0.0
        self.total_monetization_proceeds: float = 0.0
        self.total_expiry_payoffs: float = 0.0
        self.total_roll_costs: float = 0.0

    # ─────────────────────────────────────────────────────────────────
    # Main Daily Entry Point
    # ─────────────────────────────────────────────────────────────────

    def process_day(self, date: pd.Timestamp, spy_price: float,
                    regime: str, portfolio_value: float,
                    hmm_transition_probs: Optional[Dict] = None) -> float:
        """
        Process one trading day. Returns net cash flow.
        Negative = premium paid, Positive = payoff/monetization.
        """
        if not self.cfg.ENABLE_OPTIONS_HEDGE:
            return 0.0

        net_cash_flow = 0.0

        # 1. Settle expired positions
        net_cash_flow += self._settle_expired(date, spy_price)

        # 2. Monetize in stressed regimes
        if regime in self.cfg.HEDGE_EXIT_REGIMES and self.active_positions:
            net_cash_flow += self._monetize_positions(date, spy_price, regime)

        # 3. Roll near-expiry positions
        net_cash_flow += self._roll_expiring(date, spy_price, regime, portfolio_value)

        # 4. Enter new hedges in calm regimes
        if regime in self.cfg.HEDGE_ENTRY_REGIMES and not self.active_positions:
            scale = 1.0
            if hmm_transition_probs:
                deterioration_prob = sum(
                    hmm_transition_probs.get(r, 0.0)
                    for r in self.cfg.HEDGE_EXIT_REGIMES
                )
                if deterioration_prob > self.cfg.TRANSITION_PROB_THRESHOLD:
                    scale = self.cfg.TRANSITION_SCALE_UP

            cost = self._enter_new_hedge(date, spy_price, portfolio_value,
                                         regime, scale)
            net_cash_flow += cost

        # 5. Mark-to-market
        hedge_mtm = self._mark_to_market_all(date, spy_price)

        vix = self.data._get_vix(date) or 0.0
        self.daily_hedge_value.append({
            'date': date, 'regime': regime,
            'n_active': len(self.active_positions),
            'hedge_mtm_value': hedge_mtm,
            'daily_cash_flow': net_cash_flow,
            'cumulative_premium': self.total_premium_spent,
            'cumulative_payoffs': (self.total_expiry_payoffs +
                                   self.total_monetization_proceeds),
            'vix': vix,
        })

        return net_cash_flow

    # ─────────────────────────────────────────────────────────────────
    # Step 1: Settle Expired
    # ─────────────────────────────────────────────────────────────────

    def _settle_expired(self, date: pd.Timestamp, spy_price: float) -> float:
        cash_flow = 0.0
        still_active = []
        for pos in self.active_positions:
            if date >= pos.expiry:
                payoff = pos.intrinsic_value(spy_price) * pos.contracts
                cash_flow += payoff
                self.total_expiry_payoffs += payoff
                self.hedge_log.append({
                    'date': date, 'action': 'EXPIRY_SETTLE',
                    'long_strike': pos.long_strike,
                    'short_strike': pos.short_strike,
                    'contracts': pos.contracts, 'spy_price': spy_price,
                    'payoff': payoff, 'premium_paid': pos.net_premium_paid,
                    'net_pnl': payoff - pos.net_premium_paid,
                    'entry_regime': pos.entry_regime,
                })
            else:
                still_active.append(pos)
        self.active_positions = still_active
        return cash_flow

    # ─────────────────────────────────────────────────────────────────
    # Step 2: Monetize in Stressed Regimes
    # ─────────────────────────────────────────────────────────────────

    def _monetize_positions(self, date: pd.Timestamp,
                            spy_price: float, regime: str) -> float:
        cash_flow = 0.0
        remaining = []
        for pos in self.active_positions:
            long_price = self.data.get_option_price_on_date(
                date, pos.long_strike, pos.expiry)
            short_price = self.data.get_option_price_on_date(
                date, pos.short_strike, pos.expiry)
            if long_price is None or short_price is None:
                remaining.append(pos)
                continue

            sp = self.cfg.SPREAD_PENALTY_CRISIS
            sell_long = long_price['best_bid'] * (1 - sp)
            buy_short = short_price['best_offer'] * (1 + sp)
            net_close = ((sell_long - buy_short) * pos.contracts
                         * self.cfg.CONTRACT_MULTIPLIER)
            if net_close <= 0:
                remaining.append(pos)
                continue

            cash_flow += net_close
            self.total_monetization_proceeds += net_close
            self.hedge_log.append({
                'date': date, 'action': 'MONETIZE',
                'long_strike': pos.long_strike,
                'short_strike': pos.short_strike,
                'contracts': pos.contracts, 'spy_price': spy_price,
                'close_proceeds': net_close,
                'premium_paid': pos.net_premium_paid,
                'net_pnl': net_close - pos.net_premium_paid,
                'regime': regime,
            })
        self.active_positions = remaining
        return cash_flow

    # ─────────────────────────────────────────────────────────────────
    # Step 3: Roll Expiring Positions
    # ─────────────────────────────────────────────────────────────────

    def _roll_expiring(self, date, spy_price, regime, portfolio_value) -> float:
        cash_flow = 0.0
        still_active = []
        for pos in self.active_positions:
            dte = (pos.expiry - date).days
            if dte <= self.cfg.MIN_DTE_TO_HOLD:
                close_cash = self._close_single(date, pos, spy_price)
                cash_flow += close_cash
                self.total_roll_costs += abs(close_cash) if close_cash < 0 else 0
                if regime in self.cfg.HEDGE_ENTRY_REGIMES:
                    entry_cost = self._enter_new_hedge(
                        date, spy_price, portfolio_value, regime, 1.0)
                    cash_flow += entry_cost
                self.hedge_log.append({
                    'date': date, 'action': 'ROLL',
                    'old_expiry': pos.expiry,
                    'close_proceeds': close_cash, 'regime': regime,
                })
            else:
                still_active.append(pos)
        self.active_positions = still_active
        return cash_flow

    def _close_single(self, date, pos, spy_price) -> float:
        long_p = self.data.get_option_price_on_date(
            date, pos.long_strike, pos.expiry)
        short_p = self.data.get_option_price_on_date(
            date, pos.short_strike, pos.expiry)
        if long_p is None or short_p is None:
            return pos.intrinsic_value(spy_price) * pos.contracts
        sp = (self.cfg.SPREAD_PENALTY_CRISIS
              if pos.entry_regime in self.cfg.HEDGE_EXIT_REGIMES
              else self.cfg.SPREAD_PENALTY_NORMAL)
        sell_long = long_p['best_bid'] * (1 - sp)
        buy_short = short_p['best_offer'] * (1 + sp)
        return ((sell_long - buy_short) * pos.contracts
                * self.cfg.CONTRACT_MULTIPLIER)

    # ─────────────────────────────────────────────────────────────────
    # Step 4: Enter New Hedge
    # ─────────────────────────────────────────────────────────────────

    def _enter_new_hedge(self, date, spy_price, portfolio_value,
                         regime, scale=1.0) -> float:
        chain = self.data.get_put_chain(date, target_dte=self.cfg.TARGET_DTE)
        if chain is None or chain.empty:
            return 0.0

        long_target = spy_price * (1 - self.cfg.LONG_PUT_OTM_PCT)
        short_target = spy_price * (1 - self.cfg.SHORT_PUT_OTM_PCT)

        long_put = self._find_nearest_strike(chain, long_target)
        short_put = self._find_nearest_strike(chain, short_target)

        if long_put is None or short_put is None:
            return 0.0
        if long_put['strike'] <= short_put['strike']:
            return 0.0

        hedge_notional = (portfolio_value * self.cfg.MARKET_BETA
                          * self.cfg.HEDGE_NOTIONAL_FRACTION * scale)
        contracts = max(1, int(hedge_notional
                               / (spy_price * self.cfg.CONTRACT_MULTIPLIER)))

        sp = self.cfg.SPREAD_PENALTY_NORMAL
        long_cost = long_put['best_offer'] * (1 + sp)
        short_credit = short_put['best_bid'] * (1 - sp)
        net_cost_per = (long_cost - short_credit) * self.cfg.CONTRACT_MULTIPLIER

        if net_cost_per <= 0:
            return 0.0

        total_premium = net_cost_per * contracts
        max_premium = portfolio_value * 0.03
        if total_premium > max_premium:
            contracts = max(1, int(max_premium / net_cost_per))
            total_premium = net_cost_per * contracts

        pos = PutSpreadPosition(
            entry_date=date, expiry=long_put['exdate'],
            long_strike=long_put['strike'],
            short_strike=short_put['strike'],
            contracts=contracts, net_premium_paid=total_premium,
            entry_regime=regime,
            long_entry_price=long_cost, short_entry_price=short_credit,
        )
        self.active_positions.append(pos)
        self.total_premium_spent += total_premium

        self.hedge_log.append({
            'date': date, 'action': 'ENTER',
            'long_strike': pos.long_strike,
            'short_strike': pos.short_strike,
            'expiry': pos.expiry, 'contracts': contracts,
            'net_premium': total_premium, 'spy_price': spy_price,
            'regime': regime, 'scale': scale,
            'long_iv': long_put.get('impl_volatility', np.nan),
            'short_iv': short_put.get('impl_volatility', np.nan),
        })
        return -total_premium

    # ─────────────────────────────────────────────────────────────────
    # Step 5: Mark-to-Market
    # ─────────────────────────────────────────────────────────────────

    def _mark_to_market_all(self, date, spy_price) -> float:
        total_mtm = 0.0
        for pos in self.active_positions:
            long_p = self.data.get_option_price_on_date(
                date, pos.long_strike, pos.expiry)
            short_p = self.data.get_option_price_on_date(
                date, pos.short_strike, pos.expiry)
            if long_p and short_p:
                pos.mark_to_market(long_p['mid_price'], short_p['mid_price'])
                value = ((long_p['mid_price'] - short_p['mid_price'])
                         * pos.contracts * self.cfg.CONTRACT_MULTIPLIER)
                total_mtm += value
            else:
                total_mtm += pos.intrinsic_value(spy_price) * pos.contracts
        return total_mtm

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _find_nearest_strike(chain: pd.DataFrame,
                              target: float) -> Optional[dict]:
        if chain.empty:
            return None
        idx = (chain['strike'] - target).abs().idxmin()
        return chain.loc[idx].to_dict()

    # ─────────────────────────────────────────────────────────────────
    # Summary & Attribution
    # ─────────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Hedge overlay performance summary for metrics.json."""
        total_payoffs = (self.total_expiry_payoffs
                         + self.total_monetization_proceeds)
        net_hedge_pnl = total_payoffs - self.total_premium_spent

        hedge_df = pd.DataFrame(self.daily_hedge_value)
        regime_breakdown = {}
        if not hedge_df.empty:
            for regime, group in hedge_df.groupby('regime'):
                regime_breakdown[regime] = {
                    'days': len(group),
                    'total_cash_flow': group['daily_cash_flow'].sum(),
                    'avg_hedge_value': group['hedge_mtm_value'].mean(),
                    'avg_vix': group['vix'].mean(),
                }

        return {
            'pricing_method': 'Synthetic BS (VIX + skew)',
            'skew_slope': self.cfg.SKEW_SLOPE,
            'total_premium_spent': self.total_premium_spent,
            'total_expiry_payoffs': self.total_expiry_payoffs,
            'total_monetization_proceeds': self.total_monetization_proceeds,
            'total_roll_costs': self.total_roll_costs,
            'net_hedge_pnl': net_hedge_pnl,
            'total_trades': len(self.hedge_log),
            'regime_breakdown': regime_breakdown,
            'cost_as_pct_of_payoffs': (
                self.total_premium_spent / total_payoffs
                if total_payoffs > 0 else float('inf')
            ),
        }

    def get_hedge_log_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.hedge_log)

    def get_daily_hedge_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.daily_hedge_value)
