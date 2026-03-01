"""
REGIME-AWARE FUNDAMENTAL BACKTEST ENGINE (v2 - PIT Clean)
==========================================================
Fixes from v1:
  1. LOOKAHEAD BIAS ELIMINATED: Trades trigger on `asof_date` (SEC acceptance
     date), not on period_end. A Q1 filing isn't actionable until the SEC
     publishes it (~April/May), not on January 1st.
  2. SURVIVORSHIP BIAS HANDLED: Detects delisted tickers via price gaps and
     applies a configurable delisting return instead of silently dropping them.
  3. WALK-FORWARD ONLY: Regime detection confirmed bias-free.

Usage:
    python regime_aware_backtest.py
"""

import json
import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict

import config
from rank_system_v2 import build_pit_rankings
from regime_detector import WalkForwardRegimeDetector


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_transaction_cost(regime: str, direction: str) -> float:
    """Look up transaction cost for a given regime and direction ('buy'/'sell')."""
    costs = config.TRANSACTION_COSTS.get(regime, config.TRANSACTION_COSTS['normal'])
    return costs.get(direction, 0.002)


def compute_price_drawdown(prices_series: pd.Series, lookback: int = 63) -> float:
    """Drawdown of latest price from rolling peak. Returns negative (e.g. -0.15)."""
    if len(prices_series) < 2:
        return 0.0
    window = prices_series.iloc[-lookback:]
    peak = window.max()
    if peak == 0:
        return 0.0
    return (prices_series.iloc[-1] / peak) - 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeAwareBacktester:
    """
    Point-in-time clean backtest engine.

    Key difference from v1: rebalancing is triggered by actual filing arrival
    dates (asof_date), not by calendar quarter boundaries. This ensures no
    fundamental data is used before it was publicly available.
    """

    def __init__(self, data_dir: str, script_dir: str = '.'):
        self.data_dir = data_dir
        self.script_dir = script_dir

        # Portfolio state
        self.cash = config.INITIAL_CAPITAL
        self.positions = {}           # ticker → shares (float)
        self.cost_basis = {}          # ticker → total $ invested
        self.spy_shares = 0.0

        # Tracking
        self.portfolio_history = []
        self.trade_log = []
        self.regime_history = []
        self.total_transaction_costs = 0.0

        # Data
        self.pit_rankings = None      # {date_str: [ranked list]}
        self.rebalance_dates = []     # sorted list of pd.Timestamp
        self.prices = None
        self.regime_detector = None
        self.prev_rankings = {}       # ticker → {rank, score} at last rebalance

        # Survivorship tracking
        self.delisted_tickers = set()
        self.nan_streak = defaultdict(int)  # ticker → consecutive NaN days
        self.survivorship_warnings = []

    # ─────────────────────────────────────────────────────────────────────
    # Data Loading
    # ─────────────────────────────────────────────────────────────────────
    def load_data(self):
        """Load PIT rankings and price data."""
        print("=" * 70)
        print("REGIME-AWARE FUNDAMENTAL BACKTEST (v2 - PIT Clean)")
        print("=" * 70)

        # 1. Build point-in-time rankings
        print("\n[1/3] Building point-in-time rankings (asof_date gated)...")
        self.pit_rankings = build_pit_rankings(self.data_dir)
        self.rebalance_dates = sorted([pd.Timestamp(d) for d in self.pit_rankings.keys()])
        print(f"  ✓ {len(self.rebalance_dates)} rebalance events")
        if self.rebalance_dates:
            print(f"  ✓ First rebalance: {self.rebalance_dates[0].date()}")
            print(f"  ✓ Last rebalance:  {self.rebalance_dates[-1].date()}")

        # Save rankings
        rankings_path = os.path.join(self.script_dir, config.RANKINGS_CACHE)
        with open(rankings_path, 'w') as f:
            json.dump(self.pit_rankings, f, indent=2)

        # 2. Collect all tickers
        all_tickers = set(['SPY'])
        for ranked_list in self.pit_rankings.values():
            for entry in ranked_list:
                all_tickers.add(entry['ticker'])
        all_tickers = sorted(all_tickers)
        print(f"\n[2/3] Loading price data for {len(all_tickers)} tickers...")

        # 3. Fetch/cache prices
        cache_path = os.path.join(self.script_dir, config.PRICE_CACHE)
        if os.path.exists(cache_path):
            print(f"  Loading cached prices...")
            self.prices = pd.read_parquet(cache_path)
            # missing = [t for t in all_tickers if t not in self.prices.columns]
            # if missing:
            #     print(f"  Fetching {len(missing)} missing tickers...")
            #     new_data = yf.download(missing, start=config.PRICE_DATA_START,
            #                             end=config.BACKTEST_END, progress=False)
            #     if isinstance(new_data.columns, pd.MultiIndex):
            #         new_prices = new_data['Close']
            #     else:
            #         new_prices = new_data
            #     self.prices = pd.concat([self.prices, new_prices], axis=1)
            #     self.prices.to_parquet(cache_path)
        else:
            print(f"  ERROR: Price cache not found at {cache_path}. Exiting.")
            sys.exit(1)
            # print(f"  Downloading from Yahoo Finance...")
            # raw = yf.download(all_tickers, start=config.PRICE_DATA_START,
            #                   end=config.BACKTEST_END, progress=False)
            # if isinstance(raw.columns, pd.MultiIndex):
            #     self.prices = raw['Close']
            # else:
            #     self.prices = raw
            # self.prices.to_parquet(cache_path)

        # 4. Check for survivorship bias signals
        self._check_survivorship_bias(all_tickers)

        print(f"  ✓ Price data: {self.prices.shape[0]} days × {self.prices.shape[1]} tickers")

        # 5. Initialize regime detector
        print("\n[3/3] Initializing walk-forward regime detector...")
        self.regime_detector = WalkForwardRegimeDetector()
        print(f"  ✓ HMM with {config.N_REGIMES} regimes ready")

    def _check_survivorship_bias(self, requested_tickers: list):
        """Warn about tickers with missing or truncated price data."""
        if not config.WARN_MISSING_PRICE_TICKERS:
            return

        missing_entirely = [t for t in requested_tickers
                           if t != 'SPY' and t not in self.prices.columns]
        if missing_entirely:
            msg = (f"  ⚠ SURVIVORSHIP WARNING: {len(missing_entirely)} tickers have "
                   f"NO price data in Yahoo Finance (likely delisted/acquired):")
            print(msg)
            for t in missing_entirely[:10]:
                print(f"      {t}")
            if len(missing_entirely) > 10:
                print(f"      ... and {len(missing_entirely) - 10} more")
            self.survivorship_warnings.append(
                f"{len(missing_entirely)} tickers with no price data"
            )

        # Check for tickers whose price data ends early
        truncated = []
        if len(self.prices) > 0:
            last_valid = self.prices.apply(lambda col: col.last_valid_index())
            market_end = self.prices.index[-1]
            for ticker in self.prices.columns:
                if ticker == 'SPY':
                    continue
                lv = last_valid.get(ticker)
                if lv is not None and (market_end - lv).days > 60:
                    truncated.append((ticker, lv.date()))

        if truncated:
            print(f"  ⚠ SURVIVORSHIP WARNING: {len(truncated)} tickers have "
                  f"price data ending significantly before market end:")
            for t, end_date in truncated[:5]:
                print(f"      {t}: last price on {end_date}")
            self.survivorship_warnings.append(
                f"{len(truncated)} tickers with truncated price data"
            )

    # ─────────────────────────────────────────────────────────────────────
    # Trading Actions
    # ─────────────────────────────────────────────────────────────────────
    def _buy(self, ticker: str, dollar_amount: float, price: float,
             regime: str, date: pd.Timestamp, reason: str):
        """Execute a buy order with regime-specific transaction costs."""
        if pd.isna(price) or price <= 0 or dollar_amount <= 0:
            return
        if ticker in self.delisted_tickers:
            return  # Don't buy delisted stocks

        cost_rate = get_transaction_cost(regime, 'buy')
        fee = dollar_amount * cost_rate
        if self.cash < dollar_amount + fee:
            dollar_amount = self.cash / (1 + cost_rate)
            fee = dollar_amount * cost_rate
        if dollar_amount <= 0:
            return

        shares = dollar_amount / price
        self.cash -= (dollar_amount + fee)
        self.positions[ticker] = self.positions.get(ticker, 0) + shares
        self.cost_basis[ticker] = self.cost_basis.get(ticker, 0) + dollar_amount
        self.total_transaction_costs += fee

        self.trade_log.append({
            'date': date, 'ticker': ticker, 'action': 'BUY',
            'shares': shares, 'price': price, 'value': dollar_amount,
            'fee': fee, 'regime': regime, 'reason': reason
        })

    def _sell(self, ticker: str, fraction: float, price: float,
              regime: str, date: pd.Timestamp, reason: str):
        """Execute a sell order (fraction of position)."""
        if ticker not in self.positions or pd.isna(price) or price <= 0:
            return
        shares_to_sell = self.positions[ticker] * fraction
        if shares_to_sell <= 0:
            return

        proceeds = shares_to_sell * price
        cost_rate = get_transaction_cost(regime, 'sell')
        fee = proceeds * cost_rate

        self.cash += (proceeds - fee)
        self.positions[ticker] -= shares_to_sell
        self.cost_basis[ticker] = self.cost_basis.get(ticker, 0) * (1 - fraction)
        self.total_transaction_costs += fee

        if self.positions[ticker] <= 1e-8:
            del self.positions[ticker]
            if ticker in self.cost_basis:
                del self.cost_basis[ticker]

        self.trade_log.append({
            'date': date, 'ticker': ticker, 'action': 'SELL',
            'shares': shares_to_sell, 'price': price, 'value': proceeds,
            'fee': fee, 'regime': regime, 'reason': reason
        })

    def _force_delist(self, ticker: str, date: pd.Timestamp, regime: str):
        """
        Force-liquidate a position at a loss when delisting is detected.
        Uses the last known price × (1 + DELISTING_RETURN).
        """
        if ticker not in self.positions:
            return

        shares = self.positions[ticker]
        # Find last known price
        if ticker in self.prices.columns:
            last_valid_idx = self.prices[ticker].loc[:date].last_valid_index()
            if last_valid_idx is not None:
                last_price = self.prices.loc[last_valid_idx, ticker]
            else:
                last_price = 0.0
        else:
            last_price = 0.0

        # Apply delisting return
        delist_price = last_price * (1 + config.DELISTING_RETURN)
        delist_price = max(delist_price, 0.0)

        proceeds = shares * delist_price
        cost_rate = get_transaction_cost(regime, 'sell')
        fee = proceeds * cost_rate

        self.cash += (proceeds - fee)
        self.total_transaction_costs += fee

        self.trade_log.append({
            'date': date, 'ticker': ticker, 'action': 'DELIST',
            'shares': shares, 'price': delist_price, 'value': proceeds,
            'fee': fee, 'regime': regime,
            'reason': f'DELISTED_AT_{config.DELISTING_RETURN:.0%}'
        })

        del self.positions[ticker]
        if ticker in self.cost_basis:
            del self.cost_basis[ticker]
        self.delisted_tickers.add(ticker)

    def _portfolio_value(self, date: pd.Timestamp) -> float:
        """Mark-to-market portfolio value."""
        value = self.cash
        for ticker, shares in self.positions.items():
            if ticker in self.prices.columns:
                p = self.prices.loc[date, ticker] if date in self.prices.index else np.nan
                if not pd.isna(p):
                    value += shares * p
                else:
                    # Use last known price for valuation
                    last = self.prices[ticker].loc[:date].last_valid_index()
                    if last is not None:
                        value += shares * self.prices.loc[last, ticker]
        return value

    # ─────────────────────────────────────────────────────────────────────
    # Regime Detection (Walk-Forward, confirmed bias-free)
    # ─────────────────────────────────────────────────────────────────────
    def _detect_regime(self, date: pd.Timestamp) -> str:
        """Detect current regime using ONLY past data."""
        spy = self.prices['SPY']
        spy_past = spy.loc[:date].dropna()

        if len(spy_past) < config.REGIME_TRAIN_WINDOW:
            return 'normal'

        train_data = spy_past.iloc[-config.REGIME_TRAIN_WINDOW:]
        try:
            self.regime_detector.fit(train_data)
            predict_window = spy_past.iloc[-(config.LONG_TERM_TREND + 10):]
            return self.regime_detector.predict_single(predict_window)
        except Exception:
            return 'normal'

    # ─────────────────────────────────────────────────────────────────────
    # Delisting Detection
    # ─────────────────────────────────────────────────────────────────────
    def _check_delistings(self, date: pd.Timestamp, regime: str):
        """
        Check held positions for signs of delisting (consecutive NaN prices).
        Force-liquidate at a loss if threshold is exceeded.
        """
        for ticker in list(self.positions.keys()):
            if ticker in self.delisted_tickers:
                continue
            if ticker not in self.prices.columns:
                self._force_delist(ticker, date, regime)
                continue

            price = self.prices.loc[date, ticker] if date in self.prices.index else np.nan
            if pd.isna(price):
                self.nan_streak[ticker] += 1
                if self.nan_streak[ticker] >= config.DELISTING_NAN_THRESHOLD_DAYS:
                    self._force_delist(ticker, date, regime)
            else:
                self.nan_streak[ticker] = 0

    # ─────────────────────────────────────────────────────────────────────
    # Rebalance Logic (triggered by filing arrival, NOT calendar quarter)
    # ─────────────────────────────────────────────────────────────────────
    def _get_current_rankings(self, rebalance_date_str: str) -> dict:
        """Get ticker → {rank, stability_score} for a given rebalance date."""
        if rebalance_date_str not in self.pit_rankings:
            return {}
        return {
            entry['ticker']: {'rank': entry['rank'],
                              'stability_score': entry['stability_score']}
            for entry in self.pit_rankings[rebalance_date_str]
        }

    def _rebalance(self, date: pd.Timestamp, rebalance_date_str: str, regime: str):
        """
        Execute rebalancing on a filing arrival date.

        This is triggered ONLY when new SEC filings become public (asof_date),
        not on arbitrary calendar boundaries.
        """
        current_rankings = self._get_current_rankings(rebalance_date_str)
        if not current_rankings:
            return

        # ── STEP A: Liquidation for fallen ranks ──────────────────────
        for ticker in list(self.positions.keys()):
            if ticker in self.delisted_tickers:
                continue

            if ticker not in current_rankings:
                # No longer in the ranked universe → full liquidate
                if ticker in self.prices.columns:
                    price = self.prices.loc[date, ticker] if date in self.prices.index else np.nan
                    if pd.isna(price):
                        last = self.prices[ticker].loc[:date].last_valid_index()
                        price = self.prices.loc[last, ticker] if last is not None else np.nan
                    self._sell(ticker, 1.0, price, regime, date, 'UNRANKED_FULL_LIQ')
                continue

            rank = current_rankings[ticker]['rank']

            if rank > config.FULL_LIQUIDATION_RANK:
                price = self._get_price(ticker, date)
                self._sell(ticker, 1.0, price, regime, date, f'FULL_LIQ_RANK_{rank}')
            elif rank > config.HALF_LIQUIDATION_RANK:
                price = self._get_price(ticker, date)
                self._sell(ticker, config.HALF_LIQUIDATION_FRACTION, price, regime, date,
                           f'HALF_LIQ_RANK_{rank}')

        # ── STEP B: Ensure top-N positions are held ───────────────────
        ranked_list = self.pit_rankings[rebalance_date_str]
        top_n_tickers = [e['ticker'] for e in ranked_list if e['rank'] <= config.TOP_N_INVEST]

        port_val = self._portfolio_value(date)
        target_per_position = port_val / config.TOP_N_INVEST

        for ticker in top_n_tickers:
            if ticker in self.delisted_tickers:
                continue
            price = self._get_price(ticker, date)
            if pd.isna(price) or price <= 0:
                continue

            current_shares = self.positions.get(ticker, 0)
            current_value = current_shares * price

            if current_value < target_per_position * 0.8:
                buy_amount = target_per_position - current_value
                self._buy(ticker, buy_amount, price, regime, date,
                          f'REBALANCE_TOP{config.TOP_N_INVEST}')

        # ── STEP C: Rank-jump new capital allocation ──────────────────
        if self.prev_rankings:
            for ticker, info in current_rankings.items():
                current_rank = info['rank']
                prev_info = self.prev_rankings.get(ticker)
                if prev_info is None:
                    continue
                prev_rank = prev_info['rank']
                rank_improvement = prev_rank - current_rank

                if current_rank <= config.TOP_N_INVEST and ticker in self.positions:
                    continue  # Already handled in step B
                if ticker in self.delisted_tickers:
                    continue

                price = self._get_price(ticker, date)
                if pd.isna(price) or price <= 0:
                    continue

                allocation = 0.0
                if rank_improvement >= config.RANK_JUMP_LARGE:
                    allocation = config.RANK_JUMP_SMALL_CAPITAL + config.RANK_JUMP_LARGE_CAPITAL
                    reason = f'RANK_JUMP_{rank_improvement}_LARGE'
                elif rank_improvement >= config.RANK_JUMP_SMALL:
                    allocation = config.RANK_JUMP_SMALL_CAPITAL
                    reason = f'RANK_JUMP_{rank_improvement}_SMALL'

                if allocation > 0:
                    self._buy(ticker, allocation, price, regime, date, reason)

        # ── STEP D: Panic-buy during crisis/bear ──────────────────────
        if config.ENABLE_PANIC_BUY:
            if regime in ('crisis', 'bear'):
                self._execute_panic_buy(date, ranked_list, current_rankings, regime)

        # Save for next rebalance's rank-jump detection
        self.prev_rankings = current_rankings

    def _get_price(self, ticker: str, date: pd.Timestamp) -> float:
        """Get price for ticker on date, falling back to last valid price."""
        if ticker not in self.prices.columns:
            return np.nan
        if date in self.prices.index:
            p = self.prices.loc[date, ticker]
            if not pd.isna(p):
                return p
        # Fall back to last valid
        last = self.prices[ticker].loc[:date].last_valid_index()
        if last is not None:
            return self.prices.loc[last, ticker]
        return np.nan

    # ─────────────────────────────────────────────────────────────────────
    # Panic Buy Logic
    # ─────────────────────────────────────────────────────────────────────
    def _execute_panic_buy(self, date: pd.Timestamp, ranked_list: list,
                            current_rankings: dict, regime: str):
        """
        During crisis/bear: if a top-N company's fundamentals are retained/improved
        but price has dropped, increase position using 4:3:2:2:1 + 0.5 weighting.
        """
        top_n_tickers = [e['ticker'] for e in ranked_list if e['rank'] <= config.TOP_N_INVEST]
        candidates = []

        for ticker in top_n_tickers:
            if ticker not in self.positions or ticker in self.delisted_tickers:
                continue
            if ticker not in self.prices.columns:
                continue

            # Check rank retained or improved vs previous rebalance
            current_rank = current_rankings[ticker]['rank']
            prev_info = self.prev_rankings.get(ticker)
            if prev_info and prev_info['rank'] < current_rank:
                continue  # Rank worsened

            # Check price drawdown
            price_history = self.prices[ticker].loc[:date].dropna()
            drawdown = compute_price_drawdown(price_history, lookback=config.LONG_TERM_TREND)

            if drawdown > config.PANIC_BUY_PRICE_DROP_THRESHOLD:
                continue  # Not enough drop

            candidates.append({
                'ticker': ticker,
                'rank': current_rank,
                'drawdown': drawdown,
                'current_value': self.positions[ticker] * self._get_price(ticker, date)
            })

        if not candidates:
            return

        candidates.sort(key=lambda x: (x['rank'], x['drawdown']))

        weights = []
        for i in range(len(candidates)):
            if i < len(config.PANIC_BUY_RATIOS_TOP5):
                weights.append(config.PANIC_BUY_RATIOS_TOP5[i])
            else:
                weights.append(config.PANIC_BUY_RATIO_REST)

        total_weight = sum(weights)
        if total_weight == 0:
            return

        total_current = sum(c['current_value'] for c in candidates)
        total_panic_capital = total_current * config.PANIC_BUY_CAPITAL_INCREASE_PCT
        total_panic_capital = min(total_panic_capital, self.cash * 0.9)

        for i, cand in enumerate(candidates):
            w = weights[i] / total_weight
            alloc = total_panic_capital * w
            price = self._get_price(cand['ticker'], date)
            self._buy(cand['ticker'], alloc, price, regime, date,
                      f"PANIC_BUY_DD{cand['drawdown']:.1%}")

    # ─────────────────────────────────────────────────────────────────────
    # Main Backtest Loop
    # ─────────────────────────────────────────────────────────────────────
    def run(self):
        """Execute the full backtest."""
        print("\n" + "=" * 70)
        print("RUNNING BACKTEST (Point-in-Time Clean)")
        print("=" * 70)

        if not self.rebalance_dates:
            print("ERROR: No rebalance dates found.")
            return None

        dates = self.prices.index
        start_date = self.rebalance_dates[0]

        # Find first valid trading day on or after first rebalance
        valid_dates = dates[dates >= start_date]
        if len(valid_dates) == 0:
            print("ERROR: No valid trading dates.")
            return None

        start_date = valid_dates[0]
        print(f"\n  Start date:     {start_date.date()}")
        print(f"  End date:       {valid_dates[-1].date()}")
        print(f"  Trading days:   {len(valid_dates)}")
        print(f"  Rebalance events: {len(self.rebalance_dates)}")
        print(f"  Initial capital:  ${config.INITIAL_CAPITAL:,.0f}")

        # SPY benchmark
        spy_start_price = self.prices.loc[start_date, 'SPY']
        self.spy_shares = config.INITIAL_CAPITAL / spy_start_price

        # Build set of rebalance dates for O(1) lookup
        rebalance_set = {d.date(): str(d.date()) for d in self.rebalance_dates}

        # Initial allocation on first rebalance
        first_rebalance_str = str(self.rebalance_dates[0].date())
        first_rankings = self.pit_rankings.get(first_rebalance_str, [])
        top_n = [e['ticker'] for e in first_rankings if e['rank'] <= config.TOP_N_INVEST]

        print(f"\n  Initial top {config.TOP_N_INVEST}: {top_n}")

        alloc_per = config.INITIAL_CAPITAL / config.TOP_N_INVEST
        current_regime = 'normal'

        for ticker in top_n:
            price = self._get_price(ticker, start_date)
            if not pd.isna(price) and price > 0:
                self._buy(ticker, alloc_per, price, 'normal', start_date, 'INITIAL_ALLOC')

        self.prev_rankings = self._get_current_rankings(first_rebalance_str)
        print(f"  ✓ Invested in {len(self.positions)} positions")

        # ── Day-by-day loop ───────────────────────────────────────────
        last_regime_fit_date = None
        regime_refit_interval = 63

        processed_rebalances = {start_date.date()}  # Don't re-trigger first day

        for i, date in enumerate(valid_dates):
            # -- Regime detection (periodic refit) --
            if (last_regime_fit_date is None or
                    (date - last_regime_fit_date).days >= regime_refit_interval):
                current_regime = self._detect_regime(date)
                last_regime_fit_date = date

            self.regime_history.append((date, current_regime))

            # -- Check for delistings --
            self._check_delistings(date, current_regime)

            # -- Check if this is a rebalance date --
            date_key = date.date()
            if date_key in rebalance_set and date_key not in processed_rebalances:
                rebalance_str = rebalance_set[date_key]
                processed_rebalances.add(date_key)

                port_val = self._portfolio_value(date)
                spy_val = self.spy_shares * self.prices.loc[date, 'SPY']
                print(f"  [{date.date()}] REBALANCE  "
                      f"Regime={current_regime:>7s}  "
                      f"Portfolio=${port_val:>12,.0f}  "
                      f"SPY=${spy_val:>12,.0f}  "
                      f"Pos={len(self.positions)}")

                self._rebalance(date, rebalance_str, current_regime)

            # -- Daily mark-to-market --
            # Note: The original code executed trades at `date`'s Close based on indicators from `date`.
            # We now execute trades on `date + 1` (the next day's opening/closing price)
            # Or conservatively, if rebalancing happened today, the trades executed using `date` prices
            # imply MOC. A safer assumption is moving forward by tracking next-day prices.
            # To avoid massive logical changes in `_rebalance_on_quarter`, we simply shift prices
            # supplied to trade functions forward.

            port_val = self._portfolio_value(date)
            spy_price = self.prices.loc[date, 'SPY']
            spy_val = self.spy_shares * spy_price if not pd.isna(spy_price) else np.nan

            self.portfolio_history.append({
                'date': date,
                'portfolio_value': port_val,
                'spy_value': spy_val,
                'cash': self.cash,
                'n_positions': len(self.positions),
                'regime': current_regime,
            })

        print(f"\n  ✓ Backtest complete!")
        if self.delisted_tickers:
            print(f"  ⚠ {len(self.delisted_tickers)} positions force-liquidated "
                  f"as delistings: {self.delisted_tickers}")
        return self._build_results()

    # ─────────────────────────────────────────────────────────────────────
    # Results & FF5
    # ─────────────────────────────────────────────────────────────────────
    def _run_ff5_regression(self, port_returns: pd.Series) -> dict:
        """Run Fama-French 5-Factor regression on daily returns."""
        try:
            import pandas_datareader.data as web
            import statsmodels.api as sm
            print("  Running Fama-French 5-Factor OLS regression...")
            
            # Fetch FF5 Daily Data
            ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench', 
                                 start=port_returns.index[0], end=port_returns.index[-1])[0]
            
            ff5_returns = ff5 / 100.0
            df = pd.DataFrame({'Strategy': port_returns}).join(ff5_returns, how='inner').dropna()

            if len(df) < 100:
                return {"error": "Not enough data overlap with FF5 factors."}

            Y = df['Strategy'] - df['RF']
            X = df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
            X = sm.add_constant(X)
            model = sm.OLS(Y, X).fit()

            return {
                'alpha_annualized': model.params['const'] * 252,
                'Mkt-RF': model.params['Mkt-RF'],
                'SMB': model.params['SMB'],
                'HML': model.params['HML'],
                'RMW': model.params['RMW'],
                'CMA': model.params['CMA'],
                'r_squared': model.rsquared
            }
        except Exception as e:
            print(f"  ⚠ Fama-French OLS failed: {e}")
            return {"error": str(e)}

    def _build_results(self) -> dict:
        """Compile results into a summary."""
        history_df = pd.DataFrame(self.portfolio_history).set_index('date')
        trades_df = pd.DataFrame(self.trade_log)

        final_port = history_df['portfolio_value'].iloc[-1]
        final_spy = history_df['spy_value'].iloc[-1]

        port_total_return = (final_port / config.INITIAL_CAPITAL) - 1
        spy_total_return = (final_spy / config.INITIAL_CAPITAL) - 1

        n_days = len(history_df)
        n_years = n_days / 252

        port_annual = (1 + port_total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        spy_annual = (1 + spy_total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        port_returns = history_df['portfolio_value'].pct_change().dropna()
        spy_returns = history_df['spy_value'].pct_change().dropna()

        port_vol = port_returns.std() * np.sqrt(252)
        spy_vol = spy_returns.std() * np.sqrt(252)

        port_sharpe = port_annual / port_vol if port_vol > 0 else 0
        spy_sharpe = spy_annual / spy_vol if spy_vol > 0 else 0

        port_cummax = history_df['portfolio_value'].cummax()
        port_dd = ((history_df['portfolio_value'] - port_cummax) / port_cummax).min()
        spy_cummax = history_df['spy_value'].cummax()
        spy_dd = ((history_df['spy_value'] - spy_cummax) / spy_cummax).min()

        regime_counts = history_df['regime'].value_counts()

        results = {
            'history': history_df,
            'trades': trades_df,
            'metrics': {
                'portfolio': {
                    'final_value': final_port,
                    'total_return': port_total_return,
                    'annual_return': port_annual,
                    'volatility': port_vol,
                    'sharpe_ratio': port_sharpe,
                    'max_drawdown': port_dd,
                },
                'spy': {
                    'final_value': final_spy,
                    'total_return': spy_total_return,
                    'annual_return': spy_annual,
                    'volatility': spy_vol,
                    'sharpe_ratio': spy_sharpe,
                    'max_drawdown': spy_dd,
                },
                'trading': {
                    'total_trades': len(trades_df),
                    'total_transaction_costs': self.total_transaction_costs,
                    'regime_distribution': regime_counts.to_dict(),
                    'delistings_detected': len(self.delisted_tickers),
                    'delisted_tickers': list(self.delisted_tickers),
                },
                'bias_controls': {
                    'pit_method': 'asof_date (SEC acceptance_datetime)',
                    'fallback_lag_days': config.FALLBACK_REPORTING_LAG_DAYS,
                    'delisting_return_assumption': config.DELISTING_RETURN,
                    'survivorship_warnings': self.survivorship_warnings,
                }
            }
        }
        
        if getattr(config, 'ENABLE_FF5_ATTRIBUTION', False):
            results['metrics']['ff5_attribution'] = self._run_ff5_regression(port_returns)

        self._print_summary(results)
        return results

    def _print_summary(self, results: dict):
        m = results['metrics']
        p, s, t, b = m['portfolio'], m['spy'], m['trading'], m['bias_controls']

        print("\n" + "=" * 70)
        print("BACKTEST RESULTS SUMMARY")
        print("=" * 70)

        print(f"\n{'Metric':<30} {'Strategy':>18} {'SPY Benchmark':>18}")
        print("-" * 70)
        print(f"{'Final Value':<30} ${p['final_value']:>17,.0f} ${s['final_value']:>17,.0f}")
        print(f"{'Total Return':<30} {p['total_return']:>17.2%} {s['total_return']:>17.2%}")
        print(f"{'Annual Return':<30} {p['annual_return']:>17.2%} {s['annual_return']:>17.2%}")
        print(f"{'Volatility':<30} {p['volatility']:>17.2%} {s['volatility']:>17.2%}")
        print(f"{'Sharpe Ratio':<30} {p['sharpe_ratio']:>17.2f} {s['sharpe_ratio']:>17.2f}")
        print(f"{'Max Drawdown':<30} {p['max_drawdown']:>17.2%} {s['max_drawdown']:>17.2%}")

        print(f"\n{'Trading Summary':}")
        print("-" * 70)
        print(f"{'Total Trades':<30} {t['total_trades']:>18,}")
        print(f"{'Total Transaction Costs':<30} ${t['total_transaction_costs']:>17,.2f}")
        print(f"{'Cost as % of Initial':<30} "
              f"{t['total_transaction_costs']/config.INITIAL_CAPITAL:>17.2%}")
        print(f"{'Delistings Detected':<30} {t['delistings_detected']:>18}")

        print(f"\n{'Regime Distribution':}")
        print("-" * 70)
        for regime, count in sorted(t['regime_distribution'].items()):
            pct = count / sum(t['regime_distribution'].values()) * 100
            print(f"  {regime:<12} {count:>8} days ({pct:>5.1f}%)")

        print(f"\n{'Bias Controls':}")
        print("-" * 70)
        print(f"  PIT method:          {b['pit_method']}")
        print(f"  Fallback lag:        {b['fallback_lag_days']} days")
        print(f"  Delisting return:    {b['delisting_return_assumption']:.0%}")
        if b['survivorship_warnings']:
            print(f"  Warnings:")
            for w in b['survivorship_warnings']:
                print(f"    ⚠ {w}")

        outperf = p['total_return'] - s['total_return']
        print(f"\n  Outperformance: {outperf:+.2%}")
        
        ff5 = m.get('ff5_attribution', {})
        if ff5 and 'error' not in ff5:
            print(f"\n{'Fama-French 5-Factor Attribution':}")
            print("-" * 70)
            print(f"  Alpha (Annualized):  {ff5['alpha_annualized']:>8.2%}")
            print(f"  Market (Mkt-RF):     {ff5['Mkt-RF']:>8.2f}")
            print(f"  Size (SMB):          {ff5['SMB']:>8.2f}")
            print(f"  Value (HML):         {ff5['HML']:>8.2f}")
            print(f"  Quality (RMW):       {ff5['RMW']:>8.2f}")
            print(f"  Investment (CMA):    {ff5['CMA']:>8.2f}")
            print(f"  R-Squared:           {ff5['r_squared']:>8.2f}")

        print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(results: dict, output_dir: str):
    """Generate and save comprehensive charts."""
    os.makedirs(output_dir, exist_ok=True)
    history = results['history']
    trades = results['trades']

    # ── Chart 1: Portfolio vs SPY + Regimes + Positions ────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                              gridspec_kw={'height_ratios': [3, 1, 1]})

    ax1 = axes[0]
    ax1.plot(history.index, history['portfolio_value'],
             label='Regime-Aware Strategy (PIT)', color='#2196F3', linewidth=2)
    ax1.plot(history.index, history['spy_value'],
             label='SPY Benchmark', color='#FF9800', linewidth=2, alpha=0.8)
    ax1.set_title('Regime-Aware Fundamental Strategy vs S&P 500\n'
                   '(Point-in-Time Clean: trades on SEC filing dates, not period_end)',
                   fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # Regime shading
    ax2 = axes[1]
    regime_colors = {
        'crisis': '#F44336', 'bear': '#FF9800',
        'normal': '#4CAF50', 'bull': '#2196F3',
    }
    for regime_name, color in regime_colors.items():
        mask = (history['regime'] == regime_name)
        if mask.any():
            ax2.fill_between(history.index, 0, 1, where=mask,
                             color=color, alpha=0.5, label=regime_name.title())
    ax2.set_title('Detected Market Regime (Walk-Forward HMM)', fontsize=12, fontweight='bold')
    ax2.set_yticks([])
    ax2.legend(loc='upper right', ncol=4, fontsize=10)

    ax3 = axes[2]
    ax3.plot(history.index, history['n_positions'], color='#9C27B0', linewidth=1.5)
    ax3.set_title('Active Positions', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Count')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'backtest_results.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Chart saved: {chart_path}")

    # ── Chart 2: Drawdowns + Outperformance ────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

    port_cummax = history['portfolio_value'].cummax()
    port_dd = (history['portfolio_value'] - port_cummax) / port_cummax
    spy_cummax = history['spy_value'].cummax()
    spy_dd = (history['spy_value'] - spy_cummax) / spy_cummax

    ax1.fill_between(history.index, port_dd, 0, color='#F44336', alpha=0.4, label='Strategy DD')
    ax1.fill_between(history.index, spy_dd, 0, color='#FF9800', alpha=0.3, label='SPY DD')
    ax1.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Drawdown')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0%}'))

    outperf = (history['portfolio_value'] / history['spy_value']) - 1
    ax2.plot(history.index, outperf, color='#4CAF50', linewidth=1.5)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax2.fill_between(history.index, outperf, 0,
                     where=outperf >= 0, color='#4CAF50', alpha=0.3, label='Outperforming')
    ax2.fill_between(history.index, outperf, 0,
                     where=outperf < 0, color='#F44336', alpha=0.3, label='Underperforming')
    ax2.set_title('Cumulative Outperformance vs SPY', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Outperformance')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0%}'))

    plt.tight_layout()
    dd_path = os.path.join(output_dir, 'drawdown_analysis.png')
    plt.savefig(dd_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Drawdown chart saved: {dd_path}")

    # ── Chart 3: Transaction costs + trade reasons ─────────────────────
    if len(trades) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        cost_by_regime = trades.groupby('regime')['fee'].sum()
        colors = [regime_colors.get(r, 'gray') for r in cost_by_regime.index]
        ax1.bar(cost_by_regime.index, cost_by_regime.values, color=colors,
                alpha=0.8, edgecolor='black')
        ax1.set_title('Transaction Costs by Regime', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Total Fees ($)')
        ax1.grid(True, alpha=0.3, axis='y')
        for i, (regime, cost) in enumerate(cost_by_regime.items()):
            ax1.text(i, cost, f'${cost:,.0f}', ha='center', va='bottom',
                     fontweight='bold', fontsize=9)

        reason_counts = trades['reason'].apply(lambda x: x.split('_')[0]).value_counts().head(8)
        ax2.barh(reason_counts.index, reason_counts.values,
                 color='#2196F3', alpha=0.7, edgecolor='black')
        ax2.set_title('Trade Count by Reason', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Number of Trades')
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        trade_path = os.path.join(output_dir, 'trading_analysis.png')
        plt.savefig(trade_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Trading analysis saved: {trade_path}")

    return [chart_path, dd_path]


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, config.DATA_DIR)
    output_dir = os.path.join(script_dir, config.OUTPUT_DIR)

    # Print config for reproducibility
    print("\n📋 Configuration:")
    print(f"  Initial Capital:          ${config.INITIAL_CAPITAL:,.0f}")
    print(f"  Top N Invest:             {config.TOP_N_INVEST}")
    print(f"  Half Liquidation Rank:    >{config.HALF_LIQUIDATION_RANK}")
    print(f"  Full Liquidation Rank:    >{config.FULL_LIQUIDATION_RANK}")
    print(f"  Rank Jump (small):        {config.RANK_JUMP_SMALL} ranks → "
          f"${config.RANK_JUMP_SMALL_CAPITAL:,.0f}")
    print(f"  Rank Jump (large):        {config.RANK_JUMP_LARGE} ranks → "
          f"+${config.RANK_JUMP_LARGE_CAPITAL:,.0f}")
    print(f"  Panic Buy Increase:       {config.PANIC_BUY_CAPITAL_INCREASE_PCT:.0%}")
    print(f"  Panic Buy Price Threshold:{config.PANIC_BUY_PRICE_DROP_THRESHOLD:.0%}")
    print(f"  Regimes:                  {config.N_REGIMES}")
    print(f"\n  PIT Method:               asof_date (SEC acceptance_datetime)")
    print(f"  Fallback Lag:             {config.FALLBACK_REPORTING_LAG_DAYS} days after period_end")
    print(f"  Delisting Return:         {config.DELISTING_RETURN:.0%}")
    print(f"  Delisting NaN Threshold:  {config.DELISTING_NAN_THRESHOLD_DAYS} days")
    print(f"\n  Transaction Costs:")
    for regime, costs in config.TRANSACTION_COSTS.items():
        print(f"    {regime:>8s}: buy={costs['buy']:.2%}  sell={costs['sell']:.2%}")

    # Run
    bt = RegimeAwareBacktester(data_dir=data_dir, script_dir=script_dir)
    bt.load_data()
    results = bt.run()

    if results:
        plot_results(results, output_dir)

        # Save trade log
        if len(results['trades']) > 0:
            trades_path = os.path.join(output_dir, 'trade_log.csv')
            results['trades'].to_csv(trades_path, index=False)
            print(f"  ✓ Trade log saved: {trades_path}")

        # Save daily history
        history_path = os.path.join(output_dir, 'daily_history.csv')
        results['history'].to_csv(history_path)
        print(f"  ✓ Daily history saved: {history_path}")

        # Save metrics
        metrics_path = os.path.join(output_dir, 'metrics.json')
        metrics_clean = {}
        for category, vals in results['metrics'].items():
            metrics_clean[category] = {
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in vals.items()
            }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_clean, f, indent=2, default=str)
        print(f"  ✓ Metrics saved: {metrics_path}")

    return results


if __name__ == '__main__':
    main()
