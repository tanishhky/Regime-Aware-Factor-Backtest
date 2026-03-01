# Regime-Aware Fundamental Backtest (v2 - PIT Clean)

A backtesting engine that combines quarterly fundamental rankings with real-time HMM regime detection — **free of lookahead bias and survivorship bias**.

## Bias Fixes (v1 → v2)

### 1. Lookahead Bias — ELIMINATED

**v1 problem:** Traded on Q1 fundamentals on January 1st, months before those financials were actually filed with the SEC.

**v2 fix:** Uses `asof_date` from the chrono-fund engine (= `acceptance_datetime.date()` from SEC EDGAR) as the **sole gate** for when data becomes actionable. Rebalancing triggers on actual filing arrival dates, not calendar quarter boundaries.

```
v1: period_end = 2024-03-31  →  trade on 2024-01-01  ← FUTURE DATA
v2: asof_date  = 2024-05-12  →  trade on 2024-05-12  ← CORRECT
```

If `asof_date` is unavailable (older parquet extracts), falls back to `filing_date` from `filings.parquet`, or `period_end + 60 days` as a conservative lag.

### 2. Survivorship Bias — MITIGATED

**v1 problem:** Universe built from modern-day parquet tickers + Yahoo Finance, silently excluding delisted/bankrupt companies.

**v2 mitigations:**
- **Delisting detection:** Tracks consecutive NaN price days per held ticker. After 20 days of missing prices → forces liquidation at `last_known_price × (1 - 30%)`.
- **Startup warnings:** Reports tickers with no price data and tickers with truncated price histories.
- **Configurable delisting return:** `DELISTING_RETURN = -0.30` (adjustable; some delistings are M&A at premiums).
- **Recommendation:** For production use, replace Yahoo Finance with a survivorship-bias-free PIT database (Norgate Data, Sharadar via Nasdaq Data Link).

### 3. Walk-Forward Regime Detection — CONFIRMED CLEAN

The `WalkForwardRegimeDetector` was already correct in v1: fits only on `spy.loc[:date]`, predicts on trailing indicators, no future leakage.

## Architecture

```
config.py                 ← All tunable parameters (thresholds, costs, ratios, bias controls)
rank_system_v2.py         ← PIT-gated EWMA scoring (uses asof_date, not period_end)
regime_detector.py        ← Walk-forward HMM (confirmed bias-free)
regime_aware_backtest.py  ← Main engine (filing-triggered rebalancing + delisting handling)
```

## How It Works

### Step 1: Point-in-Time Rankings
For each SEC filing arrival (`asof_date`), all companies with sufficient filing history are re-scored using EWMA-smoothed fundamentals (ROIC, FCF margin, D/E, revenue growth). Rankings only include data that was publicly available at that moment.

### Step 2: Filing-Triggered Rebalancing
Instead of rebalancing on arbitrary quarter boundaries, the system rebalances **when new filings arrive**. This naturally handles the SEC reporting lag (typically 40-60 days after period end for 10-Qs, 60-90 for 10-Ks).

### Step 3: Position Management

| Condition | Action |
|---|---|
| Rank stays ≤ 10 | Hold / rebalance to equal weight |
| Rank drops to 11–15 | Hold full position |
| Rank drops to 16–18 | Sell 50% |
| Rank drops below 18 | Full liquidation |
| Company jumps up 15+ ranks | Allocate $20K new capital |
| Company jumps up 25+ ranks | Allocate additional $50K |
| Crisis/bear + strong fundamentals + price drop ≥ 10% | Increase position by 20% (4:3:2:2:1 + 0.5×) |
| Price data missing for 20+ consecutive days | Force-liquidate at -30% (delisting assumed) |

### Step 4: Regime-Specific Transaction Costs

| Regime | Buy | Sell | Rationale |
|---|---|---|---|
| Crisis | 80bps | 100bps | Wide spreads, high market impact |
| Bear | 40bps | 50bps | Moderate stress |
| Normal | 15bps | 20bps | Standard conditions |
| Bull | 10bps | 12bps | Tight spreads |

### Step 5: SPY Benchmark
Same initial capital tracked as buy-and-hold for fair comparison.

## Configuration

All parameters in `config.py`, grouped by category:

**Capital & Sizing:** `INITIAL_CAPITAL`, `TOP_N_INVEST`
**Liquidation:** `HALF_LIQUIDATION_RANK`, `FULL_LIQUIDATION_RANK`, `HALF_LIQUIDATION_FRACTION`
**Rank Jumps:** `RANK_JUMP_SMALL/LARGE`, capital amounts
**Panic Buy:** `PANIC_BUY_CAPITAL_INCREASE_PCT`, `PANIC_BUY_PRICE_DROP_THRESHOLD`, ratio weights
**Transaction Costs:** `TRANSACTION_COSTS` dict (per regime × per direction)
**PIT Controls:** `FALLBACK_REPORTING_LAG_DAYS`, `MIN_FILING_HISTORY`
**Survivorship:** `DELISTING_RETURN`, `DELISTING_NAN_THRESHOLD_DAYS`

## Usage

```bash
pip install -r requirements.txt

# Place your chrono-fund parquet files in historical_data/:
#   statements_income.parquet   (must have: ticker, period_end, asof_date, ...)
#   statements_balance.parquet
#   statements_cashflow.parquet
#   filings.parquet             (optional but recommended for precise PIT)

python regime_aware_backtest.py
```

## Output

- `results/backtest_results.png` — Portfolio vs SPY with regime shading
- `results/drawdown_analysis.png` — Drawdown + outperformance
- `results/trading_analysis.png` — Cost breakdown by regime + trade reasons
- `results/trade_log.csv` — Every trade with date, price, fee, regime, reason
- `results/daily_history.csv` — Daily portfolio value, cash, positions, regime
- `results/metrics.json` — Summary metrics including bias control report
- `pit_rankings.json` — Full PIT rankings at each filing arrival date

## Known Limitations

1. **Yahoo Finance price data** still has survivorship bias for very old delistings. For research-grade results, use Norgate Data or Sharadar.
2. **Filing lag approximation:** If your parquet files lack `asof_date` and `filings.parquet`, the 60-day fallback lag is conservative but imprecise. Run chrono-fund with full EDGAR metadata for best results.
3. **No dividend reinvestment** in the current implementation. Both strategy and SPY benchmark are price-return only.
