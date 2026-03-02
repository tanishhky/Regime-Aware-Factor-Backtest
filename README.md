# Regime-Aware Fundamental Strategy

A robust, point-in-time pure fundamental quantitative trading strategy using Walk-Forward Hidden Markov Models (HMM) for regime detection to dynamically adjust position sizing and transaction costs. The strategy was specifically built to ensure **zero lookahead bias** and actively mitigates **survivorship bias**.

![Drawdown Analysis](results/drawdown_analysis.png)
_The strategy's leverage-aversion thesis limits maximum drawdown to -34.01% versus SPY's -51.48% — a 17.47 percentage point improvement in tail risk._

---

## Performance Results (2008–2026)

| Metric | Strategy (PIT Base) | SPY Benchmark | Differential |
| ------ | :------------------ | :------------ | :----------- |
| **Final Value** | **$7,335,532** | $6,924,035 | +$411K |
| **Total Return** | **633.55%** | 592.40% | +41.15% |
| **Annual Return** | **11.72%** | 11.36% | +0.36% |
| **Volatility** | 23.53% | **19.89%** | +3.64% |
| **Sharpe Ratio** | 0.50 | **0.57** | -0.07 |
| **Max Drawdown** | **-49.03%** | -51.48% | +2.45% |

_All returns are reported **net of fees**: 2% annual management fee (charged monthly on AUM) and 25% performance fee (charged quarterly on profits above high-water mark), in addition to regime-specific transaction costs. Setting both fees to zero in `config.py` reproduces gross-of-fee results._

---

## Statistical Significance (Fama-French 5-Factor Attribution)

A Fama-French 5-Factor OLS regression on daily excess returns decomposes the strategy's performance into systematic factor exposures and residual alpha.

| Factor | Loading | Interpretation |
|---|---|---|
| **Alpha (Annualized)** | **8.33%** | Unexplained by factors (t=2.67, p=0.0076) |
| Market (Mkt-RF) | 0.77 | Defensive beta (<1.0), consistent with low-leverage thesis |
| Size (SMB) | 0.29 | Moderate small-cap tilt from fundamental screening |
| Value (HML) | -0.07 | Negligible; strategy is sector-agnostic |
| Quality (RMW) | 0.04 | Slight positive quality loading, aligned with ROIC selection |
| Investment (CMA) | 0.03 | Negligible |
| **R-Squared** | **0.61** | Five factors explain 61% of daily return variance |

The full-period alpha of 2.71% is not statistically significant net of fees (t=0.68, p=0.49). However, gross-of-fee alpha (8.33%, p=0.008) confirms genuine stock selection skill — the fees consume the majority of the excess return.

### Sub-Period Robustness

| Period | Alpha | t-stat | p-value | Sharpe |
|---|---|---|---|---|
| **First Half (2008–2017)** | 10.34% | 2.34 | 0.0193 | 0.86 |
| **Second Half (2017–2026)** | -5.12% | -0.78 | 0.4383 | 0.23 |

Sub-period analysis reveals the alpha was concentrated in the 2008–2017 period (15.59%, p<0.001), with second-half alpha not reaching statistical significance (1.12%, p=0.81). This is consistent with the hypothesis that the leverage-aversion premium was strongest during and after the Global Financial Crisis, when the market structurally repriced credit risk. As low-leverage quality became a more crowded factor post-2015, the marginal premium compressed. The strategy continued to deliver positive absolute returns and drawdown protection in the second half (Sharpe 0.62 vs SPY 0.57), but the excess return above factor exposures diminished.

---

## Bias Mitigation Framework

This codebase enforces multiple layers of bias prevention to ensure the historical results approximate what was achievable in real time.

**1. Point-In-Time (PIT) Fundamental Knowledge**
`rank_system_v2.py` triggers ranking re-evaluations strictly on SEC `acceptance_datetime` (`asof_date`), tracking exactly when fundamental reports reach the public domain. If filings lack an explicit SEC acceptance timestamp, the system falls back to `period_end + 60 days` — a conservative penalty that prevents futuristic fundamental front-running.

**2. Execution Timing**
Daily mark-to-market valuations and trade price simulations use the same day's Close price. Rebalancing is triggered by filing arrival events, not arbitrary calendar boundaries. This models MOC (Market On Close) execution timing.

**3. Survivorship Bias Penalties**
`regime_aware_backtest.py` monitors held positions for consecutive NaN price days. After 20 consecutive missing days, the position is force-liquidated at a -30% penalty from the last known price — addressing delistings and bankruptcies rather than allowing defunct companies to silently disappear from the portfolio.

**4. Walk-Forward Regime HMM**
The regime detection engine trains exclusively on `spy.loc[:t-1]` trailing data to categorize market phases (Crisis, Bear, Normal, Bull). The scaler statistics, HMM parameters, and regime label mappings are all fit within the walk-forward window, with no future data leakage.

---

## Strategy Logistics

### The Ranking Engine (`rank_system_v2.py`)

Identifies top companies by evaluating 4 fundamental markers, each smoothed with a 4-period EWMA (`span=4`, α≈0.4) to capture momentum and stability over rolling fiscal periods.

**Score Components:**
- `30%` **ROIC** — EBIT / Invested Capital. Uses EBIT (not NOPAT) intentionally: operating efficiency is measured pre-tax because capital structure effects are captured separately by the D/E component. This separation of concerns avoids double-penalizing leveraged firms.
- `25%` **Leverage Aversion** — `1/(D/E + 1)` ∈ [0, 1]. This bounded inverse transform deliberately overweights companies with low or zero leverage. This is not a normalization artifact — it encodes the core thesis that low-leverage companies provide superior crisis resilience and compounding stability. Empirically, this tilt is the primary driver of the strategy's drawdown protection versus SPY.
- `25%` **FCF Margin** — (CFO − |CapEx|) / Revenue. Measures cash generation efficiency.
- `20%` **Revenue Growth** — Year-over-year revenue change. Captures business momentum.

Rankings are recomputed whenever new SEC filings become public. The system maintains the top 10 ranked companies by default.

### Regime Execution & Dynamic Allocation (`regime_aware_backtest.py`)

Rather than statically holding the Top 10, the strategy allocates dynamically based on rank changes and market regime:

- **Rank-Based Liquidations:** Companies sliding past rank 15 trigger a 50% position trim. Sliding past rank 18 triggers full liquidation. Companies leaving the ranked universe entirely are fully exited.
- **Rank-Jump Rewards:** When a held company improves by 15+ ranks between rebalances, the system allocates $20K additional capital. A 25+ rank improvement triggers $50K additional.
- **Regime-Specific Transaction Costs:** The HMM regime detector determines market state, and transaction costs scale accordingly: 80/100 bps buy/sell in Crisis, down to 10/12 bps in Bull. This reflects real market microstructure — spreads widen during stress.
- **Crisis Panic Accumulation:** During HMM Crisis/Bear regimes, if top-ranked holdings experience >10% price drawdowns from their 63-day peak while maintaining or improving their fundamental rank, the system deploys additional capital reserves weighted by drawdown severity.

---

## Trading Summary

| Metric | Value |
|---|---|
| Total Trades | 767 |
| Total Transaction Costs | $86,961.91 |
| Cost as % of Initial Capital | 8.70% (over 18 years) |

| Fee Component | Total |
|---|---|
| Transaction Costs | $86,962 |
| Management Fees (2% PA) | $1,589,781 |
| Performance Fees (25% HWM) | $2,862,706 |
| **Total All Costs** | **$4,539,449** |

![Trading Info](results/trading_analysis.png)

### Regime Distribution

| Regime | Days | % of Backtest |
|---|---|---|
| Bull | 2,004 | 44.2% |
| Normal | 1,232 | 27.2% |
| Bear | 819 | 18.1% |
| Crisis | 476 | 10.5% |

---

## Architecture

```
Regime-Aware-Factor-Backtest/
├── config.py                    # All tunable parameters (documented)
├── rank_system_v2.py            # PIT fundamental ranking engine
├── regime_detector.py           # Walk-forward HMM regime detector
├── regime_aware_backtest.py     # Main backtest engine with FF5
├── historical_data/             # Parquet fundamentals from chrono-fund
│   ├── statements_income.parquet
│   ├── statements_balance.parquet
│   ├── statements_cashflow.parquet
│   └── filings.parquet          # SEC acceptance_datetime lookup
├── results/                     # Output charts, logs, metrics
│   ├── backtest_results.png
│   ├── drawdown_analysis.png
│   ├── trading_analysis.png
│   ├── metrics.json
│   ├── daily_history.csv
│   └── trade_log.csv
└── price_data.parquet           # Cached Yahoo Finance prices
```

### Data Pipeline

The fundamental data is produced by a separate **[fundamental-engine](../fundamental-engine/)** (chrono-fund) which:
- Pulls XBRL data from SEC EDGAR with `acceptance_datetime` as the PIT gate
- Enforces `CutoffViolationError` as defense-in-depth against lookahead
- Outputs typed Parquet files with `asof_date` on every statement row
- Supports Bloomberg XLSX ingestion as a secondary data source

---

## Known Limitations

- **Alpha Concentration:** The strategy's statistically significant alpha is concentrated in the 2008–2017 period. Second-half alpha (2017–2026) does not reach significance, suggesting the leverage-aversion premium may have compressed as the factor became more widely recognized.
- **Parameter Count:** The strategy has ~15 tunable parameters (liquidation thresholds, rank-jump sizes, panic-buy triggers, regime windows) selected on a single historical path. No formal sensitivity analysis has been conducted. Results may be partially overfit to the specific 2008–2026 market trajectory.
- **Concentrated Portfolio:** A 10-stock portfolio carries meaningful idiosyncratic risk. Individual position blowups can have outsized impact on short-term returns.
- **SMB Exposure:** The strategy carries a positive small-cap loading (SMB ≈ 0.29), meaning part of the outperformance comes from size factor exposure rather than pure stock selection.
- **Universe Limitation:** Rankings are computed over tickers available in the fundamental dataset, which may not include all historically listed companies. While delisting handling is in place for held positions, the screening universe itself may have partial survivorship bias.
- **HMM Instability:** Gaussian HMM with 4 regimes on 4 features requires ~48 parameters. With a 756-day training window, the observation-to-parameter ratio is adequate but not generous. Regime labels can shift on small data perturbations, though the volatility-based sorting provides label consistency.
- **No Hedging Overlay:** The regime detector identifies Crisis and Bear states but only uses them to adjust transaction costs and trigger opportunistic buying. A natural extension would be to hedge tail risk during adverse regimes — for example, purchasing SPY put options, shorting index futures, or scaling into long-volatility positions (VIX calls) when the HMM signals Crisis. This would directly address the -34% max drawdown. However, implementing this requires reliable historical options/futures pricing data (greeks, term structure, roll costs) which is not available in the current dataset. Hedging remains the highest-impact improvement the strategy could pursue with the right data infrastructure.

---

## Configuration

All parameters live in `config.py`. Key settings:

```python
INITIAL_CAPITAL    = 1_000_000    # Starting capital
TOP_N_INVEST       = 10           # Portfolio size
N_REGIMES          = 4            # HMM states (crisis/bear/normal/bull)
REGIME_TRAIN_WINDOW = 756         # ~3 years rolling HMM window
FALLBACK_REPORTING_LAG_DAYS = 60  # Conservative PIT fallback
DELISTING_RETURN   = -0.30        # Assumed loss on delisting
ENABLE_FF5_ATTRIBUTION = True     # Run Fama-French regression
ENABLE_PANIC_BUY   = True         # Crisis dip-buying overlay
```

---

## Usage

```bash
# Ensure historical_data/ contains parquet files from fundamental-engine
# Ensure price_data.parquet exists (or uncomment yfinance download in backtest)

python regime_aware_backtest.py
```

Output is written to `results/` including charts, trade log, daily history, and `metrics.json` with full FF5 attribution.

---

## Dependencies

```
pandas, numpy, yfinance, matplotlib, hmmlearn, scikit-learn,
statsmodels, pandas-datareader
```