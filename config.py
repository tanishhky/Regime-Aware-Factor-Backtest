"""
REGIME-AWARE FUNDAMENTAL BACKTEST - CONFIGURATION
===================================================
All tunable parameters live here. Tweak and experiment freely.

Version 2: Point-in-time clean. Uses asof_date from chrono-fund engine
           as the sole data availability gate. No lookahead bias.
"""

# =============================================================================
# 1. CAPITAL & PORTFOLIO SIZING
# =============================================================================
INITIAL_CAPITAL = 1_000_000          # Starting capital ($)

# How many top-ranked companies to initially invest in
TOP_N_INVEST = 10

# =============================================================================
# 2. LIQUIDATION THRESHOLDS (rank-based)
# =============================================================================
HALF_LIQUIDATION_RANK = 15           # Sell 50% if rank > this
FULL_LIQUIDATION_RANK = 18           # Sell 100% if rank > this
HALF_LIQUIDATION_FRACTION = 0.50     # Fraction to sell at half-liq threshold

# =============================================================================
# 3. NEW CAPITAL ALLOCATION (rank-jump rewards)
# =============================================================================
RANK_JUMP_SMALL = 15                 # Minimum rank improvement for small allocation
RANK_JUMP_SMALL_CAPITAL = 20_000     # Dollars allocated for 15-rank jump

RANK_JUMP_LARGE = 25                 # Minimum rank improvement for large allocation
RANK_JUMP_LARGE_CAPITAL = 50_000     # ADDITIONAL dollars for 25-rank jump

# =============================================================================
# 4. PANIC-BUY / DIP-BUYING LOGIC
# =============================================================================
PANIC_BUY_CAPITAL_INCREASE_PCT = 0.20  # Increase invested capital by 20%
PANIC_BUY_PRICE_DROP_THRESHOLD = -0.10  # 10% drawdown from 63-day peak

# Distribution ratio for panic-buy capital among top candidates.
# Sorted by: retained/improved rank + biggest price drop.
PANIC_BUY_RATIOS_TOP5 = [4, 3, 2, 2, 1]    # Weights for positions 1-5
PANIC_BUY_RATIO_REST = 0.5                   # Weight for positions 6-10

# =============================================================================
# 5. REGIME-SPECIFIC TRANSACTION COSTS (realistic, per-direction)
# =============================================================================
TRANSACTION_COSTS = {
    'crisis': {'buy': 0.0080, 'sell': 0.0100},   # 80bps / 100bps
    'bear':   {'buy': 0.0040, 'sell': 0.0050},   # 40bps / 50bps
    'normal': {'buy': 0.0015, 'sell': 0.0020},   # 15bps / 20bps
    'bull':   {'buy': 0.0010, 'sell': 0.0012},   # 10bps / 12bps
}

REGIME_LABEL_MAP = {}  # Auto-populated at runtime

# =============================================================================
# 6. REGIME DETECTION PARAMETERS
# =============================================================================
N_REGIMES = 4
SHORT_TERM_TREND = 21                # ~1 month
LONG_TERM_TREND = 63                 # ~1 quarter

# =============================================================================
# 7. BACKTEST TIMING
# =============================================================================
BACKTEST_START = '2006-01-01'
BACKTEST_END = '2026-03-01'
PRICE_DATA_START = '2005-01-01'      # Extra lookback for regime training

REGIME_TRAIN_WINDOW = 756            # ~3 years rolling window

# =============================================================================
# 8. POINT-IN-TIME DATA HANDLING
# =============================================================================
# The chrono-fund engine provides `asof_date` in every statement row,
# derived from SEC `acceptance_datetime`. This is the ONLY gate for when
# fundamental data becomes actionable.
#
# If `asof_date` is missing (older extracts), fall back to:
#   filing_date from filings.parquet, or period_end + this lag.
FALLBACK_REPORTING_LAG_DAYS = 60     # Conservative: ~2 months after period_end

# Minimum filings needed before a ticker is rankable
MIN_FILING_HISTORY = 2               # Need ≥2 to compute growth rates

# =============================================================================
# 9. SURVIVORSHIP BIAS CONTROLS
# =============================================================================
# If a held ticker's price data disappears, assume delisting.
DELISTING_RETURN = -0.30             # Assume 30% loss on delisting
                                     # (conservative; some delistings are M&A)

# Consecutive NaN trading days before treating as delisted
DELISTING_NAN_THRESHOLD_DAYS = 20    # ~1 month

WARN_MISSING_PRICE_TICKERS = True

# =============================================================================
# 10. DATA PATHS
# =============================================================================
DATA_DIR = "historical_data"         # Directory with parquet fundamentals
PRICE_CACHE = "price_data.parquet"   # Cached yfinance prices
RANKINGS_CACHE = "all_rankings.json" # Full rankings output
OUTPUT_DIR = "results"               # Output directory for charts/reports

# =============================================================================
# 11. STRATEGY OVERLAYS AND REPORTING
# =============================================================================
ENABLE_PANIC_BUY = True              # Discretionary overlay to buy dips in crisis
ENABLE_FF5_ATTRIBUTION = True        # Regress daily returns against Fama-French 5

# =============================================================================
# 12. SCORING FORMULA WEIGHTS
# =============================================================================
# The stability score uses EBIT-based ROIC (not NOPAT) to measure pure
# operating efficiency, while capital structure risk is captured separately
# by the bounded inverse D/E transform. This separation of concerns avoids
# double-penalizing leveraged firms.
SCORE_WEIGHT_ROIC = 0.30             # EBIT / Invested Capital
SCORE_WEIGHT_FCF_MARGIN = 0.25
SCORE_WEIGHT_DE_INVERSE = 0.25       # Uses 1/(D/E + 1) ∈ [0,1], deliberately outsized
SCORE_WEIGHT_REV_GROWTH = 0.20

# =============================================================================
# 13. FEE STRUCTURE (2-and-25 hedge fund standard)
# =============================================================================
# Management fee: charged monthly as (annual_rate / 12) × AUM
# Performance fee: charged quarterly as rate × max(0, NAV - HWM)
# Set both to 0.0 to run gross-of-fee backtest (no code changes needed).
MANAGEMENT_FEE_ANNUAL = 0.02         # 2% per annum, deducted monthly
PERFORMANCE_FEE_RATE = 0.25          # 25% of quarterly profits above HWM
