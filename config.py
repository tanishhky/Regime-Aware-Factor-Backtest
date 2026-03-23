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
INITIAL_CAPITAL = 500_000_000          # Starting capital ($)

# Capital injections: every INJECTION_INTERVAL_YEARS, add INJECTION_PCT of
# current INITIAL_CAPITAL. Set INJECTION_PCT to 0.0 to disable.
INJECTION_INTERVAL_YEARS = 2          # Inject new capital every N years
INJECTION_PCT = 0.0                   # 0% = disabled (set to 0.10 for 10% injections)

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
RANK_JUMP_SMALL_CAPITAL = 200_000     # Dollars allocated for 15-rank jump

RANK_JUMP_LARGE = 25                 # Minimum rank improvement for large allocation
RANK_JUMP_LARGE_CAPITAL = 500_000     # ADDITIONAL dollars for 25-rank jump

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
SKIP_RANKING_REBUILD = True          # Set True to load cached rankings (fast rerun)

# =============================================================================
# 12. SCORING FORMULA WEIGHTS
# =============================================================================
# The stability score intentionally overweights leverage aversion via the
# bounded inverse D/E transform. This is the core investment thesis, not
# a normalization choice. See README for rationale.
SCORE_WEIGHT_ROIC = 0.30
SCORE_WEIGHT_FCF_MARGIN = 0.25
SCORE_WEIGHT_DE_INVERSE = 0.25       # Uses 1/(D/E + 1) ∈ [0,1], deliberately outsized
SCORE_WEIGHT_REV_GROWTH = 0.20

# =============================================================================
# 13. FEE STRUCTURE (2-and-25 hedge fund standard)
# =============================================================================
# Management fee: charged monthly as (annual_rate / 12) × AUM
# Performance fee: charged quarterly as rate × max(0, NAV - HWM)
# Set both to 0.0 to run gross-of-fee backtest (no code changes needed).
MANAGEMENT_FEE_ANNUAL = 0.02          # Set to 0.02 for 2% per annum
PERFORMANCE_FEE_RATE = 0.15           # Set to 0.20 for 20% of quarterly profits above HWM

# =============================================================================
# 14. OPTIONS HEDGE OVERLAY (Regime-Conditional Put Spreads via Synthetic BS)
# =============================================================================
# Master switch — set False for A/B comparison (hedged vs unhedged)
ENABLE_OPTIONS_HEDGE = False

# Pricing: VIX-based synthetic Black-Scholes (no full OptionMetrics needed)
VIX_RF_CACHE = 'vix_rf_cache.parquet'

# Optional WRDS calibration (uses optionmsamp_us sample to fit skew model)
WRDS_USERNAME = None                  # Set to your username to enable
USE_WRDS_CALIBRATION = False          # Set True to calibrate from sample

# Hedge sizing (beta-adjusted)
HEDGE_NOTIONAL_FRACTION = 1.0         # 1.0 = full beta-adjusted exposure
HEDGE_MARKET_BETA = 0.77              # From FF5 Mkt-RF loading

# Put spread strikes
HEDGE_LONG_PUT_OTM = 0.05            # 5% OTM  → protection floor
HEDGE_SHORT_PUT_OTM = 0.15           # 15% OTM → spread lower leg
# Protection band: covers -5% to -15% SPY drawdown

# Tenor
HEDGE_TARGET_DTE = 30                 # ~1 month rolling puts
HEDGE_MIN_DTE = 5                     # Roll at ≤5 DTE remaining

# Regime rules: BUY when vol is cheap, SELL when vol is expensive
HEDGE_ENTRY_REGIMES = ('normal', 'bull')
HEDGE_EXIT_REGIMES = ('crisis', 'bear')

# HMM transition early-entry (optional)
HEDGE_TRANSITION_PROB_THRESHOLD = 0.30
HEDGE_TRANSITION_SCALE_UP = 1.5

# Synthetic pricing model parameters
# Skew: IV(K) = VIX/100 * (1 + slope * |moneyness|)
HEDGE_SKEW_SLOPE = 2.5               # Literature default; override with WRDS calibration
# Spread: half_spread = mid * (base + vix_scaling * max(0, VIX-15))
HEDGE_SPREAD_BASE_PCT = 0.03
HEDGE_SPREAD_VIX_SCALING = 0.002

# Execution slippage on top of synthetic spread
HEDGE_SPREAD_PENALTY_NORMAL = 0.10    # 10% of bid-ask
HEDGE_SPREAD_PENALTY_CRISIS = 0.30    # 30% of bid-ask (wider in stress)

# =============================================================================
# 15. ADAPTIVE OPTIMIZATION (Walk-Forward Parameter Re-Optimization)
# =============================================================================
ENABLE_ADAPTIVE_WEIGHTS = False        # Walk-forward scoring weight optimization
ADAPTIVE_REFIT_INTERVAL = 252         # Re-optimize annually (~252 trading days)
ADAPTIVE_LOOKBACK = 756               # 3-year lookback for optimization
ADAPTIVE_WEIGHT_BOUNDS = (0.10, 0.40) # Min/max per scoring factor weight
ADAPTIVE_MAX_WEIGHT_CHANGE = 0.10     # Max change per factor per refit cycle
ENABLE_EARLY_REFIT = False             # Trigger refit on staleness detection

# Regime-adaptive liquidation thresholds
ENABLE_ADAPTIVE_LIQUIDATION = False    # crisis/bear→12/15, normal→15/18, bull→18/22

# Adaptive TOP_N based on score dispersion
ENABLE_ADAPTIVE_TOP_N = False
ADAPTIVE_TOP_N_MIN = 8               # Concentrate when dispersion high
ADAPTIVE_TOP_N_MAX = 15              # Diversify when dispersion low
ADAPTIVE_DISPERSION_HIGH = 0.05      # σ of top-20 scores above this → concentrate
ADAPTIVE_DISPERSION_LOW = 0.02       # σ below this → diversify

# =============================================================================
# 16. DRAWDOWN MANAGEMENT
# =============================================================================
ENABLE_DRAWDOWN_SCALING = False
DD_START_THRESHOLD = -0.05            # Start reducing exposure at -5% from HWM
DD_FULL_THRESHOLD = -0.20             # Maximum reduction at -20%
DD_MIN_SCALAR = 0.25                  # Never below 25% exposure
DD_RECOVERY_SPEED = 0.5              # Blending speed for recovery (0=frozen, 1=instant)
ENABLE_REGIME_DD_PARAMS = False        # Use regime-specific DD thresholds

# =============================================================================
# 17. ALPHA FADE (Strategy → Passive Blend)
# =============================================================================
ENABLE_ALPHA_FADE = False
ALPHA_FADE_WINDOW = 252               # 12-month rolling alpha
ALPHA_FADE_TRIGGER_MONTHS = 3         # Consecutive negative months to trigger fade
ALPHA_FADE_MIN_STRATEGY_WEIGHT = 0.30 # Floor: 30% strategy, 70% benchmark

# =============================================================================
# 18. VOLATILITY TARGETING (Option 2)
# =============================================================================
ENABLE_VOL_TARGETING = False
VOL_TARGET = 0.15                     # Target annualized volatility
VOL_LOOKBACK = 20                     # 20-day trailing standard deviation
VOL_MIN_LEVERAGE = 0.3                # Minimum exposure
VOL_MAX_LEVERAGE = 1.5                # Maximum leverage allowed

# =============================================================================
# 19. FACTOR MOMENTUM (Option 3)
# =============================================================================
ENABLE_FACTOR_MOMENTUM = False
FACTOR_MOMENTUM_LOOKBACK = 252        # 12 months for long-short performance eval

# =============================================================================
# 20. POSITION TRAILING STOPS (Option 4)
# =============================================================================
ENABLE_TRAILING_STOPS = False
# Liquidate if price drops below peak * (1 - threshold)
TRAILING_STOP_THRESHOLDS = {
    'bull': 0.25,
    'normal': 0.20,
    'bear': 0.15,
    'crisis': 0.12
}

# =============================================================================
# 21. BENCHMARK BLEND (Option 5)
# =============================================================================
ENABLE_BENCHMARK_BLEND = True
BENCHMARK_BLEND_WEIGHT = 0.40         # 60% strategy, 40% SPY tracker
