"""
RANK SYSTEM V2 - Point-in-Time Fundamental Rankings
=====================================================
Uses `asof_date` from the chrono-fund engine as the sole gate for when
fundamental data becomes actionable. No lookahead bias.

Architecture:
  1. Load all statement data with asof_date (or fall back to period_end + lag)
  2. Pre-compute a timeline of "filing arrival events"
  3. At any date d, the PIT knowledge state = all filings with asof_date <= d
  4. EWMA scores use only PIT-available filings per ticker
  5. Rankings are recomputed whenever new filings arrive

Returns:
  - A dict mapping each rebalance_date → full ranked list of companies
  - Only filings that were actually public by that date contribute to scores
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import timedelta
from typing import Optional

import config


def _resolve_asof_date(row: pd.Series, filings_lookup: Optional[dict]) -> pd.Timestamp:
    """
    Determine when a filing's data became available to investors.

    Priority:
      1. asof_date column (from chrono-fund engine, = acceptance_datetime.date())
      2. filing_date from filings.parquet joined via accession
      3. period_end + FALLBACK_REPORTING_LAG_DAYS
    """
    # Priority 1: direct asof_date
    if 'asof_date' in row.index and pd.notna(row['asof_date']):
        return pd.Timestamp(row['asof_date'])

    # Priority 2: filing_date via accession lookup
    if filings_lookup and 'accession' in row.index and pd.notna(row.get('accession')):
        acc = row['accession']
        if acc in filings_lookup:
            return pd.Timestamp(filings_lookup[acc])

    # Priority 3: period_end + conservative lag
    if 'period_end' in row.index and pd.notna(row['period_end']):
        return pd.Timestamp(row['period_end']) + timedelta(days=config.FALLBACK_REPORTING_LAG_DAYS)

    return pd.NaT


def _load_filings_lookup(data_dir: str) -> Optional[dict]:
    """
    Load filings.parquet if it exists and build accession → filing_date lookup.
    Falls back gracefully if the file doesn't exist.
    """
    filings_path = os.path.join(data_dir, "filings.parquet")
    if not os.path.exists(filings_path):
        return None

    try:
        filings = pd.read_parquet(filings_path)
        # Prefer acceptance_datetime (most precise), fall back to filing_date
        if 'acceptance_datetime' in filings.columns:
            filings['_avail_date'] = pd.to_datetime(filings['acceptance_datetime']).dt.date
        elif 'filing_date' in filings.columns:
            filings['_avail_date'] = pd.to_datetime(filings['filing_date']).dt.date
        else:
            return None

        lookup = {}
        for _, row in filings.iterrows():
            if pd.notna(row.get('accession')) and pd.notna(row.get('_avail_date')):
                lookup[row['accession']] = row['_avail_date']
        return lookup
    except Exception as e:
        print(f"  ⚠ Could not load filings.parquet: {e}")
        return None


def _compute_raw_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute fundamental metrics from merged statement data."""
    df = df.copy()

    # Total Debt
    df['total_debt'] = df['short_term_debt'].fillna(0) + df['long_term_debt'].fillna(0)

    # Tax Rate and NOPAT
    tax_rate = df['income_tax_expense'] / df['pretax_income'].replace(0, np.nan)
    tax_rate = tax_rate.fillna(0.21)  # Default corporate tax rate
    tax_rate = tax_rate.clip(0, 1)    # Clamp between 0% and 100%
    nopat = df['ebit'] * (1 - tax_rate)

    # ROIC = NOPAT / (Debt + Equity)
    invested_capital = df['total_debt'] + df['total_equity'].replace(0, np.nan)
    df['roic'] = nopat / invested_capital

    # FCF Margin = (CFO - |CapEx|) / Revenue
    fcf = df['cfo'] - df['capex'].abs()
    df['fcf_margin'] = fcf / df['revenue'].replace(0, np.nan)

    # D/E = Total Debt / Equity
    df['d_e'] = df['total_debt'] / df['total_equity'].replace(0, np.nan)

    # Revenue Growth (YoY within each ticker, sorted by period_end)
    df = df.sort_values(['ticker', 'period_end'])
    df['rev_growth'] = df.groupby('ticker')['revenue'].pct_change()

    return df


def _compute_ewma_score(ticker_filings: pd.DataFrame) -> pd.DataFrame:
    """
    Compute EWMA stability score for a single ticker's filing history.
    Filings must be sorted by period_end (chronological order of fiscal periods).
    """
    tf = ticker_filings.copy()

    def calc_ewma(series):
        return series.ewm(span=4, adjust=False).mean()

    tf['roic_ewma'] = calc_ewma(tf['roic'])
    tf['fcf_margin_ewma'] = calc_ewma(tf['fcf_margin'])
    tf['d_e_ewma'] = calc_ewma(tf['d_e'])
    tf['rev_growth_ewma'] = calc_ewma(tf['rev_growth'])

    d_e_clamped = np.maximum(tf['d_e_ewma'], 0)
    tf['stability_score'] = (
        config.SCORE_WEIGHT_ROIC * tf['roic_ewma'] +
        config.SCORE_WEIGHT_FCF_MARGIN * tf['fcf_margin_ewma'] +
        config.SCORE_WEIGHT_DE_INVERSE * (1 / (d_e_clamped + 1)) +
        config.SCORE_WEIGHT_REV_GROWTH * tf['rev_growth_ewma']
    )

    return tf


def build_pit_rankings(data_dir: str) -> dict:
    """
    Build point-in-time rankings triggered by actual filing arrivals.

    Returns
    -------
    dict : {rebalance_date_str: [{'ticker', 'stability_score', 'rank'}, ...]}
           Keys are ISO date strings (YYYY-MM-DD) when new filings became public.
           Values are full rankings of all scorable tickers at that point in time.
    """
    print("  Loading fundamental data...")

    # Load statements
    income = pd.read_parquet(os.path.join(data_dir, "statements_income.parquet"))
    balance = pd.read_parquet(os.path.join(data_dir, "statements_balance.parquet"))
    cashflow = pd.read_parquet(os.path.join(data_dir, "statements_cashflow.parquet"))

    # Load filings lookup for asof_date resolution
    filings_lookup = _load_filings_lookup(data_dir)
    if filings_lookup:
        print(f"  ✓ Loaded {len(filings_lookup)} filing records for PIT resolution")
    else:
        print(f"  ⚠ No filings.parquet found; using period_end + {config.FALLBACK_REPORTING_LAG_DAYS}d lag")

    # Select needed columns (keep accession and asof_date if they exist)
    extra_cols = []
    for col in ['accession', 'asof_date']:
        if col in income.columns:
            extra_cols.append(col)

    inc_cols = ['ticker', 'period_end', 'revenue', 'net_income', 'ebit', 'pretax_income', 'income_tax_expense'] + extra_cols
    inc = income[[c for c in inc_cols if c in income.columns]].drop_duplicates(
        subset=['ticker', 'period_end']
    )

    bal_cols = ['ticker', 'period_end', 'total_assets', 'total_liabilities',
                'total_equity', 'short_term_debt', 'long_term_debt']
    bal_extra = [c for c in ['accession', 'asof_date'] if c in balance.columns]
    bal = balance[[c for c in bal_cols + bal_extra if c in balance.columns]].drop_duplicates(
        subset=['ticker', 'period_end']
    )

    cf_cols = ['ticker', 'period_end', 'cfo', 'capex']
    cf_extra = [c for c in ['accession', 'asof_date'] if c in cashflow.columns]
    cf = cashflow[[c for c in cf_cols + cf_extra if c in cashflow.columns]].drop_duplicates(
        subset=['ticker', 'period_end']
    )

    # Merge datasets
    df = inc.merge(bal, on=['ticker', 'period_end'], how='inner', suffixes=('', '_bal'))
    df = df.merge(cf, on=['ticker', 'period_end'], how='inner', suffixes=('', '_cf'))

    df['period_end'] = pd.to_datetime(df['period_end'])

    # ── Resolve asof_date for every row ───────────────────────────────
    # Use the BEST available asof_date across the three joined tables
    if 'asof_date' not in df.columns:
        # Check if any of the suffixed versions exist
        for candidate in ['asof_date_bal', 'asof_date_cf', 'asof_date']:
            if candidate in df.columns:
                df['asof_date'] = df[candidate]
                break

    print("  Resolving point-in-time availability dates...")
    df['available_date'] = df.apply(
        lambda row: _resolve_asof_date(row, filings_lookup), axis=1
    )

    # Drop rows with no resolvable availability date
    n_before = len(df)
    df = df.dropna(subset=['available_date'])
    n_after = len(df)
    if n_before != n_after:
        print(f"  ⚠ Dropped {n_before - n_after} rows with unresolvable availability dates")

    df['available_date'] = pd.to_datetime(df['available_date'])
    df = df.sort_values(['ticker', 'period_end'])

    # ── Compute raw metrics ───────────────────────────────────────────
    df = _compute_raw_metrics(df)

    # ── Build filing arrival timeline ─────────────────────────────────
    # Get all unique dates when new filings became available
    filing_dates = sorted(df['available_date'].unique())
    print(f"  ✓ {len(filing_dates)} distinct filing arrival dates")
    print(f"  ✓ Range: {filing_dates[0].date()} → {filing_dates[-1].date()}")
    print(f"  ✓ {df['ticker'].nunique()} unique tickers in dataset")

    # ── For each filing arrival date, compute PIT rankings ────────────
    results = {}
    prev_rebalance_scores = {}  # For detecting "no change" dates

    for i, avail_date in enumerate(filing_dates):
        # PIT filter: only filings that were public by this date
        pit_data = df[df['available_date'] <= avail_date].copy()

        # For each ticker, keep only the latest filing (by period_end)
        # that was available by this date
        latest_per_ticker = (
            pit_data
            .sort_values('period_end')
            .groupby('ticker')
            .tail(1)
            .set_index('ticker')
        )

        # But for EWMA, we need the full PIT history per ticker
        # Compute EWMA scores for each ticker using their full available history
        # But for EWMA, we need the full PIT history per ticker
        # Compute EWMA scores for each ticker using their full available history
        all_scores = {}
        for ticker, group in pit_data.groupby('ticker'):
            if len(group) < config.MIN_FILING_HISTORY:
                continue  # Not enough history to compute growth + EWMA
            scored = _compute_ewma_score(group)
            if scored['stability_score'].isna().all():
                continue
            # Use the latest score
            latest_score = scored['stability_score'].iloc[-1]
            if pd.notna(latest_score):
                all_scores[ticker] = latest_score

        if not all_scores:
            continue

        # Check if scores actually changed from last rebalance
        if all_scores == prev_rebalance_scores:
            continue  # No new information → skip

        # Rank all tickers
        sorted_tickers = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        ranked = []
        for rank_idx, (ticker, score) in enumerate(sorted_tickers, start=1):
            ranked.append({
                'ticker': ticker,
                'stability_score': float(score),
                'rank': rank_idx,
            })

        date_key = str(avail_date.date())
        results[date_key] = ranked
        prev_rebalance_scores = all_scores

    # Deduplicate by keeping only dates where rankings actually changed
    print(f"  ✓ {len(results)} effective rebalance dates (where rankings changed)")

    return results


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, config.DATA_DIR)

    print(f"Building PIT rankings from {data_path} ...")
    res = build_pit_rankings(data_path)

    output_file = os.path.join(script_dir, config.RANKINGS_CACHE)
    print(f"Saving to {output_file} ...")
    with open(output_file, 'w') as f:
        json.dump(res, f, indent=2)

    # Summary
    dates = sorted(res.keys())
    print(f"\nTotal rebalance events: {len(dates)}")
    for d in dates[-3:]:
        n = len(res[d])
        top3 = [(e['ticker'], e['rank'], f"{e['stability_score']:.4f}") for e in res[d][:3]]
        print(f"  {d}: {n} companies | Top 3: {top3}")
    print("Done!")
