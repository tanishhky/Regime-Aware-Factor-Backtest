# Alternative Risk Management Evaluation

We evaluated the four alternative risk management strategies proposed instead of the complex adaptive engine. The goal was to find a robust, parameter-lite way to handle the 2021-2022 factor decay drawdown.

## Experiment Results

| Experiment | Final Value | Ann Ret | Volatility | Sharpe | Max Drawdown | Trades |
|---|---|---|---|---|---|---|
| **Baseline (Static)** | $4,884M | 13.51% | 22.37% | 0.60 | -43.49% | 230 |
| **Option 2: Volatility Targeting** | $2,456M | 9.26% | 16.52% | 0.56 | -33.48% | 230 |
| **Option 4: Trailing Stops** | $4,671M | 13.23% | 21.46% | 0.62 | -40.41% | 1142 |
| **Option 5: Benchmark Blend (60/40)** | $4,315M | 12.73% | 20.82% | 0.61 | -39.74% | 230 |
| **Recommended: Vol Target + Stops** | $2,042M | 8.14% | 16.05% | 0.51 | -33.54% | 1142 |

*(Note: Option 3 Factor Momentum was excluded because the current pipeline's pre-computed rankings would require a disruptive architectural rewrite to dynamically re-weight in an ongoing backtest.)*

## Diagnosis & Findings

The results above strongly validate your diagnosis: **the core issue is factor death, not parameter staleness.**

1. **Volatility Targeting (Option 2)** successfully suppressed the max drawdown from -43% to -33%, but mechanically de-leveraging the portfolio when the primary factor goes flat costs massive long-term compounding (Final Value halved to $2.4B).
2. **Trailing Stops (Option 4)** marginally improved Sharpe (0.62 vs 0.60) by cutting the bleeding positions, but multiplied the trading frequency by 5x (1142 trades vs 230), introducing significant turnover drag to achieve a mere 3% drawdown reduction.
3. **The Recommended Combo (Vol Target + Stops)** performed worst overall (Sharpe 0.51). The reason is a "double penalty" during stress: trailing stops sell positions at local bottoms right as the vol-targeter cuts portfolio exposure, causing the strategy to entirely miss the V-shaped recoveries.
4. **The Benchmark Blend (Option 5)** proved to be the most pragmatic and mechanically robust solution. By keeping a constant 60/40 blend with SPY, it caps the maximum theoretical underperformance, drops the drawdown from -43% to -39%, keeps the Final Value beautifully high at $4.3B, and requires zero extra parameter fitting or turnover.

**Conclusion**: Mechanical risk overlays (Vol Targeting / Trailing Stops) cannot fix a broken underlying factor, they only modulate its exposure. The most intellectually honest approach (Option 1) is to document the factor lifecycle and apply structurally robust portfolio-level asset allocation (Option 5: 60/40 active/passive blend) rather than fitting dynamic overlays.
