import json
import config
from regime_aware_backtest import RegimeAwareBacktester

EXPERIMENTS = {
    "Baseline (Static)": {
        "ENABLE_VOL_TARGETING": False,
        "ENABLE_TRAILING_STOPS": False,
        "ENABLE_BENCHMARK_BLEND": False,
    },
    "Option 2: Volatility Targeting": {
        "ENABLE_VOL_TARGETING": True,
        "ENABLE_TRAILING_STOPS": False,
        "ENABLE_BENCHMARK_BLEND": False,
    },
    "Option 4: Trailing Stops": {
        "ENABLE_VOL_TARGETING": False,
        "ENABLE_TRAILING_STOPS": True,
        "ENABLE_BENCHMARK_BLEND": False,
    },
    "Option 5: Benchmark Blend (60/40)": {
        "ENABLE_VOL_TARGETING": False,
        "ENABLE_TRAILING_STOPS": False,
        "ENABLE_BENCHMARK_BLEND": True,
    },
    "Recommended: Vol Targeting + Trailing Stops": {
        "ENABLE_VOL_TARGETING": True,
        "ENABLE_TRAILING_STOPS": True,
        "ENABLE_BENCHMARK_BLEND": False,
    }
}

def run_all():
    results_summary = []
    
    # Ensure adaptive options are disabled for clean comparison
    config.ENABLE_ADAPTIVE_WEIGHTS = False
    config.ENABLE_DRAWDOWN_SCALING = False
    config.ENABLE_ALPHA_FADE = False
    config.ENABLE_ADAPTIVE_LIQUIDATION = False
    config.ENABLE_ADAPTIVE_TOP_N = False
    config.ENABLE_OPTIONS_HEDGE = False # Disable hedge for pure factor evaluation
    
    import os
    for name, overrides in EXPERIMENTS.items():
        print(f"\n{'='*70}\nRunning Experiment: {name}\n{'='*70}")
        
        # Apply overrides
        for k, v in overrides.items():
            setattr(config, k, v)
            
        bt = RegimeAwareBacktester(data_dir=os.path.join(os.path.dirname(__file__), config.DATA_DIR))
        bt.load_data()
        res = bt.run()
        
        if res and 'metrics' in res:
            m = res['metrics']
            results_summary.append({
                "Experiment": name,
                "Final Value ($M)": m['portfolio']['final_value'] / 1e6,
                "Ann Return (%)": m['portfolio']['annual_return'] * 100,
                "Volatility (%)": m['portfolio']['volatility'] * 100,
                "Sharpe": m['portfolio']['sharpe_ratio'],
                "Max DD (%)": m['portfolio']['max_drawdown'] * 100,
                "Trades": m['trading']['total_trades']
            })
            
    # Print markdown table
    print("\n\n### Experiment Results Summary\n")
    print("| Experiment | Final Value | Ann Ret | Volatility | Sharpe | Max Drawdown | Trades |")
    print("|---|---|---|---|---|---|---|")
    for r in results_summary:
        print(f"| {r['Experiment']} | ${r['Final Value ($M)']:.0f}M | {r['Ann Return (%)']:.2f}% | {r['Volatility (%)']:.2f}% | {r['Sharpe']:.2f} | {r['Max DD (%)']:.2f}% | {r['Trades']} |")

if __name__ == "__main__":
    run_all()
