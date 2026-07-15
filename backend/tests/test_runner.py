"""Temporary manual integration check for Runner.

This is an integration test: it verifies that Runner correctly wires
together data loading, feature engineering, feature quality
preparation, backtesting, and performance analysis, and returns
results of the expected shape. It does not re-verify the internal
logic of any of those stages - each already has its own dedicated
test.
"""

import pandas as pd

from backend.app.runner import Runner
from backend.app.strategy.rsi_macd import RSIMACDStrategy

strategy = RSIMACDStrategy()
runner = Runner(
    strategy=strategy,
    initial_cash=100000.0,
)

results = runner.run("RELIANCE")

print(f"Returned dictionary keys: {list(results.keys())}")
print(f"Shape of raw_data: {results['raw_data'].shape}")
print(f"Shape of features: {results['features'].shape}")
print(f"Shape of prepared_features: {results['prepared_features'].shape}")
print(f"Shape of backtest_results: {results['backtest_results'].shape}")
print(f"Performance summary: {results['performance']}")

# ------------------------------------------------------------
# 1. Returned dictionary keys
# ------------------------------------------------------------
expected_keys = {
    "raw_data",
    "features",
    "prepared_features",
    "backtest_results",
    "performance",
}

if set(results.keys()) != expected_keys:
    raise ValueError(
        f"Expected result keys {expected_keys}, but got {set(results.keys())}."
    )

# ------------------------------------------------------------
# 2. Returned object types
# ------------------------------------------------------------
expected_types = {
    "raw_data": pd.DataFrame,
    "features": pd.DataFrame,
    "prepared_features": pd.DataFrame,
    "backtest_results": pd.DataFrame,
    "performance": dict,
}

for key, expected_type in expected_types.items():
    actual_value = results[key]
    if not isinstance(actual_value, expected_type):
        raise ValueError(
            f"Expected '{key}' to be of type {expected_type}, "
            f"but got {type(actual_value)}."
        )

# ------------------------------------------------------------
# 3. backtest_results must contain Portfolio_Value
# ------------------------------------------------------------
if "Portfolio_Value" not in results["backtest_results"].columns:
    raise ValueError(
        "Expected column 'Portfolio_Value' is missing from backtest_results."
    )

# ------------------------------------------------------------
# 4. performance dictionary keys
# ------------------------------------------------------------
expected_performance_keys = {
    "Total Return (%)",
    "CAGR (%)",
    "Daily Volatility (%)",
    "Maximum Drawdown (%)",
    "Sharpe Ratio",
}

if set(results["performance"].keys()) != expected_performance_keys:
    raise ValueError(
        f"Expected performance keys {expected_performance_keys}, "
        f"but got {set(results['performance'].keys())}."
    )

# ------------------------------------------------------------
# 5. None of the returned DataFrames should be empty
# ------------------------------------------------------------
dataframe_keys = ["raw_data", "features", "prepared_features", "backtest_results"]

for key in dataframe_keys:
    dataframe = results[key]
    if len(dataframe) == 0:
        raise ValueError(f"Expected '{key}' to be non-empty, but it has zero rows.")

print("Runner integration test passed.")