"""Temporary manual check for PerformanceAnalyzer."""

import math

import pandas as pd

from backend.app.backtesting.performance import PerformanceAnalyzer

results_df = pd.DataFrame(
    {
        "Portfolio_Value": [100.0, 110.0, 105.0, 120.0, 90.0],
    }
)

analyzer = PerformanceAnalyzer(results_df)

total_return = analyzer.total_return()
cagr = analyzer.cagr()
daily_volatility = analyzer.daily_volatility()
max_drawdown = analyzer.max_drawdown()
sharpe_ratio = analyzer.sharpe_ratio()
summary = analyzer.summary()

print(f"Total Return (%): {total_return}")
print(f"CAGR (%): {cagr}")
print(f"Daily Volatility (%): {daily_volatility}")
print(f"Maximum Drawdown (%): {max_drawdown}")
print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Summary(): {summary}")

# ------------------------------------------------------------
# 1. Total Return
# Portfolio goes from 100.0 to 90.0:
#   ((90 / 100) - 1) * 100 = -10.0
# ------------------------------------------------------------
expected_total_return = ((90 / 100) - 1) * 100

if not math.isclose(total_return, expected_total_return, rel_tol=1e-9):
    raise ValueError(
        f"Expected Total Return of {expected_total_return}, "
        f"but got {total_return}."
    )

# ------------------------------------------------------------
# 2. Maximum Drawdown
# Running peak: 100, 110, 110, 120, 120
# Drawdown:       0,   0,  -4.5454...%, 0, -25%
# The worst drawdown is -25.0 at the final row (90 vs. peak of 120).
# ------------------------------------------------------------
expected_max_drawdown = -25.0

if not math.isclose(max_drawdown, expected_max_drawdown, rel_tol=1e-9):
    raise ValueError(
        f"Expected Maximum Drawdown of {expected_max_drawdown}, "
        f"but got {max_drawdown}."
    )

# ------------------------------------------------------------
# 3. Summary() keys
# ------------------------------------------------------------
expected_summary_keys = {
    "Total Return (%)",
    "CAGR (%)",
    "Daily Volatility (%)",
    "Maximum Drawdown (%)",
    "Sharpe Ratio",
}

if set(summary.keys()) != expected_summary_keys:
    raise ValueError(
        f"Expected summary() keys {expected_summary_keys}, "
        f"but got {set(summary.keys())}."
    )

# ------------------------------------------------------------
# 4. CAGR, Daily Volatility, and Sharpe Ratio
# These involve floating-point calculations that are not manually
# verified numerically here. Instead, only their type and basic
# sanity properties are checked.
# ------------------------------------------------------------
if not isinstance(cagr, float):
    raise ValueError(f"Expected CAGR to be a float, but got {type(cagr)}.")

if not isinstance(daily_volatility, float):
    raise ValueError(
        f"Expected Daily Volatility to be a float, but got {type(daily_volatility)}."
    )

if daily_volatility < 0:
    raise ValueError(
        f"Expected Daily Volatility to be non-negative, but got {daily_volatility}."
    )

if not isinstance(sharpe_ratio, float):
    raise ValueError(
        f"Expected Sharpe Ratio to be a float, but got {type(sharpe_ratio)}."
    )

print("PerformanceAnalyzer test passed.")