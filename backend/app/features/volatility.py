"""Volatility technical indicators for OHLCV price data.

This module is responsible for a single task: calculating volatility
indicators (the Average True Range) from pandas Series of high, low,
and close prices.

This module does NOT:
    - operate on DataFrames
    - download data
    - validate data
    - save files

Only pandas is used; no external technical-analysis libraries
(e.g. TA-Lib, pandas-ta) are required or permitted.
"""

import pandas as pd

ATR_PERIOD: int = 14


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = ATR_PERIOD,
) -> pd.Series:
    """Calculate the Average True Range (ATR) for a price series.

    ATR measures volatility by averaging the "true range" of price
    movement over a lookback period, where true range captures the
    full extent of a period's price action, including any gap from
    the previous close. Unlike a simple high-low range, ATR accounts
    for gaps between one period's close and the next period's high or
    low, which is why the previous close is required: without it, a
    gap up or down would understate how much the price actually
    moved. True range is smoothed using Wilder's exponential moving
    average (``alpha = 1 / period``), the same smoothing convention
    used for RSI.

    Args:
        high: A pandas Series of period high prices.
        low: A pandas Series of period low prices.
        close: A pandas Series of period closing prices.
        period: The lookback period for the ATR smoothing
            (standard ATR14 uses 14).

    Returns:
        A pandas Series of ATR values, aligned to the input index.
    """
    previous_close = close.shift(1)

    high_low = high - low
    high_prev_close = (high - previous_close).abs()
    low_prev_close = (low - previous_close).abs()

    true_range = pd.concat(
        [
            high_low,
            high_prev_close,
            low_prev_close,
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    return atr