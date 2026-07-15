"""Trend-following technical indicators for OHLCV price data.

This module is responsible for a single task: calculating trend
indicators (simple moving average and exponential moving average) from
a pandas Series of closing prices.

This module does NOT:
    - operate on DataFrames
    - download data
    - validate data
    - save files

Only pandas is used; no external technical-analysis libraries
(e.g. TA-Lib, pandas-ta) are required or permitted.
"""

import pandas as pd

SMA_SHORT_WINDOW: int = 20
SMA_LONG_WINDOW: int = 50
EMA_SHORT_SPAN: int = 20
EMA_LONG_SPAN: int = 50


def calculate_sma(close: pd.Series, window: int) -> pd.Series:
    """Calculate the simple moving average (SMA) of a price series.

    Args:
        close: A pandas Series of closing prices.
        window: The number of periods to average over.

    Returns:
        A pandas Series of SMA values, aligned to the input index.
    """
    return close.rolling(window=window).mean()


def calculate_ema(close: pd.Series, span: int) -> pd.Series:
    """Calculate the exponential moving average (EMA) of a price series.

    Args:
        close: A pandas Series of closing prices.
        span: The span parameter controlling the EMA's decay rate.

    Returns:
        A pandas Series of EMA values, aligned to the input index.
    """
    return close.ewm(span=span, adjust=False).mean()