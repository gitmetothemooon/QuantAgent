"""Volatility technical indicators for OHLCV price data.

This module is responsible for a single task: calculating volatility
indicators (the Average True Range and Bollinger Bands) from pandas
Series of high, low, and close prices.

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
BOLLINGER_WINDOW: int = 20
BOLLINGER_STD_DEV: int = 2


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


def calculate_bollinger_bands(
    close: pd.Series,
    window: int = BOLLINGER_WINDOW,
    num_std_dev: int = BOLLINGER_STD_DEV,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands for a price series.

    Bollinger Bands measure volatility relative to a moving average by
    plotting a band above and below it, sized according to how much
    price has recently dispersed from that average. The middle band is
    a simple moving average of ``close``, and the upper and lower
    bands are that average offset by a multiple of the rolling
    standard deviation. Rolling standard deviation is used because it
    naturally widens the bands during volatile periods and narrows
    them during calm periods, so the bands adapt to current market
    conditions rather than staying a fixed distance from price.

    Args:
        close: A pandas Series of closing prices.
        window: The number of periods used for the moving average and
            rolling standard deviation (standard Bollinger Bands use
            20).
        num_std_dev: The number of standard deviations used to offset
            the upper and lower bands from the middle band (standard
            Bollinger Bands use 2).

    Returns:
        A tuple of three pandas Series, each aligned to the input
        index, in the following order:

        - ``middle_band``: The simple moving average of ``close``.
        - ``upper_band``: ``middle_band`` plus ``num_std_dev`` rolling
          standard deviations, representing the upper volatility
          boundary.
        - ``lower_band``: ``middle_band`` minus ``num_std_dev`` rolling
          standard deviations, representing the lower volatility
          boundary.
    """
    middle_band = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()

    upper_band = middle_band + (num_std_dev * rolling_std)
    lower_band = middle_band - (num_std_dev * rolling_std)

    return middle_band, upper_band, lower_band