"""Momentum technical indicators for OHLCV price data.

This module is responsible for a single task: calculating momentum
indicators (the Relative Strength Index) from a pandas Series of
closing prices.

This module does NOT:
    - operate on DataFrames
    - download data
    - validate data
    - save files

Only pandas is used; no external technical-analysis libraries
(e.g. TA-Lib, pandas-ta) are required or permitted.
"""

import pandas as pd

RSI_PERIOD: int = 14


def calculate_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Calculate the Relative Strength Index (RSI) for a price series.

    Uses the standard Wilder's smoothing method: average gains and
    average losses are smoothed with an exponential moving average
    where ``alpha = 1 / period``, matching the conventional RSI
    definition used by most charting platforms.

    Args:
        close: A pandas Series of closing prices.
        period: The lookback period for the RSI calculation
            (standard RSI14 uses 14).

    Returns:
        A pandas Series of RSI values, aligned to the input index.
    """
    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # Divide by a version of avg_loss with zeros swapped for NaN so the
    # division itself never triggers a divide-by-zero runtime warning.
    # The resulting NaN is overwritten by the explicit rules below.
    relative_strength = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + relative_strength))

    # Explicit divide-by-zero handling: a zero average loss means price
    # only moved up over the window (RSI = 100), and a zero average gain
    # means price only moved down (RSI = 0). Rows that don't yet have a
    # full window are unaffected: avg_gain/avg_loss are NaN there, the
    # comparisons below evaluate to False, and the existing NaN in `rsi`
    # is left untouched.
    rsi = rsi.mask(avg_loss == 0, 100.0)
    rsi = rsi.mask(avg_gain == 0, 0.0)

    return rsi