"""Momentum technical indicators for OHLCV price data.

This module is responsible for a single task: calculating momentum
indicators (the Relative Strength Index and MACD) from a pandas Series
of closing prices.

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
MACD_FAST_PERIOD: int = 12
MACD_SLOW_PERIOD: int = 26
MACD_SIGNAL_PERIOD: int = 9


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


def calculate_macd(
    close: pd.Series,
    fast_period: int = MACD_FAST_PERIOD,
    slow_period: int = MACD_SLOW_PERIOD,
    signal_period: int = MACD_SIGNAL_PERIOD,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate the Moving Average Convergence Divergence (MACD).

    MACD measures momentum and trend direction by comparing a fast and
    a slow exponential moving average of price. When the fast EMA is
    above the slow EMA, momentum is considered bullish; when it is
    below, momentum is considered bearish. The signal line is a
    smoothed version of the MACD line itself, and the histogram shows
    the gap between the two, which is often used to spot momentum
    shifts before they show up in the MACD line crossing the signal
    line.

    Args:
        close: A pandas Series of closing prices.
        fast_period: The span of the fast EMA (standard MACD uses 12).
        slow_period: The span of the slow EMA (standard MACD uses 26).
        signal_period: The span of the EMA applied to the MACD line to
            produce the signal line (standard MACD uses 9).

    Returns:
        A tuple of three pandas Series, each aligned to the input
        index, in the following order:

        - ``macd_line``: The difference between the fast and slow EMA
          of ``close``, representing raw momentum.
        - ``signal_line``: An EMA of ``macd_line``, used as a trigger
          line for identifying momentum shifts.
        - ``histogram``: The difference between ``macd_line`` and
          ``signal_line``, representing the strength of the current
          momentum shift.
    """
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram