"""Volume technical indicators for OHLCV price data.

This module is responsible for a single task: calculating volume-based
technical indicators (On-Balance Volume) from pandas Series of closing
prices and traded volume.

This module does NOT:
    - operate on DataFrames
    - download data
    - validate data
    - save files

Only pandas is used; no external technical-analysis libraries
(e.g. TA-Lib, pandas-ta) are required or permitted.
"""

import pandas as pd


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume (OBV) for a price and volume series.

    OBV measures buying and selling pressure by running a cumulative
    total of volume, adding a period's volume when price closes higher
    than the previous period and subtracting it when price closes
    lower, while leaving the total unchanged on an unchanged close.
    The cumulative volume is useful because it reveals whether volume
    is flowing in the same direction as price (confirming a trend) or
    diverging from it (often an early warning that a trend is losing
    strength) - information that looking at volume or price alone
    cannot show.

    The series is initialized as ``OBV[0] = Volume[0]``, and each
    subsequent value follows:

    - ``OBV[i] = OBV[i-1] + Volume[i]`` if ``Close[i] > Close[i-1]``
    - ``OBV[i] = OBV[i-1] - Volume[i]`` if ``Close[i] < Close[i-1]``
    - ``OBV[i] = OBV[i-1]`` otherwise

    Args:
        close: A pandas Series of closing prices.
        volume: A pandas Series of traded volume, aligned to the same
            index as ``close``.

    Returns:
        A pandas Series of cumulative OBV values, aligned to the input
        index.
    """
    price_change = close.diff()

    direction = pd.Series(0.0, index=close.index)
    direction[price_change > 0] = 1.0
    direction[price_change < 0] = -1.0

    # The first row has no prior close to compare against, so it is
    # treated as an up move: this makes the first term of the
    # cumulative sum below equal to Volume[0], matching the required
    # OBV[0] = Volume[0] initialization.
    direction.iloc[0] = 1.0

    signed_volume = volume * direction
    obv = signed_volume.cumsum()

    return obv