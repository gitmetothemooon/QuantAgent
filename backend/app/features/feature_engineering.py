"""Generate technical indicators from validated historical OHLCV stock data.

This module is responsible for a single task: orchestrating the
generation of technical indicator columns (simple moving averages,
exponential moving averages, ``Daily_Return_Pct``, and ``RSI14``) from
an already-validated OHLCV pandas DataFrame. The actual indicator
calculations live in dedicated modules:

    - :mod:`backend.app.features.trend`: SMA and EMA
    - :mod:`backend.app.features.momentum`: RSI

This module does NOT:
    - download data
    - validate data
    - modify the input DataFrame in place
    - save files
    - implement indicator calculations directly

It assumes the input DataFrame has already passed through
:func:`backend.app.data.validate_data.validate_stock_data` and therefore
contains, at minimum, a ``Close`` column with no missing values.

Only pandas is used; no external technical-analysis libraries
(e.g. TA-Lib, pandas-ta) are required or permitted.
"""

import pandas as pd

from backend.app.features.trend import (
    EMA_LONG_SPAN,
    EMA_SHORT_SPAN,
    SMA_LONG_WINDOW,
    SMA_SHORT_WINDOW,
    calculate_ema,
    calculate_sma,
)
from backend.app.features.momentum import (
    RSI_PERIOD,
    calculate_rsi,
    calculate_macd,
)


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator columns to a validated OHLCV DataFrame.

    The input DataFrame is assumed to already be validated (e.g. via
    ``validate_stock_data``) and is never modified in place; a new
    DataFrame is returned with all original columns preserved plus the
    following additional columns:

    - ``SMA20``: 20-period simple moving average of ``Close``.
    - ``SMA50``: 50-period simple moving average of ``Close``.
    - ``EMA20``: 20-period exponential moving average of ``Close``.
    - ``EMA50``: 50-period exponential moving average of ``Close``.
    - ``Daily_Return_Pct``: Percentage change of ``Close`` from the
      previous row.
    - ``RSI14``: 14-period Relative Strength Index of ``Close``, using
      Wilder's smoothing method.

    Args:
        df: A validated pandas DataFrame containing historical OHLCV
            stock data with at least a ``Close`` column.

    Returns:
        A new pandas DataFrame containing all original columns plus
        ``SMA20``, ``SMA50``, ``EMA20``, ``EMA50``, ``Daily_Return_Pct``,
        and ``RSI14``.

    Raises:
        ValueError: If the input DataFrame does not contain a
            ``Close`` column.
    """
    if "Close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Close' column.")

    features_df: pd.DataFrame = df.copy()

    features_df["SMA20"] = calculate_sma(features_df["Close"], window=SMA_SHORT_WINDOW)
    features_df["SMA50"] = calculate_sma(features_df["Close"], window=SMA_LONG_WINDOW)

    features_df["EMA20"] = calculate_ema(features_df["Close"], span=EMA_SHORT_SPAN)
    features_df["EMA50"] = calculate_ema(features_df["Close"], span=EMA_LONG_SPAN)

    features_df["Daily_Return_Pct"] = features_df["Close"].pct_change() * 100

    features_df["RSI14"] = calculate_rsi(
        features_df["Close"],
        period=RSI_PERIOD,
    )

    macd_line, signal_line, histogram = calculate_macd(
        features_df["Close"]
    )

    features_df["MACD"] = macd_line
    features_df["MACD_Signal"] = signal_line
    features_df["MACD_Histogram"] = histogram

    return features_df