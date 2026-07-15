"""Generate technical indicators from validated historical OHLCV stock data.

This module is responsible for a single task: orchestrating the
generation of technical indicator columns (simple moving averages,
exponential moving averages, ``Daily_Return_Pct``, RSI, MACD, ATR, and
Bollinger Bands) from an already-validated OHLCV pandas DataFrame. The
actual indicator calculations live in dedicated modules:

    - :mod:`backend.app.features.trend`: SMA and EMA
    - :mod:`backend.app.features.momentum`: RSI and MACD
    - :mod:`backend.app.features.volatility`: ATR and Bollinger Bands

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
from backend.app.features.volatility import (
    ATR_PERIOD,
    calculate_atr,
    calculate_bollinger_bands,
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
    - ``RSI14``: 14-period Relative Strength Index.
    - ``MACD``: Moving Average Convergence Divergence line.
    - ``MACD_Signal``: Exponential moving average of the MACD line.
    - ``MACD_Histogram``: Difference between the MACD line and its
      signal line.
    - ``ATR14``: 14-period Average True Range.
    - ``BB_Middle``: 20-period simple moving average used as the
      Bollinger middle band.
    - ``BB_Upper``: Upper Bollinger Band (middle band + two standard
      deviations).
    - ``BB_Lower``: Lower Bollinger Band (middle band minus two standard
  deviations).

    Args:
        df: A validated pandas DataFrame containing historical OHLCV
            stock data with at least a ``Close`` column.

    Returns:
        A new pandas DataFrame containing all original columns plus
        ``SMA20``, ``SMA50``, ``EMA20``, ``EMA50``, ``Daily_Return_Pct``,
        ``RSI14``, ``MACD``, ``MACD_Signal``, ``MACD_Histogram``,
        ``ATR14``, ``BB_Middle``, ``BB_Upper``, and ``BB_Lower``.

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

    features_df["ATR14"] = calculate_atr(
        features_df["High"],
        features_df["Low"],
        features_df["Close"],
        period=ATR_PERIOD,
    )

    bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(
        features_df["Close"],
    )
    features_df["BB_Middle"] = bb_middle
    features_df["BB_Upper"] = bb_upper
    features_df["BB_Lower"] = bb_lower

    return features_df