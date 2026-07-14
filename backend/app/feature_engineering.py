"""Generate technical indicators from validated historical OHLCV stock data.

This module is responsible for a single task: deriving technical
indicator columns (simple moving averages, exponential moving averages,
and ``Daily_Return_Pct``) from an already-validated OHLCV pandas
DataFrame.

This module does NOT:
    - download data
    - validate data
    - modify the input DataFrame in place
    - save files

It assumes the input DataFrame has already passed through
:func:`backend.app.validate_data.validate_stock_data` and therefore
contains, at minimum, a ``Close`` column with no missing values.

Only pandas is used; no external technical-analysis libraries
(e.g. TA-Lib, pandas-ta) are required or permitted.
"""

import pandas as pd

SMA_SHORT_WINDOW: int = 20
SMA_LONG_WINDOW: int = 50
EMA_SHORT_SPAN: int = 20
EMA_LONG_SPAN: int = 50


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

    Args:
        df: A validated pandas DataFrame containing historical OHLCV
            stock data with at least a ``Close`` column.

    Returns:
        A new pandas DataFrame containing all original columns plus
        ``SMA20``, ``SMA50``, ``EMA20``, ``EMA50``, and
        ``Daily_Return_Pct``.

    Raises:
        ValueError: If the input DataFrame does not contain a
            ``Close`` column.
    """
    if "Close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Close' column.")

    features_df: pd.DataFrame = df.copy()

    features_df["SMA20"] = features_df["Close"].rolling(window=SMA_SHORT_WINDOW).mean()
    features_df["SMA50"] = features_df["Close"].rolling(window=SMA_LONG_WINDOW).mean()

    features_df["EMA20"] = features_df["Close"].ewm(span=EMA_SHORT_SPAN, adjust=False).mean()
    features_df["EMA50"] = features_df["Close"].ewm(span=EMA_LONG_SPAN, adjust=False).mean()

    features_df["Daily_Return_Pct"] = features_df["Close"].pct_change() * 100

    return features_df