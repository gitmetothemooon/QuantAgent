"""Generate technical indicators from validated historical OHLCV stock data.

This module is responsible for a single task: deriving technical
indicator columns (simple moving averages, exponential moving averages,
``Daily_Return_Pct``, and ``RSI14``) from an already-validated OHLCV
pandas DataFrame.

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
RSI_PERIOD: int = 14


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

    features_df["SMA20"] = features_df["Close"].rolling(window=SMA_SHORT_WINDOW).mean()
    features_df["SMA50"] = features_df["Close"].rolling(window=SMA_LONG_WINDOW).mean()

    features_df["EMA20"] = features_df["Close"].ewm(span=EMA_SHORT_SPAN, adjust=False).mean()
    features_df["EMA50"] = features_df["Close"].ewm(span=EMA_LONG_SPAN, adjust=False).mean()

    features_df["Daily_Return_Pct"] = features_df["Close"].pct_change() * 100

    features_df["RSI14"] = _calculate_rsi(features_df["Close"], period=RSI_PERIOD)

    return features_df


def _calculate_rsi(close: pd.Series, period: int) -> pd.Series:
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