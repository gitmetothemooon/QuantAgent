"""Prepare engineered features for downstream strategy modules.

This module is responsible for a single task: inspecting where NaN
values occur in an engineered-features DataFrame, removing only the
leading warm-up rows required by rolling-window indicators (e.g.
SMA, EMA, RSI, ATR, rolling volatility), and confirming that no other
NaN values remain, so that downstream strategy modules receive a
fully populated, index-clean DataFrame. An empty input DataFrame is
rejected outright.

This module does NOT:
    - download data
    - validate raw data
    - calculate indicators
    - save files

It assumes the input DataFrame has already been produced by
:func:`backend.app.feature_engineering.generate_features` (or an
equivalent process) and therefore may contain leading NaN values in
rolling-window-derived columns.

Only pandas is used.
"""

import pandas as pd


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leading warm-up rows from an engineered-features DataFrame.

    The input DataFrame is never modified in place. For each column,
    the position of the first valid (non-NaN) value is located via
    :meth:`pandas.Series.first_valid_index`; any NaN values before
    that position are expected to originate from rolling-window
    indicators such as SMA, EMA, RSI, ATR, or rolling volatility,
    which cannot produce values until their warm-up period has
    elapsed. The warm-up assumption is explicitly verified: a column
    may only contain NaN values before its first valid value, never
    after it. Only the minimum number of leading rows required to
    eliminate every column's expected NaN values is removed. After
    that removal, the remaining DataFrame is checked for any other
    NaN values, which are treated as unexpected data quality issues.

    Args:
        df: A pandas DataFrame containing engineered features, expected
            to contain leading NaN values produced by rolling-window
            indicators.

    Returns:
        A new pandas DataFrame with leading warm-up rows removed, all
        original columns preserved, and a fresh, reset integer index.

    Raises:
        ValueError: If the input DataFrame is None or empty.
        ValueError: If any column contains only NaN values, which
            indicates a broken feature-generation pipeline rather
            than a normal indicator warm-up period.
        ValueError: If any column contains a NaN value after its
            first valid value, which violates the warm-up assumption.
        ValueError: If NaN values remain anywhere in the DataFrame
            after the leading warm-up rows have been removed.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame must not be empty.")

    cleaned_df: pd.DataFrame = df.copy()

    # For each column, use first_valid_index() to locate the first
    # non-NaN value. Different indicators have different warm-up
    # lengths (e.g. SMA50 needs more leading rows than SMA20), so the
    # largest position across all columns determines how many rows
    # must be dropped to clear every indicator's warm-up period.
    leading_nan_counts = []
    for column_name in cleaned_df.columns:
        column = cleaned_df[column_name]
        first_valid_index = column.first_valid_index()

        if first_valid_index is None:
            # A column with no valid values at all is not a normal
            # warm-up period; it indicates a broken feature-generation
            # pipeline and must fail loudly rather than being silently
            # dropped.
            raise ValueError(
                f"Column '{column_name}' contains only NaN values."
            )

        first_valid_position = cleaned_df.index.get_loc(first_valid_index)
        leading_nan_counts.append(first_valid_position)

        # Explicitly verify the warm-up assumption: NaN values are
        # only allowed before the first valid value in this column.
        # A NaN occurring after it indicates a genuine data gap, not
        # an indicator warm-up period, and must not pass silently.
        values_after_first_valid = column.loc[first_valid_index:]
        if values_after_first_valid.isnull().any():
            raise ValueError(
                f"Column '{column_name}' contains NaN value(s) after its "
                "first valid value; NaN values are only expected before "
                "the first valid value, during the indicator warm-up "
                "period."
            )

    warmup_row_count = max(leading_nan_counts) if leading_nan_counts else 0

    # Remove only the leading warm-up rows identified above. NaN
    # values that occur anywhere else in the DataFrame have already
    # been ruled out by the warm-up assumption check above.
    cleaned_df = cleaned_df.iloc[warmup_row_count:]

    remaining_nan_columns = cleaned_df.columns[cleaned_df.isnull().any()].tolist()
    if remaining_nan_columns:
        raise ValueError(
            "Unexpected NaN values remain after removing warm-up rows "
            f"in column(s): {remaining_nan_columns}."
        )

    cleaned_df = cleaned_df.reset_index(drop=True)

    return cleaned_df