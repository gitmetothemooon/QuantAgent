"""
validate_data.py

Single-responsibility module for validating historical OHLCV stock data
contained in a pandas DataFrame.

This module does NOT:
    - download data
    - save files
    - calculate indicators
    - modify data, except for removing duplicate dates and sorting by date

It exposes a single public function, `validate_stock_data`, which performs
a series of structural and value-level checks on the input DataFrame and
raises a `ValueError` with a descriptive message if any check fails.
"""

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = ("Date", "Open", "High", "Low", "Close", "Volume")
PRICE_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close")
NUMERIC_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


def validate_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a DataFrame of historical OHLCV stock data.

    The following checks are performed, in order:
        1. The DataFrame is not empty.
        2. All required columns are present:
           Date, Open, High, Low, Close, Volume.
        3. The Date column can be converted to datetime (invalid values are
           explicitly detected and rejected).
        4. Numeric columns (Open, High, Low, Close, Volume) can be converted
           to numeric dtypes.
        5. Duplicate dates are removed (first occurrence kept).
        6. Rows are sorted by Date in ascending order.
        7. There are no missing values in Open, High, Low, Close, or Volume.
        8. All price values (Open, High, Low, Close) are greater than zero.
        9. All Volume values are greater than or equal to zero.

    Args:
        df: A pandas DataFrame expected to contain historical OHLCV stock
            data with at least the columns Date, Open, High, Low, Close,
            and Volume.

    Returns:
        A validated pandas DataFrame containing only the columns Date, Open,
        High, Low, Close, and Volume, in that exact order, with duplicate
        dates removed and rows sorted by Date in ascending order. The Date
        column is converted to datetime dtype and numeric columns are
        converted to numeric dtypes.

    Raises:
        ValueError: If the DataFrame is empty, is missing required columns,
            has a Date column containing values that cannot be converted to
            datetime, has numeric columns containing values that cannot be
            converted to numeric types, contains missing values in required
            columns, contains non-positive price values, or contains
            negative volume values.
    """
    if df is None or df.empty:
        raise ValueError("Validation failed: input DataFrame is empty.")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Validation failed: missing required column(s): {missing_columns}."
        )

    validated_df = df.copy()

    original_dates = validated_df["Date"]
    validated_df["Date"] = pd.to_datetime(validated_df["Date"], errors="coerce")
    invalid_date_mask = validated_df["Date"].isna()
    if invalid_date_mask.any():
        bad_values = original_dates[invalid_date_mask].tolist()
        raise ValueError(
            "Validation failed: 'Date' column contains value(s) that could "
            f"not be converted to datetime: {bad_values}."
        )

    for col in NUMERIC_COLUMNS:
        try:
            validated_df[col] = pd.to_numeric(validated_df[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Validation failed: column '{col}' contains value(s) that "
                f"could not be converted to a numeric type: {exc}"
            ) from exc

    validated_df = validated_df.drop_duplicates(subset="Date", keep="first")

    validated_df = validated_df.sort_values(by="Date", ascending=True).reset_index(
        drop=True
    )

    numeric_cols = list(NUMERIC_COLUMNS)
    null_counts = validated_df[numeric_cols].isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if not columns_with_nulls.empty:
        rows_with_nulls = validated_df[numeric_cols].isnull().any(axis=1)
        bad_dates = (
            validated_df.loc[rows_with_nulls, "Date"].dt.strftime("%Y-%m-%d").tolist()
        )
        raise ValueError(
            "Validation failed: missing values found in column(s) "
            f"{columns_with_nulls.to_dict()} on date(s): {bad_dates}."
        )

    for col in PRICE_COLUMNS:
        if (validated_df[col] <= 0).any():
            bad_dates = (
                validated_df.loc[validated_df[col] <= 0, "Date"]
                .dt.strftime("%Y-%m-%d")
                .tolist()
            )
            raise ValueError(
                f"Validation failed: column '{col}' contains non-positive "
                f"value(s) on date(s): {bad_dates}."
            )

    if (validated_df["Volume"] < 0).any():
        bad_dates = (
            validated_df.loc[validated_df["Volume"] < 0, "Date"]
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )
        raise ValueError(
            "Validation failed: column 'Volume' contains negative value(s) "
            f"on date(s): {bad_dates}."
        )

    validated_df = validated_df[list(REQUIRED_COLUMNS)]

    return validated_df