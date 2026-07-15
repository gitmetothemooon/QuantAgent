"""Load historical OHLCV stock data from a validated CSV file.

This module is responsible for a single task: locating the raw CSV file
for a given stock symbol, reading it into a pandas DataFrame, and
validating it via :func:`backend.app.data.validate_data.validate_stock_data`.

It does not download data, calculate indicators, modify prices, perform
feature engineering, or save files.
"""

from pathlib import Path

import pandas as pd

from backend.app.data.validate_data import validate_stock_data

# Project root:
# backend/app/data/load_data.py
#        ↑ data
#        ↑ app
#        ↑ backend
#        ↑ QuantAgent
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]


def load_stock_data(symbol: str) -> pd.DataFrame:
    """Load and validate historical OHLCV data for a stock symbol.

    Reads the CSV file located at ``data/raw/<SYMBOL>.csv`` (relative to
    the project root), then validates the resulting DataFrame using
    :func:`validate_stock_data`.

    Args:
        symbol: The stock ticker symbol (e.g. ``"RELIANCE"``). The symbol
            is uppercased to build the expected filename.

    Returns:
        A validated pandas DataFrame containing the stock's historical
        OHLCV data.

    Raises:
        FileNotFoundError: If no CSV file exists at the expected path.
        ValueError: If the loaded data fails validation in
            :func:`validate_stock_data`.
    """
    csv_path: Path = PROJECT_ROOT / "data" / "raw" / f"{symbol.upper()}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find data file for symbol '{symbol}' at expected "
            f"path: {csv_path}"
        )

    df: pd.DataFrame = pd.read_csv(csv_path)
    validated_df: pd.DataFrame = validate_stock_data(df)

    return validated_df


def main() -> None:
    """Demonstrate loading and validating data for the RELIANCE symbol."""
    df = load_stock_data("RELIANCE")
    print(df.head())


if __name__ == "__main__":
    main()