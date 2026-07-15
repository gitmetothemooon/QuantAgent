"""
download_data.py

Single-responsibility module for QuantAgent:
Download historical stock price data (OHLCV) using yfinance and save it
locally as a CSV file inside data/raw/.

Example:
    RELIANCE.NS -> data/raw/RELIANCE.csv
"""

from pathlib import Path

import yfinance as yf

# Directory where all raw CSV files are stored.
# Resolved relative to this file's location (not the current working
# directory) so it always points to QuantAgent/data/raw, regardless of
# where the script is run from:
#   QuantAgent/backend/app/download_data.py -> up 3 levels -> QuantAgent/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_stock_data(symbol: str, period: str = "5y") -> Path:
    """
    Download historical OHLCV data for a stock symbol and save it as a CSV.

    Args:
        symbol: Ticker symbol to download, e.g. "RELIANCE.NS" or "AAPL".
        period: How much historical data to fetch (default "5y").
                Accepts any period string supported by yfinance,
                e.g. "1y", "6mo", "max".

    Returns:
        Path to the saved CSV file.

    Raises:
        ValueError: If no data is returned for the given symbol.
        RuntimeError: If the download fails for any other reason.
    """
    print(f"Downloading data for '{symbol}' (period={period})...")

    try:
        data = yf.download(symbol, period=period, progress=False)
    except Exception as error:
        raise RuntimeError(
            f"Failed to download data for '{symbol}': {error}"
        ) from error

    if data is None or data.empty:
        raise ValueError(
            f"No data returned for symbol '{symbol}'. "
            "Check that the symbol is correct and try again."
        )

    # yfinance sometimes returns a MultiIndex column header (e.g. one level
    # for the field name and one for the ticker). Flatten it down to plain
    # column names like "Open", "High", "Low", "Close", "Volume".
    if data.columns.nlevels > 1:
        data.columns = data.columns.get_level_values(0)

    # Move the Date out of the index and into a regular column, then keep
    # only the columns needed for a simple OHLCV CSV header.
    data = data.reset_index()
    data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]

    # Make sure the target directory exists before writing to it.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # "RELIANCE.NS" -> "RELIANCE.csv"
    file_stem = symbol.split(".")[0]
    output_path = DATA_DIR / f"{file_stem}.csv"

    data.to_csv(output_path, index=False)
    print(f"Saved {len(data)} rows to '{output_path}'.")

    return output_path


def main() -> None:
    """Run a simple example download when this file is executed directly."""
    example_symbol = "RELIANCE.NS"

    try:
        saved_path = download_stock_data(example_symbol)
        print(f"Done. Data saved at: {saved_path}")
    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()