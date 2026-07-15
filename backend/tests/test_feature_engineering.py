"""Temporary manual check for generate_features() on RELIANCE data."""

from backend.app.data.load_data import load_stock_data
from backend.app.features.feature_engineering import generate_features

df = load_stock_data("RELIANCE")
features_df = generate_features(df)

print(f"Number of rows: {len(features_df)}")
print(f"Columns: {list(features_df.columns)}")
print(features_df.head(10))

# -------------------------
# RSI Validation
# -------------------------

rsi = features_df["RSI14"]
rsi_valid = rsi.dropna()

if rsi_valid.empty:
    raise ValueError("RSI14 contains no valid values.")

print("RSI Summary:")
print(f"Minimum RSI value: {rsi_valid.min()}")
print(f"Maximum RSI value: {rsi_valid.max()}")
print(f"Number of NaN values in RSI14: {rsi.isna().sum()}")

if not rsi_valid.between(0, 100).all():
    raise ValueError(
        "RSI14 contains values outside the valid range of 0 to 100."
    )

# -------------------------
# MACD Validation
# -------------------------

expected_columns = {
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
}

missing_columns = expected_columns - set(features_df.columns)

if missing_columns:
    raise ValueError(
        f"Missing expected MACD column(s): {sorted(missing_columns)}"
    )

macd = features_df["MACD"]
macd_signal = features_df["MACD_Signal"]
macd_histogram = features_df["MACD_Histogram"]

print("MACD Summary:")
print(f"Minimum MACD: {macd.min()}")
print(f"Maximum MACD: {macd.max()}")

print(f"Minimum Signal: {macd_signal.min()}")
print(f"Maximum Signal: {macd_signal.max()}")

print(f"Minimum Histogram: {macd_histogram.min()}")
print(f"Maximum Histogram: {macd_histogram.max()}")

print(f"MACD NaN count: {macd.isna().sum()}")
print(f"Signal NaN count: {macd_signal.isna().sum()}")
print(f"Histogram NaN count: {macd_histogram.isna().sum()}")

macd_series = {
    "MACD": macd,
    "MACD_Signal": macd_signal,
    "MACD_Histogram": macd_histogram,
}

for column_name, column in macd_series.items():
    if column.isna().all():
        raise ValueError(
            f"Column '{column_name}' consists entirely of NaN values."
        )