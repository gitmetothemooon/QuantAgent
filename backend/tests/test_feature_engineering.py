"""Temporary manual check for generate_features() on RELIANCE data."""

from backend.app.data.load_data import load_stock_data
from backend.app.features.feature_engineering import generate_features

df = load_stock_data("RELIANCE")
features_df = generate_features(df)

print(f"Number of rows: {len(features_df)}")
print(f"Columns: {list(features_df.columns)}")
print(features_df.head(10))

rsi = features_df["RSI14"]
rsi_valid = rsi.dropna()

if rsi_valid.empty:
    raise ValueError("RSI14 contains no valid values.")

print("RSI Summary:")
print(f"Minimum RSI value: {rsi_valid.min()}")
print(f"Maximum RSI value: {rsi_valid.max()}")
print(f"Number of NaN values in RSI14: {rsi.isna().sum()}")

if not rsi_valid.between(0, 100).all():
    raise ValueError("RSI14 contains values outside the valid range of 0 to 100.")

macd_columns = ["MACD", "MACD_Signal", "MACD_Histogram"]
for column in macd_columns:
    if column not in features_df.columns:
        raise ValueError(f"Expected column '{column}' is missing from features_df.")

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

if "ATR14" not in features_df.columns:
    raise ValueError("Expected column 'ATR14' is missing from features_df.")

atr = features_df["ATR14"]

print("ATR Summary:")
print(f"Minimum ATR: {atr.min()}")
print(f"Maximum ATR: {atr.max()}")
print(f"ATR NaN count: {atr.isna().sum()}")

if atr.isna().all():
    raise ValueError("Column 'ATR14' consists entirely of NaN values.")

bb_columns = ["BB_Middle", "BB_Upper", "BB_Lower"]
for column in bb_columns:
    if column not in features_df.columns:
        raise ValueError(f"Expected column '{column}' is missing from features_df.")

bb_middle = features_df["BB_Middle"]
bb_upper = features_df["BB_Upper"]
bb_lower = features_df["BB_Lower"]

print("Bollinger Bands Summary:")
print(f"Minimum BB_Middle: {bb_middle.min()}")
print(f"Maximum BB_Middle: {bb_middle.max()}")
print(f"Minimum BB_Upper: {bb_upper.min()}")
print(f"Maximum BB_Upper: {bb_upper.max()}")
print(f"Minimum BB_Lower: {bb_lower.min()}")
print(f"Maximum BB_Lower: {bb_lower.max()}")

print(f"BB_Middle NaN count: {bb_middle.isna().sum()}")
print(f"BB_Upper NaN count: {bb_upper.isna().sum()}")
print(f"BB_Lower NaN count: {bb_lower.isna().sum()}")

bb_series = {
    "BB_Middle": bb_middle,
    "BB_Upper": bb_upper,
    "BB_Lower": bb_lower,
}
for column_name, column in bb_series.items():
    if column.isna().all():
        raise ValueError(
            f"Column '{column_name}' consists entirely of NaN values."
        )

if "OBV" not in features_df.columns:
    raise ValueError("Expected column 'OBV' is missing from features_df.")

obv = features_df["OBV"]

print("OBV Summary:")
print(f"Minimum OBV: {obv.min()}")
print(f"Maximum OBV: {obv.max()}")
print(f"OBV NaN count: {obv.isna().sum()}")

if obv.isna().all():
    raise ValueError("Column 'OBV' consists  entirely of NaN values.")