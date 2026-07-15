"""Temporary manual check for generate_features() on RELIANCE data."""

from backend.app.load_data import load_stock_data
from backend.app.feature_engineering import generate_features

df = load_stock_data("RELIANCE")
features_df = generate_features(df)

print(f"Number of rows: {len(features_df)}")
print(f"Columns: {list(features_df.columns)}")
print(features_df.head(10))

rsi = features_df["RSI14"]
rsi_valid = rsi.dropna()

print("RSI Summary:")
print(f"Minimum RSI value: {rsi_valid.min()}")
print(f"Maximum RSI value: {rsi_valid.max()}")
print(f"Number of NaN values in RSI14: {rsi.isna().sum()}")

if not rsi_valid.between(0, 100).all():
    raise ValueError("RSI14 contains values outside the valid range of 0 to 100.")