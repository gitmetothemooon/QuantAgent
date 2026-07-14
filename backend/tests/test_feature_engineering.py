"""Temporary manual check for generate_features() on RELIANCE data."""

from backend.app.load_data import load_stock_data
from backend.app.feature_engineering import generate_features

df = load_stock_data("RELIANCE")
features_df = generate_features(df)

print(f"Number of rows: {len(features_df)}")
print(f"Columns: {list(features_df.columns)}")
print(features_df.head(10))