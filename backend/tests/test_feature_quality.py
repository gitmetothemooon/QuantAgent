"""Temporary manual check for feature_quality.prepare_features() on RELIANCE data."""

from backend.app.data.load_data import load_stock_data
from backend.app.features.feature_engineering import generate_features
from backend.app.features.feature_quality import prepare_features
raw_df = load_stock_data("RELIANCE")
features_df = generate_features(raw_df)
cleaned_df = prepare_features(features_df)

print(f"Original number of rows: {len(raw_df)}")
print(f"Number of rows after feature generation: {len(features_df)}")
print(f"Number of rows after feature preparation: {len(cleaned_df)}")
print(f"Columns: {list(cleaned_df.columns)}")
print(cleaned_df.head(10))
print("Remaining NaN values:")
print(cleaned_df.isnull().sum())