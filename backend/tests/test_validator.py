"""Temporary manual check for validate_stock_data() on data/raw/RELIANCE.csv."""

import pandas as pd

from backend.app.validate_data import validate_stock_data
df = pd.read_csv("data/raw/RELIANCE.csv")
validated_df = validate_stock_data(df)

print("Validation successful")
print(f"Number of rows: {len(validated_df)}")
print(validated_df.head(5))