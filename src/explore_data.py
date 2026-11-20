
import pandas as pd
from pathlib import Path

# Project directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Load dataset
DATA_PATH = PROJECT_DIR / "data" / "raw" / "hour.csv"
df = pd.read_csv(DATA_PATH)

print("\n===== DUPLICATE RECORDS =====")
print("Duplicate rows:", df.duplicated().sum())

print("\n===== TARGET VARIABLE: CNT =====")
print(df["cnt"].describe())

print("\n===== RENTAL DEMAND BY HOUR =====")
print(df.groupby("hr")["cnt"].mean().round(2))

print("\n===== RENTAL DEMAND BY SEASON =====")
print(df.groupby("season")["cnt"].mean().round(2))

print("\n===== CORRELATION WITH RENTAL DEMAND =====")

# Exclude identifiers and variables that reveal the target
features = df.drop(
    columns=["instant", "dteday", "casual", "registered"]
)

correlations = (
    features.corr(numeric_only=True)["cnt"]
    .sort_values(ascending=False)
)

print(correlations)

print("\n===== EDA COMPLETED =====")
