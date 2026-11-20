
import pandas as pd
from pathlib import Path

# Locate the project directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Load the hourly dataset
DATA_PATH = PROJECT_DIR / "data" / "raw" / "hour.csv"

df = pd.read_csv(DATA_PATH)

print("\n===== DATASET OVERVIEW =====")
print("Shape:", df.shape)

print("\n===== FIRST 5 RECORDS =====")
print(df.head())

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())
