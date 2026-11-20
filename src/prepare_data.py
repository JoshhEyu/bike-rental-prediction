
import pandas as pd
from pathlib import Path

# Project directories
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "raw" / "hour.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Convert date column
df["dteday"] = pd.to_datetime(df["dteday"])

# Extract additional date features
df["day"] = df["dteday"].dt.day

# Remove identifier, original date, and leakage columns
df = df.drop(
    columns=["instant", "dteday", "casual", "registered"]
)

# Define target
X = df.drop(columns=["cnt"])
y = df["cnt"]

# One-hot encode categorical features
categorical_features = [
    "season",
    "mnth",
    "hr",
    "weekday",
    "weathersit"
]

X = pd.get_dummies(
    X,
    columns=categorical_features,
    drop_first=True,
    dtype=int
)

# Chronological split: 70% training, 15% validation, 15% testing
total_rows = len(X)

train_end = int(total_rows * 0.70)
validation_end = int(total_rows * 0.85)

X_train = X.iloc[:train_end]
X_val = X.iloc[train_end:validation_end]
X_test = X.iloc[validation_end:]

y_train = y.iloc[:train_end]
y_val = y.iloc[train_end:validation_end]
y_test = y.iloc[validation_end:]

# Save prepared datasets
X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_val.to_csv(OUTPUT_DIR / "X_val.csv", index=False)
X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
y_val.to_csv(OUTPUT_DIR / "y_val.csv", index=False)
y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

# Display results
print("===== DATA PREPARATION COMPLETED =====")
print("Total records:", total_rows)
print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

print("\nNumber of features after encoding:", X.shape[1])

print("\nTraining period:")
print("First:", df.index[0], "| Last:", X_train.index[-1])

print("\nValidation period:")
print("First:", X_val.index[0], "| Last:", X_val.index[-1])

print("\nTesting period:")
print("First:", X_test.index[0], "| Last:", X_test.index[-1])

print("\nTarget variable:", y.name)
print("Data preparation successful!")
