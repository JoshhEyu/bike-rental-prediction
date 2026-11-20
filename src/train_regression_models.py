
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Project directories
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load prepared datasets
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_val = pd.read_csv(DATA_DIR / "X_val.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze()

# Define regression approaches
models = {
    "Univariate": ["temp"],
    "Bivariate": ["temp", "hum"],
    "Multivariate": list(X_train.columns)
}

results = []
trained_models = {}

# Train and evaluate each model
for model_name, features in models.items():

    print(f"\nTraining {model_name} Linear Regression...")

    model = LinearRegression()

    model.fit(X_train[features], y_train)

    predictions = model.predict(X_val[features])

    mae = mean_absolute_error(y_val, predictions)
    mse = mean_squared_error(y_val, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, predictions)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[model_name] = model

    # Save trained model
    joblib.dump(
        {
            "model": model,
            "features": features
        },
        MODEL_DIR / f"{model_name.lower()}_regression.pkl"
    )

    print(f"{model_name} training completed!")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

# Create comparison table
results_df = pd.DataFrame(results)

print("\n===== REGRESSION MODEL COMPARISON =====")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv(
    PROJECT_DIR / "reports" / "regression_model_comparison.csv",
    index=False
)

print("\nAll models trained and saved successfully!")
