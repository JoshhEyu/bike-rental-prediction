
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

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

# Define advanced regression models
models = {
    "Decision Tree": DecisionTreeRegressor(
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

results = []

# Train and evaluate models
for model_name, model in models.items():

    print(f"\nTraining {model_name} Regressor...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

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

    joblib.dump(
        {
            "model": model,
            "features": list(X_train.columns)
        },
        MODEL_DIR / (
            model_name.lower().replace(" ", "_")
            + "_regression.pkl"
        )
    )

    print(f"{model_name} completed!")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

# Save comparison results
results_df = pd.DataFrame(results)

print("\n===== ADVANCED REGRESSION COMPARISON =====")
print(results_df.to_string(index=False))

results_df.to_csv(
    PROJECT_DIR / "reports" / "advanced_regression_comparison.csv",
    index=False
)

print("\nAll advanced models trained and saved successfully!")
