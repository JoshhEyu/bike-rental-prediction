
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load training and validation data
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_val = pd.read_csv(DATA_DIR / "X_val.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze()

# Time-aware cross-validation
time_cv = TimeSeriesSplit(n_splits=3)

# Models and parameter search spaces
searches = {
    "Random Forest": {
        "model": RandomForestRegressor(
            random_state=42,
            n_jobs=1
        ),
        "params": {
            "n_estimators": [100, 150, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 1.0]
        }
    },

    "Gradient Boosting": {
        "model": GradientBoostingRegressor(
            random_state=42
        ),
        "params": {
            "n_estimators": [100, 150, 200, 300],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [2, 3, 4],
            "min_samples_leaf": [2, 5, 10],
            "subsample": [0.8, 1.0],
            "max_features": [None, "sqrt"]
        }
    }
}

results = []

for name, config in searches.items():

    print(f"\nTuning {name}...")

    search = RandomizedSearchCV(
        estimator=config["model"],
        param_distributions=config["params"],
        n_iter=8,
        scoring="neg_root_mean_squared_error",
        cv=time_cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # Evaluate on validation data
    predictions = best_model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    mse = mean_squared_error(y_val, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, predictions)

    # Save tuned model
    filename = name.lower().replace(" ", "_") + "_tuned.pkl"

    joblib.dump(
        {
            "model": best_model,
            "features": list(X_train.columns)
        },
        MODEL_DIR / filename
    )

    results.append({
        "Model": name,
        "Best CV RMSE": -search.best_score_,
        "Validation MAE": mae,
        "Validation RMSE": rmse,
        "Validation R2": r2,
        "Best Parameters": str(search.best_params_)
    })

    print("\nBest parameters:")
    print(search.best_params_)

    print(f"Cross-validation RMSE: {-search.best_score_:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R2: {r2:.4f}")

# Save tuning results
results_df = pd.DataFrame(results)

results_df.to_csv(
    PROJECT_DIR / "reports" / "tuned_regression_comparison.csv",
    index=False
)

print("\n===== TUNED MODEL COMPARISON =====")
print(results_df.to_string(index=False))

print("\nHyperparameter tuning completed!")
