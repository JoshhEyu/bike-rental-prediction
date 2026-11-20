
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.linear_model import LinearRegression

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

# Load datasets
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_val = pd.read_csv(DATA_DIR / "X_val.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

# Combine training and validation data
X_train_final = pd.concat(
    [X_train, X_val],
    ignore_index=True
)

y_train_final = pd.concat(
    [y_train, y_val],
    ignore_index=True
)

# Load selected tuned Gradient Boosting model
saved = joblib.load(
    MODEL_DIR / "gradient_boosting_tuned.pkl"
)

tuned_model = saved["model"]

# Create a fresh model using selected parameters
final_model = clone(tuned_model)

print("Training final Gradient Boosting model...")

final_model.fit(X_train_final, y_train_final)

# Predict on untouched test data
predictions = final_model.predict(X_test)

# Calculate final metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n===== FINAL TEST RESULTS =====")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# Save final model
joblib.dump(
    {
        "model": final_model,
        "features": list(X_train.columns)
    },
    MODEL_DIR / "final_bike_rental_model.pkl"
)

# Save test predictions
prediction_results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions,
    "Residual": y_test.values - predictions
})

prediction_results.to_csv(
    PROJECT_DIR / "reports" / "final_test_predictions.csv",
    index=False
)

# Save final metrics
metrics = pd.DataFrame([{
    "Model": "Tuned Gradient Boosting",
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2": r2
}])

metrics.to_csv(
    PROJECT_DIR / "reports" / "final_test_metrics.csv",
    index=False
)

print("\nFinal model saved successfully!")
print("Test predictions saved successfully!")
print("Final evaluation metrics saved successfully!")
