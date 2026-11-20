
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

# Load validation data
X_val = pd.read_csv(DATA_DIR / "X_val.csv")
y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze()

# Load multivariate model
saved = joblib.load(MODEL_DIR / "multivariate_regression.pkl")

model = saved["model"]
features = saved["features"]

# Generate predictions
predictions = model.predict(X_val[features])

# Calculate residuals
residuals = y_val - predictions

print("===== RESIDUAL ANALYSIS =====")

print("\nResidual statistics:")
print(pd.Series(residuals).describe())

print("\nMean residual:", round(residuals.mean(), 4))
print("Mean absolute residual:", round(np.abs(residuals).mean(), 4))

print("\n===== PREDICTION PERFORMANCE =====")
print("MAE:", round(mean_absolute_error(y_val, predictions), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_val, predictions)), 4))
print("R2:", round(r2_score(y_val, predictions), 4))

print("\n===== SAMPLE PREDICTIONS =====")

comparison = pd.DataFrame({
    "Actual": y_val.iloc[:10].values,
    "Predicted": predictions[:10].round(2),
    "Residual": residuals.iloc[:10].round(2)
})

print(comparison.to_string(index=False))

print("\nResidual analysis completed!")
