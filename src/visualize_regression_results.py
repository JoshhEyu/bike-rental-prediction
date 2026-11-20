
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load validation data
X_val = pd.read_csv(DATA_DIR / "X_val.csv")
y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze()

sns.set_theme(style="whitegrid")

model_names = ["Univariate", "Bivariate", "Multivariate"]
predictions = {}

# Generate predictions
for name in model_names:
    saved = joblib.load(
        MODEL_DIR / f"{name.lower()}_regression.pkl"
    )

    model = saved["model"]
    features = saved["features"]

    predictions[name] = model.predict(X_val[features])

# 1. Actual vs predicted plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, name in zip(axes, model_names):
    ax.scatter(y_val, predictions[name], alpha=0.3)
    ax.plot(
        [y_val.min(), y_val.max()],
        [y_val.min(), y_val.max()],
        linestyle="--"
    )
    ax.set_title(f"{name} Regression")
    ax.set_xlabel("Actual Rentals")
    ax.set_ylabel("Predicted Rentals")

plt.suptitle("Actual vs Predicted Bike Rental Demand")
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "actual_vs_predicted.png",
    dpi=300
)
plt.close()

# 2. Residual distribution for multivariate model
residuals = y_val - predictions["Multivariate"]

plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=40, kde=True)
plt.title("Residual Distribution - Multivariate Regression")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "residual_distribution.png",
    dpi=300
)
plt.close()

# 3. Model metric comparison
results = pd.read_csv(
    PROJECT_DIR / "reports" / "regression_model_comparison.csv"
)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

metrics = ["MAE", "RMSE", "R2"]

for ax, metric in zip(axes, metrics):
    sns.barplot(
        data=results,
        x="Model",
        y=metric,
        ax=ax
    )
    ax.set_title(f"Model Comparison - {metric}")
    ax.tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "regression_metrics_comparison.png",
    dpi=300
)
plt.close()

print("Regression visualizations generated successfully!")

print("\nGenerated files:")
for file in [
    "actual_vs_predicted.png",
    "residual_distribution.png",
    "regression_metrics_comparison.png"
]:
    print(file)
