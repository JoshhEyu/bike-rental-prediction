import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Load final test predictions
results = pd.read_csv(
    REPORT_DIR / "final_test_predictions.csv"
)

actual = results["Actual"]
predicted = results["Predicted"]
residuals = results["Residual"]

sns.set_theme(style="whitegrid")

# 1. Actual vs Predicted
plt.figure(figsize=(9, 6))
sns.scatterplot(x=actual, y=predicted, alpha=0.5)

plt.plot(
    [actual.min(), actual.max()],
    [actual.min(), actual.max()],
    linestyle="--",
    color="red"
)

plt.xlabel("Actual Bike Rentals")
plt.ylabel("Predicted Bike Rentals")
plt.title("Actual vs Predicted Bike Rental Demand")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "final_actual_vs_predicted.png",
    dpi=300
)
plt.close()

# 2. Residual Distribution
plt.figure(figsize=(9, 6))
sns.histplot(residuals, kde=True)

plt.axvline(0, color="red", linestyle="--")

plt.xlabel("Prediction Error (Residual)")
plt.ylabel("Frequency")
plt.title("Distribution of Final Model Residuals")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "final_residual_distribution.png",
    dpi=300
)
plt.close()

# 3. Residuals vs Predicted
plt.figure(figsize=(9, 6))
sns.scatterplot(x=predicted, y=residuals, alpha=0.5)

plt.axhline(0, color="red", linestyle="--")

plt.xlabel("Predicted Bike Rentals")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Bike Rental Demand")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "final_residuals_vs_predicted.png",
    dpi=300
)
plt.close()

print("Final test visualizations generated successfully!")
print("1. final_actual_vs_predicted.png")
print("2. final_residual_distribution.png")
print("3. final_residuals_vs_predicted.png")