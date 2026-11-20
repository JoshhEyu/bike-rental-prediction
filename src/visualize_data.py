
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Load dataset
DATA_PATH = PROJECT_DIR / "data" / "raw" / "hour.csv"
df = pd.read_csv(DATA_PATH)

# Create output directory
FIGURE_DIR = PROJECT_DIR / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Average rental demand by hour
hourly_demand = df.groupby("hr")["cnt"].mean()

plt.figure(figsize=(10, 5))
sns.lineplot(x=hourly_demand.index, y=hourly_demand.values, marker="o")
plt.title("Average Bike Rental Demand by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Number of Rentals")
plt.xticks(range(24))
plt.tight_layout()
plt.savefig(FIGURE_DIR / "hourly_demand.png", dpi=300)
plt.close()

# 2. Average rental demand by season
seasonal_demand = df.groupby("season")["cnt"].mean()

plt.figure(figsize=(8, 5))
sns.barplot(x=seasonal_demand.index, y=seasonal_demand.values)
plt.title("Average Bike Rental Demand by Season")
plt.xlabel("Season (1 = Spring, 2 = Summer, 3 = Fall, 4 = Winter)")
plt.ylabel("Average Number of Rentals")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "seasonal_demand.png", dpi=300)
plt.close()

# 3. Temperature versus rental demand
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="temp", y="cnt", alpha=0.25)
plt.title("Temperature vs. Bike Rental Demand")
plt.xlabel("Normalized Temperature")
plt.ylabel("Number of Rentals")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "temperature_vs_demand.png", dpi=300)
plt.close()

# 4. Correlation heatmap
features = df.drop(
    columns=["instant", "dteday", "casual", "registered"]
)

plt.figure(figsize=(11, 8))
sns.heatmap(
    features.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Heatmap of Bike Sharing Features")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=300)
plt.close()

print("All visualizations generated successfully!")

print("\nSaved visualization files:")
for file in FIGURE_DIR.glob("*.png"):
    print(file.name)
