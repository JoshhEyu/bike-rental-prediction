
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Model evaluation results
models = ["Decision Tree", "Random Forest", "Gradient Boosting"]

mae = [129.6616, 103.2356, 108.0438]
rmse = [181.8981, 142.9614, 131.3270]
r2 = [0.3361, 0.5899, 0.6539]

# Create three comparison charts
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

metrics = [
    ("MAE", mae),
    ("RMSE", rmse),
    ("R² Score", r2)
]

for ax, (title, values) in zip(axes, metrics):
    bars = ax.bar(models, values)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=25)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "Advanced Regression Models – Validation Performance",
    fontsize=15
)

plt.tight_layout()

# Save the figure
output_dir = Path(__file__).resolve().parent.parent / "reports" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "advanced_regression_comparison.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Figure saved successfully: {output_path}")
