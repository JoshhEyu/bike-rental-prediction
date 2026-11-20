
import matplotlib.pyplot as plt
from pathlib import Path

# Tuned model validation results
models = ["Tuned Random Forest", "Tuned Gradient Boosting"]

mae = [95.7373, 74.5293]
rmse = [134.0099, 95.1655]
r2 = [0.6396, 0.8183]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = [
    ("MAE", mae),
    ("RMSE", rmse),
    ("R² Score", r2)
]

for ax, (title, values) in zip(axes, metrics):
    bars = ax.bar(models, values)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "Tuned Regression Models – Validation Performance",
    fontsize=15
)

plt.tight_layout()

output_dir = Path(__file__).resolve().parent.parent / "reports" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "tuned_regression_comparison.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Figure saved successfully: {output_path}")
