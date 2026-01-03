"""
Compare trained models using cross-validated metrics
and generate comparison plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def compare_results(metrics_path: str = "results/metrics.csv") -> None:
    metrics = pd.read_csv(metrics_path)

    metrics = metrics.sort_values("rmse")

    print("Model Comparison")
    print("----------------")
    print(metrics)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # 1️⃣ RMSE comparison (bar chart)
    plt.figure(figsize=(8, 5))
    plt.barh(metrics["model"], metrics["rmse"])
    plt.xlabel("Cross-Validated RMSE")
    plt.ylabel("Model")
    plt.title("Model Performance Comparison (Lower is Better)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(results_dir / "rmse_comparison.png")
    plt.show()

    # 2️⃣ RMSE spread visualization
    plt.figure(figsize=(7, 5))
    plt.plot(metrics["model"], metrics["rmse"], marker="o")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.title("RMSE Across Models")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(results_dir / "rmse_trend.png")
    plt.show()


if __name__ == "__main__":
    compare_results()
