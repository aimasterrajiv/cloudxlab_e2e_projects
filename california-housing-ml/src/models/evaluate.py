"""
Evaluate the best trained model on a hold-out test set
and generate diagnostic plots.
"""

import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.data.load_data import load_california
from src.utils.seed import set_seed
import yaml

def evaluate(config_path) -> None:
    
    config = yaml.safe_load(open(config_path))
    set_seed(config["project"]["random_seed"])


    # Load test data saved during training
    X_test = pd.read_parquet("data/processed/X_test.parquet")
    y_test = pd.read_parquet("data/processed/y_test.parquet")["MedHouseVal"]


    # Load trained model
    model = joblib.load("results/best_model.joblib")

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Evaluation Metrics")
    print("------------------")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    # Save predictions
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    predictions_df = pd.DataFrame(
        {"actual": y_test.values, "predicted": y_pred}
    )
    predictions_df.to_csv(results_dir / "predictions.csv", index=False)

    # -------------------------
    # Diagnostic plots
    # -------------------------

    # 1️⃣ Actual vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--",
    )
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(results_dir / "actual_vs_predicted.png")
    plt.show()

    # 2️⃣ Residual plot
    residuals = y_test - y_pred

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, residuals, alpha=0.3)
    plt.axhline(0)
    plt.xlabel("Actual Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(results_dir / "residuals.png")
    plt.show()

    # 3️⃣ Residual distribution
    plt.figure(figsize=(6, 5))
    plt.hist(residuals, bins=40)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(results_dir / "residual_distribution.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    evaluate(args.config)
