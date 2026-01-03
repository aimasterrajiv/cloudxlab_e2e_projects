import argparse
import yaml
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline


from src.data.load_data import load_california
from src.etl.preprocessing import build_preprocessor
from src.utils.seed import set_seed
from src.utils.metrics import rmse

MODEL_REGISTRY = {
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def train(config_path):


    config = yaml.safe_load(open(config_path))
    set_seed(config["project"]["random_seed"])


    df = load_california()
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=config["data"]["test_size"],
            random_state=config["project"]["random_seed"]
        )
    
    # 20% Test Data saved for future use
    X_test.to_parquet("data/processed/X_test.parquet")
    y_test.to_frame(name="MedHouseVal").to_parquet(
    "data/processed/y_test.parquet"
    )   

    results = []
    best_score = float("inf")
    best_model = None


    #for name, model in models.items():
    for model_name, model_cfg in config["models"].items():
        if not model_cfg.get("enabled", False):
            continue

        params = model_cfg.get("params", {})
        model_cls = MODEL_REGISTRY[model_name]        

        pipe = Pipeline(
            steps=[
            ("prep", build_preprocessor(df)),
            ("model", model_cls(**params))
            ]
        )

        print(f"Training model: {model_name}")

        scores = cross_val_score(pipe, X_trainval, y_trainval, cv=5, scoring="neg_root_mean_squared_error")
        rmse_score = -scores.mean()
        results.append({"model": model_name, "rmse": rmse_score})


        if rmse_score < best_score:
            best_score = rmse_score
            best_model = pipe


    best_model.fit(X_trainval, y_trainval)
    joblib.dump(best_model, "results/best_model.joblib")
    pd.DataFrame(results).to_csv("results/metrics.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)