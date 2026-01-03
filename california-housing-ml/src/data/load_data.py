from sklearn.datasets import fetch_california_housing
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def load_california():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.to_parquet(RAW_DIR / "california_housing.parquet")
    return df
