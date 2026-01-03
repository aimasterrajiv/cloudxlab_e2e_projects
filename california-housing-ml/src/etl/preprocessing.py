import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def build_preprocessor(df: pd.DataFrame):
    num_cols = df.drop(columns=["MedHouseVal"]).columns


    numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
    ])


    return ColumnTransformer([
    ("num", numeric_pipeline, num_cols)
    ])