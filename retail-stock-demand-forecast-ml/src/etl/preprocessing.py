import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer


def build_preprocessor(df: pd.DataFrame):
    

    numeric_features = ['Inventory Level','Units Sold','Units Ordered','Price','Discount','Competitor Pricing']
    categorical_features = ['Store ID','Product ID','Category','Region','Weather Condition','Seasonality']
    passthrough_features = ['Promotion','Epidemic']
    drop_features = ['Date']

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])


    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        ))
    ])


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features),
            ('pass', 'passthrough', passthrough_features),
            ('drop', 'drop', drop_features)
        ],
        remainder='drop'   # Explicit is better than implicit
    )

    return preprocessor