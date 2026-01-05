# Retail stock demand forecast (Local ML)


This project trains and compares multiple ML algorithms on the **Retail stock data** using Python and scikit‑learn.


## Setup (VS Code)
```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

## End to End Flow
Load Retail Stock Data
        ↓
Preprocess features (impute + scale)
        ↓
Train multiple ML algorithms
        ↓
Cross-validate each model
        ↓
Compare RMSE scores
        ↓
Save best model + metrics locally

## Repository Structure
retail-stock-demand-forecast-ml/
│
├── README.md          → How to run the project
├── requirements.txt   → Python libraries needed
├── .gitignore         → Prevent large/temporary files from Git
│
├── configs/
│   └── default.yaml   → Central config (seed, CV, models)
│
├── data/
│   ├── raw/           → Original dataset (unchanged)
│   ├── processed/     → Future engineered datasets
│   └── README.md      → Dataset description
│
├── notebooks/
│   ├── 01_eda.ipynb   → Exploratory Data Analysis
│   └── 02_model_comparison.ipynb → Visual comparison
│
├── src/
│   ├── data/          → Data loading
│   ├── etl/           → Feature engineering
│   ├── models/        → Training & evaluation
│   └── utils/         → Reusable helpers
│
├── results/           → Outputs (metrics + best model)
│
└── tests/             → Unit tests

## Requirements.txt - which library exists
numpy           → Numerical operations
pandas          → DataFrames
scikit-learn    → Core ML framework
xgboost         → Gradient boosting (strong baseline)
lightgbm        → Fast GBM, handles non-linearities
catboost        → GBM with strong defaults
optuna          → Hyperparameter tuning (future use)
shap            → Model explainability
joblib          → Save/load trained models
pyyaml          → Read config YAML
matplotlib      → Plotting
seaborn         → Statistical plots
pytest          → Unit testing
pyarrow         → for supporting Parquet,as it is not a native Pandas format.

