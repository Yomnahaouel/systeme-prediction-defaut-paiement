"""
config.py — Central Configuration

Production-ready configuration for Credit Default Risk Prediction System.
All parameters are centralized here for reproducibility and easy tuning.

Author: Yomna Haouel
"""

from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
FEATURE_MATRIX_PATH = DATA_DIR / "feature_matrix.csv"

MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR_EDA = PLOT_DIR / "eda"
PLOT_DIR_TRAINING = PLOT_DIR / "training"
PLOT_DIR_COMPARISON = PLOT_DIR / "comparison"

for d in [MODEL_DIR, PLOT_DIR, PLOT_DIR_EDA, PLOT_DIR_TRAINING, PLOT_DIR_COMPARISON]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# DATA SETTINGS
# ─────────────────────────────────────────────

TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
SET_COL = "set"

TEST_SIZE = 0.2
RANDOM_STATE = 42

MISSING_THRESHOLD = 0.60      # Drop columns with > 60% missing


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

# Financial ratios: new_col -> (numerator, denominator)
FINANCIAL_RATIOS = {
    "CREDIT_INCOME_RATIO": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
    "ANNUITY_INCOME_RATIO": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
    "CREDIT_TERM": ("AMT_CREDIT", "AMT_ANNUITY"),
    "CREDIT_GOODS_RATIO": ("AMT_CREDIT", "AMT_GOODS_PRICE"),
    "INCOME_PER_PERSON": ("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
    "ANNUITY_CREDIT_RATIO": ("AMT_ANNUITY", "AMT_CREDIT"),
}

AGE_COLUMN = "DAYS_BIRTH"
EMPLOYMENT_COLUMN = "DAYS_EMPLOYED"


# ─────────────────────────────────────────────
# FEATURE SELECTION
# ─────────────────────────────────────────────

CORRELATION_THRESHOLD = 0.85
MI_TOP_K = 120
IMPORTANCE_THRESHOLD = 0.001
SELECTION_STRATEGY = "union"      # "union" or "intersection"


# ─────────────────────────────────────────────
# IMBALANCE HANDLING
# ─────────────────────────────────────────────

SMOTE_SAMPLING_STRATEGY = 0.5     # Ratio minoritaire / majoritaire après SMOTE


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────

CV_FOLDS = 5
SCORING_METRIC = "roc_auc"

MODELS_TO_TRAIN = [
    "LogisticRegression",
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]


# ─────────────────────────────────────────────
# HYPERPARAMETER GRIDS
# ─────────────────────────────────────────────

PARAM_GRIDS = {
    "LogisticRegression": {
        "C": [0.01, 0.1, 1.0],
        "penalty": ["l2"],
        "class_weight": ["balanced"],
        "max_iter": [1000],
    },
    "RandomForest": {
        "n_estimators": [200, 300],
        "max_depth": [12, 18],
        "min_samples_split": [5, 10],
        "class_weight": ["balanced"],
    },
    "XGBoost": {
        "n_estimators": [300, 500],
        "max_depth": [4, 6],
        "learning_rate": [0.03, 0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "scale_pos_weight": [10],
        "eval_metric": ["auc"],
        "verbosity": [0],
    },
    "LightGBM": {
        "n_estimators": [500, 800],
        "max_depth": [6, 8],
        "learning_rate": [0.03, 0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "is_unbalance": [True],
        "verbose": [-1],
    },
    "CatBoost": {
        "iterations": [500],
        "depth": [6, 8],
        "learning_rate": [0.03, 0.05],
        "auto_class_weights": ["Balanced"],
        "verbose": [0],
    },
}


# ─────────────────────────────────────────────
# MLFLOW TRACKING
# ─────────────────────────────────────────────

USE_MLFLOW = True
MLFLOW_EXPERIMENT_NAME = "credit_default_risk"
MLFLOW_TRACKING_URI = (BASE_DIR / "mlruns").as_uri()


# ─────────────────────────────────────────────
# FASTAPI
# ─────────────────────────────────────────────

API_HOST = "0.0.0.0"
API_PORT = 8000
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
SELECTED_FEATURES_PATH = MODEL_DIR / "selected_features.json"
