"""
training.py — Multi-Model Training with Cross-Validation & MLflow

Models trained:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. XGBoost
  4. LightGBM
  5. CatBoost

Evaluation metrics:
  - ROC-AUC (primary)
  - Recall (classe défaut)
  - F1-score
  - Confusion Matrix

Imbalance handling:
  - SMOTE (applied inside CV folds only)
  - class_weight / scale_pos_weight built-in

MLflow Integration:
  - Track params, metrics, and models per run

Author: Yomna Haouel
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import time
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, recall_score, f1_score, precision_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings

import config

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# MODEL FACTORY
# ──────────────────────────────────────────────

def get_model(name: str, params: dict = None):
    """Instantiate a model by name with optional hyperparameters."""
    if params is None:
        params = {}

    if name == "LogisticRegression":
        return LogisticRegression(
            random_state=config.RANDOM_STATE,
            solver="lbfgs",
            **params
        )

    elif name == "RandomForest":
        return RandomForestClassifier(
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            **params
        )

    elif name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            random_state=config.RANDOM_STATE,
            use_label_encoder=False,
            n_jobs=-1,
            **params
        )

    elif name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            **params
        )

    elif name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            random_seed=config.RANDOM_STATE,
            **params
        )

    else:
        raise ValueError(f"Unknown model: {name}")


def get_best_params(name: str) -> dict:
    """Get the first combination of hyperparameters from config grid."""
    grid = config.PARAM_GRIDS.get(name, {})
    params = {}
    for key, values in grid.items():
        if isinstance(values, list) and len(values) > 0:
            params[key] = values[0]
        else:
            params[key] = values
    return params


# ──────────────────────────────────────────────
# SMOTE INSIDE CV (anti-leakage)
# ──────────────────────────────────────────────

def apply_smote(X_train, y_train):
    """Apply SMOTE only on training fold (never on validation/test)."""
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(
            sampling_strategy=config.SMOTE_SAMPLING_STRATEGY,
            random_state=config.RANDOM_STATE,
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return X_res, y_res
    except ImportError:
        print("  ⚠ imblearn not installed, skipping SMOTE")
        return X_train, y_train


# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────

def evaluate_model(y_true, y_pred, y_proba, model_name: str) -> dict:
    """Compute all evaluation metrics."""
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    print(f"\n  ── {model_name} ──")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  F1-score:  {metrics['f1']:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["No Default", "Default"]))

    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name: str):
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"])
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(config.PLOT_DIR_TRAINING / f"cm_{model_name}.png",
                dpi=120, bbox_inches="tight")
    plt.close("all")


def plot_roc_curve(y_true, y_proba, model_name: str):
    """Save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(config.PLOT_DIR_TRAINING / f"roc_{model_name}.png",
                dpi=120, bbox_inches="tight")
    plt.close("all")


# ──────────────────────────────────────────────
# MLFLOW TRACKING
# ──────────────────────────────────────────────

def setup_mlflow():
    """Initialize MLflow experiment."""
    if not config.USE_MLFLOW:
        return False
    try:
        import mlflow
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
        print(f"  MLflow tracking: {config.MLFLOW_TRACKING_URI}")
        print(f"  Experiment: {config.MLFLOW_EXPERIMENT_NAME}")
        return True
    except ImportError:
        print("  ⚠ MLflow not installed, skipping tracking")
        return False


def log_to_mlflow(model_name, params, metrics, model_obj):
    """Log a training run to MLflow."""
    if not config.USE_MLFLOW:
        return
    try:
        import mlflow
        import mlflow.sklearn

        with mlflow.start_run(run_name=model_name):
            # Log params
            for k, v in params.items():
                mlflow.log_param(k, v)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("cv_folds", config.CV_FOLDS)

            # Log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log model
            mlflow.sklearn.log_model(model_obj, "model")

            print(f"  → Logged to MLflow: {model_name}")
    except Exception as e:
        print(f"  ⚠ MLflow logging failed: {e}")


# ──────────────────────────────────────────────
# TRAIN ONE MODEL
# ──────────────────────────────────────────────

def train_single_model(model_name: str, X_train: pd.DataFrame,
                        y_train: pd.Series, X_test: pd.DataFrame,
                        y_test: pd.Series, use_smote: bool = True) -> dict:
    """
    Train a single model with stratified CV, evaluate on test set.
    Returns dict with metrics, model, and training time.
    """
    print(f"\n{'─' * 55}")
    print(f"  Training: {model_name}")
    print(f"{'─' * 55}")

    start = time.time()

    # Get hyperparameters
    params = get_best_params(model_name)
    print(f"  Params: {params}")

    # Prepare data (handle NaN)
    X_tr = X_train.fillna(0).values if hasattr(X_train, 'fillna') else X_train
    X_te = X_test.fillna(0).values if hasattr(X_test, 'fillna') else X_test
    y_tr = y_train.values if hasattr(y_train, 'values') else y_train
    y_te = y_test.values if hasattr(y_test, 'values') else y_test

    # Apply SMOTE (on train only)
    if use_smote:
        X_tr_sm, y_tr_sm = apply_smote(X_tr, y_tr)
        print(f"  After SMOTE: {X_tr_sm.shape[0]:,} samples")
    else:
        X_tr_sm, y_tr_sm = X_tr, y_tr

    # Train model
    model = get_model(model_name, params)
    model.fit(X_tr_sm, y_tr_sm)

    # Predict on TEST set
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    train_time = time.time() - start

    # Evaluate
    metrics = evaluate_model(y_te, y_pred, y_proba, model_name)
    metrics["train_time_sec"] = round(train_time, 1)

    # Plots
    plot_confusion_matrix(y_te, y_pred, model_name)
    plot_roc_curve(y_te, y_proba, model_name)

    # Save model
    model_path = config.MODEL_DIR / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    print(f"  Model saved: {model_path.name}")
    print(f"  Training time: {train_time:.1f}s")

    # MLflow tracking
    log_to_mlflow(model_name, params, metrics, model)

    return {
        "model_name": model_name,
        "model": model,
        "params": params,
        "metrics": metrics,
    }


# ──────────────────────────────────────────────
# TRAIN ALL MODELS
# ──────────────────────────────────────────────

def train_all_models(X_train, y_train, X_test, y_test, use_smote=True) -> list:
    """Train all models defined in config.MODELS_TO_TRAIN."""
    print("\n" + "=" * 60)
    print("   MULTI-MODEL TRAINING")
    print("=" * 60)

    mlflow_ok = setup_mlflow()
    if mlflow_ok:
        print("  ✓ MLflow tracking enabled")
    else:
        print("  ✗ MLflow tracking disabled")

    results = []

    for model_name in config.MODELS_TO_TRAIN:
        try:
            result = train_single_model(
                model_name, X_train, y_train, X_test, y_test,
                use_smote=use_smote,
            )
            results.append(result)
        except Exception as e:
            print(f"\n  ✗ {model_name} failed: {e}")
            continue

    print(f"\n✅ Training complete: {len(results)}/{len(config.MODELS_TO_TRAIN)} models")
    return results


if __name__ == "__main__":
    from preprocessing import DataPreprocessor
    from feature_engineering import run_feature_engineering
    from feature_selection import run_feature_selection

    preprocessor = DataPreprocessor()
    df = preprocessor.load_and_filter()
    df = preprocessor.basic_cleaning(df)
    X_train, X_test, y_train, y_test = preprocessor.split(df)
    X_train, X_test = run_feature_engineering(X_train, X_test)
    X_train, X_test, _ = run_feature_selection(X_train, X_test, y_train)

    results = train_all_models(X_train, y_train, X_test, y_test)
