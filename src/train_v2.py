#!/usr/bin/env python3
"""
train_v2.py — Fixed Training Pipeline for Credit Default Prediction

Key improvements over v1:
  - NO SMOTE (was causing issues) — uses class_weight='balanced' only
  - Proper missing value handling (drop >50% missing, impute rest)
  - LabelEncoder for categoricals (saved for API use)
  - LightGBM feature importance for selection (top 100 features)
  - Better hyperparameters (deeper trees, more estimators)
  - Clean, production-ready code

Outputs:
  - models/best_model_v2.joblib
  - models/selected_features_v2.json
  - models/label_encoders.joblib

Author: Fixed by AI Agent
Date: 2026-03-05
"""

import sys
import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Data files (try v2 first, then regular)
DATA_FILES = [
    DATA_DIR / "feature_matrix_final.csv",
    DATA_DIR / "feature_matrix_v2.csv",
    DATA_DIR / "feature_matrix.csv",
]

TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MISSING_THRESHOLD = 0.50  # Drop columns with >50% missing
TOP_FEATURES = 100        # Select top 100 features via LightGBM importance


# ═══════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """Load feature matrix from available data file."""
    for fpath in DATA_FILES:
        if fpath.exists():
            print(f"📂 Loading data from: {fpath.name}")
            df = pd.read_csv(fpath)
            print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            return df
    raise FileNotFoundError(f"No data file found. Tried: {[f.name for f in DATA_FILES]}")


def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Preprocess the data:
    1. Separate target and features
    2. Drop columns with >50% missing
    3. Fill remaining NaN (median for numeric, mode for categorical)
    4. Encode categorical columns with LabelEncoder
    5. Stratified train/test split
    
    Returns: X_train, X_test, y_train, y_test, label_encoders, feature_names
    """
    print("\n🔧 Preprocessing data...")
    
    # Separate target
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in data")
    
    y = df[TARGET_COL].copy()
    X = df.drop(columns=[TARGET_COL])
    
    # Remove ID column if present
    if ID_COL in X.columns:
        X = X.drop(columns=[ID_COL])
        print(f"   Dropped ID column: {ID_COL}")
    
    # Step 1: Drop columns with >50% missing
    missing_pct = X.isnull().sum() / len(X)
    cols_to_drop = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    if cols_to_drop:
        print(f"   Dropping {len(cols_to_drop)} columns with >{MISSING_THRESHOLD*100:.0f}% missing")
        X = X.drop(columns=cols_to_drop)
    
    # Step 2: Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"   Numeric columns: {len(numeric_cols)}")
    print(f"   Categorical columns: {len(categorical_cols)}")
    
    # Step 3: Fill missing values
    # Numeric: median
    for col in numeric_cols:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
    
    # Categorical: mode
    for col in categorical_cols:
        if X[col].isnull().any():
            mode_val = X[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else "UNKNOWN"
            X[col] = X[col].fillna(fill_val)
    
    # Step 4: Encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Add "UNKNOWN" to handle unseen categories at inference time
        unique_vals = list(X[col].unique()) + ["UNKNOWN"]
        le.fit(unique_vals)
        X[col] = le.transform(X[col])
        label_encoders[col] = le
    
    if label_encoders:
        print(f"   Encoded {len(label_encoders)} categorical columns")
    
    # Step 5: Verify no missing values remain
    remaining_nan = X.isnull().sum().sum()
    if remaining_nan > 0:
        print(f"   ⚠ Warning: {remaining_nan} NaN values remain, filling with 0")
        X = X.fillna(0)
    
    # Step 6: Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"\n   Train set: {len(X_train):,} samples")
    print(f"   Test set:  {len(X_test):,} samples")
    print(f"   Class distribution (train): {y_train.value_counts().to_dict()}")
    
    feature_names = X.columns.tolist()
    
    return X_train, X_test, y_train, y_test, label_encoders, feature_names


# ═══════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════

def select_features_lgbm(X_train: pd.DataFrame, y_train: pd.Series, 
                         X_test: pd.DataFrame, top_k: int = 100) -> tuple:
    """
    Select top K features using LightGBM feature importance.
    Returns: X_train_selected, X_test_selected, selected_features
    """
    print(f"\n🎯 Feature Selection (LightGBM importance, top {top_k})...")
    
    from lightgbm import LGBMClassifier
    
    # Train LightGBM for feature importance
    lgbm = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        is_unbalance=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    lgbm.fit(X_train, y_train)
    
    # Get feature importances
    importances = pd.Series(lgbm.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)
    
    # Select top K
    selected_features = importances.head(top_k).index.tolist()
    
    print(f"   Selected {len(selected_features)} features")
    print(f"   Top 10: {selected_features[:10]}")
    
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]
    
    return X_train_sel, X_test_sel, selected_features


# ═══════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════

def get_models() -> dict:
    """
    Return dictionary of models with optimized hyperparameters.
    All use class_weight='balanced' or equivalent — NO SMOTE.
    """
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    
    models = {
        "LogisticRegression": LogisticRegression(
            C=0.1,
            penalty="l2",
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        
        "XGBoost": XGBClassifier(
            n_estimators=800,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=10,  # Handle imbalance
            eval_metric="auc",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        ),
        
        "LightGBM": LGBMClassifier(
            n_estimators=1000,
            max_depth=10,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,  # Handle imbalance
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        ),
        
        "CatBoost": CatBoostClassifier(
            iterations=800,
            depth=8,
            learning_rate=0.03,
            auto_class_weights="Balanced",  # Handle imbalance
            random_seed=RANDOM_STATE,
            verbose=0
        )
    }
    
    return models


# ═══════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0)
    }
    
    return metrics, y_pred, y_proba


def train_all_models(X_train, X_test, y_train, y_test) -> list:
    """Train all models and return results."""
    print("\n" + "=" * 70)
    print("   🚀 TRAINING MODELS (class_weight='balanced', NO SMOTE)")
    print("=" * 70)
    
    models = get_models()
    results = []
    
    for name, model in models.items():
        print(f"\n▶ Training {name}...")
        
        try:
            import time
            start = time.time()
            
            model.fit(X_train, y_train)
            train_time = time.time() - start
            
            metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)
            metrics["Time (s)"] = round(train_time, 1)
            
            results.append({
                "name": name,
                "model": model,
                "metrics": metrics
            })
            
            print(f"   ✓ ROC-AUC: {metrics['ROC-AUC']:.4f} | "
                  f"Precision: {metrics['Precision']:.4f} | "
                  f"Recall: {metrics['Recall']:.4f} | "
                  f"F1: {metrics['F1']:.4f} | "
                  f"Time: {train_time:.1f}s")
            
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            continue
    
    return results


def print_comparison_table(results: list):
    """Print a nice comparison table of all models."""
    print("\n" + "=" * 70)
    print("   📊 MODEL COMPARISON")
    print("=" * 70)
    
    # Create DataFrame for display
    rows = []
    for r in results:
        row = {"Model": r["name"]}
        row.update(r["metrics"])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by ROC-AUC descending
    df = df.sort_values("ROC-AUC", ascending=False)
    
    # Format for display
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
    
    # Best model
    best = df.iloc[0]
    print(f"\n🏆 Best Model: {best['Model']} (ROC-AUC: {best['ROC-AUC']:.4f})")
    
    return df


def save_outputs(results: list, selected_features: list, label_encoders: dict):
    """Save best model, selected features, and label encoders."""
    print("\n" + "=" * 70)
    print("   💾 SAVING OUTPUTS")
    print("=" * 70)
    
    # Find best model by ROC-AUC
    best_result = max(results, key=lambda x: x["metrics"]["ROC-AUC"])
    best_model = best_result["model"]
    best_name = best_result["name"]
    
    # Save best model
    model_path = MODEL_DIR / "best_model_v2.joblib"
    joblib.dump(best_model, model_path)
    print(f"   ✓ Best model saved: {model_path.name} ({best_name})")
    
    # Save selected features
    features_path = MODEL_DIR / "selected_features_v2.json"
    with open(features_path, "w") as f:
        json.dump(selected_features, f, indent=2)
    print(f"   ✓ Selected features saved: {features_path.name} ({len(selected_features)} features)")
    
    # Save label encoders (serialize classes_ as list)
    encoders_serializable = {}
    for col, le in label_encoders.items():
        encoders_serializable[col] = le.classes_.tolist()
    
    encoders_path = MODEL_DIR / "label_encoders.joblib"
    joblib.dump(label_encoders, encoders_path)
    print(f"   ✓ Label encoders saved: {encoders_path.name} ({len(label_encoders)} encoders)")
    
    # Also save as JSON for reference
    encoders_json_path = MODEL_DIR / "label_encoders.json"
    with open(encoders_json_path, "w") as f:
        json.dump(encoders_serializable, f, indent=2)
    print(f"   ✓ Label encoders (JSON): {encoders_json_path.name}")
    
    return best_name


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    """Main training pipeline."""
    print("\n" + "═" * 70)
    print("   CREDIT DEFAULT PREDICTION — TRAINING PIPELINE v2")
    print("   (Fixed: NO SMOTE, class_weight='balanced' only)")
    print("═" * 70)
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Preprocess
    X_train, X_test, y_train, y_test, label_encoders, all_features = preprocess_data(df)
    
    # Step 3: Feature selection
    X_train_sel, X_test_sel, selected_features = select_features_lgbm(
        X_train, y_train, X_test, top_k=TOP_FEATURES
    )
    
    # Step 4: Train all models
    results = train_all_models(X_train_sel, X_test_sel, y_train, y_test)
    
    if not results:
        print("\n❌ No models trained successfully!")
        return 1
    
    # Step 5: Print comparison
    comparison_df = print_comparison_table(results)
    
    # Step 6: Save outputs
    best_name = save_outputs(results, selected_features, label_encoders)
    
    print("\n" + "═" * 70)
    print(f"   ✅ TRAINING COMPLETE")
    print(f"   Best model: {best_name}")
    print(f"   Outputs in: {MODEL_DIR}")
    print("═" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
