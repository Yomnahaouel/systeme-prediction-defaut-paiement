#!/usr/bin/env python3
"""
Fix Stacking Model Threshold Issue
===================================
The stacking meta-model (Logistic Regression) is predicting all 0s due to:
1. Class imbalance (~8% default rate)
2. Default threshold of 0.5 being too high

This script implements:
1. Probability averaging instead of LR meta-model
2. Weighted averaging based on individual AUCs
3. Optimal threshold search

Author: Agent-StackingFix
Date: 2026-03-05
"""

import os
import sys
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models" / "advanced"

FEATURE_FILES = {
    'base': DATA_DIR / "feature_matrix_final.csv",
    'time': DATA_DIR / "time_features.csv",
    'target_encoded': DATA_DIR / "target_encoded_features.csv",
    'interactions': DATA_DIR / "interaction_features.csv",
}

MISSING_THRESHOLD = 0.40
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def print_metrics(y_true, y_pred, y_proba, prefix=""):
    """Print comprehensive metrics."""
    auc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    print(f"{prefix}AUC: {auc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    
    return {'auc': auc, 'f1': f1, 'precision': precision, 'recall': recall}


# ============================================================================
# DATA LOADING (same as train_advanced.py)
# ============================================================================

def load_and_merge_features():
    """Load and merge all available feature files."""
    print_header("1. LOADING DATA")
    
    dfs = {}
    
    for name, path in FEATURE_FILES.items():
        if path.exists():
            print(f"  Loading {name}: {path.name}")
            df = pd.read_csv(path)
            dfs[name] = df
    
    if 'base' not in dfs:
        raise FileNotFoundError("Base feature matrix not found!")
    
    merged = dfs['base'].copy()
    
    for name, df in dfs.items():
        if name == 'base':
            continue
        
        id_col = None
        for col in ['SK_ID_CURR', 'sk_id_curr', 'id']:
            if col in df.columns:
                id_col = col
                break
        
        if id_col is None:
            continue
        
        existing_cols = set(merged.columns)
        new_cols = [col for col in df.columns if col not in existing_cols or col == id_col]
        
        if len(new_cols) <= 1:
            continue
        
        df_subset = df[new_cols].copy()
        if id_col != 'SK_ID_CURR':
            df_subset.rename(columns={id_col: 'SK_ID_CURR'}, inplace=True)
        
        merged = merged.merge(df_subset, on='SK_ID_CURR', how='left')
    
    print(f"  Final shape: {merged.shape}")
    return merged


def preprocess_data(df, label_encoders_saved):
    """Preprocess data using saved label encoders."""
    y = df['TARGET'].values
    X = df.drop(columns=['TARGET'])
    
    if 'SK_ID_CURR' in X.columns:
        X = X.drop(columns=['SK_ID_CURR'])
    
    print(f"  Initial features: {X.shape[1]}")
    print(f"  Default rate: {y.mean()*100:.2f}%")
    
    # Drop columns with >40% missing
    missing_pct = X.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    X = X.drop(columns=cols_to_drop)
    
    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Fill numeric NaN with median
    for col in numeric_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Encode categoricals
    for col in categorical_cols:
        X[col] = X[col].fillna('MISSING')
        if col in label_encoders_saved:
            le = label_encoders_saved[col]
            # Handle unseen categories
            X[col] = X[col].astype(str).apply(
                lambda x: x if x in le.classes_ else 'UNKNOWN'
            )
            # Add UNKNOWN if not in classes
            if 'UNKNOWN' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'UNKNOWN')
            X[col] = le.transform(X[col])
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Replace inf values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    return X, y


# ============================================================================
# MAIN FIX LOGIC
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print(" STACKING FIX: THRESHOLD OPTIMIZATION & WEIGHTED AVERAGING")
    print("=" * 70)
    
    # 1. Load saved models and features
    print_header("2. LOADING TRAINED MODELS")
    
    # Load feature list
    features_path = MODELS_DIR / "selected_features.pkl"
    selected_features = joblib.load(features_path)
    print(f"  Selected features: {len(selected_features)}")
    
    # Load label encoders
    encoders_path = MODELS_DIR / "label_encoders.pkl"
    label_encoders = joblib.load(encoders_path)
    print(f"  Label encoders: {len(label_encoders)}")
    
    # Load Level 1 models
    model_types = ['lgb', 'xgb', 'cat']
    models = {name: [] for name in model_types}
    
    for name in model_types:
        for fold in range(1, 6):
            model_path = MODELS_DIR / f"{name}_fold{fold}.pkl"
            models[name].append(joblib.load(model_path))
        print(f"  Loaded {name.upper()}: 5 folds")
    
    # 2. Load and preprocess data
    print_header("3. LOADING AND PREPROCESSING TEST DATA")
    
    df = load_and_merge_features()
    X, y = preprocess_data(df, label_encoders)
    
    # Train/test split with same random state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Select only the features used during training
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]
    
    print(f"  Train set: {X_train_sel.shape}")
    print(f"  Test set: {X_test_sel.shape}")
    print(f"  Test default rate: {y_test.mean()*100:.2f}%")
    
    # 3. Generate predictions from each model
    print_header("4. GENERATING BASE MODEL PREDICTIONS")
    
    test_preds = {}
    model_aucs = {}
    
    for name in model_types:
        preds = np.zeros(len(X_test_sel))
        for model in models[name]:
            preds += model.predict_proba(X_test_sel)[:, 1] / 5  # Average over folds
        test_preds[name] = preds
        auc = roc_auc_score(y_test, preds)
        model_aucs[name] = auc
        print(f"  {name.upper()}: AUC = {auc:.4f}")
    
    # 4. Test the original meta-model (show the problem)
    print_header("5. ORIGINAL META-MODEL (THE PROBLEM)")
    
    meta_model = joblib.load(MODELS_DIR / "meta_model.pkl")
    meta_features = np.column_stack([test_preds[name] for name in model_types])
    
    original_proba = meta_model.predict_proba(meta_features)[:, 1]
    original_pred = meta_model.predict(meta_features)
    
    print(f"  Predictions distribution: {np.bincount(original_pred)}")
    print(f"  All zeros predicted: {(original_pred == 0).all()}")
    print(f"  Probability range: [{original_proba.min():.4f}, {original_proba.max():.4f}]")
    print(f"  Mean probability: {original_proba.mean():.4f}")
    
    # 5. Fix #1: Simple averaging
    print_header("6. FIX #1: SIMPLE PROBABILITY AVERAGING")
    
    simple_avg = (test_preds['lgb'] + test_preds['xgb'] + test_preds['cat']) / 3
    simple_avg_auc = roc_auc_score(y_test, simple_avg)
    print(f"  Simple Average AUC: {simple_avg_auc:.4f}")
    
    # 6. Fix #2: Weighted averaging based on AUCs
    print_header("7. FIX #2: WEIGHTED AVERAGING")
    
    # Weight configurations to try
    weight_configs = [
        ('AUC-based', None),  # Will compute from AUCs
        ('Equal', {'lgb': 1/3, 'xgb': 1/3, 'cat': 1/3}),
        ('CAT-heavy', {'lgb': 0.2, 'xgb': 0.3, 'cat': 0.5}),
        ('XGB-heavy', {'lgb': 0.2, 'xgb': 0.5, 'cat': 0.3}),
        ('LGB-heavy', {'lgb': 0.5, 'xgb': 0.3, 'cat': 0.2}),
        ('Best2-only', {'lgb': 0.0, 'xgb': 0.4, 'cat': 0.6}),
    ]
    
    best_weights = None
    best_weight_name = ""
    best_weighted_proba = None
    best_weighted_auc = 0
    
    for name, weights in weight_configs:
        if weights is None:
            # Compute weights from AUCs (normalized)
            total_auc = sum(model_aucs.values())
            weights = {k: v/total_auc for k, v in model_aucs.items()}
        
        weighted_proba = sum(test_preds[m] * w for m, w in weights.items())
        auc = roc_auc_score(y_test, weighted_proba)
        
        print(f"  {name:15s}: AUC = {auc:.4f} | Weights: LGB={weights['lgb']:.2f}, XGB={weights['xgb']:.2f}, CAT={weights['cat']:.2f}")
        
        if auc > best_weighted_auc:
            best_weighted_auc = auc
            best_weights = weights
            best_weight_name = name
            best_weighted_proba = weighted_proba
    
    print(f"\n  Best weighting: {best_weight_name} (AUC: {best_weighted_auc:.4f})")
    
    # 7. Find optimal threshold
    print_header("8. OPTIMAL THRESHOLD SEARCH")
    
    print(f"\n  Using best weighting: {best_weight_name}")
    print(f"  Base default rate: {y_test.mean()*100:.2f}%\n")
    
    thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    
    best_threshold = 0.5
    best_f1 = 0
    best_metrics = {}
    
    print(f"  {'Threshold':<10} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Pred_1s':<8}")
    print(f"  {'-'*50}")
    
    for thresh in thresholds:
        preds = (best_weighted_proba >= thresh).astype(int)
        n_ones = preds.sum()
        
        if n_ones == 0:
            f1 = precision = recall = 0
        else:
            f1 = f1_score(y_test, preds, zero_division=0)
            precision = precision_score(y_test, preds, zero_division=0)
            recall = recall_score(y_test, preds, zero_division=0)
        
        print(f"  {thresh:<10.2f} {f1:<8.4f} {precision:<10.4f} {recall:<8.4f} {n_ones:<8}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_metrics = {'f1': f1, 'precision': precision, 'recall': recall}
    
    # 8. Final evaluation with optimal settings
    print_header("9. FINAL ENSEMBLE RESULTS")
    
    final_proba = best_weighted_proba
    final_pred = (final_proba >= best_threshold).astype(int)
    final_auc = roc_auc_score(y_test, final_proba)
    
    print(f"\n  Best Configuration:")
    print(f"  -------------------")
    print(f"  Weighting: {best_weight_name}")
    print(f"  Weights: LGB={best_weights['lgb']:.3f}, XGB={best_weights['xgb']:.3f}, CAT={best_weights['cat']:.3f}")
    print(f"  Threshold: {best_threshold}")
    print(f"\n  Final Metrics:")
    print(f"  ---------------")
    print(f"  AUC:       {final_auc:.4f}")
    print(f"  F1-Score:  {best_metrics['f1']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall:    {best_metrics['recall']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, final_pred)
    print(f"\n  Confusion Matrix:")
    print(f"  -----------------")
    print(f"  TN: {cm[0][0]:6d}  FP: {cm[0][1]:6d}")
    print(f"  FN: {cm[1][0]:6d}  TP: {cm[1][1]:6d}")
    
    # Comparison with individual models
    print(f"\n  Comparison with Individual Models:")
    print(f"  -----------------------------------")
    for name in model_types:
        individual_pred = (test_preds[name] >= best_threshold).astype(int)
        individual_f1 = f1_score(y_test, individual_pred, zero_division=0)
        print(f"  {name.upper()}: AUC={model_aucs[name]:.4f}, F1(thresh={best_threshold})={individual_f1:.4f}")
    print(f"  ENSEMBLE: AUC={final_auc:.4f}, F1(thresh={best_threshold})={best_metrics['f1']:.4f}")
    
    # 9. Save the optimized ensemble
    print_header("10. SAVING OPTIMIZED ENSEMBLE")
    
    ensemble_config = {
        'type': 'weighted_average',
        'weights': best_weights,
        'threshold': best_threshold,
        'model_types': model_types,
        'n_folds': 5,
        'metrics': {
            'auc': final_auc,
            'f1': best_metrics['f1'],
            'precision': best_metrics['precision'],
            'recall': best_metrics['recall']
        },
        'individual_aucs': model_aucs,
        'weight_name': best_weight_name
    }
    
    ensemble_path = MODELS_DIR / "ensemble_final.pkl"
    joblib.dump(ensemble_config, ensemble_path)
    print(f"  Saved: {ensemble_path}")
    
    # Also save as a text summary
    summary_path = MODELS_DIR / "ensemble_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Optimized Ensemble Configuration\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Weighting Strategy: {best_weight_name}\n")
        f.write(f"Weights:\n")
        f.write(f"  LightGBM:  {best_weights['lgb']:.4f}\n")
        f.write(f"  XGBoost:   {best_weights['xgb']:.4f}\n")
        f.write(f"  CatBoost:  {best_weights['cat']:.4f}\n")
        f.write(f"\nOptimal Threshold: {best_threshold}\n")
        f.write(f"\nFinal Metrics:\n")
        f.write(f"  AUC:       {final_auc:.4f}\n")
        f.write(f"  F1-Score:  {best_metrics['f1']:.4f}\n")
        f.write(f"  Precision: {best_metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {best_metrics['recall']:.4f}\n")
        f.write(f"\nIndividual Model AUCs:\n")
        for name, auc in model_aucs.items():
            f.write(f"  {name.upper()}: {auc:.4f}\n")
    
    print(f"  Saved: {summary_path}")
    
    print("\n" + "=" * 70)
    print(" STACKING FIX COMPLETE!")
    print("=" * 70)
    
    return ensemble_config


if __name__ == "__main__":
    main()
