#!/usr/bin/env python3
"""
Smart Tuning - Competition-Level Hyperparameter Configuration
==============================================================
Based on Home Credit Default Risk Kaggle winners (0.80+ AUC)

Key differences from standard approach:
1. More iterations (2000-5000) with lower learning rate (0.01-0.03)
2. Early stopping (200-500 patience) - CRITICAL
3. Proper regularization
4. Optimized ensemble weights via scipy.minimize
5. Business-cost-aware threshold optimization

Author: Agent-SmartTuning
Date: 2026-03-05
"""

import os
import sys
import warnings
from pathlib import Path
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    precision_recall_curve, classification_report, confusion_matrix
)
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - COMPETITION-LEVEL PARAMETERS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models" / "smart_tuning"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5
TEST_SIZE = 0.15  # Smaller test set, more training data
MISSING_THRESHOLD = 0.40

# Feature files to merge
FEATURE_FILES = {
    'base': DATA_DIR / "feature_matrix_final.csv",
    'time': DATA_DIR / "time_features.csv",
    'target_encoded': DATA_DIR / "target_encoded_features.csv",
    'interactions': DATA_DIR / "interaction_features.csv",
}

# ============================================================================
# COMPETITION-LEVEL HYPERPARAMETERS
# ============================================================================

# CatBoost - based on winning solutions
CATBOOST_PARAMS = {
    'iterations': 3000,  # Winners: 2000-5000
    'learning_rate': 0.02,  # Winners: 0.01-0.03
    'depth': 7,  # Winners: 6-8
    'l2_leaf_reg': 5,  # Winners: 3-10
    'border_count': 200,  # Winners: 128-254
    'random_strength': 0.5,
    'bagging_temperature': 0.8,
    'auto_class_weights': 'Balanced',
    'random_state': RANDOM_STATE,
    'thread_count': -1,
    'verbose': 0,
    # Early stopping configured during fit
}

# LightGBM - based on winning solutions
LIGHTGBM_PARAMS = {
    'n_estimators': 5000,  # Winners: 2000-10000
    'learning_rate': 0.015,  # Winners: 0.005-0.02
    'num_leaves': 40,  # Winners: 30-50 (conservative)
    'max_depth': -1,  # Winners: unlimited
    'min_data_in_leaf': 200,  # Winners: 100-300
    'feature_fraction': 0.7,  # Winners: 0.6-0.8
    'bagging_fraction': 0.7,  # Winners: 0.6-0.8
    'bagging_freq': 1,
    'lambda_l1': 0.5,  # Winners: 0.1-5
    'lambda_l2': 1.0,  # Winners: 0.1-5
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1,
    # Early stopping configured during fit
}

# Early stopping patience
EARLY_STOPPING_ROUNDS = 300  # Winners: 200-500


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(text, char="="):
    """Print a formatted header."""
    width = 80
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


def print_subheader(text):
    """Print a subheader."""
    print(f"\n>>> {text}")
    print("-" * 60)


# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_and_merge_features():
    """Load and merge all available feature files."""
    print_header("1. LOADING AND MERGING FEATURES")
    
    dfs = {}
    for name, path in FEATURE_FILES.items():
        if path.exists():
            print(f"  Loading {name}...")
            df = pd.read_csv(path)
            print(f"    Shape: {df.shape}")
            dfs[name] = df
        else:
            print(f"  Skipping {name}: File not found")
    
    if 'base' not in dfs:
        raise FileNotFoundError("Base feature matrix required!")
    
    merged = dfs['base'].copy()
    
    for name, df in dfs.items():
        if name == 'base':
            continue
        
        id_col = None
        for col in ['SK_ID_CURR', 'sk_id_curr']:
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
        print(f"  After merging {name}: {merged.shape}")
    
    print(f"\n  Final merged shape: {merged.shape}")
    return merged


# ============================================================================
# 2. PREPROCESSING
# ============================================================================

def preprocess_data(df):
    """Preprocess the data with competition-level handling."""
    print_header("2. PREPROCESSING")
    
    y = df['TARGET'].values
    X = df.drop(columns=['TARGET'])
    
    if 'SK_ID_CURR' in X.columns:
        X = X.drop(columns=['SK_ID_CURR'])
    
    print(f"  Initial features: {X.shape[1]}")
    print(f"  Target distribution: {np.bincount(y.astype(int))}")
    print(f"  Default rate: {y.mean()*100:.2f}%")
    
    # Drop high-missing columns
    missing_pct = X.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    X = X.drop(columns=cols_to_drop)
    print(f"  Dropped {len(cols_to_drop)} columns with >{MISSING_THRESHOLD*100:.0f}% missing")
    
    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"  Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")
    
    # Fill numeric with median
    for col in numeric_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Encode categoricals
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = X[col].fillna('MISSING')
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Handle inf values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"  Final shape: {X.shape}")
    return X, y, label_encoders, categorical_cols


# ============================================================================
# 3. FEATURE SELECTION
# ============================================================================

def select_features(X_train, y_train, X_test, n_features=200):
    """Select top features using a quick LightGBM."""
    print_header("3. FEATURE SELECTION")
    
    print("  Training quick LightGBM for feature importance...")
    
    selector = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        class_weight='balanced',
        n_jobs=-1,
        verbose=-1,
        random_state=RANDOM_STATE
    )
    selector.fit(X_train, y_train)
    
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': selector.feature_importances_
    }).sort_values('importance', ascending=False)
    
    n_features = min(n_features, len(importances))
    top_features = importances.head(n_features)['feature'].tolist()
    
    print(f"  Selected top {n_features} features")
    print(f"\n  Top 15 features:")
    for _, row in importances.head(15).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    return X_train[top_features], X_test[top_features], top_features


# ============================================================================
# 4. TRAIN MODELS WITH EARLY STOPPING (COMPETITION STYLE)
# ============================================================================

def train_catboost_with_early_stopping(X_train, y_train, X_test, cat_features=None):
    """Train CatBoost with proper early stopping (competition style)."""
    print_subheader("Training CatBoost (Competition-Level)")
    print(f"  Parameters: {CATBOOST_PARAMS['iterations']} iterations, lr={CATBOOST_PARAMS['learning_rate']}, depth={CATBOOST_PARAMS['depth']}")
    print(f"  Early stopping: {EARLY_STOPPING_ROUNDS} rounds")
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_aucs = []
    best_iterations = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train[val_idx]
        
        model = CatBoostClassifier(**CATBOOST_PARAMS)
        
        # Train with early stopping
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=(X_fold_val, y_fold_val),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            use_best_model=True,
            verbose=False
        )
        
        # Get predictions
        oof_preds[val_idx] = model.predict_proba(X_fold_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
        
        fold_auc = roc_auc_score(y_fold_val, oof_preds[val_idx])
        fold_aucs.append(fold_auc)
        best_iterations.append(model.best_iteration_)
        models.append(model)
        
        print(f"  Fold {fold+1}: AUC = {fold_auc:.5f}, best_iter = {model.best_iteration_}")
    
    oof_auc = roc_auc_score(y_train, oof_preds)
    print(f"\n  CatBoost OOF AUC: {oof_auc:.5f} (mean fold: {np.mean(fold_aucs):.5f} ± {np.std(fold_aucs):.5f})")
    print(f"  Avg best iteration: {np.mean(best_iterations):.0f}")
    
    return oof_preds, test_preds, models, oof_auc


def train_lightgbm_with_early_stopping(X_train, y_train, X_test):
    """Train LightGBM with proper early stopping (competition style)."""
    print_subheader("Training LightGBM (Competition-Level)")
    print(f"  Parameters: {LIGHTGBM_PARAMS['n_estimators']} estimators, lr={LIGHTGBM_PARAMS['learning_rate']}, leaves={LIGHTGBM_PARAMS['num_leaves']}")
    print(f"  Early stopping: {EARLY_STOPPING_ROUNDS} rounds")
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_aucs = []
    best_iterations = []
    models = []
    
    # Prepare params without n_estimators (we'll use callbacks)
    params = LIGHTGBM_PARAMS.copy()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train[val_idx]
        
        model = LGBMClassifier(**params)
        
        # Train with early stopping via callbacks
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            eval_metric='auc',
            callbacks=[
                # Early stopping
                __import__('lightgbm').early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                __import__('lightgbm').log_evaluation(period=0)  # Suppress logging
            ]
        )
        
        # Get predictions
        oof_preds[val_idx] = model.predict_proba(X_fold_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
        
        fold_auc = roc_auc_score(y_fold_val, oof_preds[val_idx])
        fold_aucs.append(fold_auc)
        best_iterations.append(model.best_iteration_)
        models.append(model)
        
        print(f"  Fold {fold+1}: AUC = {fold_auc:.5f}, best_iter = {model.best_iteration_}")
    
    oof_auc = roc_auc_score(y_train, oof_preds)
    print(f"\n  LightGBM OOF AUC: {oof_auc:.5f} (mean fold: {np.mean(fold_aucs):.5f} ± {np.std(fold_aucs):.5f})")
    print(f"  Avg best iteration: {np.mean(best_iterations):.0f}")
    
    return oof_preds, test_preds, models, oof_auc


# ============================================================================
# 5. ENSEMBLE WITH WEIGHT OPTIMIZATION
# ============================================================================

def optimize_ensemble_weights(y_true, predictions_dict):
    """Optimize ensemble weights using scipy.minimize (competition approach)."""
    print_subheader("Optimizing Ensemble Weights")
    
    model_names = list(predictions_dict.keys())
    preds_array = np.column_stack([predictions_dict[name] for name in model_names])
    
    def objective(weights):
        """Negative AUC to minimize."""
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        blended = preds_array @ weights
        return -roc_auc_score(y_true, blended)
    
    # Initial weights (equal)
    n_models = len(model_names)
    initial_weights = [1.0 / n_models] * n_models
    
    # Constraints: weights must be positive and sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    
    # Optimize
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    optimal_weights = result.x
    optimal_weights = optimal_weights / optimal_weights.sum()  # Re-normalize
    
    # Calculate AUCs
    equal_blend = preds_array @ np.array(initial_weights)
    equal_auc = roc_auc_score(y_true, equal_blend)
    
    optimal_blend = preds_array @ optimal_weights
    optimal_auc = roc_auc_score(y_true, optimal_blend)
    
    print(f"  Equal weights AUC:    {equal_auc:.5f}")
    print(f"  Optimized weights AUC: {optimal_auc:.5f}")
    print(f"  Improvement: {(optimal_auc - equal_auc)*10000:.2f} bps")
    
    print(f"\n  Optimal weights:")
    for name, weight in zip(model_names, optimal_weights):
        print(f"    {name}: {weight:.4f}")
    
    return dict(zip(model_names, optimal_weights)), optimal_blend


def apply_ensemble_weights(test_preds_dict, weights):
    """Apply optimized weights to test predictions."""
    model_names = list(weights.keys())
    preds_array = np.column_stack([test_preds_dict[name] for name in model_names])
    weight_array = np.array([weights[name] for name in model_names])
    return preds_array @ weight_array


# ============================================================================
# 6. OPTIMAL THRESHOLD FINDING
# ============================================================================

def find_optimal_threshold(y_true, y_proba, method='f1'):
    """Find optimal threshold using F1 or business cost metric."""
    print_subheader(f"Finding Optimal Threshold (Method: {method})")
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Calculate F1 for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    if method == 'f1':
        # Find threshold with best F1
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
    elif method == 'cost':
        # Business cost: FN costs 5x more than FP (missing a default is worse)
        # Cost = FP * cost_fp + FN * cost_fn
        cost_fp = 1
        cost_fn = 5
        
        best_cost = float('inf')
        best_threshold = 0.5
        
        for thresh in np.arange(0.1, 0.9, 0.01):
            y_pred = (y_proba >= thresh).astype(int)
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            cost = fp * cost_fp + fn * cost_fn
            
            if cost < best_cost:
                best_cost = cost
                best_threshold = thresh
    
    else:  # balanced
        # Find threshold where precision ≈ recall
        pr_diff = np.abs(precisions - recalls)
        best_idx = np.argmin(pr_diff)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    # Calculate metrics at optimal threshold
    y_pred_opt = (y_proba >= best_threshold).astype(int)
    f1_opt = f1_score(y_true, y_pred_opt)
    precision_opt = precision_score(y_true, y_pred_opt)
    recall_opt = recall_score(y_true, y_pred_opt)
    
    # Compare with default threshold
    y_pred_default = (y_proba >= 0.5).astype(int)
    f1_default = f1_score(y_true, y_pred_default)
    
    print(f"  Default threshold (0.50):")
    print(f"    F1 = {f1_default:.4f}")
    
    print(f"\n  Optimal threshold ({best_threshold:.3f}):")
    print(f"    F1 = {f1_opt:.4f}")
    print(f"    Precision = {precision_opt:.4f}")
    print(f"    Recall = {recall_opt:.4f}")
    
    print(f"\n  F1 improvement: {(f1_opt - f1_default)*100:.2f}%")
    
    return best_threshold, f1_opt


# ============================================================================
# 7. EVALUATION
# ============================================================================

def evaluate_final_model(y_true, y_proba, threshold, model_name="Ensemble"):
    """Comprehensive evaluation of the final model."""
    print_header(f"FINAL EVALUATION: {model_name}")
    
    auc = roc_auc_score(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    
    print(f"\n  ROC-AUC Score: {auc:.5f}")
    print(f"  Threshold: {threshold:.3f}")
    
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Default', 'Default']))
    
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"    TN: {cm[0][0]:6d}  FP: {cm[0][1]:6d}")
    print(f"    FN: {cm[1][0]:6d}  TP: {cm[1][1]:6d}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\n  Additional Metrics:")
    print(f"    Specificity (TNR): {specificity:.4f}")
    print(f"    NPV: {npv:.4f}")
    print(f"    False Positive Rate: {fp/(fp+tn):.4f}")
    print(f"    False Negative Rate: {fn/(fn+tp):.4f}")
    
    return {
        'auc': auc,
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'threshold': threshold
    }


# ============================================================================
# 8. SAVE RESULTS
# ============================================================================

def save_models_and_results(models_dict, weights, threshold, top_features, results):
    """Save trained models and results."""
    print_header("SAVING MODELS AND RESULTS")
    
    # Save models
    for name, models in models_dict.items():
        model_path = MODELS_DIR / f"{name}_models.joblib"
        joblib.dump(models, model_path)
        print(f"  Saved {name} models to {model_path}")
    
    # Save weights and threshold
    config = {
        'weights': weights,
        'threshold': threshold,
        'top_features': top_features,
        'params': {
            'catboost': CATBOOST_PARAMS,
            'lightgbm': LIGHTGBM_PARAMS
        },
        'results': results
    }
    
    config_path = MODELS_DIR / "ensemble_config.joblib"
    joblib.dump(config, config_path)
    print(f"  Saved ensemble config to {config_path}")
    
    # Save summary
    summary_path = MODELS_DIR / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SMART TUNING - COMPETITION-LEVEL TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("RESULTS:\n")
        for key, value in results.items():
            f.write(f"  {key}: {value}\n")
        
        f.write(f"\nENSEMBLE WEIGHTS:\n")
        for name, weight in weights.items():
            f.write(f"  {name}: {weight:.4f}\n")
        
        f.write(f"\nOPTIMAL THRESHOLD: {threshold:.3f}\n")
    
    print(f"  Saved summary to {summary_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline."""
    start_time = time.time()
    
    print("\n" + "=" * 80)
    print(" SMART TUNING - COMPETITION-LEVEL HYPERPARAMETERS")
    print(" Based on Home Credit Default Risk Winners (0.80+ AUC)")
    print("=" * 80)
    
    # 1. Load and merge data
    df = load_and_merge_features()
    
    # 2. Preprocess
    X, y, label_encoders, cat_features = preprocess_data(df)
    
    # 3. Train/Test split
    print_header("TRAIN/TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} samples ({y_train.mean()*100:.2f}% default)")
    print(f"  Test:  {X_test.shape[0]} samples ({y_test.mean()*100:.2f}% default)")
    
    # 4. Feature selection
    X_train_sel, X_test_sel, top_features = select_features(X_train, y_train, X_test, n_features=200)
    
    # 5. Train models with early stopping
    print_header("4. TRAINING MODELS (Competition-Level)")
    
    cat_oof, cat_test, cat_models, cat_auc = train_catboost_with_early_stopping(
        X_train_sel, y_train, X_test_sel
    )
    
    lgb_oof, lgb_test, lgb_models, lgb_auc = train_lightgbm_with_early_stopping(
        X_train_sel, y_train, X_test_sel
    )
    
    # 6. Optimize ensemble weights
    print_header("5. ENSEMBLE OPTIMIZATION")
    
    oof_preds = {
        'catboost': cat_oof,
        'lightgbm': lgb_oof
    }
    
    test_preds = {
        'catboost': cat_test,
        'lightgbm': lgb_test
    }
    
    weights, ensemble_oof = optimize_ensemble_weights(y_train, oof_preds)
    ensemble_test = apply_ensemble_weights(test_preds, weights)
    
    # 7. Find optimal threshold
    print_header("6. THRESHOLD OPTIMIZATION")
    
    # Try different methods
    threshold_f1, f1_at_threshold = find_optimal_threshold(y_train, ensemble_oof, method='f1')
    threshold_cost, _ = find_optimal_threshold(y_train, ensemble_oof, method='cost')
    
    # Use F1 threshold by default (more balanced)
    best_threshold = threshold_f1
    
    # 8. Final evaluation
    results = evaluate_final_model(y_test, ensemble_test, best_threshold)
    
    # Compare with individual models
    print_header("COMPARISON: Individual vs Ensemble")
    
    cat_test_auc = roc_auc_score(y_test, cat_test)
    lgb_test_auc = roc_auc_score(y_test, lgb_test)
    ensemble_auc = roc_auc_score(y_test, ensemble_test)
    
    print(f"\n  Test Set AUC:")
    print(f"    CatBoost:  {cat_test_auc:.5f}")
    print(f"    LightGBM:  {lgb_test_auc:.5f}")
    print(f"    Ensemble:  {ensemble_auc:.5f}")
    
    best_single = max(cat_test_auc, lgb_test_auc)
    improvement = (ensemble_auc - best_single) * 10000
    print(f"\n  Ensemble vs best single: {improvement:+.2f} bps")
    
    # 9. Save everything
    models_dict = {
        'catboost': cat_models,
        'lightgbm': lgb_models
    }
    
    save_models_and_results(models_dict, weights, best_threshold, top_features, results)
    
    # Summary
    elapsed = time.time() - start_time
    print_header("TRAINING COMPLETE")
    print(f"\n  Total time: {elapsed/60:.1f} minutes")
    print(f"\n  FINAL RESULTS:")
    print(f"    Test AUC:     {results['auc']:.5f}")
    print(f"    Test F1:      {results['f1']:.5f}")
    print(f"    Test Precision: {results['precision']:.5f}")
    print(f"    Test Recall:    {results['recall']:.5f}")
    print(f"    Threshold:    {results['threshold']:.3f}")
    
    print(f"\n  Models saved to: {MODELS_DIR}")
    
    return results


if __name__ == "__main__":
    main()
