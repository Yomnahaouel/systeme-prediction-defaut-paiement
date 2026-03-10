#!/usr/bin/env python3
"""
Optuna Hyperparameter Tuning for CatBoost
==========================================
This script uses Optuna to find optimal hyperparameters for CatBoost,
our best-performing model from the advanced training pipeline.

Author: Agent-Optuna
Date: 2026-03-05
"""

import os
import sys
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import joblib

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models" / "advanced"

# Feature files to merge
FEATURE_FILES = {
    'base': DATA_DIR / "feature_matrix_final.csv",
    'time': DATA_DIR / "time_features.csv",
    'target_encoded': DATA_DIR / "target_encoded_features.csv",
    'interactions': DATA_DIR / "interaction_features.csv",
}

# Preprocessing parameters
MISSING_THRESHOLD = 0.40
TEST_SIZE = 0.20
RANDOM_STATE = 42
N_TOP_FEATURES = 200

# Optuna parameters
N_TRIALS = 50
TIMEOUT_SECONDS = 600  # 10 minutes max

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


# ============================================================================
# DATA LOADING (Same as train_advanced.py)
# ============================================================================

def load_and_merge_features():
    """Load and merge all available feature files."""
    print_header("1. LOADING AND MERGING FEATURES")
    
    dfs = {}
    
    for name, path in FEATURE_FILES.items():
        if path.exists():
            print(f"  Loading {name}: {path.name}")
            df = pd.read_csv(path)
            print(f"    Shape: {df.shape}")
            dfs[name] = df
        else:
            print(f"  Skipping {name}: File not found ({path.name})")
    
    if 'base' not in dfs:
        raise FileNotFoundError("Base feature matrix (feature_matrix_final.csv) is required!")
    
    # Start with base features
    merged = dfs['base'].copy()
    print(f"\n  Base features shape: {merged.shape}")
    
    # Merge additional features on SK_ID_CURR
    for name, df in dfs.items():
        if name == 'base':
            continue
        
        id_col = None
        for col in ['SK_ID_CURR', 'sk_id_curr', 'id']:
            if col in df.columns:
                id_col = col
                break
        
        if id_col is None:
            print(f"  Warning: No ID column found in {name}, skipping...")
            continue
        
        existing_cols = set(merged.columns)
        new_cols = [col for col in df.columns if col not in existing_cols or col == id_col]
        
        if len(new_cols) <= 1:
            print(f"  No new columns to add from {name}")
            continue
        
        df_subset = df[new_cols].copy()
        if id_col != 'SK_ID_CURR':
            df_subset.rename(columns={id_col: 'SK_ID_CURR'}, inplace=True)
        
        merged = merged.merge(df_subset, on='SK_ID_CURR', how='left')
        print(f"  After merging {name}: {merged.shape}")
    
    print(f"\n  Final merged shape: {merged.shape}")
    return merged


def preprocess_data(df):
    """Preprocess the data: handle missing values and encode categoricals."""
    print_header("2. PREPROCESSING")
    
    if 'TARGET' not in df.columns:
        raise ValueError("TARGET column not found in data!")
    
    y = df['TARGET'].values
    X = df.drop(columns=['TARGET'])
    
    if 'SK_ID_CURR' in X.columns:
        X = X.drop(columns=['SK_ID_CURR'])
    
    print(f"  Initial features: {X.shape[1]}")
    print(f"  Class distribution: {np.bincount(y.astype(int))} (0: no default, 1: default)")
    print(f"  Default rate: {y.mean()*100:.2f}%")
    
    # Drop columns with >40% missing
    missing_pct = X.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    X = X.drop(columns=cols_to_drop)
    print(f"  Dropped {len(cols_to_drop)} columns with >{MISSING_THRESHOLD*100:.0f}% missing")
    print(f"  Remaining features: {X.shape[1]}")
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"  Numeric columns: {len(numeric_cols)}")
    print(f"  Categorical columns: {len(categorical_cols)}")
    
    # Fill numeric NaN with median
    for col in numeric_cols:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
    
    # Encode categoricals with LabelEncoder
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = X[col].fillna('MISSING')
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Replace inf values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    print(f"  Preprocessing complete. Shape: {X.shape}")
    
    return X, y, label_encoders


def split_data(X, y):
    """Perform stratified train/test split."""
    print_header("3. TRAIN/TEST SPLIT")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    
    print(f"  Train set: {X_train.shape[0]} samples ({y_train.mean()*100:.2f}% default)")
    print(f"  Test set:  {X_test.shape[0]} samples ({y_test.mean()*100:.2f}% default)")
    
    return X_train, X_test, y_train, y_test


def select_features(X_train, y_train, X_test, n_features=N_TOP_FEATURES):
    """Select top features using LightGBM feature importance."""
    print_header("4. FEATURE SELECTION")
    
    print(f"  Training LightGBM for feature importance...")
    
    selector_model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        class_weight='balanced',
        n_jobs=-1,
        verbose=-1,
        random_state=RANDOM_STATE
    )
    
    selector_model.fit(X_train, y_train)
    
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': selector_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    n_features = min(n_features, len(importances))
    top_features = importances.head(n_features)['feature'].tolist()
    
    print(f"  Selected top {n_features} features")
    
    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]
    
    return X_train_selected, X_test_selected, top_features


# ============================================================================
# OPTUNA HYPERPARAMETER TUNING
# ============================================================================

def create_objective(X, y):
    """Create the Optuna objective function for CatBoost tuning."""
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 1),
            'auto_class_weights': 'Balanced',
            'verbose': 0,
            'thread_count': -1,
            'random_state': RANDOM_STATE,
        }
        
        # 3-fold stratified CV for speed
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        
        model = CatBoostClassifier(**params)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
        
        return cv_scores.mean()
    
    return objective


def run_optuna_tuning(X_train, y_train):
    """Run Optuna hyperparameter tuning."""
    print_header("5. OPTUNA HYPERPARAMETER TUNING")
    
    print(f"  Starting Optuna optimization...")
    print(f"  Max trials: {N_TRIALS}")
    print(f"  Timeout: {TIMEOUT_SECONDS} seconds (10 minutes)")
    print(f"  Objective: Maximize ROC-AUC with 3-fold CV")
    print()
    
    # Create study
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name='catboost_tuning'
    )
    
    # Create objective function
    objective = create_objective(X_train, y_train)
    
    # Run optimization with progress callback
    def callback(study, trial):
        if trial.number % 5 == 0 or trial.number == 0:
            print(f"  Trial {trial.number:3d}: AUC = {trial.value:.4f} (Best so far: {study.best_value:.4f})")
    
    study.optimize(
        objective, 
        n_trials=N_TRIALS, 
        timeout=TIMEOUT_SECONDS,
        callbacks=[callback],
        show_progress_bar=False
    )
    
    print(f"\n  Optimization complete!")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Best CV AUC: {study.best_value:.4f}")
    
    return study


# ============================================================================
# TRAIN AND EVALUATE FINAL MODEL
# ============================================================================

def get_default_params():
    """Get default CatBoost parameters (from train_advanced.py)."""
    return {
        'iterations': 1000,
        'learning_rate': 0.02,
        'depth': 6,
        'auto_class_weights': 'Balanced',
        'verbose': 0,
        'thread_count': -1,
        'random_state': RANDOM_STATE,
    }


def train_and_compare(X_train, y_train, X_test, y_test, best_params):
    """Train models with default and tuned params, then compare."""
    print_header("6. TRAINING AND COMPARISON")
    
    # Prepare tuned params
    tuned_params = best_params.copy()
    tuned_params['auto_class_weights'] = 'Balanced'
    tuned_params['verbose'] = 0
    tuned_params['thread_count'] = -1
    tuned_params['random_state'] = RANDOM_STATE
    
    default_params = get_default_params()
    
    # Train default model
    print("  Training model with DEFAULT parameters...")
    default_model = CatBoostClassifier(**default_params)
    default_model.fit(X_train, y_train)
    
    default_train_proba = default_model.predict_proba(X_train)[:, 1]
    default_test_proba = default_model.predict_proba(X_test)[:, 1]
    default_test_pred = default_model.predict(X_test)
    
    default_train_auc = roc_auc_score(y_train, default_train_proba)
    default_test_auc = roc_auc_score(y_test, default_test_proba)
    default_f1 = f1_score(y_test, default_test_pred)
    
    print(f"    Train AUC: {default_train_auc:.4f}")
    print(f"    Test AUC:  {default_test_auc:.4f}")
    print(f"    Test F1:   {default_f1:.4f}")
    
    # Train tuned model
    print("\n  Training model with TUNED parameters...")
    tuned_model = CatBoostClassifier(**tuned_params)
    tuned_model.fit(X_train, y_train)
    
    tuned_train_proba = tuned_model.predict_proba(X_train)[:, 1]
    tuned_test_proba = tuned_model.predict_proba(X_test)[:, 1]
    tuned_test_pred = tuned_model.predict(X_test)
    
    tuned_train_auc = roc_auc_score(y_train, tuned_train_proba)
    tuned_test_auc = roc_auc_score(y_test, tuned_test_proba)
    tuned_f1 = f1_score(y_test, tuned_test_pred)
    tuned_precision = precision_score(y_test, tuned_test_pred)
    tuned_recall = recall_score(y_test, tuned_test_pred)
    
    print(f"    Train AUC:  {tuned_train_auc:.4f}")
    print(f"    Test AUC:   {tuned_test_auc:.4f}")
    print(f"    Test F1:    {tuned_f1:.4f}")
    print(f"    Precision:  {tuned_precision:.4f}")
    print(f"    Recall:     {tuned_recall:.4f}")
    
    results = {
        'default': {
            'train_auc': default_train_auc,
            'test_auc': default_test_auc,
            'f1': default_f1,
        },
        'tuned': {
            'train_auc': tuned_train_auc,
            'test_auc': tuned_test_auc,
            'f1': tuned_f1,
            'precision': tuned_precision,
            'recall': tuned_recall,
        }
    }
    
    return tuned_model, results


def print_final_report(study, results, best_params):
    """Print the final report."""
    print_header("7. FINAL REPORT")
    
    print("\n  BEST PARAMETERS FOUND:")
    print("-" * 50)
    for param, value in best_params.items():
        if isinstance(value, float):
            print(f"    {param}: {value:.6f}")
        else:
            print(f"    {param}: {value}")
    
    print("\n  PERFORMANCE COMPARISON:")
    print("-" * 50)
    print(f"  {'Metric':<15} {'Default':>12} {'Tuned':>12} {'Change':>12}")
    print("-" * 50)
    
    default_auc = results['default']['test_auc']
    tuned_auc = results['tuned']['test_auc']
    auc_change = tuned_auc - default_auc
    auc_pct = (auc_change / default_auc) * 100
    
    default_f1 = results['default']['f1']
    tuned_f1 = results['tuned']['f1']
    f1_change = tuned_f1 - default_f1
    f1_pct = (f1_change / default_f1) * 100 if default_f1 > 0 else 0
    
    print(f"  {'Test AUC':<15} {default_auc:>12.4f} {tuned_auc:>12.4f} {auc_change:>+12.4f} ({auc_pct:+.2f}%)")
    print(f"  {'Test F1':<15} {default_f1:>12.4f} {tuned_f1:>12.4f} {f1_change:>+12.4f} ({f1_pct:+.2f}%)")
    
    print("\n  OPTUNA STUDY STATISTICS:")
    print("-" * 50)
    print(f"    Total trials completed: {len(study.trials)}")
    print(f"    Best CV AUC (3-fold):   {study.best_value:.4f}")
    print(f"    Best trial number:      {study.best_trial.number}")
    
    print("\n  SUMMARY:")
    print("-" * 50)
    if auc_change > 0:
        print(f"  ✓ Hyperparameter tuning IMPROVED Test AUC by {auc_change:.4f} ({auc_pct:+.2f}%)")
    else:
        print(f"  ✗ Hyperparameter tuning did not improve Test AUC ({auc_change:+.4f})")
    
    print(f"\n  Final tuned model Test AUC: {tuned_auc:.4f}")


def save_tuned_model(model, study, best_params, results, top_features):
    """Save the tuned model and metadata."""
    print_header("8. SAVING TUNED MODEL")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the tuned model
    model_path = MODELS_DIR / "catboost_tuned.pkl"
    joblib.dump(model, model_path)
    print(f"  Saved tuned model: {model_path.name}")
    
    # Save tuning results
    tuning_results = {
        'best_params': best_params,
        'best_cv_auc': study.best_value,
        'results': results,
        'n_trials': len(study.trials),
        'feature_count': len(top_features),
    }
    
    results_path = MODELS_DIR / "optuna_tuning_results.pkl"
    joblib.dump(tuning_results, results_path)
    print(f"  Saved tuning results: {results_path.name}")
    
    # Save a text summary
    summary_path = MODELS_DIR / "optuna_tuning_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Optuna Hyperparameter Tuning Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Best Parameters:\n")
        for param, value in best_params.items():
            if isinstance(value, float):
                f.write(f"  {param}: {value:.6f}\n")
            else:
                f.write(f"  {param}: {value}\n")
        
        f.write(f"\nBest CV AUC (3-fold): {study.best_value:.4f}\n")
        f.write(f"Test AUC (default):   {results['default']['test_auc']:.4f}\n")
        f.write(f"Test AUC (tuned):     {results['tuned']['test_auc']:.4f}\n")
        
        improvement = results['tuned']['test_auc'] - results['default']['test_auc']
        f.write(f"Improvement:          {improvement:+.4f}\n")
        
        f.write(f"\nTotal trials: {len(study.trials)}\n")
    
    print(f"  Saved summary: {summary_path.name}")
    print(f"\n  All files saved to: {MODELS_DIR}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the complete Optuna tuning pipeline."""
    print("\n" + "=" * 70)
    print(" OPTUNA HYPERPARAMETER TUNING FOR CATBOOST")
    print("=" * 70)
    
    try:
        # 1. Load and merge features
        df = load_and_merge_features()
        
        # 2. Preprocess
        X, y, label_encoders = preprocess_data(df)
        
        # 3. Train/test split
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # 4. Feature selection
        X_train_sel, X_test_sel, top_features = select_features(X_train, y_train, X_test)
        
        # 5. Run Optuna tuning
        study = run_optuna_tuning(X_train_sel, y_train)
        best_params = study.best_params
        
        # 6. Train and compare models
        tuned_model, results = train_and_compare(
            X_train_sel, y_train, X_test_sel, y_test, best_params
        )
        
        # 7. Print final report
        print_final_report(study, results, best_params)
        
        # 8. Save model
        save_tuned_model(tuned_model, study, best_params, results, top_features)
        
        print("\n" + "=" * 70)
        print(" TUNING COMPLETE!")
        print("=" * 70)
        print(f"\n  Best tuned Test AUC: {results['tuned']['test_auc']:.4f}")
        print(f"  Model saved to: {MODELS_DIR / 'catboost_tuned.pkl'}")
        print("\n")
        
        return study, results
        
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
