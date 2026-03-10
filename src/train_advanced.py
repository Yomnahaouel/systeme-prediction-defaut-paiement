#!/usr/bin/env python3
"""
Advanced Training Pipeline with Model Stacking
==============================================
This script implements a sophisticated ensemble approach using:
- Level 1: LightGBM, XGBoost, CatBoost with 5-fold CV
- Level 2: Logistic Regression meta-model on OOF predictions

Author: Agent-AdvancedTraining
Date: 2026-03-05
"""

import os
import sys
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import joblib

warnings.filterwarnings('ignore')

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
MISSING_THRESHOLD = 0.40  # Drop columns with >40% missing
TEST_SIZE = 0.20
RANDOM_STATE = 42
N_TOP_FEATURES = 200
N_FOLDS = 5

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def print_metrics(y_true, y_pred, y_proba, model_name="Model"):
    """Print comprehensive metrics for a model."""
    auc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    
    return {'auc': auc, 'f1': f1, 'precision': precision, 'recall': recall}


# ============================================================================
# 1. DATA LOADING AND MERGING
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
        
        # Find the ID column
        id_col = None
        for col in ['SK_ID_CURR', 'sk_id_curr', 'id']:
            if col in df.columns:
                id_col = col
                break
        
        if id_col is None:
            print(f"  Warning: No ID column found in {name}, skipping...")
            continue
        
        # Get non-ID columns that aren't already in merged
        existing_cols = set(merged.columns)
        new_cols = [col for col in df.columns if col not in existing_cols or col == id_col]
        
        if len(new_cols) <= 1:  # Only ID column
            print(f"  No new columns to add from {name}")
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
    """Preprocess the data: handle missing values and encode categoricals."""
    print_header("2. PREPROCESSING")
    
    # Separate target and features
    if 'TARGET' not in df.columns:
        raise ValueError("TARGET column not found in data!")
    
    y = df['TARGET'].values
    X = df.drop(columns=['TARGET'])
    
    # Keep track of SK_ID_CURR if present (for reference, but don't use as feature)
    if 'SK_ID_CURR' in X.columns:
        ids = X['SK_ID_CURR'].values
        X = X.drop(columns=['SK_ID_CURR'])
    else:
        ids = None
    
    print(f"  Initial features: {X.shape[1]}")
    print(f"  Class distribution: {np.bincount(y.astype(int))} (0: no default, 1: default)")
    print(f"  Default rate: {y.mean()*100:.2f}%")
    
    # 2a. Drop columns with >40% missing
    missing_pct = X.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    X = X.drop(columns=cols_to_drop)
    print(f"  Dropped {len(cols_to_drop)} columns with >{MISSING_THRESHOLD*100:.0f}% missing")
    print(f"  Remaining features: {X.shape[1]}")
    
    # 2b. Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"  Numeric columns: {len(numeric_cols)}")
    print(f"  Categorical columns: {len(categorical_cols)}")
    
    # 2c. Fill numeric NaN with median
    for col in numeric_cols:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
    
    # 2d. Encode categoricals with LabelEncoder
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle NaN values
        X[col] = X[col].fillna('MISSING')
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Replace any remaining inf values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    print(f"  Preprocessing complete. Shape: {X.shape}")
    
    return X, y, label_encoders


# ============================================================================
# 3. TRAIN/TEST SPLIT
# ============================================================================

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


# ============================================================================
# 4. FEATURE SELECTION
# ============================================================================

def select_features(X_train, y_train, X_test, n_features=N_TOP_FEATURES):
    """Select top features using LightGBM feature importance."""
    print_header("4. FEATURE SELECTION")
    
    print(f"  Training LightGBM for feature importance...")
    
    # Train a quick LightGBM model
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
    
    # Get feature importances
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': selector_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select top N features
    n_features = min(n_features, len(importances))
    top_features = importances.head(n_features)['feature'].tolist()
    
    print(f"  Selected top {n_features} features")
    print(f"  Top 10 features:")
    for i, row in importances.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]
    
    return X_train_selected, X_test_selected, top_features


# ============================================================================
# 5. LEVEL 1 MODELS (OUT-OF-FOLD PREDICTIONS)
# ============================================================================

def train_level1_models(X_train, y_train, X_test, y_test):
    """Train Level 1 models with 5-fold CV and get OOF predictions."""
    print_header("5. LEVEL 1 MODELS (5-Fold CV)")
    
    # Define Level 1 models
    models = {
        'lgb': LGBMClassifier(
            n_estimators=1000, 
            learning_rate=0.02, 
            max_depth=8,
            num_leaves=64, 
            colsample_bytree=0.8, 
            subsample=0.8,
            class_weight='balanced', 
            n_jobs=-1,
            verbose=-1,
            random_state=RANDOM_STATE
        ),
        'xgb': XGBClassifier(
            n_estimators=800, 
            learning_rate=0.02, 
            max_depth=6,
            colsample_bytree=0.8, 
            subsample=0.8, 
            scale_pos_weight=11,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        'cat': CatBoostClassifier(
            iterations=1000, 
            learning_rate=0.02, 
            depth=6,
            auto_class_weights='Balanced', 
            verbose=0,
            random_state=RANDOM_STATE
        ),
    }
    
    # Initialize arrays for OOF and test predictions
    oof_preds = {name: np.zeros(len(X_train)) for name in models}
    test_preds = {name: np.zeros(len(X_test)) for name in models}
    trained_models = {name: [] for name in models}
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    for name, model_template in models.items():
        print(f"\n  Training {name.upper()}...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"    Fold {fold + 1}/{N_FOLDS}...", end=" ")
            
            # Split data for this fold
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train[val_idx]
            
            # Clone the model for this fold
            if name == 'lgb':
                model = LGBMClassifier(**model_template.get_params())
            elif name == 'xgb':
                model = XGBClassifier(**model_template.get_params())
            else:  # cat
                model = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=6, auto_class_weights="Balanced", verbose=0, thread_count=-1)
            
            # Train
            model.fit(X_fold_train, y_fold_train)
            
            # Get OOF predictions
            oof_preds[name][val_idx] = model.predict_proba(X_fold_val)[:, 1]
            
            # Get test predictions (average across folds)
            test_preds[name] += model.predict_proba(X_test)[:, 1] / N_FOLDS
            
            # Save trained model
            trained_models[name].append(model)
            
            # Print fold score
            fold_auc = roc_auc_score(y_fold_val, oof_preds[name][val_idx])
            print(f"AUC: {fold_auc:.4f}")
    
    # Evaluate individual models
    print("\n  Individual Model Scores (on validation data):")
    l1_results = {}
    for name in models:
        oof_auc = roc_auc_score(y_train, oof_preds[name])
        test_auc = roc_auc_score(y_test, test_preds[name])
        print(f"    {name.upper()}: OOF AUC = {oof_auc:.4f}, Test AUC = {test_auc:.4f}")
        l1_results[name] = {'oof_auc': oof_auc, 'test_auc': test_auc}
    
    return oof_preds, test_preds, trained_models, l1_results


# ============================================================================
# 6. LEVEL 2 STACKING
# ============================================================================

def train_level2_model(oof_preds, test_preds, y_train, y_test):
    """Train Level 2 meta-model using OOF predictions."""
    print_header("6. LEVEL 2 STACKING (Meta-Model)")
    
    # Stack OOF predictions as meta-features
    meta_train = np.column_stack([oof_preds[name] for name in oof_preds])
    meta_test = np.column_stack([test_preds[name] for name in test_preds])
    
    print(f"  Meta-features shape (train): {meta_train.shape}")
    print(f"  Meta-features shape (test):  {meta_test.shape}")
    
    # Train meta-model
    meta_model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    
    meta_model.fit(meta_train, y_train)
    
    # Get stacked predictions
    stacked_oof_proba = meta_model.predict_proba(meta_train)[:, 1]
    stacked_test_proba = meta_model.predict_proba(meta_test)[:, 1]
    stacked_test_pred = meta_model.predict(meta_test)
    
    # Print coefficients (model weights)
    print(f"\n  Meta-model coefficients:")
    for i, name in enumerate(oof_preds.keys()):
        print(f"    {name.upper()}: {meta_model.coef_[0][i]:.4f}")
    print(f"    Intercept: {meta_model.intercept_[0]:.4f}")
    
    return meta_model, stacked_oof_proba, stacked_test_proba, stacked_test_pred


# ============================================================================
# 7. EVALUATION
# ============================================================================

def evaluate_models(y_train, y_test, oof_preds, test_preds, 
                   stacked_oof_proba, stacked_test_proba, stacked_test_pred, l1_results):
    """Comprehensive evaluation of all models."""
    print_header("7. FINAL EVALUATION")
    
    results = {}
    
    # Individual model results on test set
    print("\n  INDIVIDUAL MODELS (Test Set):")
    print("-" * 50)
    
    best_individual_auc = 0
    best_individual_name = ""
    
    for name in test_preds:
        test_auc = roc_auc_score(y_test, test_preds[name])
        test_pred_binary = (test_preds[name] > 0.5).astype(int)
        test_f1 = f1_score(y_test, test_pred_binary)
        
        results[name] = {'auc': test_auc, 'f1': test_f1}
        print(f"  {name.upper():4s}: AUC = {test_auc:.4f}, F1 = {test_f1:.4f}")
        
        if test_auc > best_individual_auc:
            best_individual_auc = test_auc
            best_individual_name = name
    
    # Stacked model results
    print("\n  STACKED MODEL (Test Set):")
    print("-" * 50)
    
    stacked_auc = roc_auc_score(y_test, stacked_test_proba)
    stacked_f1 = f1_score(y_test, stacked_test_pred)
    stacked_precision = precision_score(y_test, stacked_test_pred)
    stacked_recall = recall_score(y_test, stacked_test_pred)
    
    results['stacked'] = {
        'auc': stacked_auc, 
        'f1': stacked_f1,
        'precision': stacked_precision,
        'recall': stacked_recall
    }
    
    print(f"  STACKED: AUC = {stacked_auc:.4f}, F1 = {stacked_f1:.4f}")
    print(f"           Precision = {stacked_precision:.4f}, Recall = {stacked_recall:.4f}")
    
    # Improvement analysis
    print("\n  IMPROVEMENT ANALYSIS:")
    print("-" * 50)
    
    auc_improvement = stacked_auc - best_individual_auc
    auc_improvement_pct = (auc_improvement / best_individual_auc) * 100
    
    print(f"  Best individual model: {best_individual_name.upper()} (AUC: {best_individual_auc:.4f})")
    print(f"  Stacked model: AUC = {stacked_auc:.4f}")
    print(f"  Improvement: {auc_improvement:+.4f} ({auc_improvement_pct:+.2f}%)")
    
    if auc_improvement > 0:
        print(f"\n  ✓ Stacking IMPROVED performance!")
    else:
        print(f"\n  ✗ Stacking did not improve over best individual model")
        print(f"    Consider: using more diverse models or tuning hyperparameters")
    
    # Confusion matrix for stacked model
    print("\n  STACKED MODEL CONFUSION MATRIX:")
    print("-" * 50)
    cm = confusion_matrix(y_test, stacked_test_pred)
    print(f"  TN: {cm[0][0]:6d}  FP: {cm[0][1]:6d}")
    print(f"  FN: {cm[1][0]:6d}  TP: {cm[1][1]:6d}")
    
    # Classification report
    print("\n  CLASSIFICATION REPORT:")
    print("-" * 50)
    print(classification_report(y_test, stacked_test_pred, target_names=['No Default', 'Default']))
    
    return results


# ============================================================================
# 8. SAVE MODELS
# ============================================================================

def save_models(trained_models, meta_model, top_features, label_encoders, results):
    """Save all trained models and metadata."""
    print_header("8. SAVING MODELS")
    
    # Create directory if needed
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save Level 1 models
    for name, fold_models in trained_models.items():
        for i, model in enumerate(fold_models):
            path = MODELS_DIR / f"{name}_fold{i+1}.pkl"
            joblib.dump(model, path)
        print(f"  Saved {name} models ({len(fold_models)} folds)")
    
    # Save meta-model
    meta_path = MODELS_DIR / "meta_model.pkl"
    joblib.dump(meta_model, meta_path)
    print(f"  Saved meta-model: {meta_path.name}")
    
    # Save feature list
    features_path = MODELS_DIR / "selected_features.pkl"
    joblib.dump(top_features, features_path)
    print(f"  Saved feature list: {features_path.name}")
    
    # Save label encoders
    encoders_path = MODELS_DIR / "label_encoders.pkl"
    joblib.dump(label_encoders, encoders_path)
    print(f"  Saved label encoders: {encoders_path.name}")
    
    # Save results summary
    results_path = MODELS_DIR / "training_results.pkl"
    joblib.dump(results, results_path)
    print(f"  Saved results: {results_path.name}")
    
    # Save a text summary
    summary_path = MODELS_DIR / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Advanced Training Pipeline Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of features: {len(top_features)}\n")
        f.write(f"Number of folds: {N_FOLDS}\n\n")
        
        f.write("Model Performance (Test AUC):\n")
        for name, metrics in results.items():
            f.write(f"  {name.upper()}: {metrics['auc']:.4f}\n")
        
        f.write(f"\nTop 20 Features:\n")
        for feat in top_features[:20]:
            f.write(f"  - {feat}\n")
    
    print(f"  Saved summary: {summary_path.name}")
    print(f"\n  All models saved to: {MODELS_DIR}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run the complete advanced training pipeline."""
    print("\n" + "=" * 70)
    print(" ADVANCED TRAINING PIPELINE WITH MODEL STACKING")
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
        
        # 5. Train Level 1 models
        oof_preds, test_preds, trained_models, l1_results = train_level1_models(
            X_train_sel, y_train, X_test_sel, y_test
        )
        
        # 6. Train Level 2 meta-model
        meta_model, stacked_oof, stacked_test_proba, stacked_test_pred = train_level2_model(
            oof_preds, test_preds, y_train, y_test
        )
        
        # 7. Evaluate
        results = evaluate_models(
            y_train, y_test, oof_preds, test_preds,
            stacked_oof, stacked_test_proba, stacked_test_pred, l1_results
        )
        
        # 8. Save models
        save_models(trained_models, meta_model, top_features, label_encoders, results)
        
        print("\n" + "=" * 70)
        print(" TRAINING COMPLETE!")
        print("=" * 70)
        print(f"\n  Best stacked AUC: {results['stacked']['auc']:.4f}")
        print(f"  Models saved to: {MODELS_DIR}")
        print("\n")
        
        return results
        
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
