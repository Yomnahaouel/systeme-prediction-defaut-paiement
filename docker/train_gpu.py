"""
GPU-Accelerated Training with Optuna Hyperparameter Tuning
Run this in Docker with: docker run --gpus all -v /path/to/data:/app/data gpu-training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from catboost import CatBoostClassifier
import optuna
import joblib
import warnings
import time

warnings.filterwarnings('ignore')

print("="*70)
print("   GPU-ACCELERATED TRAINING WITH OPTUNA")
print("="*70)

# Load FULL dataset (300K+ samples)
print("\n[1] Loading FULL dataset...")
df = pd.read_csv('data/application_train.csv')
print(f"  Loaded: {df.shape}")

# Quick preprocessing
y = df['TARGET']
X = df.drop(['TARGET', 'SK_ID_CURR'], axis=1)

# Drop high missing
missing_pct = X.isnull().mean()
X = X.drop(columns=missing_pct[missing_pct > 0.4].index)
print(f"  After dropping >40% missing: {X.shape[1]} features")

# Fill & encode
X = X.fillna(-999)
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype('category').cat.codes

print(f"  Final: {X.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# Optuna objective
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'auto_class_weights': 'Balanced',
        'task_type': 'GPU',  # USE GPU!
        'devices': '0',
        'verbose': 0,
        'random_seed': 42,
        'early_stopping_rounds': 100,
    }
    
    # 3-fold CV for speed
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))
    
    return np.mean(scores)

# Run Optuna
print("\n[2] Running Optuna (100 trials)...")
start = time.time()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

elapsed = time.time() - start
print(f"  Time: {elapsed/60:.1f} minutes")
print(f"  Best CV AUC: {study.best_value:.4f}")
print(f"  Best params: {study.best_params}")

# Train final model with best params
print("\n[3] Training final model...")
best_params = study.best_params
best_params['auto_class_weights'] = 'Balanced'
best_params['task_type'] = 'GPU'
best_params['devices'] = '0'
best_params['verbose'] = 500
best_params['random_seed'] = 42
best_params['early_stopping_rounds'] = 200

final_model = CatBoostClassifier(**best_params)
final_model.fit(X_train, y_train, eval_set=(X_test, y_test))

# Evaluate
preds_proba = final_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, preds_proba)

# Threshold optimization
best_f1, best_thresh = 0, 0.5
for t in np.arange(0.1, 0.5, 0.02):
    f1 = f1_score(y_test, (preds_proba >= t).astype(int))
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

final_preds = (preds_proba >= best_thresh).astype(int)

print("\n" + "="*70)
print("   🏆 FINAL RESULTS")
print("="*70)
print(f"  ROC-AUC:    {auc:.4f}")
print(f"  F1-Score:   {f1_score(y_test, final_preds):.4f}")
print(f"  Precision:  {precision_score(y_test, final_preds):.4f}")
print(f"  Recall:     {recall_score(y_test, final_preds):.4f}")
print(f"  Threshold:  {best_thresh:.2f}")
print("="*70)

# Save
joblib.dump(final_model, 'models/catboost_gpu_tuned.joblib')
joblib.dump(study.best_params, 'models/best_params.joblib')
print("\n✅ Models saved!")
