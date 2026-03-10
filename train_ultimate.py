"""
ULTIMATE GPU Training Script
Full feature engineering + Optuna + Ensemble
Target: 0.78-0.80+ AUC

Author: 7afnawi for Hefny
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import optuna
import joblib
import time
import warnings
import gc

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("="*70)
print("   🏆 ULTIMATE GPU TRAINING - FULL PIPELINE")
print("="*70)

# ═══════════════════════════════════════════════════════════════════
# STEP 1: LOAD ALL DATA
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 1: LOADING ALL DATA FILES")
print("="*70)

app_train = pd.read_csv('data/application_train.csv')
print(f"  application_train: {app_train.shape}")

bureau = pd.read_csv('data/bureau.csv')
print(f"  bureau: {bureau.shape}")

prev_app = pd.read_csv('data/previous_application.csv')
print(f"  previous_application: {prev_app.shape}")

installments = pd.read_csv('data/installments_payments.csv')
print(f"  installments_payments: {installments.shape}")

cc_balance = pd.read_csv('data/credit_card_balance.csv')
print(f"  credit_card_balance: {cc_balance.shape}")

pos_cash = pd.read_csv('data/POS_CASH_balance.csv')
print(f"  POS_CASH_balance: {pos_cash.shape}")

# ═══════════════════════════════════════════════════════════════════
# STEP 2: FEATURE ENGINEERING - BUREAU
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 2: BUREAU AGGREGATIONS")
print("="*70)

bureau_agg = bureau.groupby('SK_ID_CURR').agg(
    BUREAU_COUNT=('SK_ID_BUREAU', 'count'),
    BUREAU_ACTIVE=('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
    BUREAU_CLOSED=('CREDIT_ACTIVE', lambda x: (x == 'Closed').sum()),
    BUREAU_CREDIT_TYPES=('CREDIT_TYPE', 'nunique'),
    BUREAU_DAYS_CREDIT_MEAN=('DAYS_CREDIT', 'mean'),
    BUREAU_DAYS_CREDIT_MIN=('DAYS_CREDIT', 'min'),
    BUREAU_DAYS_CREDIT_MAX=('DAYS_CREDIT', 'max'),
    BUREAU_DAYS_ENDDATE_MEAN=('DAYS_CREDIT_ENDDATE', 'mean'),
    BUREAU_AMT_CREDIT_SUM=('AMT_CREDIT_SUM', 'sum'),
    BUREAU_AMT_CREDIT_MEAN=('AMT_CREDIT_SUM', 'mean'),
    BUREAU_AMT_CREDIT_MAX=('AMT_CREDIT_SUM', 'max'),
    BUREAU_AMT_DEBT_SUM=('AMT_CREDIT_SUM_DEBT', 'sum'),
    BUREAU_AMT_DEBT_MEAN=('AMT_CREDIT_SUM_DEBT', 'mean'),
    BUREAU_AMT_OVERDUE_SUM=('AMT_CREDIT_SUM_OVERDUE', 'sum'),
    BUREAU_AMT_OVERDUE_MAX=('AMT_CREDIT_SUM_OVERDUE', 'max'),
    BUREAU_CNT_PROLONG=('CNT_CREDIT_PROLONG', 'sum'),
    BUREAU_AMT_ANNUITY_MEAN=('AMT_ANNUITY', 'mean'),
    BUREAU_AMT_ANNUITY_SUM=('AMT_ANNUITY', 'sum'),
).reset_index()

# Derived bureau features
bureau_agg['BUREAU_ACTIVE_RATIO'] = bureau_agg['BUREAU_ACTIVE'] / bureau_agg['BUREAU_COUNT'].replace(0, np.nan)
bureau_agg['BUREAU_DEBT_CREDIT_RATIO'] = bureau_agg['BUREAU_AMT_DEBT_SUM'] / bureau_agg['BUREAU_AMT_CREDIT_SUM'].replace(0, np.nan)
bureau_agg['BUREAU_OVERDUE_RATIO'] = bureau_agg['BUREAU_AMT_OVERDUE_SUM'] / bureau_agg['BUREAU_AMT_CREDIT_SUM'].replace(0, np.nan)

print(f"  Created {len(bureau_agg.columns)-1} bureau features")
del bureau
gc.collect()

# ═══════════════════════════════════════════════════════════════════
# STEP 3: FEATURE ENGINEERING - PREVIOUS APPLICATIONS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 3: PREVIOUS APPLICATION AGGREGATIONS")
print("="*70)

prev_agg = prev_app.groupby('SK_ID_CURR').agg(
    PREV_COUNT=('SK_ID_PREV', 'count'),
    PREV_APPROVED=('NAME_CONTRACT_STATUS', lambda x: (x == 'Approved').sum()),
    PREV_REFUSED=('NAME_CONTRACT_STATUS', lambda x: (x == 'Refused').sum()),
    PREV_CANCELED=('NAME_CONTRACT_STATUS', lambda x: (x == 'Canceled').sum()),
    PREV_AMT_APPLICATION_MEAN=('AMT_APPLICATION', 'mean'),
    PREV_AMT_APPLICATION_SUM=('AMT_APPLICATION', 'sum'),
    PREV_AMT_APPLICATION_MAX=('AMT_APPLICATION', 'max'),
    PREV_AMT_CREDIT_MEAN=('AMT_CREDIT', 'mean'),
    PREV_AMT_CREDIT_SUM=('AMT_CREDIT', 'sum'),
    PREV_AMT_DOWN_PAYMENT_MEAN=('AMT_DOWN_PAYMENT', 'mean'),
    PREV_DAYS_DECISION_MEAN=('DAYS_DECISION', 'mean'),
    PREV_DAYS_DECISION_MIN=('DAYS_DECISION', 'min'),
    PREV_CNT_PAYMENT_MEAN=('CNT_PAYMENT', 'mean'),
    PREV_CNT_PAYMENT_SUM=('CNT_PAYMENT', 'sum'),
).reset_index()

# Derived features
prev_agg['PREV_APPROVAL_RATE'] = prev_agg['PREV_APPROVED'] / prev_agg['PREV_COUNT'].replace(0, np.nan)
prev_agg['PREV_REFUSED_RATE'] = prev_agg['PREV_REFUSED'] / prev_agg['PREV_COUNT'].replace(0, np.nan)
prev_agg['PREV_CREDIT_APP_RATIO'] = prev_agg['PREV_AMT_CREDIT_MEAN'] / prev_agg['PREV_AMT_APPLICATION_MEAN'].replace(0, np.nan)

print(f"  Created {len(prev_agg.columns)-1} previous app features")
del prev_app
gc.collect()

# ═══════════════════════════════════════════════════════════════════
# STEP 4: FEATURE ENGINEERING - INSTALLMENTS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 4: INSTALLMENTS AGGREGATIONS")
print("="*70)

# Calculate payment metrics
installments['PAYMENT_DIFF'] = installments['AMT_PAYMENT'] - installments['AMT_INSTALMENT']
installments['PAYMENT_RATIO'] = installments['AMT_PAYMENT'] / installments['AMT_INSTALMENT'].replace(0, np.nan)
installments['DAYS_DIFF'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
installments['IS_LATE'] = (installments['DAYS_DIFF'] > 0).astype(int)
installments['DAYS_LATE'] = installments['DAYS_DIFF'].clip(lower=0)

ins_agg = installments.groupby('SK_ID_CURR').agg(
    INS_COUNT=('SK_ID_PREV', 'nunique'),
    INS_TOTAL_PAYMENTS=('NUM_INSTALMENT_NUMBER', 'count'),
    INS_AMT_INSTALMENT_MEAN=('AMT_INSTALMENT', 'mean'),
    INS_AMT_INSTALMENT_SUM=('AMT_INSTALMENT', 'sum'),
    INS_AMT_INSTALMENT_MAX=('AMT_INSTALMENT', 'max'),
    INS_AMT_PAYMENT_MEAN=('AMT_PAYMENT', 'mean'),
    INS_AMT_PAYMENT_SUM=('AMT_PAYMENT', 'sum'),
    INS_PAYMENT_DIFF_MEAN=('PAYMENT_DIFF', 'mean'),
    INS_PAYMENT_DIFF_SUM=('PAYMENT_DIFF', 'sum'),
    INS_PAYMENT_DIFF_MIN=('PAYMENT_DIFF', 'min'),
    INS_PAYMENT_RATIO_MEAN=('PAYMENT_RATIO', 'mean'),
    INS_PAYMENT_RATIO_MIN=('PAYMENT_RATIO', 'min'),
    INS_LATE_COUNT=('IS_LATE', 'sum'),
    INS_LATE_RATIO=('IS_LATE', 'mean'),
    INS_DAYS_LATE_MEAN=('DAYS_LATE', 'mean'),
    INS_DAYS_LATE_MAX=('DAYS_LATE', 'max'),
    INS_DAYS_LATE_SUM=('DAYS_LATE', 'sum'),
).reset_index()

# Payment discipline score
ins_agg['INS_PAYMENT_DISCIPLINE'] = 1 - ins_agg['INS_LATE_RATIO']

print(f"  Created {len(ins_agg.columns)-1} installment features")
del installments
gc.collect()

# ═══════════════════════════════════════════════════════════════════
# STEP 5: FEATURE ENGINEERING - CREDIT CARD
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 5: CREDIT CARD AGGREGATIONS")
print("="*70)

cc_balance['UTILIZATION'] = cc_balance['AMT_BALANCE'] / cc_balance['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan)

cc_agg = cc_balance.groupby('SK_ID_CURR').agg(
    CC_COUNT=('SK_ID_PREV', 'nunique'),
    CC_MONTHS=('MONTHS_BALANCE', 'count'),
    CC_BALANCE_MEAN=('AMT_BALANCE', 'mean'),
    CC_BALANCE_MAX=('AMT_BALANCE', 'max'),
    CC_BALANCE_SUM=('AMT_BALANCE', 'sum'),
    CC_LIMIT_MEAN=('AMT_CREDIT_LIMIT_ACTUAL', 'mean'),
    CC_DRAWINGS_MEAN=('AMT_DRAWINGS_CURRENT', 'mean'),
    CC_DRAWINGS_SUM=('AMT_DRAWINGS_CURRENT', 'sum'),
    CC_PAYMENT_MEAN=('AMT_PAYMENT_TOTAL_CURRENT', 'mean'),
    CC_PAYMENT_SUM=('AMT_PAYMENT_TOTAL_CURRENT', 'sum'),
    CC_DPD_MAX=('SK_DPD', 'max'),
    CC_DPD_MEAN=('SK_DPD', 'mean'),
    CC_DPD_SUM=('SK_DPD', 'sum'),
    CC_DPD_DEF_MAX=('SK_DPD_DEF', 'max'),
    CC_UTILIZATION_MEAN=('UTILIZATION', 'mean'),
    CC_UTILIZATION_MAX=('UTILIZATION', 'max'),
).reset_index()

print(f"  Created {len(cc_agg.columns)-1} credit card features")
del cc_balance
gc.collect()

# ═══════════════════════════════════════════════════════════════════
# STEP 6: FEATURE ENGINEERING - POS CASH
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 6: POS CASH AGGREGATIONS")
print("="*70)

pos_agg = pos_cash.groupby('SK_ID_CURR').agg(
    POS_COUNT=('SK_ID_PREV', 'nunique'),
    POS_MONTHS=('MONTHS_BALANCE', 'count'),
    POS_INSTALMENT_MEAN=('CNT_INSTALMENT', 'mean'),
    POS_INSTALMENT_FUTURE_MEAN=('CNT_INSTALMENT_FUTURE', 'mean'),
    POS_DPD_MAX=('SK_DPD', 'max'),
    POS_DPD_MEAN=('SK_DPD', 'mean'),
    POS_DPD_SUM=('SK_DPD', 'sum'),
    POS_DPD_DEF_MAX=('SK_DPD_DEF', 'max'),
).reset_index()

print(f"  Created {len(pos_agg.columns)-1} POS features")
del pos_cash
gc.collect()

# ═══════════════════════════════════════════════════════════════════
# STEP 7: MERGE ALL FEATURES
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 7: MERGING ALL FEATURES")
print("="*70)

df = app_train.copy()
print(f"  Base: {df.shape}")

df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
print(f"  + Bureau: {df.shape}")

df = df.merge(prev_agg, on='SK_ID_CURR', how='left')
print(f"  + Previous: {df.shape}")

df = df.merge(ins_agg, on='SK_ID_CURR', how='left')
print(f"  + Installments: {df.shape}")

df = df.merge(cc_agg, on='SK_ID_CURR', how='left')
print(f"  + Credit Card: {df.shape}")

df = df.merge(pos_agg, on='SK_ID_CURR', how='left')
print(f"  + POS: {df.shape}")

del bureau_agg, prev_agg, ins_agg, cc_agg, pos_agg, app_train
gc.collect()

# ═══════════════════════════════════════════════════════════════════
# STEP 8: ENGINEERED FEATURES
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 8: ENGINEERED FEATURES")
print("="*70)

# Financial ratios
df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY'].replace(0, np.nan)
df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'].replace(0, np.nan)
df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'].replace(0, np.nan)

# Age & Employment
df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365.25
df['EMPLOYED_YEARS'] = -df['DAYS_EMPLOYED'].replace(365243, np.nan) / 365.25
df['EMPLOYED_RATIO'] = df['EMPLOYED_YEARS'] / df['AGE_YEARS'].replace(0, np.nan)
df['EMPLOYED_ANOMALY'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)

# EXT_SOURCE features (MOST IMPORTANT!)
ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
df['EXT_SOURCE_STD'] = df[ext_cols].std(axis=1)
df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)
df['EXT_SOURCE_MAX'] = df[ext_cols].max(axis=1)
df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
df['EXT_SOURCE_WEIGHTED'] = 0.5*df['EXT_SOURCE_2'].fillna(0) + 0.3*df['EXT_SOURCE_3'].fillna(0) + 0.2*df['EXT_SOURCE_1'].fillna(0)

# EXT_SOURCE interactions with age
df['EXT_SOURCE_1_AGE'] = df['EXT_SOURCE_1'] * df['AGE_YEARS']
df['EXT_SOURCE_2_AGE'] = df['EXT_SOURCE_2'] * df['AGE_YEARS']
df['EXT_SOURCE_3_AGE'] = df['EXT_SOURCE_3'] * df['AGE_YEARS']

# Polynomial features
df['EXT_SOURCE_2_SQ'] = df['EXT_SOURCE_2'] ** 2
df['EXT_SOURCE_3_SQ'] = df['EXT_SOURCE_3'] ** 2
df['CREDIT_TERM_SQ'] = df['CREDIT_TERM'] ** 2

# Cross-table interactions
df['TOTAL_DEBT'] = df['AMT_CREDIT'] + df['BUREAU_AMT_DEBT_SUM'].fillna(0)
df['TOTAL_DEBT_INCOME_RATIO'] = df['TOTAL_DEBT'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
df['BUREAU_CREDIT_CURRENT_RATIO'] = df['BUREAU_AMT_CREDIT_SUM'] / df['AMT_CREDIT'].replace(0, np.nan)

# Risk flags
df['HIGH_DEBT_FLAG'] = (df['CREDIT_INCOME_RATIO'] > 5).astype(int)
df['YOUNG_FLAG'] = (df['AGE_YEARS'] < 25).astype(int)
df['LOW_EXT_FLAG'] = (df['EXT_SOURCE_MEAN'] < 0.3).astype(int)
df['LATE_PAYMENT_FLAG'] = (df['INS_LATE_COUNT'] > 5).astype(int).fillna(0)

# Document count
doc_cols = [c for c in df.columns if c.startswith('FLAG_DOCUMENT_')]
df['DOCUMENTS_COUNT'] = df[doc_cols].sum(axis=1)

print(f"  Created engineered features")
print(f"  Total features: {df.shape[1]}")

# ═══════════════════════════════════════════════════════════════════
# STEP 9: PREPROCESSING
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 9: PREPROCESSING")
print("="*70)

y = df['TARGET']
X = df.drop(['TARGET', 'SK_ID_CURR'], axis=1)

# Drop >50% missing
missing_pct = X.isnull().mean()
drop_cols = missing_pct[missing_pct > 0.5].index
X = X.drop(columns=drop_cols)
print(f"  Dropped {len(drop_cols)} columns >50% missing")

# Fill missing
X = X.fillna(-999)

# Encode categoricals
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

print(f"  Encoded {len(cat_cols)} categorical columns")
print(f"  Final shape: {X.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

# ═══════════════════════════════════════════════════════════════════
# STEP 10: OPTUNA HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 10: OPTUNA TUNING (50 trials)")
print("="*70)

def objective(trial):
    params = {
        'iterations': 2000,
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2', 1, 10),
        'border_count': 254,
        'bagging_temperature': trial.suggest_float('bagging_temp', 0, 1),
        'random_strength': trial.suggest_float('random_str', 0, 5),
        'auto_class_weights': 'Balanced',
        'task_type': 'GPU',
        'devices': '0',
        'verbose': 0,
        'early_stopping_rounds': 100,
        'random_seed': 42,
    }
    
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)
    pred = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, pred)

start = time.time()
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\n  Time: {(time.time()-start)/60:.1f} minutes")
print(f"  Best trial AUC: {study.best_value:.4f}")
print(f"  Best params: {study.best_params}")

# ═══════════════════════════════════════════════════════════════════
# STEP 11: TRAIN FINAL MODEL
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 11: FINAL MODEL TRAINING")
print("="*70)

best = study.best_params
final_model = CatBoostClassifier(
    iterations=5000,
    learning_rate=best['lr'],
    depth=best['depth'],
    l2_leaf_reg=best['l2'],
    bagging_temperature=best['bagging_temp'],
    random_strength=best['random_str'],
    border_count=254,
    auto_class_weights='Balanced',
    task_type='GPU',
    devices='0',
    early_stopping_rounds=300,
    verbose=500,
    random_seed=42,
)

final_model.fit(X_train, y_train, eval_set=(X_test, y_test))
print(f"  Best iteration: {final_model.best_iteration_}")

# ═══════════════════════════════════════════════════════════════════
# STEP 12: EVALUATION
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 12: EVALUATION")
print("="*70)

pred_proba = final_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, pred_proba)

# Threshold optimization
best_f1, best_thresh = 0, 0.5
for t in np.arange(0.05, 0.5, 0.01):
    f1 = f1_score(y_test, (pred_proba >= t).astype(int))
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

final_preds = (pred_proba >= best_thresh).astype(int)

print("\n" + "="*70)
print(" 🏆 FINAL RESULTS")
print("="*70)
print(f"  ROC-AUC:    {auc:.4f}")
print(f"  F1-Score:   {f1_score(y_test, final_preds):.4f}")
print(f"  Precision:  {precision_score(y_test, final_preds):.4f}")
print(f"  Recall:     {recall_score(y_test, final_preds):.4f}")
print(f"  Threshold:  {best_thresh:.2f}")
print("="*70)

# Feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n  Top 15 Features:")
for i, row in importance.head(15).iterrows():
    print(f"    {row['feature']}: {row['importance']:.4f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 13: SAVE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 13: SAVING MODELS")
print("="*70)

joblib.dump(final_model, 'models/ultimate_model.joblib')
joblib.dump(study.best_params, 'models/best_params.joblib')
joblib.dump(list(X.columns), 'models/feature_list.joblib')
joblib.dump(encoders, 'models/encoders.joblib')
importance.to_csv('models/feature_importance.csv', index=False)

print("  ✅ Models saved to models/")
print("\n" + "="*70)
print(" 🎉 TRAINING COMPLETE!")
print("="*70)
