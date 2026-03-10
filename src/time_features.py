#!/usr/bin/env python3
"""
Time-Windowed Feature Engineering
=================================
Creates features for multiple time windows (3, 6, 12, 24 months) from:
- installments_payments.csv
- credit_card_balance.csv  
- bureau.csv

This is the #1 technique from Kaggle winners - temporal patterns reveal default risk.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = 'data'
OUTPUT_FILE = f'{DATA_DIR}/time_features.csv'

print("=" * 60)
print("TIME-WINDOWED FEATURE ENGINEERING")
print("=" * 60)


def load_data():
    """Load source files."""
    print("\n📂 Loading data files...")
    
    installments = pd.read_csv(f'{DATA_DIR}/installments_payments.csv')
    print(f"   installments_payments: {len(installments):,} rows")
    
    cc_balance = pd.read_csv(f'{DATA_DIR}/credit_card_balance.csv')
    print(f"   credit_card_balance: {len(cc_balance):,} rows")
    
    bureau = pd.read_csv(f'{DATA_DIR}/bureau.csv')
    print(f"   bureau: {len(bureau):,} rows")
    
    return installments, cc_balance, bureau


def create_installment_features(df):
    """
    Create time-windowed features from installments_payments.
    Uses DAYS_INSTALMENT (negative days from application) to determine time window.
    """
    print("\n🔧 Creating INSTALLMENT time features...")
    
    df = df.copy()
    
    # Convert days to months (approximate: 30 days per month)
    df['MONTHS_AGO'] = (-df['DAYS_INSTALMENT'] / 30).clip(lower=0)
    
    # Calculate if payment was late (DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT means late)
    df['IS_LATE'] = (df['DAYS_ENTRY_PAYMENT'] > df['DAYS_INSTALMENT']).astype(int)
    df['DAYS_LATE'] = (df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']).clip(lower=0)
    
    # Payment ratio (how much of installment was paid)
    df['PAYMENT_RATIO'] = df['AMT_PAYMENT'] / df['AMT_INSTALMENT'].replace(0, np.nan)
    df['PAYMENT_RATIO'] = df['PAYMENT_RATIO'].clip(0, 2)  # Cap at 2x
    
    # Payment difference
    df['PAYMENT_DIFF'] = df['AMT_PAYMENT'] - df['AMT_INSTALMENT']
    
    time_windows = [3, 6, 12, 24]
    features_list = []
    
    for window in time_windows:
        # Filter to time window
        mask = df['MONTHS_AGO'] <= window
        window_df = df[mask].groupby('SK_ID_CURR')
        
        agg = window_df.agg({
            'IS_LATE': ['sum', 'mean'],
            'DAYS_LATE': ['sum', 'mean', 'max'],
            'PAYMENT_RATIO': ['mean', 'std', 'min'],
            'PAYMENT_DIFF': ['sum', 'mean'],
            'SK_ID_PREV': 'count'
        })
        
        # Flatten column names
        agg.columns = [f'INS_{col[0]}_{col[1].upper()}_{window}M' for col in agg.columns]
        agg = agg.rename(columns={
            f'INS_IS_LATE_SUM_{window}M': f'INS_LATE_COUNT_{window}M',
            f'INS_IS_LATE_MEAN_{window}M': f'INS_LATE_RATIO_{window}M',
            f'INS_SK_ID_PREV_COUNT_{window}M': f'INS_PAYMENT_COUNT_{window}M'
        })
        
        features_list.append(agg)
    
    # Merge all windows
    result = features_list[0]
    for feat_df in features_list[1:]:
        result = result.join(feat_df, how='outer')
    
    # Create TREND features (recent vs historical)
    if 'INS_PAYMENT_RATIO_MEAN_3M' in result.columns and 'INS_PAYMENT_RATIO_MEAN_12M' in result.columns:
        result['INS_PAYMENT_RATIO_TREND'] = result['INS_PAYMENT_RATIO_MEAN_3M'] - result['INS_PAYMENT_RATIO_MEAN_12M']
    
    if 'INS_LATE_RATIO_3M' in result.columns and 'INS_LATE_RATIO_12M' in result.columns:
        result['INS_LATE_TREND'] = result['INS_LATE_RATIO_3M'] - result['INS_LATE_RATIO_12M']
    
    result = result.reset_index()
    print(f"   Created {len(result.columns) - 1} installment features for {len(result):,} clients")
    
    return result


def create_credit_card_features(df):
    """
    Create time-windowed features from credit_card_balance.
    Uses MONTHS_BALANCE (negative months from now, -1 = last month).
    """
    print("\n💳 Creating CREDIT CARD time features...")
    
    df = df.copy()
    
    # Convert MONTHS_BALANCE to positive months ago
    df['MONTHS_AGO'] = -df['MONTHS_BALANCE']
    
    # Calculate utilization
    df['UTILIZATION'] = df['AMT_BALANCE'] / df['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan)
    df['UTILIZATION'] = df['UTILIZATION'].clip(0, 2)
    
    time_windows = [3, 6, 12, 24]
    features_list = []
    
    for window in time_windows:
        mask = df['MONTHS_AGO'] <= window
        window_df = df[mask].groupby('SK_ID_CURR')
        
        agg = window_df.agg({
            'AMT_BALANCE': ['mean', 'max', 'std'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['mean', 'max'],
            'UTILIZATION': ['mean', 'max', 'min'],
            'SK_DPD': ['sum', 'mean', 'max'],  # Days past due
            'SK_DPD_DEF': ['sum', 'max'],  # Days past due (defaulted)
            'AMT_DRAWINGS_CURRENT': ['sum', 'mean'],
            'AMT_PAYMENT_CURRENT': ['sum', 'mean'],
            'CNT_DRAWINGS_CURRENT': ['sum'],
            'MONTHS_BALANCE': 'count'
        })
        
        agg.columns = [f'CC_{col[0]}_{col[1].upper()}_{window}M' for col in agg.columns]
        agg = agg.rename(columns={
            f'CC_AMT_BALANCE_MEAN_{window}M': f'CC_BALANCE_{window}M',
            f'CC_UTILIZATION_MEAN_{window}M': f'CC_UTILIZATION_{window}M',
            f'CC_SK_DPD_SUM_{window}M': f'CC_DPD_{window}M',
            f'CC_MONTHS_BALANCE_COUNT_{window}M': f'CC_RECORD_COUNT_{window}M'
        })
        
        features_list.append(agg)
    
    # Merge all windows
    result = features_list[0]
    for feat_df in features_list[1:]:
        result = result.join(feat_df, how='outer')
    
    # Create TREND features
    if 'CC_BALANCE_3M' in result.columns and 'CC_BALANCE_12M' in result.columns:
        result['CC_BALANCE_TREND'] = result['CC_BALANCE_3M'] / result['CC_BALANCE_12M'].replace(0, np.nan)
        result['CC_BALANCE_TREND'] = result['CC_BALANCE_TREND'].clip(0, 10)
    
    if 'CC_UTILIZATION_3M' in result.columns and 'CC_UTILIZATION_12M' in result.columns:
        result['CC_UTILIZATION_TREND'] = result['CC_UTILIZATION_3M'] - result['CC_UTILIZATION_12M']
    
    if 'CC_DPD_3M' in result.columns and 'CC_DPD_12M' in result.columns:
        result['CC_DPD_TREND'] = result['CC_DPD_3M'] - result['CC_DPD_12M']
    
    result = result.reset_index()
    print(f"   Created {len(result.columns) - 1} credit card features for {len(result):,} clients")
    
    return result


def create_bureau_features(df):
    """
    Create time-windowed features from bureau.
    Uses DAYS_CREDIT (when credit was opened, negative days from application).
    """
    print("\n🏦 Creating BUREAU time features...")
    
    df = df.copy()
    
    # Convert days to months
    df['MONTHS_AGO'] = (-df['DAYS_CREDIT'] / 30).clip(lower=0)
    df['DAYS_CREDIT_UPDATE_MONTHS'] = (-df['DAYS_CREDIT_UPDATE'] / 30).clip(lower=0)
    
    # Active credit flag
    df['IS_ACTIVE'] = (df['CREDIT_ACTIVE'] == 'Active').astype(int)
    
    # Has overdue
    df['HAS_OVERDUE'] = (df['CREDIT_DAY_OVERDUE'] > 0).astype(int)
    
    time_windows = [3, 6, 12, 24]
    features_list = []
    
    for window in time_windows:
        # Credits opened in window
        mask_opened = df['MONTHS_AGO'] <= window
        opened_df = df[mask_opened].groupby('SK_ID_CURR')
        
        agg_opened = opened_df.agg({
            'SK_ID_BUREAU': 'count',
            'IS_ACTIVE': 'sum',
            'AMT_CREDIT_SUM': ['sum', 'mean'],
            'AMT_CREDIT_SUM_DEBT': ['sum', 'mean'],
            'AMT_CREDIT_SUM_OVERDUE': ['sum', 'max'],
            'CREDIT_DAY_OVERDUE': ['sum', 'max'],
            'HAS_OVERDUE': 'sum'
        })
        
        agg_opened.columns = [f'BUREAU_{col[0]}_{col[1].upper()}_{window}M' for col in agg_opened.columns]
        agg_opened = agg_opened.rename(columns={
            f'BUREAU_SK_ID_BUREAU_COUNT_{window}M': f'BUREAU_CREDITS_OPENED_{window}M',
            f'BUREAU_IS_ACTIVE_SUM_{window}M': f'BUREAU_ACTIVE_{window}M',
            f'BUREAU_HAS_OVERDUE_SUM_{window}M': f'BUREAU_OVERDUE_COUNT_{window}M'
        })
        
        # Recent updates (enquiries proxy)
        mask_updated = df['DAYS_CREDIT_UPDATE_MONTHS'] <= window
        updated_df = df[mask_updated].groupby('SK_ID_CURR')
        
        agg_updated = updated_df.agg({
            'SK_ID_BUREAU': 'count'
        })
        agg_updated.columns = [f'BUREAU_ENQUIRIES_{window}M']
        
        # Combine
        combined = agg_opened.join(agg_updated, how='outer')
        features_list.append(combined)
    
    # Merge all windows
    result = features_list[0]
    for feat_df in features_list[1:]:
        result = result.join(feat_df, how='outer')
    
    # Trend features
    if 'BUREAU_CREDITS_OPENED_3M' in result.columns and 'BUREAU_CREDITS_OPENED_12M' in result.columns:
        result['BUREAU_CREDIT_VELOCITY'] = result['BUREAU_CREDITS_OPENED_3M'] / result['BUREAU_CREDITS_OPENED_12M'].replace(0, np.nan)
    
    result = result.reset_index()
    print(f"   Created {len(result.columns) - 1} bureau features for {len(result):,} clients")
    
    return result


def main():
    # Load data
    installments, cc_balance, bureau = load_data()
    
    # Create features from each source
    ins_features = create_installment_features(installments)
    cc_features = create_credit_card_features(cc_balance)
    bureau_features = create_bureau_features(bureau)
    
    # Merge all features
    print("\n🔗 Merging all time features...")
    
    # Start with installment features
    result = ins_features
    
    # Merge credit card features
    result = result.merge(cc_features, on='SK_ID_CURR', how='outer')
    
    # Merge bureau features
    result = result.merge(bureau_features, on='SK_ID_CURR', how='outer')
    
    # Fill NaN with 0 for count features, leave others as NaN
    count_cols = [c for c in result.columns if 'COUNT' in c or 'CREDITS_OPENED' in c or 'ENQUIRIES' in c]
    result[count_cols] = result[count_cols].fillna(0)
    
    print(f"\n✅ Final dataset: {len(result):,} clients × {len(result.columns)} columns")
    
    # Save
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Saved to {OUTPUT_FILE}")
    
    # Summary
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)
    
    feature_groups = {
        'Installment (INS_)': [c for c in result.columns if c.startswith('INS_')],
        'Credit Card (CC_)': [c for c in result.columns if c.startswith('CC_')],
        'Bureau (BUREAU_)': [c for c in result.columns if c.startswith('BUREAU_')]
    }
    
    for group, cols in feature_groups.items():
        print(f"\n{group}: {len(cols)} features")
        # Show a few examples
        for col in cols[:5]:
            print(f"   • {col}")
        if len(cols) > 5:
            print(f"   ... and {len(cols) - 5} more")
    
    # Show trend features specifically
    trend_cols = [c for c in result.columns if 'TREND' in c or 'VELOCITY' in c]
    print(f"\n🎯 TREND FEATURES ({len(trend_cols)} total):")
    for col in trend_cols:
        print(f"   • {col}")
    
    # Sample
    print("\n📊 Sample (first 3 rows):")
    print(result.head(3).T)
    
    return result


if __name__ == '__main__':
    main()
