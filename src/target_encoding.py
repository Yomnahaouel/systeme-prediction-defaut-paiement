#!/usr/bin/env python3
"""
Target Encoding for Categorical Variables
==========================================
Implements target encoding with smoothing and 5-fold cross-validation
to avoid target leakage.

Author: Agent-TargetEncoding
Date: 2026-03-05
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = 'data/feature_matrix_final.csv'
OUTPUT_PATH = 'data/target_encoded_features.csv'
ENCODER_PATH = 'models/target_encoders.json'

# Smoothing parameter (higher = more regularization toward global mean)
SMOOTHING = 10
N_FOLDS = 5
RANDOM_STATE = 42


def calculate_target_encoding(df, column, target_col='TARGET', smoothing=SMOOTHING):
    """
    Calculate smoothed target encoding for a categorical column.
    
    Smoothing formula:
    smooth_mean = (category_mean * count + global_mean * smoothing) / (count + smoothing)
    
    This pulls rare categories toward the global mean, reducing overfitting.
    """
    global_mean = df[target_col].mean()
    
    # Calculate stats per category
    agg = df.groupby(column)[target_col].agg(['mean', 'count'])
    
    # Apply smoothing
    smooth_mean = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)
    
    return smooth_mean.to_dict(), global_mean


def target_encode_cv(df, categorical_columns, target_col='TARGET', n_folds=N_FOLDS, smoothing=SMOOTHING):
    """
    Apply target encoding using K-fold cross-validation to prevent target leakage.
    
    For each fold:
    - Calculate encoding on the training portion
    - Apply encoding to the validation portion
    
    This ensures no data point sees its own target in the encoding calculation.
    """
    print(f"\n{'='*60}")
    print(f"TARGET ENCODING WITH {n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*60}")
    print(f"Smoothing parameter: {smoothing}")
    print(f"Number of categorical columns: {len(categorical_columns)}")
    
    # Initialize encoded columns
    encoded_df = pd.DataFrame(index=df.index)
    encoded_df['SK_ID_CURR'] = df['SK_ID_CURR']
    
    # Store encoder mappings (computed on full data for future predictions)
    encoder_mappings = {
        'global_mean': float(df[target_col].mean()),
        'smoothing': smoothing,
        'columns': {}
    }
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    for col in categorical_columns:
        col_name = f"{col}_TARGET_ENC"
        encoded_df[col_name] = np.nan
        
        # Get non-null indices for this column
        valid_mask = df[col].notna()
        valid_indices = df[valid_mask].index
        
        print(f"\nEncoding: {col}")
        print(f"  - Unique values: {df[col].nunique()}")
        print(f"  - Missing values: {df[col].isna().sum()} ({df[col].isna().mean()*100:.2f}%)")
        
        # K-fold target encoding for valid values
        fold_encodings = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(valid_indices)):
            # Get actual DataFrame indices
            train_indices = valid_indices[train_idx]
            val_indices = valid_indices[val_idx]
            
            # Calculate encoding on training fold
            train_data = df.loc[train_indices]
            encoding_map, _ = calculate_target_encoding(train_data, col, target_col, smoothing)
            
            # Apply to validation fold
            val_categories = df.loc[val_indices, col]
            global_mean = df.loc[train_indices, target_col].mean()
            
            # Map categories, use global mean for unseen categories
            encoded_values = val_categories.map(encoding_map).fillna(global_mean)
            encoded_df.loc[val_indices, col_name] = encoded_values
            
            fold_encodings.append(len(encoding_map))
        
        # For missing values, use the global mean
        missing_mask = df[col].isna()
        if missing_mask.sum() > 0:
            encoded_df.loc[missing_mask, col_name] = encoder_mappings['global_mean']
        
        # Calculate full-data encoding for future predictions
        full_encoding, global_mean = calculate_target_encoding(df[valid_mask], col, target_col, smoothing)
        encoder_mappings['columns'][col] = {
            'encoding': {str(k): float(v) for k, v in full_encoding.items()},
            'feature_name': col_name,
            'n_categories': len(full_encoding),
            'default_value': float(global_mean)
        }
        
        print(f"  - Encoded feature: {col_name}")
        print(f"  - Encoding range: [{encoded_df[col_name].min():.4f}, {encoded_df[col_name].max():.4f}]")
    
    return encoded_df, encoder_mappings


def analyze_encodings(encoded_df, original_df, encoder_mappings):
    """Print summary statistics for encoded features."""
    print(f"\n{'='*60}")
    print("ENCODING SUMMARY")
    print(f"{'='*60}")
    
    # Get encoded columns (exclude SK_ID_CURR)
    encoded_cols = [c for c in encoded_df.columns if c.endswith('_TARGET_ENC')]
    
    print(f"\nTotal encoded features: {len(encoded_cols)}")
    print(f"Total rows: {len(encoded_df)}")
    
    # Summary statistics
    summary_data = []
    for col in encoded_cols:
        orig_col = col.replace('_TARGET_ENC', '')
        summary_data.append({
            'Feature': col,
            'Original': orig_col,
            'Min': encoded_df[col].min(),
            'Max': encoded_df[col].max(),
            'Mean': encoded_df[col].mean(),
            'Std': encoded_df[col].std(),
            'NaN': encoded_df[col].isna().sum()
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nEncoded Features Statistics:")
    print(summary_df.to_string(index=False))
    
    # Correlation with target
    print(f"\n{'='*60}")
    print("CORRELATION WITH TARGET")
    print(f"{'='*60}")
    
    target = original_df['TARGET']
    correlations = []
    for col in encoded_cols:
        corr = encoded_df[col].corr(target)
        correlations.append((col, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\nFeatures ranked by absolute correlation with TARGET:")
    for feat, corr in correlations:
        print(f"  {feat}: {corr:+.4f}")
    
    return summary_df


def main():
    """Main execution function."""
    print("="*60)
    print("TARGET ENCODING FOR CATEGORICAL VARIABLES")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Data shape: {df.shape}")
    print(f"Target distribution: {df['TARGET'].value_counts().to_dict()}")
    print(f"Target mean (default rate): {df['TARGET'].mean():.4f}")
    
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include='object').columns.tolist()
    print(f"\nIdentified {len(categorical_columns)} categorical columns:")
    for col in categorical_columns:
        print(f"  - {col}: {df[col].nunique()} unique values")
    
    # Apply target encoding with CV
    encoded_df, encoder_mappings = target_encode_cv(df, categorical_columns)
    
    # Analyze encodings
    summary = analyze_encodings(encoded_df, df, encoder_mappings)
    
    # Save encoded features
    print(f"\n{'='*60}")
    print("SAVING OUTPUTS")
    print(f"{'='*60}")
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(ENCODER_PATH), exist_ok=True)
    
    encoded_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved encoded features to: {OUTPUT_PATH}")
    print(f"  Shape: {encoded_df.shape}")
    
    with open(ENCODER_PATH, 'w') as f:
        json.dump(encoder_mappings, f, indent=2)
    print(f"✓ Saved encoder mappings to: {ENCODER_PATH}")
    print(f"  Contains {len(encoder_mappings['columns'])} column encodings")
    
    # Final summary
    print(f"\n{'='*60}")
    print("TARGET ENCODING COMPLETE")
    print(f"{'='*60}")
    print(f"""
Summary:
- Input: {DATA_PATH}
- Categorical columns encoded: {len(categorical_columns)}
- Encoding method: Smoothed target mean with {N_FOLDS}-fold CV
- Smoothing parameter: {SMOOTHING}
- Output features: {len([c for c in encoded_df.columns if c.endswith('_TARGET_ENC')])}
- Output file: {OUTPUT_PATH}
- Encoder mappings: {ENCODER_PATH}

Encoded Features:
""")
    for col in categorical_columns:
        print(f"  {col} → {col}_TARGET_ENC")
    
    return encoded_df, encoder_mappings


if __name__ == '__main__':
    encoded_df, encoder_mappings = main()
