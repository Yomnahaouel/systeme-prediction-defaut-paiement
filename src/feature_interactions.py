#!/usr/bin/env python3
"""
Feature Interactions - Kaggle Winner-Style Feature Engineering
Creates powerful interaction features from existing features.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def create_ext_source_interactions(df):
    """Create EXT_SOURCE interaction features - most important predictors!"""
    interactions = pd.DataFrame(index=df.index)
    
    # Age in years (positive value)
    age_years = df['DAYS_BIRTH'] / -365
    
    # EXT_SOURCE x Age interactions
    if 'EXT_SOURCE_1' in df.columns:
        interactions['EXT_SOURCE_1_x_DAYS_BIRTH'] = df['EXT_SOURCE_1'] * age_years
        interactions['EXT_SOURCE_1_x_DAYS_EMPLOYED'] = df['EXT_SOURCE_1'] * df['DAYS_EMPLOYED']
    
    if 'EXT_SOURCE_2' in df.columns:
        interactions['EXT_SOURCE_2_x_DAYS_BIRTH'] = df['EXT_SOURCE_2'] * age_years
        interactions['EXT_SOURCE_2_x_AMT_CREDIT'] = df['EXT_SOURCE_2'] * df['AMT_CREDIT']
        
    if 'EXT_SOURCE_3' in df.columns:
        interactions['EXT_SOURCE_3_x_DAYS_BIRTH'] = df['EXT_SOURCE_3'] * age_years
    
    # Weighted EXT_SOURCE combination
    ext1 = df.get('EXT_SOURCE_1', pd.Series(0, index=df.index)).fillna(0)
    ext2 = df.get('EXT_SOURCE_2', pd.Series(0, index=df.index)).fillna(0)
    ext3 = df.get('EXT_SOURCE_3', pd.Series(0, index=df.index)).fillna(0)
    interactions['EXT_SOURCE_WEIGHTED'] = 0.5 * ext1 + 0.3 * ext2 + 0.2 * ext3
    
    # Additional EXT_SOURCE combinations
    interactions['EXT_SOURCE_1_x_2'] = ext1 * ext2
    interactions['EXT_SOURCE_2_x_3'] = ext2 * ext3
    interactions['EXT_SOURCE_1_x_3'] = ext1 * ext3
    interactions['EXT_SOURCE_MIN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
    interactions['EXT_SOURCE_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    
    return interactions


def create_financial_interactions(df):
    """Create financial burden and ratio features."""
    interactions = pd.DataFrame(index=df.index)
    
    # Age in years
    age_years = df['DAYS_BIRTH'] / -365
    employment_years = df['DAYS_EMPLOYED'] / -365
    
    # Debt burden calculations
    bureau_debt = df.get('BUREAU_AMT_DEBT_SUM', pd.Series(0, index=df.index)).fillna(0)
    interactions['DEBT_BURDEN'] = (df['AMT_CREDIT'] + bureau_debt) / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    
    # Annuity burden (using available data)
    interactions['ANNUITY_BURDEN'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    
    # Credit normalized by age/employment
    interactions['CREDIT_PER_YEAR'] = df['AMT_CREDIT'] / age_years.replace(0, np.nan)
    interactions['CREDIT_PER_EMPLOYMENT_YEAR'] = df['AMT_CREDIT'] / employment_years.replace(0, np.nan).replace({np.inf: np.nan, -np.inf: np.nan})
    
    # Income-related ratios
    interactions['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'].replace(0, 1)
    interactions['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (df['CNT_CHILDREN'] + 1)
    interactions['ANNUITY_PER_INCOME_PERSON'] = df['AMT_ANNUITY'] / interactions['INCOME_PER_PERSON'].replace(0, np.nan)
    
    # Goods price ratios
    interactions['GOODS_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT'].replace(0, np.nan)
    interactions['GOODS_INCOME_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    
    # Employment stability
    interactions['EMPLOYMENT_RATIO'] = employment_years / age_years.replace(0, np.nan)
    interactions['DAYS_EMPLOYED_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'].replace(0, np.nan)
    
    return interactions


def create_risk_interactions(df):
    """Create risk combination features."""
    interactions = pd.DataFrame(index=df.index)
    
    # Risk score 1: Low external score + high credit ratio
    ext_mean = df.get('EXT_SOURCE_MEAN', pd.Series(0.5, index=df.index)).fillna(0.5)
    credit_ratio = df.get('CREDIT_INCOME_RATIO', df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan))
    interactions['RISK_SCORE_1'] = (1 - ext_mean) * credit_ratio.fillna(0)
    
    # Risk score 2: Late payments + bureau debt
    ins_late = df.get('INS_LATE_RATIO', pd.Series(0, index=df.index)).fillna(0)
    bureau_credit = df.get('BUREAU_AMT_CREDIT_SUM', pd.Series(1, index=df.index)).fillna(1).replace(0, 1)
    bureau_debt = df.get('BUREAU_AMT_DEBT_SUM', pd.Series(0, index=df.index)).fillna(0)
    bureau_debt_ratio = bureau_debt / bureau_credit
    interactions['RISK_SCORE_2'] = ins_late * bureau_debt_ratio.fillna(0)
    
    # Young + high credit risk
    age_years = df.get('AGE_YEARS', df['DAYS_BIRTH'] / -365)
    interactions['YOUNG_HIGH_CREDIT'] = ((age_years < 30) * (credit_ratio > 4)).astype(int)
    
    # Risk score 3: Combined risk factors
    interactions['RISK_SCORE_3'] = (1 - ext_mean) * ins_late
    
    # Late payment risk
    ins_late_count = df.get('INS_LATE_COUNT', pd.Series(0, index=df.index)).fillna(0)
    interactions['LATE_PAYMENT_RISK'] = ins_late_count * credit_ratio.fillna(0)
    
    # Bureau risk
    bureau_dpd = df.get('BUREAU_DPD_MAX', pd.Series(0, index=df.index)).fillna(0)
    interactions['BUREAU_RISK'] = bureau_dpd * (1 - ext_mean)
    
    # Age-based risk segments
    interactions['YOUNG_RISK'] = ((age_years < 25) * (1 - ext_mean)).fillna(0)
    interactions['OLD_HIGH_DEBT'] = ((age_years > 55) * credit_ratio).fillna(0)
    
    return interactions


def create_polynomial_features(df):
    """Create polynomial features for top predictors."""
    interactions = pd.DataFrame(index=df.index)
    
    # EXT_SOURCE polynomials
    if 'EXT_SOURCE_2' in df.columns:
        interactions['EXT_SOURCE_2_SQUARED'] = df['EXT_SOURCE_2'] ** 2
        interactions['EXT_SOURCE_2_CUBED'] = df['EXT_SOURCE_2'] ** 3
    
    if 'EXT_SOURCE_3' in df.columns:
        interactions['EXT_SOURCE_3_SQUARED'] = df['EXT_SOURCE_3'] ** 2
    
    if 'EXT_SOURCE_1' in df.columns:
        interactions['EXT_SOURCE_1_SQUARED'] = df['EXT_SOURCE_1'] ** 2
    
    # Credit term polynomials
    if 'CREDIT_TERM' in df.columns:
        interactions['CREDIT_TERM_SQUARED'] = df['CREDIT_TERM'] ** 2
    
    # Age polynomials
    age_years = df['DAYS_BIRTH'] / -365
    interactions['AGE_SQUARED'] = age_years ** 2
    interactions['AGE_LOG'] = np.log1p(age_years)
    
    # Credit amount transforms
    interactions['AMT_CREDIT_LOG'] = np.log1p(df['AMT_CREDIT'])
    interactions['AMT_INCOME_LOG'] = np.log1p(df['AMT_INCOME_TOTAL'])
    
    # Ratio polynomials
    if 'CREDIT_INCOME_RATIO' in df.columns:
        interactions['CREDIT_INCOME_RATIO_SQUARED'] = df['CREDIT_INCOME_RATIO'] ** 2
    
    return interactions


def create_domain_interactions(df):
    """Create domain-specific interaction features."""
    interactions = pd.DataFrame(index=df.index)
    
    # Document submission count (flag columns)
    doc_cols = [c for c in df.columns if c.startswith('FLAG_DOCUMENT')]
    if doc_cols:
        interactions['DOC_COUNT'] = df[doc_cols].sum(axis=1)
    
    # Social circle defaults
    def_cols = [c for c in df.columns if 'DEF_' in c and 'OBS_' not in c]
    obs_cols = [c for c in df.columns if 'OBS_' in c]
    if def_cols and obs_cols:
        interactions['SOCIAL_DEF_RATIO'] = df[def_cols].sum(axis=1) / (df[obs_cols].sum(axis=1) + 1)
    
    # Housing score (own vs renting indicators)
    own_realty = df.get('FLAG_OWN_REALTY', pd.Series(0, index=df.index))
    own_car = df.get('FLAG_OWN_CAR', pd.Series(0, index=df.index))
    # Convert Y/N to 1/0 if needed
    if own_realty.dtype == object:
        own_realty = (own_realty == 'Y').astype(int)
    if own_car.dtype == object:
        own_car = (own_car == 'Y').astype(int)
    interactions['HOUSING_STABILITY'] = own_realty * 2 + own_car
    
    # Phone/email contact availability
    contact_cols = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL']
    available_contact = [c for c in contact_cols if c in df.columns]
    if available_contact:
        interactions['CONTACT_SCORE'] = df[available_contact].sum(axis=1)
    
    # Registration consistency
    reg_cols = [c for c in df.columns if 'REG_' in c and 'NOT' in c]
    if reg_cols:
        interactions['REG_INCONSISTENCY'] = df[reg_cols].sum(axis=1)
    
    # Bureau query frequency
    bureau_req_cols = [c for c in df.columns if c.startswith('AMT_REQ_CREDIT_BUREAU')]
    if bureau_req_cols:
        interactions['BUREAU_QUERY_TOTAL'] = df[bureau_req_cols].sum(axis=1)
        interactions['BUREAU_QUERY_RECENT'] = df.get('AMT_REQ_CREDIT_BUREAU_MON', 0) + df.get('AMT_REQ_CREDIT_BUREAU_QRT', 0)
    
    return interactions


def main():
    print("=" * 60)
    print("FEATURE INTERACTIONS - Kaggle Winner Style")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading feature matrix...")
    df = pd.read_csv('data/feature_matrix_final.csv')
    print(f"   Shape: {df.shape}")
    
    # Store SK_ID_CURR and TARGET
    sk_id = df['SK_ID_CURR']
    target = df['TARGET'] if 'TARGET' in df.columns else None
    
    # Create interaction features
    print("\n🔧 Creating interaction features...")
    
    print("   → EXT_SOURCE interactions...")
    ext_interactions = create_ext_source_interactions(df)
    print(f"      Created {len(ext_interactions.columns)} features")
    
    print("   → Financial interactions...")
    fin_interactions = create_financial_interactions(df)
    print(f"      Created {len(fin_interactions.columns)} features")
    
    print("   → Risk combinations...")
    risk_interactions = create_risk_interactions(df)
    print(f"      Created {len(risk_interactions.columns)} features")
    
    print("   → Polynomial features...")
    poly_interactions = create_polynomial_features(df)
    print(f"      Created {len(poly_interactions.columns)} features")
    
    print("   → Domain interactions...")
    domain_interactions = create_domain_interactions(df)
    print(f"      Created {len(domain_interactions.columns)} features")
    
    # Combine all interactions
    interactions_df = pd.concat([
        ext_interactions,
        fin_interactions,
        risk_interactions,
        poly_interactions,
        domain_interactions
    ], axis=1)
    
    # Add SK_ID_CURR and TARGET
    interactions_df.insert(0, 'SK_ID_CURR', sk_id)
    if target is not None:
        interactions_df.insert(1, 'TARGET', target)
    
    # Handle infinite values
    interactions_df = interactions_df.replace([np.inf, -np.inf], np.nan)
    
    print(f"\n📊 Total interaction features created: {len(interactions_df.columns) - 2}")
    
    # Calculate correlations with TARGET
    if target is not None:
        print("\n🎯 Top 20 features most correlated with TARGET:")
        print("-" * 50)
        
        feature_cols = [c for c in interactions_df.columns if c not in ['SK_ID_CURR', 'TARGET']]
        correlations = {}
        
        target_numeric = pd.to_numeric(interactions_df['TARGET'], errors='coerce')
        for col in feature_cols:
            col_numeric = pd.to_numeric(interactions_df[col], errors='coerce')
            corr = col_numeric.corr(target_numeric)
            if not np.isnan(corr):
                correlations[col] = corr
        
        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for i, (feature, corr) in enumerate(sorted_corr[:20], 1):
            direction = "⬆️" if corr > 0 else "⬇️"
            print(f"   {i:2d}. {direction} {feature:40s} {corr:+.4f}")
    
    # Save results
    output_path = 'data/interaction_features.csv'
    interactions_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to {output_path}")
    print(f"   Shape: {interactions_df.shape}")
    
    # Summary statistics
    print("\n📈 Feature Statistics:")
    print(f"   Non-null counts range: {interactions_df[feature_cols].count().min()} - {interactions_df[feature_cols].count().max()}")
    print(f"   Features with >50% missing: {(interactions_df[feature_cols].isnull().sum() > len(df)/2).sum()}")
    
    print("\n✅ Feature interactions complete!")
    
    return interactions_df


if __name__ == "__main__":
    main()
