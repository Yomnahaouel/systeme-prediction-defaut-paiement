"""
data_engineering.py — Complete Data Engineering Pipeline

Creates a production-ready feature matrix by:
1. Loading and aggregating ALL source tables
2. Creating domain-specific engineered features
3. Handling missing values properly
4. Saving a complete, ready-to-use feature matrix

This runs ONCE to prepare data. The output can be used directly
for training AND inference (API).

Author: 7afnawi (fixed version)
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import json

warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "feature_matrix_final.csv"
FEATURE_INFO_FILE = DATA_DIR / "feature_info.json"


class DataEngineer:
    """Complete data engineering pipeline."""
    
    def __init__(self):
        self.feature_info = {
            "original_features": [],
            "aggregated_features": [],
            "engineered_features": [],
            "total_features": 0
        }
    
    # ══════════════════════════════════════════════════════════════
    # STEP 1: LOAD BASE DATA
    # ══════════════════════════════════════════════════════════════
    
    def load_application_data(self) -> pd.DataFrame:
        """Load main application data."""
        print("\n[1/6] Loading application data...")
        
        train = pd.read_csv(DATA_DIR / "application_train.csv")
        print(f"  Loaded: {train.shape}")
        
        self.feature_info["original_features"] = list(train.columns)
        return train
    
    # ══════════════════════════════════════════════════════════════
    # STEP 2: AGGREGATE BUREAU DATA
    # ══════════════════════════════════════════════════════════════
    
    def aggregate_bureau(self) -> pd.DataFrame:
        """Aggregate bureau.csv — client's credit history at other institutions."""
        print("\n[2/6] Aggregating BUREAU data...")
        
        bureau = pd.read_csv(DATA_DIR / "bureau.csv")
        print(f"  Loaded bureau.csv: {bureau.shape}")
        
        # Basic aggregations
        agg = bureau.groupby('SK_ID_CURR').agg({
            'SK_ID_BUREAU': 'count',                          # Total credits
            'CREDIT_ACTIVE': lambda x: (x == 'Active').sum(), # Active credits
            'CREDIT_TYPE': 'nunique',                         # Credit type diversity
            'DAYS_CREDIT': ['mean', 'min', 'max'],           # Credit recency
            'DAYS_CREDIT_ENDDATE': ['mean', 'min'],          # End dates
            'DAYS_CREDIT_UPDATE': 'mean',                     # Update recency
            'AMT_CREDIT_SUM': ['sum', 'mean', 'max'],        # Credit amounts
            'AMT_CREDIT_SUM_DEBT': ['sum', 'mean'],          # Current debt
            'AMT_CREDIT_SUM_OVERDUE': ['sum', 'max'],        # Overdue amounts
            'AMT_CREDIT_SUM_LIMIT': 'sum',                   # Credit limits
            'CNT_CREDIT_PROLONG': 'sum',                      # Prolongations
            'AMT_ANNUITY': ['mean', 'sum'],                  # Annuities
        })
        
        # Flatten column names
        agg.columns = ['BUREAU_' + '_'.join(col).strip('_').upper() for col in agg.columns]
        agg = agg.reset_index()
        
        # Derived features
        agg['BUREAU_CREDIT_ACTIVE_RATIO'] = agg['BUREAU_CREDIT_ACTIVE'] / agg['BUREAU_SK_ID_BUREAU_COUNT'].replace(0, np.nan)
        agg['BUREAU_DEBT_CREDIT_RATIO'] = agg['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] / agg['BUREAU_AMT_CREDIT_SUM_SUM'].replace(0, np.nan)
        agg['BUREAU_OVERDUE_RATIO'] = agg['BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM'] / agg['BUREAU_AMT_CREDIT_SUM_SUM'].replace(0, np.nan)
        
        new_features = [c for c in agg.columns if c != 'SK_ID_CURR']
        self.feature_info["aggregated_features"].extend(new_features)
        
        print(f"  Created {len(new_features)} bureau features")
        return agg
    
    # ══════════════════════════════════════════════════════════════
    # STEP 3: AGGREGATE PREVIOUS APPLICATIONS
    # ══════════════════════════════════════════════════════════════
    
    def aggregate_previous_applications(self) -> pd.DataFrame:
        """Aggregate previous_application.csv — previous loan applications at Home Credit."""
        print("\n[3/6] Aggregating PREVIOUS APPLICATIONS...")
        
        prev = pd.read_csv(DATA_DIR / "previous_application.csv")
        print(f"  Loaded previous_application.csv: {prev.shape}")
        
        # Application status counts
        status_counts = prev.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].value_counts().unstack(fill_value=0)
        status_counts.columns = ['PREV_' + str(c).upper().replace(' ', '_') + '_COUNT' for c in status_counts.columns]
        
        # Basic aggregations
        agg = prev.groupby('SK_ID_CURR').agg({
            'SK_ID_PREV': 'count',                            # Total applications
            'AMT_APPLICATION': ['mean', 'sum', 'max'],       # Applied amounts
            'AMT_CREDIT': ['mean', 'sum', 'max'],            # Approved amounts
            'AMT_DOWN_PAYMENT': ['mean', 'sum'],             # Down payments
            'AMT_GOODS_PRICE': ['mean', 'sum'],              # Goods prices
            'DAYS_DECISION': ['mean', 'min', 'max'],         # Decision timing
            'DAYS_FIRST_DRAWING': 'mean',                    # First drawing
            'DAYS_FIRST_DUE': 'mean',                        # First due
            'CNT_PAYMENT': ['mean', 'sum'],                  # Payment counts
            'RATE_DOWN_PAYMENT': 'mean',                     # Down payment rate
        })
        
        agg.columns = ['PREV_' + '_'.join(col).strip('_').upper() for col in agg.columns]
        agg = agg.reset_index()
        
        # Merge with status counts
        agg = agg.merge(status_counts.reset_index(), on='SK_ID_CURR', how='left')
        
        # Derived features
        agg['PREV_APPROVAL_RATIO'] = agg.get('PREV_APPROVED_COUNT', 0) / agg['PREV_SK_ID_PREV_COUNT'].replace(0, np.nan)
        agg['PREV_REFUSED_RATIO'] = agg.get('PREV_REFUSED_COUNT', 0) / agg['PREV_SK_ID_PREV_COUNT'].replace(0, np.nan)
        agg['PREV_CREDIT_APPLICATION_RATIO'] = agg['PREV_AMT_CREDIT_MEAN'] / agg['PREV_AMT_APPLICATION_MEAN'].replace(0, np.nan)
        
        new_features = [c for c in agg.columns if c != 'SK_ID_CURR']
        self.feature_info["aggregated_features"].extend(new_features)
        
        print(f"  Created {len(new_features)} previous application features")
        return agg
    
    # ══════════════════════════════════════════════════════════════
    # STEP 4: AGGREGATE PAYMENT BEHAVIOR
    # ══════════════════════════════════════════════════════════════
    
    def aggregate_payments(self) -> pd.DataFrame:
        """Aggregate installments_payments.csv — payment history."""
        print("\n[4/6] Aggregating PAYMENT BEHAVIOR...")
        
        ins = pd.read_csv(DATA_DIR / "installments_payments.csv")
        print(f"  Loaded installments_payments.csv: {ins.shape}")
        
        # Calculate payment metrics
        ins['PAYMENT_DIFF'] = ins['AMT_PAYMENT'] - ins['AMT_INSTALMENT']
        ins['PAYMENT_RATIO'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT'].replace(0, np.nan)
        ins['DAYS_DIFF'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
        ins['IS_LATE'] = (ins['DAYS_DIFF'] > 0).astype(int)
        ins['IS_EARLY'] = (ins['DAYS_DIFF'] < 0).astype(int)
        ins['DAYS_LATE'] = ins['DAYS_DIFF'].clip(lower=0)
        ins['DAYS_EARLY'] = (-ins['DAYS_DIFF']).clip(lower=0)
        
        agg = ins.groupby('SK_ID_CURR').agg({
            'SK_ID_PREV': 'nunique',                          # Number of loans
            'NUM_INSTALMENT_NUMBER': 'max',                   # Max installments
            'AMT_INSTALMENT': ['mean', 'sum', 'std'],        # Installment amounts
            'AMT_PAYMENT': ['mean', 'sum', 'std'],           # Payment amounts
            'PAYMENT_DIFF': ['mean', 'sum', 'min'],          # Payment differences
            'PAYMENT_RATIO': ['mean', 'min'],                 # Payment ratios
            'DAYS_DIFF': ['mean', 'max'],                     # Timing
            'IS_LATE': ['sum', 'mean'],                       # Late payments
            'IS_EARLY': ['sum', 'mean'],                      # Early payments
            'DAYS_LATE': ['mean', 'max', 'sum'],             # Days late
            'DAYS_EARLY': ['mean', 'max'],                    # Days early
        })
        
        agg.columns = ['INS_' + '_'.join(col).strip('_').upper() for col in agg.columns]
        agg = agg.reset_index()
        
        new_features = [c for c in agg.columns if c != 'SK_ID_CURR']
        self.feature_info["aggregated_features"].extend(new_features)
        
        print(f"  Created {len(new_features)} payment features")
        return agg
    
    # ══════════════════════════════════════════════════════════════
    # STEP 5: AGGREGATE CREDIT CARD & POS
    # ══════════════════════════════════════════════════════════════
    
    def aggregate_credit_pos(self) -> tuple:
        """Aggregate credit card and POS cash balance data."""
        print("\n[5/6] Aggregating CREDIT CARD & POS data...")
        
        # Credit Card
        cc = pd.read_csv(DATA_DIR / "credit_card_balance.csv")
        print(f"  Loaded credit_card_balance.csv: {cc.shape}")
        
        cc['UTILIZATION'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan)
        
        cc_agg = cc.groupby('SK_ID_CURR').agg({
            'SK_ID_PREV': 'nunique',
            'MONTHS_BALANCE': ['count', 'min'],
            'AMT_BALANCE': ['mean', 'max', 'sum'],
            'AMT_CREDIT_LIMIT_ACTUAL': 'mean',
            'AMT_DRAWINGS_ATM_CURRENT': ['mean', 'sum'],
            'AMT_DRAWINGS_CURRENT': ['mean', 'sum'],
            'AMT_PAYMENT_TOTAL_CURRENT': ['mean', 'sum'],
            'AMT_TOTAL_RECEIVABLE': ['mean', 'max'],
            'CNT_DRAWINGS_ATM_CURRENT': ['mean', 'sum'],
            'CNT_DRAWINGS_CURRENT': 'mean',
            'SK_DPD': ['max', 'mean', 'sum'],
            'SK_DPD_DEF': ['max', 'sum'],
            'UTILIZATION': ['mean', 'max'],
        })
        
        cc_agg.columns = ['CC_' + '_'.join(col).strip('_').upper() for col in cc_agg.columns]
        cc_agg = cc_agg.reset_index()
        
        # POS Cash
        pos = pd.read_csv(DATA_DIR / "POS_CASH_balance.csv")
        print(f"  Loaded POS_CASH_balance.csv: {pos.shape}")
        
        pos_agg = pos.groupby('SK_ID_CURR').agg({
            'SK_ID_PREV': 'nunique',
            'MONTHS_BALANCE': ['count', 'min'],
            'CNT_INSTALMENT': ['mean', 'max'],
            'CNT_INSTALMENT_FUTURE': ['mean', 'min'],
            'SK_DPD': ['max', 'mean', 'sum'],
            'SK_DPD_DEF': ['max', 'sum'],
        })
        
        pos_agg.columns = ['POS_' + '_'.join(col).strip('_').upper() for col in pos_agg.columns]
        pos_agg = pos_agg.reset_index()
        
        # Track features
        cc_features = [c for c in cc_agg.columns if c != 'SK_ID_CURR']
        pos_features = [c for c in pos_agg.columns if c != 'SK_ID_CURR']
        self.feature_info["aggregated_features"].extend(cc_features)
        self.feature_info["aggregated_features"].extend(pos_features)
        
        print(f"  Created {len(cc_features)} credit card features")
        print(f"  Created {len(pos_features)} POS features")
        
        return cc_agg, pos_agg
    
    # ══════════════════════════════════════════════════════════════
    # STEP 6: DOMAIN FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════════
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific engineered features."""
        print("\n[6/6] Creating ENGINEERED FEATURES...")
        
        engineered = []
        
        # ─────────────────────────────────────────
        # FINANCIAL RATIOS
        # ─────────────────────────────────────────
        print("  → Financial ratios...")
        
        if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
            engineered.append('CREDIT_INCOME_RATIO')
        
        if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
            engineered.append('ANNUITY_INCOME_RATIO')
        
        if 'AMT_CREDIT' in df.columns and 'AMT_ANNUITY' in df.columns:
            df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY'].replace(0, np.nan)
            engineered.append('CREDIT_TERM')
        
        if 'AMT_CREDIT' in df.columns and 'AMT_GOODS_PRICE' in df.columns:
            df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'].replace(0, np.nan)
            engineered.append('CREDIT_GOODS_RATIO')
        
        if 'AMT_INCOME_TOTAL' in df.columns and 'CNT_FAM_MEMBERS' in df.columns:
            df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'].replace(0, np.nan)
            engineered.append('INCOME_PER_PERSON')
        
        if 'AMT_ANNUITY' in df.columns and 'AMT_CREDIT' in df.columns:
            df['ANNUITY_CREDIT_RATIO'] = df['AMT_ANNUITY'] / df['AMT_CREDIT'].replace(0, np.nan)
            engineered.append('ANNUITY_CREDIT_RATIO')
        
        # ─────────────────────────────────────────
        # AGE & EMPLOYMENT
        # ─────────────────────────────────────────
        print("  → Age & employment...")
        
        if 'DAYS_BIRTH' in df.columns:
            df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365.25).round(1)
            engineered.append('AGE_YEARS')
        
        if 'DAYS_EMPLOYED' in df.columns:
            # Anomaly flag (365243 = ~1000 years = unemployed)
            df['EMPLOYED_ANOMALY'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
            engineered.append('EMPLOYED_ANOMALY')
            
            # Clean employment years
            emp_clean = df['DAYS_EMPLOYED'].replace(365243, np.nan)
            df['EMPLOYMENT_YEARS'] = (-emp_clean / 365.25).round(1)
            engineered.append('EMPLOYMENT_YEARS')
            
            if 'DAYS_BIRTH' in df.columns:
                df['EMPLOYMENT_AGE_RATIO'] = df['EMPLOYMENT_YEARS'] / df['AGE_YEARS'].replace(0, np.nan)
                engineered.append('EMPLOYMENT_AGE_RATIO')
        
        # ─────────────────────────────────────────
        # EXT_SOURCE INTERACTIONS
        # ─────────────────────────────────────────
        print("  → External source interactions...")
        
        ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in df.columns]
        
        if len(ext_cols) >= 2:
            df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
            df['EXT_SOURCE_STD'] = df[ext_cols].std(axis=1)
            df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)
            df['EXT_SOURCE_MAX'] = df[ext_cols].max(axis=1)
            df['EXT_SOURCE_RANGE'] = df['EXT_SOURCE_MAX'] - df['EXT_SOURCE_MIN']
            engineered.extend(['EXT_SOURCE_MEAN', 'EXT_SOURCE_STD', 'EXT_SOURCE_MIN', 'EXT_SOURCE_MAX', 'EXT_SOURCE_RANGE'])
        
        if 'EXT_SOURCE_1' in df.columns and 'EXT_SOURCE_2' in df.columns:
            df['EXT_SOURCE_1x2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
            engineered.append('EXT_SOURCE_1x2')
        
        if 'EXT_SOURCE_2' in df.columns and 'EXT_SOURCE_3' in df.columns:
            df['EXT_SOURCE_2x3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
            engineered.append('EXT_SOURCE_2x3')
        
        if 'EXT_SOURCE_1' in df.columns and 'EXT_SOURCE_3' in df.columns:
            df['EXT_SOURCE_1x3'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
            engineered.append('EXT_SOURCE_1x3')
        
        if len(ext_cols) == 3:
            df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
            engineered.append('EXT_SOURCE_PROD')
        
        # ─────────────────────────────────────────
        # DOCUMENT FLAGS
        # ─────────────────────────────────────────
        print("  → Document aggregations...")
        
        doc_cols = [c for c in df.columns if c.startswith('FLAG_DOCUMENT_')]
        if doc_cols:
            df['DOCUMENTS_PROVIDED_COUNT'] = df[doc_cols].sum(axis=1)
            engineered.append('DOCUMENTS_PROVIDED_COUNT')
        
        # ─────────────────────────────────────────
        # CONTACT & SOCIAL
        # ─────────────────────────────────────────
        print("  → Contact & social features...")
        
        contact_cols = [c for c in ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL'] if c in df.columns]
        if contact_cols:
            df['CONTACT_INFO_COUNT'] = df[contact_cols].sum(axis=1)
            engineered.append('CONTACT_INFO_COUNT')
        
        region_cols = [c for c in df.columns if 'REG_' in c and 'NOT' in c]
        if region_cols:
            df['REGION_MISMATCH_SUM'] = df[region_cols].sum(axis=1)
            engineered.append('REGION_MISMATCH_SUM')
        
        # ─────────────────────────────────────────
        # CROSS-TABLE INTERACTIONS
        # ─────────────────────────────────────────
        print("  → Cross-table interactions...")
        
        # Bureau vs Current application
        if 'BUREAU_AMT_CREDIT_SUM_SUM' in df.columns and 'AMT_CREDIT' in df.columns:
            df['BUREAU_CURRENT_CREDIT_RATIO'] = df['BUREAU_AMT_CREDIT_SUM_SUM'] / df['AMT_CREDIT'].replace(0, np.nan)
            engineered.append('BUREAU_CURRENT_CREDIT_RATIO')
        
        if 'BUREAU_AMT_CREDIT_SUM_DEBT_SUM' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            df['BUREAU_DEBT_INCOME_RATIO'] = df['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
            engineered.append('BUREAU_DEBT_INCOME_RATIO')
        
        # Payment behavior vs Current
        if 'INS_IS_LATE_MEAN' in df.columns:
            df['PAYMENT_DISCIPLINE'] = 1 - df['INS_IS_LATE_MEAN']
            engineered.append('PAYMENT_DISCIPLINE')
        
        # Total DPD across sources
        dpd_cols = [c for c in df.columns if 'DPD' in c and 'MAX' in c]
        if dpd_cols:
            df['TOTAL_DPD_MAX'] = df[dpd_cols].max(axis=1)
            engineered.append('TOTAL_DPD_MAX')
        
        # ─────────────────────────────────────────
        # RISK INDICATORS
        # ─────────────────────────────────────────
        print("  → Risk indicators...")
        
        # High debt flag
        if 'CREDIT_INCOME_RATIO' in df.columns:
            df['HIGH_DEBT_FLAG'] = (df['CREDIT_INCOME_RATIO'] > 5).astype(int)
            engineered.append('HIGH_DEBT_FLAG')
        
        # Young age risk
        if 'AGE_YEARS' in df.columns:
            df['YOUNG_AGE_FLAG'] = (df['AGE_YEARS'] < 25).astype(int)
            engineered.append('YOUNG_AGE_FLAG')
        
        # Low external score
        if 'EXT_SOURCE_MEAN' in df.columns:
            df['LOW_EXT_SCORE_FLAG'] = (df['EXT_SOURCE_MEAN'] < 0.3).astype(int)
            engineered.append('LOW_EXT_SCORE_FLAG')
        
        self.feature_info["engineered_features"] = engineered
        print(f"  Created {len(engineered)} engineered features")
        
        return df
    
    # ══════════════════════════════════════════════════════════════
    # MAIN PIPELINE
    # ══════════════════════════════════════════════════════════════
    
    def run(self) -> pd.DataFrame:
        """Execute complete data engineering pipeline."""
        print("\n" + "=" * 70)
        print("   COMPLETE DATA ENGINEERING PIPELINE")
        print("=" * 70)
        
        # Step 1: Load base data
        df = self.load_application_data()
        
        # Step 2: Bureau aggregations
        bureau_agg = self.aggregate_bureau()
        df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
        print(f"  → After bureau merge: {df.shape}")
        
        # Step 3: Previous applications
        prev_agg = self.aggregate_previous_applications()
        df = df.merge(prev_agg, on='SK_ID_CURR', how='left')
        print(f"  → After previous merge: {df.shape}")
        
        # Step 4: Payment behavior
        ins_agg = self.aggregate_payments()
        df = df.merge(ins_agg, on='SK_ID_CURR', how='left')
        print(f"  → After payments merge: {df.shape}")
        
        # Step 5: Credit card & POS
        cc_agg, pos_agg = self.aggregate_credit_pos()
        df = df.merge(cc_agg, on='SK_ID_CURR', how='left')
        df = df.merge(pos_agg, on='SK_ID_CURR', how='left')
        print(f"  → After CC/POS merge: {df.shape}")
        
        # Step 6: Feature engineering
        df = self.engineer_features(df)
        
        # Clean up
        print("\n  Cleaning up...")
        
        # Replace infinities
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN in aggregated columns with 0 (no history = 0)
        agg_cols = self.feature_info["aggregated_features"]
        for col in agg_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Update feature info
        self.feature_info["total_features"] = len(df.columns)
        
        # Save
        print(f"\n  Saving to {OUTPUT_FILE}...")
        df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"\n  Saving feature info to {FEATURE_INFO_FILE}...")
        with open(FEATURE_INFO_FILE, 'w') as f:
            json.dump(self.feature_info, f, indent=2)
        
        # Summary
        print("\n" + "=" * 70)
        print("   DATA ENGINEERING COMPLETE")
        print("=" * 70)
        print(f"   Original features:    {len(self.feature_info['original_features'])}")
        print(f"   Aggregated features:  {len(self.feature_info['aggregated_features'])}")
        print(f"   Engineered features:  {len(self.feature_info['engineered_features'])}")
        print(f"   ─────────────────────────────────────")
        print(f"   TOTAL FEATURES:       {df.shape[1]}")
        print(f"   TOTAL ROWS:           {df.shape[0]}")
        print(f"   Output: {OUTPUT_FILE}")
        print("=" * 70)
        
        return df


if __name__ == "__main__":
    engineer = DataEngineer()
    df = engineer.run()
