"""
feature_engineering.py — Domain-driven Feature Engineering

Creates meaningful financial features AFTER train/test split.
All transformations are applied identically to train and test sets.

New features:
  1. Financial ratios (CREDIT/INCOME, ANNUITY/INCOME, etc.)
  2. Age and employment in years
  3. Interaction features (EXT_SOURCE products)
  4. Aggregation features (income per family member)
  5. Domain flags (anomaly detection in DAYS_EMPLOYED)

Author: Yomna Haouel
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings

import config

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """
    Creates domain-specific features for credit scoring.

    Principle: Same transformations applied to train AND test.
    No fitting required — purely deterministic transformations.
    """

    def __init__(self):
        self.created_features = []

    def _safe_ratio(self, df: pd.DataFrame, num_col: str, den_col: str) -> pd.Series:
        """Compute a safe ratio (handle division by zero)."""
        if num_col not in df.columns or den_col not in df.columns:
            return None
        return df[num_col] / df[den_col].replace(0, np.nan)

    # ──────────────────────────────────────────
    # 1. FINANCIAL RATIOS
    # ──────────────────────────────────────────

    def create_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create financial ratios from config.FINANCIAL_RATIOS.

        Justification métier:
        - CREDIT_INCOME_RATIO: capacité d'endettement (dette / revenu)
        - ANNUITY_INCOME_RATIO: charge mensuelle / revenu → effort financier
        - CREDIT_TERM: durée du crédit en nombre de mensualités
        - CREDIT_GOODS_RATIO: surévaluation du crédit vs prix du bien
        - INCOME_PER_PERSON: revenu par personne du foyer
        - ANNUITY_CREDIT_RATIO: part de la mensualité dans le crédit total
        """
        print("  Creating financial ratios...")

        for feat_name, (num, den) in config.FINANCIAL_RATIOS.items():
            ratio = self._safe_ratio(df, num, den)
            if ratio is not None:
                df[feat_name] = ratio
                self.created_features.append(feat_name)

        return df

    # ──────────────────────────────────────────
    # 2. AGE & EMPLOYMENT
    # ──────────────────────────────────────────

    def create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert negative day counts to years.

        Justification: DAYS_BIRTH et DAYS_EMPLOYED sont en jours négatifs.
        La conversion en années facilite l'interprétation.
        """
        print("  Creating age & employment features...")

        if config.AGE_COLUMN in df.columns:
            df["AGE_YEARS"] = (-df[config.AGE_COLUMN] / 365.25).round(1)
            self.created_features.append("AGE_YEARS")

        if config.EMPLOYMENT_COLUMN in df.columns:
            # Flag anomaly: 365243 days = ~1000 years = unemployed/retired
            df["EMPLOYED_ANOMALY"] = (df[config.EMPLOYMENT_COLUMN] == 365243).astype(int)
            self.created_features.append("EMPLOYED_ANOMALY")

            # Clean employment and convert to years
            emp_clean = df[config.EMPLOYMENT_COLUMN].replace(365243, np.nan)
            df["EMPLOYMENT_YEARS"] = (-emp_clean / 365.25).round(1)
            self.created_features.append("EMPLOYMENT_YEARS")

            # Employment to age ratio
            if config.AGE_COLUMN in df.columns:
                df["EMPLOYMENT_AGE_RATIO"] = (
                    df["EMPLOYMENT_YEARS"] / df["AGE_YEARS"].replace(0, np.nan)
                )
                self.created_features.append("EMPLOYMENT_AGE_RATIO")

        return df

    # ──────────────────────────────────────────
    # 3. EXTERNAL SOURCE INTERACTIONS
    # ──────────────────────────────────────────

    def create_ext_source_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features from EXT_SOURCE variables.

        Justification: EXT_SOURCE_1/2/3 sont les meilleurs prédicteurs
        univariés. Leurs combinaisons capturent des patterns non-linéaires.
        """
        print("  Creating EXT_SOURCE interactions...")

        ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
                    if c in df.columns]

        if len(ext_cols) >= 2:
            # Mean, min, max of available EXT_SOURCE
            df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
            df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
            df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
            df["EXT_SOURCE_MAX"] = df[ext_cols].max(axis=1)
            self.created_features.extend([
                "EXT_SOURCE_MEAN", "EXT_SOURCE_STD",
                "EXT_SOURCE_MIN", "EXT_SOURCE_MAX"
            ])

        if "EXT_SOURCE_1" in df.columns and "EXT_SOURCE_2" in df.columns:
            df["EXT_SOURCE_1x2"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
            self.created_features.append("EXT_SOURCE_1x2")

        if "EXT_SOURCE_2" in df.columns and "EXT_SOURCE_3" in df.columns:
            df["EXT_SOURCE_2x3"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
            self.created_features.append("EXT_SOURCE_2x3")

        if "EXT_SOURCE_1" in df.columns and "EXT_SOURCE_3" in df.columns:
            df["EXT_SOURCE_1x3"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]
            self.created_features.append("EXT_SOURCE_1x3")

        return df

    # ──────────────────────────────────────────
    # 4. DOCUMENT FLAGS AGGREGATION
    # ──────────────────────────────────────────

    def create_document_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate FLAG_DOCUMENT_* columns.

        Justification: Le nombre total de documents fournis peut indiquer
        le sérieux du dossier de crédit.
        """
        print("  Creating document aggregation features...")

        doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
        if doc_cols:
            df["DOCUMENTS_PROVIDED_COUNT"] = df[doc_cols].sum(axis=1)
            self.created_features.append("DOCUMENTS_PROVIDED_COUNT")

        return df

    # ──────────────────────────────────────────
    # 5. REGISTRATION & SOCIAL FEATURES
    # ──────────────────────────────────────────

    def create_social_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine regional and contact features.

        Justification: La concordance des adresses (résidence, travail,
        enregistrement) est un indicateur de stabilité.
        """
        print("  Creating social features...")

        region_cols = [c for c in df.columns if "REG_" in c and "NOT" in c]
        if region_cols:
            df["REGION_MISMATCH_SUM"] = df[region_cols].sum(axis=1)
            self.created_features.append("REGION_MISMATCH_SUM")

        contact_cols = [c for c in ["FLAG_MOBIL", "FLAG_EMP_PHONE",
                                     "FLAG_WORK_PHONE", "FLAG_PHONE",
                                     "FLAG_EMAIL"] if c in df.columns]
        if contact_cols:
            df["CONTACT_INFO_COUNT"] = df[contact_cols].sum(axis=1)
            self.created_features.append("CONTACT_INFO_COUNT")

        return df

    # ──────────────────────────────────────────
    # RUN ALL
    # ──────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        print(f"\n[FEATURE ENGINEERING] Input shape: {df.shape}")

        df = self.create_financial_ratios(df)
        df = self.create_age_features(df)
        df = self.create_ext_source_features(df)
        df = self.create_document_features(df)
        df = self.create_social_features(df)

        # Replace infinities with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        print(f"  Created features: {len(self.created_features)}")
        print(f"  Output shape: {df.shape}")

        return df


def run_feature_engineering(X_train, X_test):
    """Apply identical feature engineering to train and test."""
    print("\n" + "=" * 60)
    print("   FEATURE ENGINEERING")
    print("=" * 60)

    engineer = FeatureEngineer()

    X_train = engineer.transform(X_train.copy())
    # Reset to apply same logic (no fitting)
    engineer.created_features = []
    X_test = engineer.transform(X_test.copy())

    print(f"\n✅ Feature Engineering complete.")
    print(f"   New features created: {len(engineer.created_features)}")

    return X_train, X_test


if __name__ == "__main__":
    from preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor()
    df = preprocessor.load_and_filter()
    df = preprocessor.basic_cleaning(df)
    X_train, X_test, y_train, y_test = preprocessor.split(df)

    X_train, X_test = run_feature_engineering(X_train, X_test)
    print(f"\nFinal: X_train={X_train.shape}, X_test={X_test.shape}")
