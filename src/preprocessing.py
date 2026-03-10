"""
preprocessing.py — Data Preprocessing Pipeline

Anti data-leakage approach:
  1. Load raw data → keep train rows only
  2. Basic cleaning (drop ID, high-missing columns)
  3. Stratified train/test split BEFORE any transformation
  4. Fit transformers on TRAIN only, then transform both sets
  5. Sklearn Pipeline for reproducible preprocessing

Author: Yomna Haouel
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import warnings
import joblib

import config

warnings.filterwarnings("ignore")


class DataPreprocessor:
    """
    Handles load → clean → split → preprocess.

    Key principle: split FIRST, then fit on train only.
    """

    def __init__(self):
        self.pipeline = None
        self.numeric_cols = []
        self.categorical_cols = []

    # ──────────────────────────────────────────
    # STEP 1: Load and filter
    # ──────────────────────────────────────────

    def load_and_filter(self) -> pd.DataFrame:
        """Load CSV, keep train rows, drop ID column."""
        print("\n[PREPROCESSING] Step 1 — Load & Filter")

        df = pd.read_csv(config.FEATURE_MATRIX_PATH)
        print(f"  Raw shape: {df.shape}")

        # Keep only training rows
        if config.SET_COL in df.columns:
            df = df[df[config.SET_COL] == "train"].drop(columns=[config.SET_COL])

        # Remove rows without valid target
        df = df[df[config.TARGET_COL].notnull() & (df[config.TARGET_COL] != -999)]
        df[config.TARGET_COL] = df[config.TARGET_COL].astype(int)

        # Drop ID column
        df = df.drop(columns=[config.ID_COL], errors="ignore")

        print(f"  Train shape: {df.shape}")
        print(f"  Default rate: {df[config.TARGET_COL].mean():.4f}")

        return df

    # ──────────────────────────────────────────
    # STEP 2: Basic cleaning
    # ──────────────────────────────────────────

    def basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns with too many missing values."""
        print("\n[PREPROCESSING] Step 2 — Basic Cleaning")

        missing_pct = df.isnull().mean()
        high_missing = missing_pct[missing_pct > config.MISSING_THRESHOLD].index.tolist()
        # Don't drop target
        high_missing = [c for c in high_missing if c != config.TARGET_COL]
        df = df.drop(columns=high_missing)

        print(f"  Dropped {len(high_missing)} columns (>{config.MISSING_THRESHOLD*100:.0f}% missing)")
        print(f"  Remaining shape: {df.shape}")

        return df

    # ──────────────────────────────────────────
    # STEP 3: Stratified split
    # ──────────────────────────────────────────

    def split(self, df: pd.DataFrame):
        """Stratified train/test split. Split BEFORE preprocessing to avoid leakage."""
        print("\n[PREPROCESSING] Step 3 — Stratified Split")

        X = df.drop(columns=[config.TARGET_COL])
        y = df[config.TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y,
        )

        print(f"  X_train: {X_train.shape}")
        print(f"  X_test:  {X_test.shape}")
        print(f"  Train default rate: {y_train.mean():.4f}")
        print(f"  Test  default rate: {y_test.mean():.4f}")

        return X_train, X_test, y_train, y_test

    # ──────────────────────────────────────────
    # STEP 4: Build sklearn Pipeline
    # ──────────────────────────────────────────

    def build_pipeline(self, X_train: pd.DataFrame):
        """
        Build a ColumnTransformer pipeline:
          - Numeric: median impute → StandardScaler
          - Categorical: mode impute → OrdinalEncoder

        Fitted on X_train only — no leakage.
        """
        print("\n[PREPROCESSING] Step 4 — Build Sklearn Pipeline")

        self.numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        print(f"  Numeric features:     {len(self.numeric_cols)}")
        print(f"  Categorical features: {len(self.categorical_cols)}")

        # Numeric sub-pipeline
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        # Categorical sub-pipeline
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",
                                       unknown_value=-1)),
        ])

        # Combined transformer
        transformers = []
        if self.numeric_cols:
            transformers.append(("num", numeric_pipeline, self.numeric_cols))
        if self.categorical_cols:
            transformers.append(("cat", categorical_pipeline, self.categorical_cols))

        self.pipeline = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            n_jobs=-1,
        )

        return self.pipeline

    # ──────────────────────────────────────────
    # STEP 5: Fit & Transform
    # ──────────────────────────────────────────

    def fit_transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Fit pipeline on X_train, transform both sets."""
        print("\n[PREPROCESSING] Step 5 — Fit & Transform")

        if self.pipeline is None:
            self.build_pipeline(X_train)

        # Fit on train only
        X_train_processed = self.pipeline.fit_transform(X_train)
        X_test_processed = self.pipeline.transform(X_test)

        feature_names = self.numeric_cols + self.categorical_cols

        X_train_df = pd.DataFrame(X_train_processed, columns=feature_names,
                                  index=X_train.index)
        X_test_df = pd.DataFrame(X_test_processed, columns=feature_names,
                                 index=X_test.index)

        print(f"  X_train processed: {X_train_df.shape}")
        print(f"  X_test  processed: {X_test_df.shape}")
        print(f"  Any NaN remaining: {X_train_df.isnull().any().any()}")

        # Save pipeline for deployment
        pipeline_path = config.MODEL_DIR / "preprocessing_pipeline.joblib"
        joblib.dump(self.pipeline, pipeline_path)
        print(f"  Pipeline saved: {pipeline_path.name}")

        return X_train_df, X_test_df

    # ──────────────────────────────────────────
    # ALL-IN-ONE
    # ──────────────────────────────────────────

    def run(self):
        """Complete preprocessing: load → clean → split → transform."""
        df = self.load_and_filter()
        df = self.basic_cleaning(df)
        X_train, X_test, y_train, y_test = self.split(df)
        X_train, X_test = self.fit_transform(X_train, X_test)

        print("\n✅ Preprocessing complete.")
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.run()
    print(f"\nFinal shapes: X_train={X_train.shape}, X_test={X_test.shape}")
