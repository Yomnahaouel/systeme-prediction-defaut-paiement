"""
feature_selection.py — Optimized Feature Selection Pipeline

Three-step approach (no data leakage — fitted on train only):
  1. Correlation filter (remove redundant features)
  2. Mutual Information (filter top-K informative features)
  3. Tree-based importance (LightGBM for speed)
  4. Final combination: union or intersection strategy

Optimizations:
  - Sampling for correlation & MI computation
  - LightGBM instead of RandomForest (10x faster)
  - Parallel computation (n_jobs=-1)

Author: Yomna Haouel
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
from sklearn.feature_selection import mutual_info_classif
import warnings

import config

warnings.filterwarnings("ignore")


class FeatureSelector:
    """
    Multi-method feature selection, fit on training data only.
    """

    def __init__(self):
        self.selected_features = None

    # ─────────────────────────────────────────
    # 1. CORRELATION FILTER
    # ─────────────────────────────────────────

    def remove_high_correlation(self, X_train: pd.DataFrame,
                                 threshold: float = 0.85) -> list:
        """
        Remove one of each pair of features with |correlation| > threshold.
        Uses sampling for speed on large datasets.
        """
        print("\n[FS] Step 1: Correlation Filter")

        X_numeric = X_train.select_dtypes(include=[np.number])
        print(f"  Numeric features: {X_numeric.shape[1]}")

        # Sample for speed
        if X_numeric.shape[0] > 50000:
            X_sample = X_numeric.sample(50000, random_state=config.RANDOM_STATE)
            print("  Sampling 50,000 rows for speed")
        else:
            X_sample = X_numeric

        corr_matrix = X_sample.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        remaining = [c for c in X_train.columns if c not in to_drop]

        print(f"  Removed: {len(to_drop)} highly correlated features")
        print(f"  Remaining: {len(remaining)}")

        return remaining

    # ─────────────────────────────────────────
    # 2. MUTUAL INFORMATION
    # ─────────────────────────────────────────

    def mutual_information_selection(self, X_train: pd.DataFrame,
                                      y_train: pd.Series,
                                      top_k: int = 120) -> list:
        """
        Select top-K features by Mutual Information with TARGET.
        Uses sampling for speed.
        """
        print(f"\n[FS] Step 2: Mutual Information (top {top_k})")

        sample_size = min(80000, len(X_train))
        idx = X_train.sample(sample_size, random_state=config.RANDOM_STATE).index
        X_sample = X_train.loc[idx].select_dtypes(include=[np.number]).fillna(0)
        y_sample = y_train.loc[idx]

        mi_scores = mutual_info_classif(
            X_sample, y_sample,
            random_state=config.RANDOM_STATE,
            n_neighbors=5,
        )

        mi_series = pd.Series(mi_scores, index=X_sample.columns)
        mi_series = mi_series.sort_values(ascending=False)

        selected = mi_series.head(top_k).index.tolist()

        print(f"  Top MI scores:")
        for feat, score in mi_series.head(5).items():
            print(f"    {feat}: {score:.4f}")
        print(f"  Selected {len(selected)} features")

        return selected

    # ─────────────────────────────────────────
    # 3. TREE-BASED IMPORTANCE (LightGBM)
    # ─────────────────────────────────────────

    def tree_importance_selection(self, X_train: pd.DataFrame,
                                   y_train: pd.Series,
                                   threshold: float = 0.001) -> list:
        """
        Select features based on LightGBM feature importance.
        Much faster than RandomForest for high-dimensional data.
        """
        print("\n[FS] Step 3: Tree-based Importance (LightGBM)")

        try:
            import lightgbm as lgb

            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                is_unbalance=True,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
        except ImportError:
            print("  LightGBM not available, falling back to RandomForest")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=50, max_depth=8,
                random_state=config.RANDOM_STATE,
                n_jobs=-1, class_weight="balanced",
            )

        X_num = X_train.select_dtypes(include=[np.number]).fillna(0)
        model.fit(X_num, y_train)

        importance = pd.Series(
            model.feature_importances_, index=X_num.columns
        ).sort_values(ascending=False)

        # Normalize to sum=1
        importance = importance / importance.sum()

        selected = importance[importance > threshold].index.tolist()

        print(f"  Top important features:")
        for feat, imp in importance.head(5).items():
            print(f"    {feat}: {imp:.4f}")
        print(f"  Selected {len(selected)} features (importance > {threshold})")

        return selected

    # ─────────────────────────────────────────
    # 4. FINAL COMBINATION
    # ─────────────────────────────────────────

    def select_features(self, X_train: pd.DataFrame, y_train: pd.Series) -> list:
        """
        Run all three selection steps and combine results.

        Strategy:
        - 'intersection': features selected by BOTH MI and Tree (strict)
        - 'union':        features selected by EITHER method (permissive)
        """
        print("\n" + "=" * 60)
        print("   FEATURE SELECTION PIPELINE")
        print("=" * 60)

        # Step 1: Remove redundancy
        after_corr = self.remove_high_correlation(
            X_train, config.CORRELATION_THRESHOLD
        )
        X_filtered = X_train[after_corr]

        # Step 2: Mutual Information
        mi_selected = self.mutual_information_selection(
            X_filtered, y_train, config.MI_TOP_K
        )

        # Step 3: Tree importance
        tree_selected = self.tree_importance_selection(
            X_filtered, y_train, config.IMPORTANCE_THRESHOLD
        )

        # Combine
        strategy = config.SELECTION_STRATEGY
        if strategy == "intersection":
            final = list(set(mi_selected) & set(tree_selected))
            print(f"\n  Strategy: INTERSECTION")
        else:
            final = list(set(mi_selected) | set(tree_selected))
            print(f"\n  Strategy: UNION")

        # Fallback if too few features
        if len(final) < 20:
            print("  ⚠ Too few features with intersection, falling back to MI")
            final = mi_selected

        final.sort()
        self.selected_features = final

        # Save selected features
        features_path = config.SELECTED_FEATURES_PATH
        with open(features_path, "w") as f:
            json.dump(self.selected_features, f, indent=2)

        print(f"\n  FINAL FEATURES: {len(self.selected_features)}")
        print(f"  Saved to: {features_path.name}")

        return self.selected_features

    # ─────────────────────────────────────────
    # APPLY
    # ─────────────────────────────────────────

    def transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Apply selection to both train and test."""
        if self.selected_features is None:
            raise ValueError("Run select_features() first")

        # Only keep features that exist in both sets
        valid = [f for f in self.selected_features
                 if f in X_train.columns and f in X_test.columns]

        return X_train[valid], X_test[valid]


def run_feature_selection(X_train, X_test, y_train):
    """Run the full feature selection pipeline."""
    selector = FeatureSelector()
    selector.select_features(X_train, y_train)
    X_train_sel, X_test_sel = selector.transform(X_train, X_test)

    print(f"\n✅ Feature Selection complete.")
    print(f"   X_train: {X_train_sel.shape}, X_test: {X_test_sel.shape}")

    return X_train_sel, X_test_sel, selector


if __name__ == "__main__":
    from preprocessing import DataPreprocessor
    from feature_engineering import run_feature_engineering

    preprocessor = DataPreprocessor()
    df = preprocessor.load_and_filter()
    df = preprocessor.basic_cleaning(df)
    X_train, X_test, y_train, y_test = preprocessor.split(df)
    X_train, X_test = run_feature_engineering(X_train, X_test)
    X_train_sel, X_test_sel, _ = run_feature_selection(X_train, X_test, y_train)
