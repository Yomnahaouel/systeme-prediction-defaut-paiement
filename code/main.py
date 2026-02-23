"""
main.py — Pipeline Orchestrator

Runs the complete credit scoring pipeline:
  1. EDA (optional, run with --eda flag)
  2. Preprocessing (load → clean → split → transform)
  3. Feature Engineering (financial ratios, interactions)
  4. Feature Selection (correlation, MI, tree importance)
  5. Training (5 models, SMOTE, CV, MLflow)
  6. Model Comparison & Best Model Selection

Usage:
  python code/main.py           # Run full pipeline (skip EDA)
  python code/main.py --eda     # Run EDA first, then pipeline
  python code/main.py --eda-only  # Run only EDA

Author: Yomna Haouel
"""

import sys
import os
import time
import argparse

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

import config


def main():
    parser = argparse.ArgumentParser(description="Credit Default Risk Pipeline")
    parser.add_argument("--eda", action="store_true", help="Run EDA before training")
    parser.add_argument("--eda-only", action="store_true", help="Run only EDA")
    args = parser.parse_args()

    start_total = time.time()

    print("+" + "=" * 58 + "+")
    print("|   CREDIT DEFAULT RISK - PREDICTION PIPELINE              |")
    print("|   Author: Yomna Haouel                                   |")
    print("+" + "=" * 58 + "+")

    # ───── STEP 0: EDA (optional) ─────
    if args.eda or args.eda_only:
        from exploration_donnees import run_eda
        run_eda()
        if args.eda_only:
            print("\n✅ EDA-only mode complete.")
            return

    # ───── STEP 1: PREPROCESSING ─────
    print("\n" + "=" * 60)
    print("   STEP 1 — PREPROCESSING")
    print("=" * 60)

    from preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor()
    df = preprocessor.load_and_filter()
    df = preprocessor.basic_cleaning(df)
    X_train, X_test, y_train, y_test = preprocessor.split(df)

    del df  # Free memory

    # ───── STEP 2: FEATURE ENGINEERING ─────
    print("\n" + "=" * 60)
    print("   STEP 2 — FEATURE ENGINEERING")
    print("=" * 60)

    from feature_engineering import run_feature_engineering

    X_train, X_test = run_feature_engineering(X_train, X_test)

    # ───── STEP 3: FEATURE SELECTION ─────
    print("\n" + "=" * 60)
    print("   STEP 3 — FEATURE SELECTION")
    print("=" * 60)

    from feature_selection import run_feature_selection

    X_train_sel, X_test_sel, selector = run_feature_selection(
        X_train, X_test, y_train
    )

    del X_train, X_test  # Free memory

    # ───── STEP 4: MULTI-MODEL TRAINING ─────
    print("\n" + "=" * 60)
    print("   STEP 4 — MULTI-MODEL TRAINING")
    print("=" * 60)

    from training import train_all_models

    results = train_all_models(X_train_sel, y_train, X_test_sel, y_test)

    # ───── STEP 5: MODEL COMPARISON ─────
    print("\n" + "=" * 60)
    print("   STEP 5 — MODEL COMPARISON & SELECTION")
    print("=" * 60)

    from model_comparison import compare_models

    best = compare_models(results, X_test_sel, y_test)

    # ───── SUMMARY ─────
    elapsed = time.time() - start_total

    print("\n" + "+" + "=" * 58 + "+")
    print("|   PIPELINE COMPLETE                                      |")
    print("+" + "=" * 58 + "+")
    print(f"|   Best Model:   {best['model_name']:<41s}|")
    print(f"|   ROC-AUC:      {best['metrics']['roc_auc']:<41.4f}|")
    print(f"|   Recall:       {best['metrics']['recall']:<41.4f}|")
    print(f"|   F1-Score:     {best['metrics']['f1']:<41.4f}|")
    print(f"|   Total Time:   {elapsed:<41.1f}|")
    print("+" + "=" * 58 + "+")
    print("|   Next steps:                                            |")
    print("|     1. Review plots/ for visual analysis                  |")
    print("|     2. Launch API: uvicorn api.app:app --reload           |")
    print("|     3. View MLflow: mlflow ui --port 5000                 |")
    print("+" + "=" * 58 + "+")


if __name__ == "__main__":
    main()
