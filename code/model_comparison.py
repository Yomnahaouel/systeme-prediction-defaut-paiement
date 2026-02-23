"""
model_comparison.py — Compare Models & Select Best

Comparison criteria:
  1. ROC-AUC (discriminative power)
  2. Recall (detection des défauts — business critical)
  3. F1-score (harmonic mean precision/recall)
  4. Training time (efficiency)
  5. Interpretability assessment

Outputs:
  - Comparison table (console + CSV)
  - Combined ROC curves plot
  - Best model selection with justification
  - SHAP interpretability for the best model

Author: Yomna Haouel
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import warnings

import config

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# BUILD COMPARISON TABLE
# ──────────────────────────────────────────────

def build_comparison_table(results: list) -> pd.DataFrame:
    """Build a comparison DataFrame from training results."""
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({
            "Model": r["model_name"],
            "ROC-AUC": round(m["roc_auc"], 4),
            "Recall": round(m["recall"], 4),
            "Precision": round(m["precision"], 4),
            "F1-Score": round(m["f1"], 4),
            "Train Time (s)": m["train_time_sec"],
        })

    df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# COMBINED ROC CURVES
# ──────────────────────────────────────────────

def plot_combined_roc(results: list, X_test, y_test):
    """Plot all models' ROC curves on a single chart."""
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    for i, r in enumerate(results):
        model = r["model"]
        name = r["model_name"]
        color = colors[i % len(colors)]

        X_te = X_test.fillna(0) if hasattr(X_test, 'fillna') else X_test
        y_proba = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)

        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={auc_val:.4f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.PLOT_DIR_COMPARISON / "roc_all_models.png",
                dpi=120, bbox_inches="tight")
    plt.close("all")
    print("  → Saved: roc_all_models.png")


# ──────────────────────────────────────────────
# METRICS BAR CHART
# ──────────────────────────────────────────────

def plot_metrics_comparison(comparison_df: pd.DataFrame):
    """Bar chart comparing key metrics across models."""
    metrics_to_plot = ["ROC-AUC", "Recall", "F1-Score"]
    df_melted = comparison_df.melt(
        id_vars="Model",
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Score"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric",
                palette=["#e74c3c", "#3498db", "#2ecc71"], ax=ax)
    ax.set_title("Model Comparison — Key Metrics", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.PLOT_DIR_COMPARISON / "metrics_comparison.png",
                dpi=120, bbox_inches="tight")
    plt.close("all")
    print("  → Saved: metrics_comparison.png")


# ──────────────────────────────────────────────
# SELECT BEST MODEL
# ──────────────────────────────────────────────

def select_best_model(results: list, comparison_df: pd.DataFrame) -> dict:
    """
    Select the best model based on weighted criteria.

    Scoring:
      - 50% ROC-AUC (primary metric)
      - 30% Recall (detecting defaults is business-critical)
      - 20% F1-Score (balance precision/recall)
    """
    print("\n" + "─" * 55)
    print("  BEST MODEL SELECTION")
    print("─" * 55)

    # Weighted score
    comparison_df = comparison_df.copy()
    comparison_df["Weighted_Score"] = (
        0.50 * comparison_df["ROC-AUC"] +
        0.30 * comparison_df["Recall"] +
        0.20 * comparison_df["F1-Score"]
    )

    best_idx = comparison_df["Weighted_Score"].idxmax()
    best_name = comparison_df.loc[best_idx, "Model"]

    best_result = [r for r in results if r["model_name"] == best_name][0]
    best_model = best_result["model"]

    print(f"\n  [BEST] Best Model: {best_name}")
    print(f"     ROC-AUC:  {comparison_df.loc[best_idx, 'ROC-AUC']:.4f}")
    print(f"     Recall:   {comparison_df.loc[best_idx, 'Recall']:.4f}")
    print(f"     F1-Score: {comparison_df.loc[best_idx, 'F1-Score']:.4f}")
    print(f"     Weighted: {comparison_df.loc[best_idx, 'Weighted_Score']:.4f}")

    # Save best model
    joblib.dump(best_model, config.BEST_MODEL_PATH)
    print(f"\n  Best model saved: {config.BEST_MODEL_PATH.name}")

    return best_result


# ──────────────────────────────────────────────
# SHAP INTERPRETABILITY
# ──────────────────────────────────────────────

def shap_analysis(best_result: dict, X_test: pd.DataFrame):
    """Generate SHAP summary plot for the best model."""
    try:
        import shap
    except ImportError:
        print("  ⚠ SHAP not installed, skipping interpretability analysis")
        return

    print("\n  Running SHAP analysis...")

    model = best_result["model"]
    model_name = best_result["model_name"]

    X_sample = X_test.fillna(0).sample(
        min(500, len(X_test)), random_state=config.RANDOM_STATE
    )

    try:
        if model_name in ["XGBoost", "LightGBM", "CatBoost", "RandomForest"]:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_sample)

        shap_values = explainer.shap_values(X_sample)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 (default)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar",
                         max_display=20, show=False)
        plt.title(f"SHAP Feature Importance — {model_name}",
                 fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(config.PLOT_DIR_COMPARISON / "shap_importance.png",
                    dpi=120, bbox_inches="tight")
        plt.close("all")
        print("  → Saved: shap_importance.png")

    except Exception as e:
        print(f"  ⚠ SHAP analysis failed: {e}")


# ──────────────────────────────────────────────
# JUSTIFICATION TEXT
# ──────────────────────────────────────────────

def print_justification(best_result: dict, comparison_df: pd.DataFrame):
    """Print model selection justification for presentation."""
    name = best_result["model_name"]
    m = best_result["metrics"]

    interpretability = {
        "LogisticRegression": "Very High (linear coefficients)",
        "RandomForest": "Medium (feature importance)",
        "XGBoost": "Medium (SHAP, feature importance)",
        "LightGBM": "Medium (SHAP, feature importance)",
        "CatBoost": "Medium (SHAP, feature importance)",
    }

    print(f"""
+{"=" * 58}+
|           JUSTIFICATION DU CHOIX DU MODELE               |
+{"=" * 58}+
|
|  Modele retenu : {name}
|
|  Performance :
|    - ROC-AUC   = {m['roc_auc']:.4f}
|    - Recall    = {m['recall']:.4f}  (detection des defauts)
|    - F1-Score  = {m['f1']:.4f}
|    - Temps     = {m['train_time_sec']:.1f}s
|
|  Interpretabilite : {interpretability.get(name, 'N/A')}
|
|  Justification :
|    -> Meilleur compromis AUC / Recall / Vitesse
|    -> Le Recall est prioritaire (cout d'un defaut non
|       detecte >> cout d'un faux positif)
|    -> Compatible avec SHAP pour l'interpretabilite
|
+{"=" * 58}+
""")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def compare_models(results: list, X_test, y_test) -> dict:
    """Full model comparison pipeline."""
    print("\n" + "=" * 60)
    print("   MODEL COMPARISON & FINAL SELECTION")
    print("=" * 60)

    # Comparison table
    comparison_df = build_comparison_table(results)
    print("\n  Comparison Table:")
    print(comparison_df.to_string(index=False))

    # Save table
    csv_path = config.PLOT_DIR_COMPARISON / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n  → Table saved: {csv_path.name}")

    # Plots
    plot_combined_roc(results, X_test, y_test)
    plot_metrics_comparison(comparison_df)

    # Select best
    best = select_best_model(results, comparison_df)

    # SHAP
    shap_analysis(best, X_test)

    # Justification
    print_justification(best, comparison_df)

    return best
