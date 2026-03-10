"""
eda.py — Exploratory Data Analysis (EDA)

Analyses complètes :
  1. Distribution de TARGET et taux de défaut
  2. Analyse des variables numériques (distributions, outliers)
  3. Analyse des variables catégorielles (taux de défaut par modalité)
  4. Valeurs manquantes
  5. Corrélation avec TARGET
  6. Recommandations métier

Toutes les figures sont sauvegardées dans plots/eda/

Author: Yomna Haouel
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time
from functools import wraps
import warnings

import config

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 11

SAMPLE_SIZE = 50000


# ──────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────

def timer(func):
    """Decorator to time each EDA step."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        print(f"\n▶ {func.__name__}")
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  ✓ Done in {elapsed:.1f}s")
        return result
    return wrapper


def save_plot(name: str):
    """Save current figure and close."""
    path = config.PLOT_DIR_EDA / name
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    print(f"  → Saved: {path.name}")


# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────

@timer
def load_data():
    """Load feature_matrix.csv, keep only train rows."""
    df = pd.read_csv(config.FEATURE_MATRIX_PATH)
    print(f"  Raw shape: {df.shape}")

    # Keep train only (where TARGET != -999 and set == 'train')
    if config.SET_COL in df.columns:
        df = df[df[config.SET_COL] == "train"].drop(columns=[config.SET_COL])

    df = df[df[config.TARGET_COL].notnull() & (df[config.TARGET_COL] != -999)]
    df[config.TARGET_COL] = df[config.TARGET_COL].astype(int)

    # Downcast to save memory
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"  Train shape: {df.shape} | Memory: {mem_mb:.1f} MB")
    return df


# ──────────────────────────────────────────────
# 2. TARGET DISTRIBUTION
# ──────────────────────────────────────────────

@timer
def analyze_target(df):
    """Analyze target class distribution and imbalance ratio."""
    counts = df[config.TARGET_COL].value_counts()
    default_rate = df[config.TARGET_COL].mean() * 100

    print(f"  Class 0 (No default): {counts.get(0, 0):,}")
    print(f"  Class 1 (Default):    {counts.get(1, 0):,}")
    print(f"  Default rate: {default_rate:.2f}%")
    print(f"  Imbalance ratio: 1:{counts.get(0,1) // max(counts.get(1,1), 1)}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bar chart
    sns.countplot(x=config.TARGET_COL, data=df, ax=axes[0],
                  palette=["#2ecc71", "#e74c3c"])
    axes[0].set_title("Target Distribution (Count)")
    for p in axes[0].patches:
        axes[0].annotate(f'{int(p.get_height()):,}',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=10)

    # Pie chart
    axes[1].pie(counts.values, labels=["No Default (0)", "Default (1)"],
                autopct='%1.1f%%', colors=["#2ecc71", "#e74c3c"],
                startangle=90, explode=(0, 0.05))
    axes[1].set_title("Target Proportion")

    plt.suptitle("TARGET — Class Imbalance Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_plot("01_target_distribution.png")

    return default_rate


# ──────────────────────────────────────────────
# 3. MISSING VALUES ANALYSIS
# ──────────────────────────────────────────────

@timer
def analyze_missing(df):
    """Identify columns with high missing rates."""
    missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    n_with_missing = len(missing_pct)
    n_high_missing = (missing_pct > config.MISSING_THRESHOLD * 100).sum()

    print(f"  Columns with any missing: {n_with_missing}")
    print(f"  Columns > {config.MISSING_THRESHOLD*100:.0f}% missing: {n_high_missing}")

    # Plot top 30
    top = missing_pct.head(30).sort_values()
    fig, ax = plt.subplots(figsize=(8, 10))
    top.plot(kind="barh", ax=ax, color="#3498db")
    ax.axvline(x=config.MISSING_THRESHOLD * 100, color="red",
               linestyle="--", label=f"Threshold ({config.MISSING_THRESHOLD*100:.0f}%)")
    ax.set_title("Top 30 Features — Missing Values (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Missing (%)")
    ax.legend()
    plt.tight_layout()
    save_plot("02_missing_values.png")

    return missing_pct


# ──────────────────────────────────────────────
# 4. NUMERIC FEATURES ANALYSIS
# ──────────────────────────────────────────────

@timer
def analyze_numeric(df):
    """Distributions of key numeric features by TARGET."""
    key_cols = [c for c in [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "DAYS_REGISTRATION", "CNT_FAM_MEMBERS"
    ] if c in df.columns]

    if not key_cols:
        print("  No key numeric columns found, skipping.")
        return

    n = len(key_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    sample = df.sample(min(SAMPLE_SIZE, len(df)), random_state=config.RANDOM_STATE)

    for i, col in enumerate(key_cols):
        ax = axes[i]
        for label, color in [(0, "#2ecc71"), (1, "#e74c3c")]:
            subset = sample[sample[config.TARGET_COL] == label][col].dropna()
            ax.hist(subset, bins=50, alpha=0.5, color=color,
                    label=f"Class {label}", density=True)
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Numeric Feature Distributions by TARGET", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_plot("03_numeric_distributions.png")


# ──────────────────────────────────────────────
# 5. OUTLIER DETECTION
# ──────────────────────────────────────────────

@timer
def detect_outliers(df):
    """Detect outliers using IQR method on key financial columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    key_cols = [c for c in [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "DAYS_EMPLOYED", "AMT_GOODS_PRICE"
    ] if c in numeric_cols]

    outlier_stats = {}
    for col in key_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = n_outliers / len(df) * 100
        outlier_stats[col] = {"count": n_outliers, "pct": round(pct, 2)}
        print(f"  {col}: {n_outliers:,} outliers ({pct:.1f}%)")

    if key_cols:
        fig, axes = plt.subplots(1, len(key_cols), figsize=(4 * len(key_cols), 5))
        if len(key_cols) == 1:
            axes = [axes]
        sample = df.sample(min(SAMPLE_SIZE, len(df)), random_state=config.RANDOM_STATE)
        for ax, col in zip(axes, key_cols):
            sns.boxplot(x=config.TARGET_COL, y=col, data=sample, ax=ax,
                        palette=["#2ecc71", "#e74c3c"])
            ax.set_title(col, fontsize=10)
        plt.suptitle("Outlier Analysis (Box Plots)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        save_plot("04_outliers_boxplots.png")

    return outlier_stats


# ──────────────────────────────────────────────
# 6. CATEGORICAL ANALYSIS
# ──────────────────────────────────────────────

@timer
def analyze_categorical(df):
    """Default rate by category for top categorical features."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != config.TARGET_COL and c != config.ID_COL][:6]

    if not cat_cols:
        print("  No categorical columns found.")
        return

    ncols = 2
    nrows = (len(cat_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, col in enumerate(cat_cols):
        rates = df.groupby(col)[config.TARGET_COL].mean().sort_values(ascending=False)
        if len(rates) > 15:
            rates = rates.head(15)
        rates.plot(kind="barh", ax=axes[i], color="#e67e22")
        axes[i].set_title(f"Default Rate — {col}", fontsize=10)
        axes[i].set_xlabel("Default Rate")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Categorical Features — Default Rate", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_plot("05_categorical_default_rate.png")


# ──────────────────────────────────────────────
# 7. CORRELATION WITH TARGET
# ──────────────────────────────────────────────

@timer
def analyze_correlation(df):
    """Top features correlated with TARGET."""
    sample = df.sample(min(SAMPLE_SIZE, len(df)), random_state=config.RANDOM_STATE)
    numeric = sample.select_dtypes(include=[np.number])

    target_corr = numeric.corr()[config.TARGET_COL].abs().sort_values(ascending=False)
    target_corr = target_corr.drop(config.TARGET_COL, errors="ignore")

    print("  Top 15 correlated features with TARGET:")
    for feat, val in target_corr.head(15).items():
        print(f"    {feat}: {val:.4f}")

    # Plot top 20
    top20 = target_corr.head(20).sort_values()
    fig, ax = plt.subplots(figsize=(8, 8))
    top20.plot(kind="barh", ax=ax, color="#9b59b6")
    ax.set_title("Top 20 Features — Correlation with TARGET",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("|Pearson Correlation|")
    plt.tight_layout()
    save_plot("06_top_correlations.png")

    # Correlation heatmap (top 12 features)
    top12_cols = target_corr.head(12).index.tolist() + [config.TARGET_COL]
    corr_sub = numeric[top12_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_sub, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, square=True)
    ax.set_title("Correlation Heatmap — Top Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_plot("07_correlation_heatmap.png")

    return target_corr


# ──────────────────────────────────────────────
# 8. EDA SUMMARY & RECOMMENDATIONS
# ──────────────────────────────────────────────

def print_recommendations(default_rate, missing_pct, outlier_stats, target_corr):
    """Print business recommendations based on EDA findings."""
    print("\n" + "=" * 65)
    print("   RECOMMANDATIONS MÉTIER (Business Insights)")
    print("=" * 65)

    print(f"""
1. DÉSÉQUILIBRE DE CLASSES
   → Taux de défaut = {default_rate:.1f}%
   → Utiliser SMOTE, class_weight='balanced', ou threshold tuning.
   → Métrique principale : AUC-ROC + Recall (classe défaut).

2. VALEURS MANQUANTES
   → {(missing_pct > config.MISSING_THRESHOLD * 100).sum()} colonnes dépassent
     le seuil de {config.MISSING_THRESHOLD*100:.0f}% → à supprimer.
   → Imputation médiane pour les numériques, mode pour catégorielles.

3. OUTLIERS
   → Variables financières (AMT_INCOME, AMT_CREDIT) contiennent des
     valeurs extrêmes. Clipper au 99e percentile ou utiliser des
     modèles robustes (arbres de décision).

4. FEATURES PRÉDICTIVES
   → EXT_SOURCE_1/2/3 sont les meilleurs prédicteurs univariés.
   → Créer des ratios financiers : CREDIT/INCOME, ANNUITY/INCOME.
   → L'âge (DAYS_BIRTH) et l'ancienneté emploi (DAYS_EMPLOYED) sont
     discriminants.

5. VARIABLES CATÉGORIELLES
   → ORGANIZATION_TYPE et OCCUPATION_TYPE montrent des écarts
     significatifs de taux de défaut entre modalités.
   → Envisager un encodage cible (target encoding) avec
     régularisation pour éviter l'overfitting.
""")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run_eda():
    """Run complete EDA pipeline."""
    print("\n" + "=" * 65)
    print("   EXPLORATORY DATA ANALYSIS — Credit Default Risk")
    print("=" * 65)

    df = load_data()

    default_rate = analyze_target(df)
    missing_pct = analyze_missing(df)
    analyze_numeric(df)
    outlier_stats = detect_outliers(df)
    analyze_categorical(df)
    target_corr = analyze_correlation(df)

    print_recommendations(default_rate, missing_pct, outlier_stats, target_corr)

    del df
    gc.collect()
    print("\n✅ EDA complete. Plots saved to:", config.PLOT_DIR_EDA)


if __name__ == "__main__":
    run_eda()
