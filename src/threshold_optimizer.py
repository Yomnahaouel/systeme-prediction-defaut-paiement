"""
threshold_optimizer.py — Business Cost-Sensitive Threshold Optimization

Finds the optimal classification threshold based on business costs:
- Cost of False Negative (approving bad loan): HIGH
- Cost of False Positive (rejecting good customer): LOW

Author: 7afnawi
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    roc_curve, precision_recall_curve, 
    confusion_matrix, f1_score, roc_auc_score
)
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


# ═══════════════════════════════════════════════════════════════════
# BUSINESS COSTS (Configurable)
# ═══════════════════════════════════════════════════════════════════

COST_FN = 10000  # Cost of approving a bad loan (False Negative)
COST_FP = 500    # Cost of rejecting a good customer (False Positive)
COST_TP = 0      # Correctly rejecting bad loan
COST_TN = 0      # Correctly approving good loan


def calculate_total_cost(y_true, y_pred, cost_fp=COST_FP, cost_fn=COST_FN):
    """Calculate total business cost for given predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    return total_cost, {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}


def find_optimal_threshold(y_true, y_proba, cost_fp=COST_FP, cost_fn=COST_FN):
    """
    Find the threshold that minimizes total business cost.
    
    Returns:
        dict with optimal threshold and metrics
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    results = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        cost, cm = calculate_total_cost(y_true, y_pred, cost_fp, cost_fn)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate metrics
        precision = cm['tp'] / (cm['tp'] + cm['fp']) if (cm['tp'] + cm['fp']) > 0 else 0
        recall = cm['tp'] / (cm['tp'] + cm['fn']) if (cm['tp'] + cm['fn']) > 0 else 0
        
        results.append({
            'threshold': thresh,
            'total_cost': cost,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'tp': cm['tp'],
            'fp': cm['fp'],
            'fn': cm['fn'],
            'tn': cm['tn']
        })
    
    df_results = pd.DataFrame(results)
    
    # Find optimal by minimum cost
    optimal_idx = df_results['total_cost'].idxmin()
    optimal = df_results.loc[optimal_idx].to_dict()
    
    # Also find F1-optimal for comparison
    f1_optimal_idx = df_results['f1_score'].idxmax()
    f1_optimal = df_results.loc[f1_optimal_idx].to_dict()
    
    return {
        'cost_optimal': optimal,
        'f1_optimal': f1_optimal,
        'all_results': df_results,
        'cost_fn': cost_fn,
        'cost_fp': cost_fp
    }


def plot_threshold_analysis(results, save_path=None):
    """Create visualization of threshold analysis."""
    df = results['all_results']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Cost vs Threshold
    ax1 = axes[0, 0]
    ax1.plot(df['threshold'], df['total_cost'], 'b-', linewidth=2)
    ax1.axvline(results['cost_optimal']['threshold'], color='r', linestyle='--', 
                label=f"Optimal: {results['cost_optimal']['threshold']:.2f}")
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Total Business Cost ($)')
    ax1.set_title('Business Cost vs Classification Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: F1 Score vs Threshold
    ax2 = axes[0, 1]
    ax2.plot(df['threshold'], df['f1_score'], 'g-', linewidth=2)
    ax2.axvline(results['f1_optimal']['threshold'], color='r', linestyle='--',
                label=f"F1 Optimal: {results['f1_optimal']['threshold']:.2f}")
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score vs Classification Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Precision-Recall vs Threshold
    ax3 = axes[1, 0]
    ax3.plot(df['threshold'], df['precision'], 'b-', linewidth=2, label='Precision')
    ax3.plot(df['threshold'], df['recall'], 'r-', linewidth=2, label='Recall')
    ax3.axvline(results['cost_optimal']['threshold'], color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Score')
    ax3.set_title('Precision & Recall vs Threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    THRESHOLD OPTIMIZATION SUMMARY
    ══════════════════════════════════════
    
    Business Costs:
    • Cost of False Negative (bad loan approved): ${results['cost_fn']:,}
    • Cost of False Positive (good customer rejected): ${results['cost_fp']:,}
    
    Cost-Optimal Threshold: {results['cost_optimal']['threshold']:.3f}
    ────────────────────────────────────
    • Total Cost: ${results['cost_optimal']['total_cost']:,.0f}
    • Precision: {results['cost_optimal']['precision']:.3f}
    • Recall: {results['cost_optimal']['recall']:.3f}
    • F1 Score: {results['cost_optimal']['f1_score']:.3f}
    
    F1-Optimal Threshold: {results['f1_optimal']['threshold']:.3f}
    ────────────────────────────────────
    • Total Cost: ${results['f1_optimal']['total_cost']:,.0f}
    • F1 Score: {results['f1_optimal']['f1_score']:.3f}
    
    Recommendation: Use threshold {results['cost_optimal']['threshold']:.2f}
    for production deployment.
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Threshold analysis plot saved: {save_path}")
    
    plt.close()
    return fig


def run_threshold_optimization(X_test, y_test, model=None, save=True):
    """
    Run full threshold optimization pipeline.
    
    Args:
        X_test: Test features
        y_test: Test labels
        model: Trained model (loads best model if None)
        save: Whether to save results
    
    Returns:
        dict with optimization results
    """
    # Load model if not provided
    if model is None:
        model_path = config.MODEL_DIR / "catboost_optimized.joblib"
        if not model_path.exists():
            model_path = config.MODEL_DIR / "best_model_v2.joblib"
        if not model_path.exists():
            model_path = config.MODEL_DIR / "best_model.joblib"
        model = joblib.load(model_path)
        print(f"✓ Model loaded: {model_path.name}")
    
    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate AUC
    auc = roc_auc_score(y_test, y_proba)
    print(f"✓ Model ROC-AUC: {auc:.4f}")
    
    # Find optimal threshold
    results = find_optimal_threshold(y_test, y_proba)
    results['auc'] = auc
    
    print(f"\n{'═' * 50}")
    print("THRESHOLD OPTIMIZATION RESULTS")
    print(f"{'═' * 50}")
    print(f"Cost-Optimal Threshold: {results['cost_optimal']['threshold']:.3f}")
    print(f"  → Total Cost: ${results['cost_optimal']['total_cost']:,.0f}")
    print(f"  → F1 Score: {results['cost_optimal']['f1_score']:.3f}")
    print(f"  → Recall: {results['cost_optimal']['recall']:.3f}")
    print(f"\nF1-Optimal Threshold: {results['f1_optimal']['threshold']:.3f}")
    print(f"  → F1 Score: {results['f1_optimal']['f1_score']:.3f}")
    print(f"{'═' * 50}\n")
    
    if save:
        # Save plot
        plot_path = config.PLOTS_DIR / "threshold_optimization.png"
        plot_threshold_analysis(results, save_path=plot_path)
        
        # Save threshold config
        threshold_config = {
            'optimal_threshold': results['cost_optimal']['threshold'],
            'f1_optimal_threshold': results['f1_optimal']['threshold'],
            'cost_fn': COST_FN,
            'cost_fp': COST_FP,
            'metrics': {
                'auc': auc,
                'precision': results['cost_optimal']['precision'],
                'recall': results['cost_optimal']['recall'],
                'f1_score': results['cost_optimal']['f1_score']
            }
        }
        
        config_path = config.MODEL_DIR / "threshold_config.json"
        with open(config_path, 'w') as f:
            json.dump(threshold_config, f, indent=2)
        print(f"✓ Threshold config saved: {config_path}")
    
    return results


if __name__ == "__main__":
    # Load test data
    print("Loading test data...")
    
    # Try to load existing split
    data_path = config.DATA_DIR / "feature_matrix.csv"
    if data_path.exists():
        df = pd.read_csv(data_path, nrows=50000)  # Sample for speed
        
        # Load features
        features_path = config.MODEL_DIR / "selected_features_v2.json"
        if not features_path.exists():
            features_path = config.MODEL_DIR / "selected_features.json"
        
        with open(features_path) as f:
            features = json.load(f)
        
        # Prepare data
        X = df[features].fillna(0)
        y = df['TARGET']
        
        # Split
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Run optimization
        results = run_threshold_optimization(X_test, y_test)
    else:
        print("✗ Data file not found. Please run preprocessing first.")
