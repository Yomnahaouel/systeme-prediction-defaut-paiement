"""
model_ensemble.py — Ensemble Model Combining Top Performers

Creates a weighted voting ensemble of the best models:
- CatBoost (best single model)
- LightGBM (fast, good performance)
- XGBoost (solid baseline)

Author: 7afnawi
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


class WeightedEnsemble:
    """
    Weighted ensemble of multiple models.
    
    Combines predictions using weighted average of probabilities.
    """
    
    def __init__(self, models=None, weights=None):
        """
        Initialize ensemble.
        
        Args:
            models: List of (name, model) tuples
            weights: List of weights (same length as models)
        """
        self.models = models or []
        self.weights = weights
        self.fitted = False
    
    def add_model(self, name, model, weight=1.0):
        """Add a model to the ensemble."""
        self.models.append((name, model))
        if self.weights is None:
            self.weights = [weight]
        else:
            self.weights.append(weight)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using weighted average.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of shape (n_samples, 2) with probabilities
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Normalize weights
        weights = np.array(self.weights) / np.sum(self.weights)
        
        # Collect predictions
        predictions = []
        for (name, model), weight in zip(self.models, weights):
            try:
                proba = model.predict_proba(X)
                predictions.append(proba * weight)
            except Exception as e:
                print(f"Warning: Model {name} failed: {e}")
        
        # Weighted average
        if predictions:
            final_proba = np.sum(predictions, axis=0)
            return final_proba
        
        raise ValueError("All models failed")
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
            threshold: Classification threshold
        
        Returns:
            Array of predictions (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)


def load_trained_models():
    """
    Load all trained models from the models directory.
    
    Returns:
        Dictionary of {name: model}
    """
    models = {}
    model_dir = config.MODEL_DIR
    
    # Priority order: optimized > v2 > original
    model_files = [
        ("catboost_optimized", "catboost_optimized.joblib"),
        ("catboost", "CatBoost.joblib"),
        ("lightgbm", "LightGBM.joblib"),
        ("lightgbm_competition", "lgb_competition.joblib"),
        ("xgboost", "XGBoost.joblib"),
        ("random_forest", "RandomForest.joblib"),
    ]
    
    for name, filename in model_files:
        path = model_dir / filename
        if path.exists():
            try:
                models[name] = joblib.load(path)
                print(f"✓ Loaded {name}: {filename}")
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
    
    return models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate all models on test set.
    
    Args:
        models: Dictionary of {name: model}
        X_test: Test features
        y_test: Test labels
    
    Returns:
        DataFrame with evaluation metrics
    """
    results = []
    
    for name, model in models.items():
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            
            results.append({
                'model': name,
                'auc': roc_auc_score(y_test, y_proba),
                'f1': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred)
            })
        except Exception as e:
            print(f"Warning: Could not evaluate {name}: {e}")
    
    df = pd.DataFrame(results).sort_values('auc', ascending=False)
    return df


def create_ensemble(models, evaluation_results, top_n=3):
    """
    Create weighted ensemble from top models.
    
    Args:
        models: Dictionary of {name: model}
        evaluation_results: DataFrame from evaluate_models
        top_n: Number of top models to include
    
    Returns:
        WeightedEnsemble instance
    """
    # Get top N models by AUC
    top_models = evaluation_results.head(top_n)
    
    # Calculate weights based on AUC (normalized)
    auc_scores = top_models['auc'].values
    weights = auc_scores / auc_scores.sum()
    
    # Create ensemble
    ensemble = WeightedEnsemble()
    
    print(f"\nCreating ensemble with top {top_n} models:")
    for (_, row), weight in zip(top_models.iterrows(), weights):
        name = row['model']
        if name in models:
            ensemble.add_model(name, models[name], weight)
            print(f"  • {name}: weight={weight:.3f} (AUC={row['auc']:.4f})")
    
    return ensemble


def run_ensemble_pipeline(X_test, y_test, save=True):
    """
    Full ensemble creation and evaluation pipeline.
    
    Args:
        X_test: Test features
        y_test: Test labels
        save: Whether to save the ensemble
    
    Returns:
        Dictionary with ensemble and results
    """
    print("\n" + "═" * 50)
    print("MODEL ENSEMBLE PIPELINE")
    print("═" * 50 + "\n")
    
    # Load models
    print("Loading trained models...")
    models = load_trained_models()
    
    if len(models) < 2:
        print("✗ Need at least 2 models for ensemble")
        return None
    
    # Evaluate individual models
    print("\nEvaluating individual models...")
    eval_results = evaluate_models(models, X_test, y_test)
    print("\nIndividual Model Performance:")
    print(eval_results.to_string(index=False))
    
    # Create ensemble
    ensemble = create_ensemble(models, eval_results, top_n=3)
    
    # Evaluate ensemble
    print("\nEvaluating ensemble...")
    y_proba_ensemble = ensemble.predict_proba(X_test)[:, 1]
    y_pred_ensemble = (y_proba_ensemble >= 0.5).astype(int)
    
    ensemble_metrics = {
        'model': 'ENSEMBLE',
        'auc': roc_auc_score(y_test, y_proba_ensemble),
        'f1': f1_score(y_test, y_pred_ensemble),
        'precision': precision_score(y_test, y_pred_ensemble),
        'recall': recall_score(y_test, y_pred_ensemble)
    }
    
    # Compare
    best_single = eval_results.iloc[0]
    
    print(f"\n{'─' * 50}")
    print("ENSEMBLE vs BEST SINGLE MODEL:")
    print(f"{'─' * 50}")
    print(f"Best Single ({best_single['model']}):")
    print(f"  AUC: {best_single['auc']:.4f}")
    print(f"  F1:  {best_single['f1']:.4f}")
    print(f"\nEnsemble:")
    print(f"  AUC: {ensemble_metrics['auc']:.4f}")
    print(f"  F1:  {ensemble_metrics['f1']:.4f}")
    
    improvement = ensemble_metrics['auc'] - best_single['auc']
    if improvement > 0:
        print(f"\n✓ Ensemble improves AUC by {improvement:.4f}")
    else:
        print(f"\n→ Best single model is better (diff: {improvement:.4f})")
    
    if save:
        # Save ensemble
        ensemble_path = config.MODEL_DIR / "ensemble_model.joblib"
        joblib.dump(ensemble, ensemble_path)
        print(f"\n✓ Ensemble saved: {ensemble_path}")
        
        # Save comparison results
        all_results = pd.concat([
            eval_results,
            pd.DataFrame([ensemble_metrics])
        ], ignore_index=True)
        
        results_path = config.MODEL_DIR / "ensemble_comparison.csv"
        all_results.to_csv(results_path, index=False)
        print(f"✓ Comparison saved: {results_path}")
    
    print("\n" + "═" * 50)
    print("Ensemble Pipeline Complete!")
    print("═" * 50 + "\n")
    
    return {
        'ensemble': ensemble,
        'individual_results': eval_results,
        'ensemble_metrics': ensemble_metrics,
        'improvement': improvement
    }


if __name__ == "__main__":
    # Load test data
    print("Loading test data...")
    
    data_path = config.DATA_DIR / "feature_matrix.csv"
    if data_path.exists():
        # Load features
        features_path = config.MODEL_DIR / "selected_features_v2.json"
        if not features_path.exists():
            features_path = config.MODEL_DIR / "selected_features.json"
        
        with open(features_path) as f:
            features = json.load(f)
        
        df = pd.read_csv(data_path, nrows=50000)
        X = df[features].fillna(0)
        y = df['TARGET']
        
        # Split
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Run pipeline
        results = run_ensemble_pipeline(X_test, y_test)
    else:
        print("✗ Data file not found")
