"""
shap_explainer.py — SHAP Model Explainability

Provides global and local explanations for model predictions:
- Global feature importance (summary plot)
- Individual prediction explanations (waterfall/force plots)

Author: 7afnawi
"""

import numpy as np
import pandas as pd
import joblib
import json
import shap
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


class SHAPExplainer:
    """SHAP-based model explainer."""
    
    def __init__(self, model=None, feature_names=None):
        """
        Initialize explainer.
        
        Args:
            model: Trained model (loads best model if None)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.background_data = None
        
        if model is None:
            self._load_model()
        
        if feature_names is None:
            self._load_features()
    
    def _load_model(self):
        """Load the best trained model."""
        model_path = config.MODEL_DIR / "catboost_optimized.joblib"
        if not model_path.exists():
            model_path = config.MODEL_DIR / "best_model_v2.joblib"
        if not model_path.exists():
            model_path = config.MODEL_DIR / "best_model.joblib"
        
        self.model = joblib.load(model_path)
        print(f"✓ Model loaded: {model_path.name}")
    
    def _load_features(self):
        """Load feature names."""
        features_path = config.MODEL_DIR / "selected_features_v2.json"
        if not features_path.exists():
            features_path = config.MODEL_DIR / "selected_features.json"
        
        with open(features_path) as f:
            self.feature_names = json.load(f)
        print(f"✓ Features loaded: {len(self.feature_names)} features")
    
    def create_explainer(self, background_data=None, max_samples=100):
        """
        Create SHAP explainer with background data.
        
        Args:
            background_data: DataFrame for background distribution
            max_samples: Max samples for background (for speed)
        """
        if background_data is not None:
            # Sample for speed
            if len(background_data) > max_samples:
                background_data = background_data.sample(n=max_samples, random_state=42)
            self.background_data = background_data
        
        # Use TreeExplainer for tree-based models
        model_type = type(self.model).__name__
        
        if 'CatBoost' in model_type or 'LGBM' in model_type or 'XGB' in model_type or 'Forest' in model_type:
            self.explainer = shap.TreeExplainer(self.model)
            print(f"✓ TreeExplainer created for {model_type}")
        else:
            # Fallback to KernelExplainer
            if self.background_data is not None:
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    self.background_data
                )
                print(f"✓ KernelExplainer created for {model_type}")
            else:
                raise ValueError("Background data required for KernelExplainer")
        
        return self.explainer
    
    def calculate_shap_values(self, X, check_additivity=False):
        """
        Calculate SHAP values for given data.
        
        Args:
            X: DataFrame or array of features
            check_additivity: Whether to verify SHAP additivity
        
        Returns:
            shap_values array
        """
        if self.explainer is None:
            self.create_explainer()
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        self.shap_values = self.explainer.shap_values(X_array, check_additivity=check_additivity)
        
        # For binary classification, shap_values might be a list
        if isinstance(self.shap_values, list):
            # Take class 1 (default) values
            self.shap_values = self.shap_values[1]
        
        return self.shap_values
    
    def get_feature_importance(self, X=None, shap_values=None):
        """
        Get global feature importance based on SHAP values.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if shap_values is None:
            if self.shap_values is None:
                if X is not None:
                    self.calculate_shap_values(X)
                else:
                    raise ValueError("Need X data or pre-calculated shap_values")
            shap_values = self.shap_values
        
        # Mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)
        
        df_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df_importance
    
    def plot_summary(self, X, max_display=20, save_path=None):
        """
        Create SHAP summary plot (global feature importance).
        
        Args:
            X: DataFrame of features
            max_display: Number of top features to show
            save_path: Path to save plot
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            self.shap_values, 
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        plt.title("SHAP Feature Importance (Impact on Default Prediction)", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Summary plot saved: {save_path}")
        
        plt.close()
    
    def plot_bar(self, X, max_display=20, save_path=None):
        """Create SHAP bar plot (simpler view)."""
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            plot_type="bar",
            show=False
        )
        
        plt.title("SHAP Feature Importance", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Bar plot saved: {save_path}")
        
        plt.close()
    
    def explain_prediction(self, X_single, return_dict=True):
        """
        Explain a single prediction.
        
        Args:
            X_single: Single row DataFrame or array
            return_dict: Return as dictionary
        
        Returns:
            Dictionary with explanation or SHAP values
        """
        if self.explainer is None:
            self.create_explainer()
        
        if isinstance(X_single, pd.DataFrame):
            X_array = X_single.values
        else:
            X_array = X_single.reshape(1, -1) if X_single.ndim == 1 else X_single
        
        shap_vals = self.explainer.shap_values(X_array, check_additivity=False)
        
        # Handle list output for binary classification
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        
        if return_dict:
            # Get top contributing features
            contributions = list(zip(self.feature_names, shap_vals[0]))
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Get base value
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1]  # Class 1
            
            return {
                'base_value': float(base_value),
                'prediction_contribution': float(np.sum(shap_vals[0])),
                'top_features': [
                    {'feature': f, 'contribution': float(c)}
                    for f, c in contributions[:10]
                ],
                'all_contributions': {f: float(c) for f, c in contributions}
            }
        
        return shap_vals
    
    def plot_waterfall(self, X_single, save_path=None):
        """Create waterfall plot for single prediction."""
        if self.explainer is None:
            self.create_explainer()
        
        if isinstance(X_single, pd.DataFrame):
            X_array = X_single.values
        else:
            X_array = X_single.reshape(1, -1) if X_single.ndim == 1 else X_single
        
        # Create explanation object
        shap_vals = self.explainer(X_array)
        
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap_vals[0], max_display=15, show=False)
        plt.title("SHAP Waterfall Plot - Individual Prediction Explanation", fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Waterfall plot saved: {save_path}")
        
        plt.close()


def generate_shap_analysis(X_sample, save_plots=True):
    """
    Generate full SHAP analysis and save results.
    
    Args:
        X_sample: DataFrame with features (sample of data)
        save_plots: Whether to save plots
    
    Returns:
        SHAPExplainer instance with results
    """
    print("\n" + "═" * 50)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("═" * 50 + "\n")
    
    # Create explainer
    explainer = SHAPExplainer()
    explainer.create_explainer()
    
    # Calculate SHAP values
    print("Calculating SHAP values...")
    explainer.calculate_shap_values(X_sample)
    
    # Get feature importance
    importance_df = explainer.get_feature_importance()
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    if save_plots:
        # Create plots directory
        plots_dir = config.PLOTS_DIR / "shap"
        plots_dir.mkdir(exist_ok=True)
        
        # Generate plots
        print("\nGenerating SHAP plots...")
        explainer.plot_summary(X_sample, save_path=plots_dir / "shap_summary.png")
        explainer.plot_bar(X_sample, save_path=plots_dir / "shap_bar.png")
        
        # Example individual explanation
        explainer.plot_waterfall(X_sample.iloc[[0]], save_path=plots_dir / "shap_waterfall_example.png")
        
        # Save feature importance
        importance_df.to_csv(config.MODEL_DIR / "shap_feature_importance.csv", index=False)
        print(f"✓ Feature importance saved to models/shap_feature_importance.csv")
    
    print("\n" + "═" * 50)
    print("SHAP Analysis Complete!")
    print("═" * 50 + "\n")
    
    return explainer


if __name__ == "__main__":
    # Load sample data
    print("Loading data for SHAP analysis...")
    
    data_path = config.DATA_DIR / "feature_matrix.csv"
    if data_path.exists():
        # Load features
        features_path = config.MODEL_DIR / "selected_features_v2.json"
        if not features_path.exists():
            features_path = config.MODEL_DIR / "selected_features.json"
        
        with open(features_path) as f:
            features = json.load(f)
        
        # Load sample (500 rows for speed)
        df = pd.read_csv(data_path, nrows=5000)
        X_sample = df[features].fillna(0).sample(n=500, random_state=42)
        
        # Run analysis
        explainer = generate_shap_analysis(X_sample)
    else:
        print("✗ Data file not found")
