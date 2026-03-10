"""
Model Tests - Unit tests for ML model
Run with: pytest tests/test_model.py -v

Author: 7afnawi
"""

import pytest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


class TestModelLoading:
    """Test model loading and basic functionality."""
    
    def test_model_file_exists(self):
        """Test that model file exists."""
        model_files = list(MODEL_DIR.glob("*.joblib"))
        assert len(model_files) > 0, "No model files found"
    
    def test_model_loads_successfully(self):
        """Test model can be loaded without errors."""
        model_path = MODEL_DIR / "best_model_v2.joblib"
        if not model_path.exists():
            model_path = MODEL_DIR / "best_model.joblib"
        
        model = joblib.load(model_path)
        assert model is not None
    
    def test_model_has_predict_proba(self):
        """Test model has predict_proba method."""
        model_path = MODEL_DIR / "best_model_v2.joblib"
        if not model_path.exists():
            model_path = MODEL_DIR / "best_model.joblib"
        
        model = joblib.load(model_path)
        assert hasattr(model, 'predict_proba')


class TestModelPredictions:
    """Test model predictions."""
    
    @pytest.fixture
    def model(self):
        """Load the model."""
        model_path = MODEL_DIR / "best_model_v2.joblib"
        if not model_path.exists():
            model_path = MODEL_DIR / "best_model.joblib"
        return joblib.load(model_path)
    
    @pytest.fixture
    def features(self):
        """Load feature list."""
        import json
        feat_path = MODEL_DIR / "selected_features_v2.json"
        if not feat_path.exists():
            feat_path = MODEL_DIR / "selected_features.json"
        with open(feat_path) as f:
            return json.load(f)
    
    def test_prediction_returns_probability(self, model, features):
        """Test prediction returns valid probability."""
        # Create dummy input
        X = pd.DataFrame({feat: [0.5] for feat in features})
        
        proba = model.predict_proba(X)
        assert proba.shape[1] == 2  # Binary classification
        assert 0 <= proba[0][1] <= 1  # Valid probability
    
    def test_prediction_range(self, model, features):
        """Test all predictions are in valid range."""
        # Multiple samples
        n_samples = 10
        X = pd.DataFrame({feat: np.random.random(n_samples) for feat in features})
        
        proba = model.predict_proba(X)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
    
    def test_prediction_sums_to_one(self, model, features):
        """Test probabilities sum to 1."""
        X = pd.DataFrame({feat: [0.5] for feat in features})
        
        proba = model.predict_proba(X)
        assert np.isclose(proba[0].sum(), 1.0)


class TestFeatureImportance:
    """Test feature importance data."""
    
    def test_feature_importance_exists(self):
        """Test feature importance file exists."""
        importance_path = MODEL_DIR / "feature_importance.csv"
        assert importance_path.exists() or len(list(MODEL_DIR.glob("*importance*"))) > 0
    
    def test_feature_importance_format(self):
        """Test feature importance has correct format."""
        importance_path = MODEL_DIR / "feature_importance.csv"
        if importance_path.exists():
            df = pd.read_csv(importance_path)
            assert "feature" in df.columns
            assert "importance" in df.columns
            assert len(df) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
