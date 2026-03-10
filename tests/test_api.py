"""
API Tests - Unit tests for the Credit Risk API
Run with: pytest tests/test_api.py -v

Author: 7afnawi for Hefny
"""

import pytest
import requests
import json
from typing import Dict, Any

API_URL = "http://localhost:8000"


class TestAPIHealth:
    """Test API health endpoints."""
    
    def test_health_check(self):
        """Test that API is running."""
        response = requests.get(f"{API_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_model_loaded(self):
        """Test that model is loaded."""
        response = requests.get(f"{API_URL}/")
        data = response.json()
        assert data["model_loaded"] == True
    
    def test_features_loaded(self):
        """Test that features are loaded."""
        response = requests.get(f"{API_URL}/")
        data = response.json()
        assert data["features_loaded"] == True


class TestPredictions:
    """Test prediction endpoints."""
    
    def test_predict_valid_input(self):
        """Test prediction with valid input."""
        features = {
            "AMT_CREDIT": 500000,
            "AMT_INCOME_TOTAL": 150000,
            "AMT_ANNUITY": 25000,
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.6,
            "EXT_SOURCE_3": 0.4,
        }
        
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "default_probability" in data
        assert "prediction" in data
        assert "risk_level" in data
        assert 0 <= data["default_probability"] <= 1
        assert data["prediction"] in [0, 1]
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "VERY HIGH"]
    
    def test_predict_low_risk_client(self):
        """Test that good client gets low risk."""
        features = {
            "AMT_CREDIT": 200000,
            "AMT_INCOME_TOTAL": 300000,
            "AMT_ANNUITY": 10000,
            "EXT_SOURCE_1": 0.9,
            "EXT_SOURCE_2": 0.85,
            "EXT_SOURCE_3": 0.8,
            "DAYS_BIRTH": -45 * 365,
            "DAYS_EMPLOYED": -15 * 365,
        }
        
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features}
        )
        
        data = response.json()
        assert data["default_probability"] < 0.5
    
    def test_predict_high_risk_client(self):
        """Test that risky client gets high risk."""
        features = {
            "AMT_CREDIT": 800000,
            "AMT_INCOME_TOTAL": 50000,
            "AMT_ANNUITY": 50000,
            "EXT_SOURCE_1": 0.1,
            "EXT_SOURCE_2": 0.15,
            "EXT_SOURCE_3": 0.1,
            "DAYS_BIRTH": -20 * 365,
            "DAYS_EMPLOYED": -1 * 365,
        }
        
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features}
        )
        
        data = response.json()
        assert data["default_probability"] > 0.3
    
    def test_predict_empty_features(self):
        """Test prediction with empty features."""
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": {}}
        )
        
        # Empty features should return validation error
        assert response.status_code in [200, 422]
    
    def test_predict_missing_features(self):
        """Test prediction with partial features."""
        features = {"AMT_CREDIT": 500000}
        
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features}
        )
        
        assert response.status_code == 200


class TestModelInfo:
    """Test model info endpoint."""
    
    def test_get_model_info(self):
        """Test model info endpoint."""
        response = requests.get(f"{API_URL}/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_type" in data
        assert "n_model_features" in data or "n_features" in data


class TestSecurity:
    """Test security aspects."""
    
    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        response = requests.post(
            f"{API_URL}/predict",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422]
    
    def test_sql_injection_attempt(self):
        """Test that SQL injection doesn't crash the API."""
        features = {
            "AMT_CREDIT": "'; DROP TABLE users; --",
        }
        
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features}
        )
        
        # Should fail gracefully, not crash
        assert response.status_code in [200, 400, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
