"""
Pytest Configuration and Fixtures
Credit Risk Prediction System

Author: 7afnawi
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def api_url():
    """API base URL for testing."""
    return "http://localhost:8000"


@pytest.fixture
def sample_low_risk_client():
    """Sample low-risk client features."""
    return {
        "AMT_INCOME_TOTAL": 300000,
        "AMT_CREDIT": 200000,
        "AMT_ANNUITY": 10000,
        "AMT_GOODS_PRICE": 180000,
        "DAYS_BIRTH": -45 * 365,
        "DAYS_EMPLOYED": -15 * 365,
        "CNT_FAM_MEMBERS": 3,
        "CNT_CHILDREN": 1,
        "EXT_SOURCE_1": 0.8,
        "EXT_SOURCE_2": 0.85,
        "EXT_SOURCE_3": 0.75,
    }


@pytest.fixture
def sample_high_risk_client():
    """Sample high-risk client features."""
    return {
        "AMT_INCOME_TOTAL": 40000,
        "AMT_CREDIT": 500000,
        "AMT_ANNUITY": 40000,
        "AMT_GOODS_PRICE": 480000,
        "DAYS_BIRTH": -22 * 365,
        "DAYS_EMPLOYED": -180,
        "CNT_FAM_MEMBERS": 5,
        "CNT_CHILDREN": 3,
        "EXT_SOURCE_1": 0.15,
        "EXT_SOURCE_2": 0.20,
        "EXT_SOURCE_3": 0.10,
    }


@pytest.fixture
def batch_clients(sample_low_risk_client, sample_high_risk_client):
    """Batch of clients for testing."""
    return [sample_low_risk_client, sample_high_risk_client]
