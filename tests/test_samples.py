"""
Test Samples - Realistic client profiles for model testing
Run with: python tests/test_samples.py

Author: 7afnawi for Hefny
"""

import requests
import json
from typing import Dict, List

API_URL = "http://localhost:8000"

# ═══════════════════════════════════════════════════════════════════
# TEST CASES - 10 Realistic Client Profiles
# ═══════════════════════════════════════════════════════════════════

TEST_CASES: List[Dict] = [
    # ─────────────────────────────────────────────────────────────────
    # LOW RISK CLIENTS (Expected: < 20% probability)
    # ─────────────────────────────────────────────────────────────────
    {
        "name": "1. Perfect Client - Senior Executive",
        "expected_risk": "LOW",
        "features": {
            "AMT_INCOME_TOTAL": 450000,
            "AMT_CREDIT": 300000,
            "AMT_ANNUITY": 15000,
            "AMT_GOODS_PRICE": 280000,
            "DAYS_BIRTH": -50 * 365,  # 50 years old
            "DAYS_EMPLOYED": -20 * 365,  # 20 years employed
            "CNT_FAM_MEMBERS": 4,
            "CNT_CHILDREN": 2,
            "EXT_SOURCE_1": 0.85,
            "EXT_SOURCE_2": 0.90,
            "EXT_SOURCE_3": 0.80,
        }
    },
    {
        "name": "2. Stable Professional - Doctor",
        "expected_risk": "LOW",
        "features": {
            "AMT_INCOME_TOTAL": 300000,
            "AMT_CREDIT": 400000,
            "AMT_ANNUITY": 20000,
            "AMT_GOODS_PRICE": 380000,
            "DAYS_BIRTH": -42 * 365,
            "DAYS_EMPLOYED": -15 * 365,
            "CNT_FAM_MEMBERS": 3,
            "CNT_CHILDREN": 1,
            "EXT_SOURCE_1": 0.75,
            "EXT_SOURCE_2": 0.82,
            "EXT_SOURCE_3": 0.70,
        }
    },
    {
        "name": "3. Homeowner with History - Teacher",
        "expected_risk": "LOW",
        "features": {
            "AMT_INCOME_TOTAL": 120000,
            "AMT_CREDIT": 150000,
            "AMT_ANNUITY": 8000,
            "AMT_GOODS_PRICE": 140000,
            "DAYS_BIRTH": -55 * 365,
            "DAYS_EMPLOYED": -25 * 365,
            "CNT_FAM_MEMBERS": 2,
            "CNT_CHILDREN": 0,
            "EXT_SOURCE_1": 0.70,
            "EXT_SOURCE_2": 0.75,
            "EXT_SOURCE_3": 0.65,
        }
    },
    
    # ─────────────────────────────────────────────────────────────────
    # MEDIUM RISK CLIENTS (Expected: 20-50% probability)
    # ─────────────────────────────────────────────────────────────────
    {
        "name": "4. Young Professional - Entry Level",
        "expected_risk": "MEDIUM",
        "features": {
            "AMT_INCOME_TOTAL": 80000,
            "AMT_CREDIT": 250000,
            "AMT_ANNUITY": 18000,
            "AMT_GOODS_PRICE": 230000,
            "DAYS_BIRTH": -28 * 365,
            "DAYS_EMPLOYED": -3 * 365,
            "CNT_FAM_MEMBERS": 2,
            "CNT_CHILDREN": 0,
            "EXT_SOURCE_1": 0.50,
            "EXT_SOURCE_2": 0.55,
            "EXT_SOURCE_3": 0.45,
        }
    },
    {
        "name": "5. Self-Employed - Small Business",
        "expected_risk": "MEDIUM",
        "features": {
            "AMT_INCOME_TOTAL": 150000,
            "AMT_CREDIT": 500000,
            "AMT_ANNUITY": 35000,
            "AMT_GOODS_PRICE": 480000,
            "DAYS_BIRTH": -38 * 365,
            "DAYS_EMPLOYED": -5 * 365,
            "CNT_FAM_MEMBERS": 4,
            "CNT_CHILDREN": 2,
            "EXT_SOURCE_1": 0.45,
            "EXT_SOURCE_2": 0.50,
            "EXT_SOURCE_3": 0.40,
        }
    },
    {
        "name": "6. Divorced Single Parent",
        "expected_risk": "MEDIUM",
        "features": {
            "AMT_INCOME_TOTAL": 70000,
            "AMT_CREDIT": 200000,
            "AMT_ANNUITY": 15000,
            "AMT_GOODS_PRICE": 180000,
            "DAYS_BIRTH": -35 * 365,
            "DAYS_EMPLOYED": -6 * 365,
            "CNT_FAM_MEMBERS": 3,
            "CNT_CHILDREN": 2,
            "EXT_SOURCE_1": 0.40,
            "EXT_SOURCE_2": 0.50,
            "EXT_SOURCE_3": 0.35,
        }
    },
    {
        "name": "7. Recent Graduate - First Job",
        "expected_risk": "MEDIUM",
        "features": {
            "AMT_INCOME_TOTAL": 55000,
            "AMT_CREDIT": 120000,
            "AMT_ANNUITY": 9000,
            "AMT_GOODS_PRICE": 110000,
            "DAYS_BIRTH": -24 * 365,
            "DAYS_EMPLOYED": -1 * 365,
            "CNT_FAM_MEMBERS": 1,
            "CNT_CHILDREN": 0,
            "EXT_SOURCE_1": 0.55,
            "EXT_SOURCE_2": 0.48,
            "EXT_SOURCE_3": 0.42,
        }
    },
    
    # ─────────────────────────────────────────────────────────────────
    # HIGH RISK CLIENTS (Expected: > 50% probability)
    # ─────────────────────────────────────────────────────────────────
    {
        "name": "8. Overextended Borrower",
        "expected_risk": "HIGH",
        "features": {
            "AMT_INCOME_TOTAL": 45000,
            "AMT_CREDIT": 600000,
            "AMT_ANNUITY": 45000,  # Annuity = 100% of income!
            "AMT_GOODS_PRICE": 550000,
            "DAYS_BIRTH": -30 * 365,
            "DAYS_EMPLOYED": -2 * 365,
            "CNT_FAM_MEMBERS": 5,
            "CNT_CHILDREN": 3,
            "EXT_SOURCE_1": 0.20,
            "EXT_SOURCE_2": 0.25,
            "EXT_SOURCE_3": 0.15,
        }
    },
    {
        "name": "9. Young Unemployed",
        "expected_risk": "HIGH",
        "features": {
            "AMT_INCOME_TOTAL": 25000,
            "AMT_CREDIT": 200000,
            "AMT_ANNUITY": 20000,
            "AMT_GOODS_PRICE": 180000,
            "DAYS_BIRTH": -21 * 365,  # Very young
            "DAYS_EMPLOYED": 365243,  # Unemployed flag
            "CNT_FAM_MEMBERS": 1,
            "CNT_CHILDREN": 0,
            "EXT_SOURCE_1": 0.15,
            "EXT_SOURCE_2": 0.20,
            "EXT_SOURCE_3": 0.10,
        }
    },
    {
        "name": "10. Multiple Red Flags",
        "expected_risk": "HIGH",
        "features": {
            "AMT_INCOME_TOTAL": 35000,
            "AMT_CREDIT": 450000,
            "AMT_ANNUITY": 35000,
            "AMT_GOODS_PRICE": 400000,
            "DAYS_BIRTH": -22 * 365,
            "DAYS_EMPLOYED": -0.5 * 365,  # 6 months
            "CNT_FAM_MEMBERS": 6,
            "CNT_CHILDREN": 4,
            "EXT_SOURCE_1": 0.10,
            "EXT_SOURCE_2": 0.15,
            "EXT_SOURCE_3": 0.08,
        }
    },
]


def run_tests():
    """Run all test cases and display results."""
    print("=" * 70)
    print("   🧪 CREDIT RISK MODEL - TEST SAMPLES")
    print("=" * 70)
    
    # Check API health
    try:
        r = requests.get(f"{API_URL}/", timeout=5)
        if r.status_code != 200:
            print("❌ API not responding!")
            return
        print(f"✅ API Online: {r.json().get('model_loaded', False)}")
    except:
        print("❌ Cannot connect to API. Start it with:")
        print("   uvicorn api.app_v2:app --port 8000")
        return
    
    print("=" * 70)
    
    results = []
    correct = 0
    
    for case in TEST_CASES:
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"features": case["features"]},
                timeout=30
            )
            data = response.json()
            
            prob = data.get("default_probability", 0)
            risk = data.get("risk_level", "UNKNOWN")
            
            # Check if prediction matches expected
            expected = case["expected_risk"]
            match = "✅" if risk == expected else "⚠️"
            if risk == expected:
                correct += 1
            
            print(f"\n{match} {case['name']}")
            print(f"   Expected: {expected} | Got: {risk}")
            print(f"   Probability: {prob*100:.1f}%")
            
            results.append({
                "name": case["name"],
                "expected": expected,
                "actual": risk,
                "probability": prob,
                "match": risk == expected
            })
            
        except Exception as e:
            print(f"\n❌ {case['name']}")
            print(f"   Error: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print(f"   SUMMARY: {correct}/{len(TEST_CASES)} predictions matched expected risk")
    print("=" * 70)
    
    return results


def test_batch_endpoint():
    """Test batch prediction endpoint."""
    print("\n" + "=" * 70)
    print("   🧪 BATCH ENDPOINT TEST")
    print("=" * 70)
    
    # Take first 3 test cases
    batch_features = [case["features"] for case in TEST_CASES[:3]]
    
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"batch": batch_features},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch endpoint works!")
            print(f"   Predictions: {len(data.get('predictions', []))}")
        else:
            print(f"⚠️ Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Batch test failed: {e}")


if __name__ == "__main__":
    results = run_tests()
    test_batch_endpoint()
