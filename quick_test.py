"""
Quick test script to verify the LLM evaluation system is working
"""

import requests
import json

# Test data
test_request = {
    "question": "What is the difference between supervised and unsupervised learning in machine learning?",
    "candidate_answer": "Supervised learning uses labeled training data to learn patterns, while unsupervised learning finds hidden patterns in unlabeled data. For example, classification and regression are supervised learning tasks, while clustering and dimensionality reduction are unsupervised learning tasks.",
    "expected_keywords": ["supervised", "unsupervised", "classification", "regression", "clustering"],
    "experience_level": "intermediate",
    "question_type": "technical", 
    "context": "Machine Learning Interview",
    "time_taken": 120
}

def test_health_endpoint():
    """Test health check endpoint"""
    try:
        response = requests.get("http://127.0.0.1:8000/")
        print(f"Health Check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_comprehensive_evaluation():
    """Test comprehensive evaluation endpoint"""
    try:
        response = requests.post(
            "http://127.0.0.1:8000/evaluate/comprehensive",
            json=test_request,
            timeout=30
        )
        print(f"Comprehensive Evaluation: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Overall Score: {result.get('overall_score', 'N/A')}")
            print(f"Technical Accuracy: {result.get('technical_accuracy', 'N/A')}")
            print(f"Communication: {result.get('communication', 'N/A')}")
            print(f"Anti-cheat Status: {result.get('anti_cheat_result', {}).get('is_cheating', 'N/A')}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Comprehensive evaluation failed: {e}")
        return False

def test_risk_engine():
    """Test risk engine endpoint"""
    try:
        response = requests.post(
            "http://127.0.0.1:8000/evaluate/risk-engine",
            json=test_request,
            timeout=30
        )
        print(f"Risk Engine: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Overall Risk Level: {result.get('overall_risk_level', 'N/A')}")
            print(f"Risk Score: {result.get('risk_score', 'N/A')}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Risk engine test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing LLM Evaluation System")
    print("=" * 40)
    
    # Test all endpoints
    tests = [
        ("Health Check", test_health_endpoint),
        ("Comprehensive Evaluation", test_comprehensive_evaluation),
        ("Risk Engine", test_risk_engine)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        success = test_func()
        results.append((test_name, success))
        print(f"✅ {test_name}: {'PASSED' if success else 'FAILED'}")
    
    print("\n" + "=" * 40)
    print("📊 Test Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")