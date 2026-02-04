"""
Comprehensive API testing script for the LLM evaluation module.
Run this to test all endpoints and functionality.
"""
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_root():
    """Test if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print("❌ Server returned unexpected status")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running. Start it first!")
        return False

def test_evaluate_endpoint():
    """Test the /evaluate endpoint with sample data"""
    print("\n" + "="*60)
    print("Testing /evaluate endpoint")
    print("="*60)
    
    test_cases = [
        {
            "name": "Fresher - Machine Learning",
            "data": {
                "question": "What is machine learning?",
                "answer": "Machine learning is a subset of AI that allows systems to learn from data and improve automatically without being explicitly programmed.",
                "experience_level": "fresher"
            }
        },
        {
            "name": "Intermediate - Python",
            "data": {
                "question": "Explain the difference between list and tuple in Python",
                "answer": "Lists are mutable, meaning you can change their content. Tuples are immutable. Lists use square brackets while tuples use parentheses. Lists are slower than tuples.",
                "experience_level": "intermediate"
            }
        },
        {
            "name": "Advanced - System Design",
            "data": {
                "question": "How would you design a scalable URL shortening service?",
                "answer": "I would use a distributed hash table with consistent hashing for the database layer, implement a REST API with rate limiting, use Redis for caching frequently accessed URLs, and deploy it on Kubernetes for auto-scaling.",
                "experience_level": "advanced"
            }
        },
        {
            "name": "Short Answer Test",
            "data": {
                "question": "What is REST?",
                "answer": "REST is an architectural style for web services.",
                "experience_level": "fresher"
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['name']} ---")
        print(f"Question: {test_case['data']['question']}")
        print(f"Answer: {test_case['data']['answer'][:80]}...")
        print(f"Level: {test_case['data']['experience_level']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/evaluate",
                json=test_case['data'],
                headers={"Content-Type": "application/json"}
            )
            
            print(f"\nStatus Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("\n📊 Evaluation Results:")
                print(f"  - Relevance Score: {result.get('relevance_score', 'N/A')}")
                print(f"  - Clarity Score: {result.get('clarity_score', 'N/A')}")
                print(f"  - Technical Accuracy: {result.get('technical_accuracy', 'N/A')}")
                print(f"  - Communication Score: {result.get('communication_score', 'N/A')}")
                print(f"  - Overall Score: {result.get('overall_score', 'N/A')}")
                print(f"\n💬 Feedback: {result.get('feedback', 'N/A')[:150]}...")
                print("✅ Test PASSED")
            else:
                print(f"❌ Test FAILED")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 60)

def test_invalid_requests():
    """Test error handling with invalid requests"""
    print("\n" + "="*60)
    print("Testing Error Handling")
    print("="*60)
    
    invalid_cases = [
        {
            "name": "Missing question field",
            "data": {
                "answer": "Some answer",
                "experience_level": "fresher"
            }
        },
        {
            "name": "Missing answer field",
            "data": {
                "question": "What is AI?",
                "experience_level": "fresher"
            }
        },
        {
            "name": "Missing experience_level field",
            "data": {
                "question": "What is AI?",
                "answer": "Artificial Intelligence"
            }
        },
        {
            "name": "Empty request body",
            "data": {}
        }
    ]
    
    for i, test_case in enumerate(invalid_cases, 1):
        print(f"\n--- Invalid Test {i}: {test_case['name']} ---")
        
        try:
            response = requests.post(
                f"{BASE_URL}/evaluate",
                json=test_case['data'],
                headers={"Content-Type": "application/json"}
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 422:
                print("✅ Correctly returned 422 Unprocessable Entity")
                error_detail = response.json().get('detail', [])
                print(f"Error details: {error_detail[0] if error_detail else 'N/A'}")
            else:
                print(f"⚠️ Unexpected status code: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 GPT LLM Module - Comprehensive Test Suite")
    print("="*60)
    
    # Test if server is running
    if not test_root():
        print("\n⚠️ Please start the server first:")
        print('Run: python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000')
        return
    
    # Run tests
    test_evaluate_endpoint()
    test_invalid_requests()
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
    print("\n📝 Summary:")
    print("- Valid requests: Check scores and feedback above")
    print("- Invalid requests: Should return 422 errors")
    print("- Server health: Running properly")
    print("\n🌐 API Documentation: http://127.0.0.1:8000/docs")

if __name__ == "__main__":
    main()
