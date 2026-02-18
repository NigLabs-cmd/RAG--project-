"""
Backend Test Script
Tests the FastAPI backend endpoints
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_root():
    """Test root endpoint."""
    print("\n=== Testing Root Endpoint ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_health():
    """Test health endpoint."""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_query(question: str):
    """Test query endpoint."""
    print(f"\n=== Testing Query: '{question}' ===")
    response = requests.post(
        f"{BASE_URL}/query",
        json={"question": question}
    )
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nAnswer: {data['answer']}")
        print(f"Confidence: {data['confidence']:.4f}")
        print(f"Has Answer: {data['has_answer']}")
        print(f"Citations: {data['citations']}")
        print(f"Sources: {len(data['sources'])}")
        print(f"Query Time: {data['query_time']:.2f}s")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def main():
    """Run all tests."""
    print("=" * 80)
    print("Backend API Test Suite")
    print("=" * 80)
    print("\nMake sure the backend is running: python backend/app.py")
    print("Press Enter to continue...")
    input()
    
    try:
        # Test root
        test_root()
        
        # Test health
        test_health()
        
        # Test queries
        test_queries = [
            "What is machine learning?",
            "How to use Python for data analysis?",
            "What is quantum computing?"  # Should return fallback
        ]
        
        for query in test_queries:
            test_query(query)
        
        print("\n" + "=" * 80)
        print("All tests completed!")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to backend server")
        print("Make sure the backend is running: python backend/app.py")
    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    main()
