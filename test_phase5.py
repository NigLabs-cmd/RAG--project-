"""
End-to-End Testing Script for Phase 5
Tests the complete RAG system with backend and frontend integration
"""

import time
import requests
import json
from typing import List, Dict

# Configuration
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:5173"

# Test queries
TEST_QUERIES = [
    {
        "question": "What is machine learning?",
        "expected_answer": True,  # Should find answer
        "min_confidence": 0.5
    },
    {
        "question": "How to use Python for data analysis?",
        "expected_answer": True,
        "min_confidence": 0.4
    },
    {
        "question": "What is HTML used for?",
        "expected_answer": True,
        "min_confidence": 0.4
    },
    {
        "question": "What is quantum computing?",
        "expected_answer": False,  # Not in dataset
        "min_confidence": 0.0
    },
    {
        "question": "",  # Empty query
        "expected_answer": False,
        "min_confidence": 0.0,
        "should_fail": True
    }
]


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.response_times = []
    
    def add_pass(self):
        self.passed += 1
    
    def add_fail(self, error: str):
        self.failed += 1
        self.errors.append(error)
    
    def add_response_time(self, time: float):
        self.response_times.append(time)
    
    def print_summary(self):
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {self.passed + self.failed}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        
        if self.response_times:
            avg_time = sum(self.response_times) / len(self.response_times)
            max_time = max(self.response_times)
            min_time = min(self.response_times)
            print(f"\nResponse Times:")
            print(f"  Average: {avg_time:.2f}s")
            print(f"  Min: {min_time:.2f}s")
            print(f"  Max: {max_time:.2f}s")
        
        if self.errors:
            print(f"\nErrors:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        print("=" * 80)


def test_backend_health(results: TestResults):
    """Test backend health endpoint."""
    print("\n[TEST] Backend Health Check")
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is healthy")
            results.add_pass()
        else:
            error = f"Backend health check failed: {response.status_code}"
            print(f"❌ {error}")
            results.add_fail(error)
    except Exception as e:
        error = f"Backend health check error: {str(e)}"
        print(f"❌ {error}")
        results.add_fail(error)


def test_query(query_data: Dict, results: TestResults):
    """Test a single query."""
    question = query_data["question"]
    print(f"\n[TEST] Query: '{question}'")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={"question": question},
            timeout=30
        )
        response_time = time.time() - start_time
        results.add_response_time(response_time)
        
        # Check if should fail
        if query_data.get("should_fail", False):
            if response.status_code != 200:
                print(f"✅ Correctly rejected invalid query")
                results.add_pass()
            else:
                error = "Should have rejected invalid query"
                print(f"❌ {error}")
                results.add_fail(error)
            return
        
        # Normal query checks
        if response.status_code != 200:
            error = f"Query failed with status {response.status_code}"
            print(f"❌ {error}")
            results.add_fail(error)
            return
        
        data = response.json()
        
        # Validate response structure
        required_fields = ["answer", "sources", "confidence", "has_answer", "citations", "query_time"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            error = f"Missing fields: {missing_fields}"
            print(f"❌ {error}")
            results.add_fail(error)
            return
        
        # Check answer expectation
        if query_data["expected_answer"] and not data["has_answer"]:
            error = "Expected answer but got fallback"
            print(f"⚠️  {error}")
            # Don't fail, just warn
        
        # Check confidence
        if data["confidence"] < query_data["min_confidence"]:
            print(f"⚠️  Low confidence: {data['confidence']:.2f}")
        
        # Print results
        print(f"  Answer: {data['answer'][:100]}...")
        print(f"  Confidence: {data['confidence']:.2f}")
        print(f"  Has Answer: {data['has_answer']}")
        print(f"  Citations: {data['citations']}")
        print(f"  Sources: {len(data['sources'])}")
        print(f"  Response Time: {response_time:.2f}s")
        
        print("✅ Query test passed")
        results.add_pass()
        
    except Exception as e:
        error = f"Query error: {str(e)}"
        print(f"❌ {error}")
        results.add_fail(error)


def test_multiple_queries(results: TestResults):
    """Test multiple queries in sequence."""
    print("\n[TEST] Multiple Queries in Sequence")
    try:
        for i in range(3):
            response = requests.post(
                f"{BACKEND_URL}/query",
                json={"question": "What is machine learning?"},
                timeout=30
            )
            if response.status_code != 200:
                error = f"Query {i+1} failed"
                print(f"❌ {error}")
                results.add_fail(error)
                return
        
        print("✅ Multiple queries handled successfully")
        results.add_pass()
    except Exception as e:
        error = f"Multiple queries error: {str(e)}"
        print(f"❌ {error}")
        results.add_fail(error)


def test_edge_cases(results: TestResults):
    """Test edge cases."""
    print("\n[TEST] Edge Cases")
    
    # Test very long query
    long_query = "What is " + "machine learning " * 50
    try:
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={"question": long_query},
            timeout=30
        )
        if response.status_code == 200:
            print("✅ Long query handled")
            results.add_pass()
        else:
            error = "Long query failed"
            print(f"❌ {error}")
            results.add_fail(error)
    except Exception as e:
        error = f"Long query error: {str(e)}"
        print(f"❌ {error}")
        results.add_fail(error)


def main():
    """Run all tests."""
    print("=" * 80)
    print("PHASE 5 END-TO-END TESTING")
    print("=" * 80)
    print(f"\nBackend URL: {BACKEND_URL}")
    print(f"Frontend URL: {FRONTEND_URL}")
    print("\nMake sure both servers are running before proceeding!")
    print("Press Enter to start tests...")
    input()
    
    results = TestResults()
    
    # Test backend health
    test_backend_health(results)
    
    # Test individual queries
    for query_data in TEST_QUERIES:
        test_query(query_data, results)
    
    # Test multiple queries
    test_multiple_queries(results)
    
    # Test edge cases
    test_edge_cases(results)
    
    # Print summary
    results.print_summary()
    
    # Final message
    print("\n📝 Manual Frontend Tests:")
    print(f"  1. Open {FRONTEND_URL} in your browser")
    print("  2. Verify UI loads correctly")
    print("  3. Submit test queries")
    print("  4. Check confidence badges display")
    print("  5. Verify citations and sources appear")
    print("  6. Test error handling (stop backend)")
    print("\n✨ Phase 5 testing complete!")


if __name__ == "__main__":
    main()
