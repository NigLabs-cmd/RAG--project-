import requests
import json

# Test the query endpoint
url = "http://localhost:8000/query"
data = {"question": "What is machine learning?"}

try:
    response = requests.post(url, json=data, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
