#!/usr/bin/env python3
"""
Test script for the KrunchWrapper server
"""
import os
import sys
import json
import argparse
import requests

def test_chat_completion(server_url, model_name):
    """Test a chat completion request to the server."""
    print(f"Testing chat completion with model {model_name}...")
    
    # Sample Python code to compress
    python_code = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Calculate the 10th Fibonacci number
result = fibonacci(10)
print(f"The 10th Fibonacci number is {result}")
"""

    # Prepare request payload
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Explain this Python code:\n\n```python\n{python_code}\n```"}
        ],
        "filename": "example.py"  # Provide language hint
    }
    
    try:
        # Send request to the server
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Check response
        if response.status_code == 200:
            data = response.json()
            print("✅ Chat completion successful!")
            print(f"Model: {data.get('model', 'unknown')}")
            print(f"Response: {data['choices'][0]['message']['content'][:100]}...")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return False
    
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test the KrunchWrapper server")
    parser.add_argument("--url", default="http://localhost:5001", help="Server URL")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Model name to use")
    
    args = parser.parse_args()
    
    print(f"🧪 Testing KrunchWrapper server at {args.url}")
    success = test_chat_completion(args.url, args.model)
    
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 