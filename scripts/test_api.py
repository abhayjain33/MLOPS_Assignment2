"""
Example script to test the API with sample images.
"""

import requests
from pathlib import Path
import sys


def test_api(api_url: str, image_path: str):
    """
    Test the API with an image.
    
    Args:
        api_url: URL of the API
        image_path: Path to image file
    """
    print(f"Testing API at: {api_url}")
    print(f"Image: {image_path}")
    print("-" * 60)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    response = requests.get(f"{api_url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test info endpoint
    print("\n2. Testing info endpoint...")
    response = requests.get(f"{api_url}/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test prediction endpoint
    print("\n3. Testing prediction endpoint...")
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{api_url}/predict", files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction Results:")
        print(f"  Predicted class: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"    {class_name}: {prob:.2%}")
        print(f"  Latency: {result['latency_ms']:.2f}ms")
    else:
        print(f"Error: {response.text}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_api.py <image_path> [api_url]")
        print("Example: python scripts/test_api.py test_image.jpg http://localhost:8000")
        sys.exit(1)
    
    image_path = sys.argv[1]
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    test_api(api_url, image_path)
