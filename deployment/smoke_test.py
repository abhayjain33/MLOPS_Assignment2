"""
Smoke tests for post-deployment validation.
"""

import requests
import sys
import time
from pathlib import Path
import argparse


class SmokeTest:
    """Smoke test suite for deployed API."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize smoke test.
        
        Args:
            base_url: Base URL of the API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.passed = 0
        self.failed = 0
        
    def log(self, message: str, level: str = "INFO"):
        """Log test message."""
        print(f"[{level}] {message}")
        
    def test_health_endpoint(self) -> bool:
        """Test health check endpoint."""
        self.log("Testing /health endpoint...")
        
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                if data.get('status') == 'healthy' and data.get('model_loaded'):
                    self.log("✓ Health check passed", "SUCCESS")
                    self.log(f"  Uptime: {data.get('uptime_seconds')}s")
                    self.log(f"  Device: {data.get('device')}")
                    self.passed += 1
                    return True
                else:
                    self.log("✗ Health check failed: unhealthy status", "ERROR")
                    self.failed += 1
                    return False
            else:
                self.log(f"✗ Health check failed: status {response.status_code}", "ERROR")
                self.failed += 1
                return False
                
        except Exception as e:
            self.log(f"✗ Health check failed: {e}", "ERROR")
            self.failed += 1
            return False
    
    def test_root_endpoint(self) -> bool:
        """Test root endpoint."""
        self.log("Testing / endpoint...")
        
        try:
            response = requests.get(
                f"{self.base_url}/",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'message' in data and 'endpoints' in data:
                    self.log("✓ Root endpoint passed", "SUCCESS")
                    self.passed += 1
                    return True
                else:
                    self.log("✗ Root endpoint missing fields", "ERROR")
                    self.failed += 1
                    return False
            else:
                self.log(f"✗ Root endpoint failed: status {response.status_code}", "ERROR")
                self.failed += 1
                return False
                
        except Exception as e:
            self.log(f"✗ Root endpoint failed: {e}", "ERROR")
            self.failed += 1
            return False
    
    def test_info_endpoint(self) -> bool:
        """Test model info endpoint."""
        self.log("Testing /info endpoint...")
        
        try:
            response = requests.get(
                f"{self.base_url}/info",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['model_type', 'classes', 'input_size', 'device']
                
                if all(field in data for field in required_fields):
                    self.log("✓ Info endpoint passed", "SUCCESS")
                    self.log(f"  Model: {data.get('model_type')}")
                    self.log(f"  Classes: {data.get('classes')}")
                    self.passed += 1
                    return True
                else:
                    self.log("✗ Info endpoint missing required fields", "ERROR")
                    self.failed += 1
                    return False
            else:
                self.log(f"✗ Info endpoint failed: status {response.status_code}", "ERROR")
                self.failed += 1
                return False
                
        except Exception as e:
            self.log(f"✗ Info endpoint failed: {e}", "ERROR")
            self.failed += 1
            return False
    
    def test_metrics_endpoint(self) -> bool:
        """Test Prometheus metrics endpoint."""
        self.log("Testing /metrics endpoint...")
        
        try:
            response = requests.get(
                f"{self.base_url}/metrics",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                content = response.text
                
                # Check for expected metrics
                if 'prediction_requests_total' in content:
                    self.log("✓ Metrics endpoint passed", "SUCCESS")
                    self.passed += 1
                    return True
                else:
                    self.log("✗ Metrics endpoint missing expected metrics", "ERROR")
                    self.failed += 1
                    return False
            else:
                self.log(f"✗ Metrics endpoint failed: status {response.status_code}", "ERROR")
                self.failed += 1
                return False
                
        except Exception as e:
            self.log(f"✗ Metrics endpoint failed: {e}", "ERROR")
            self.failed += 1
            return False
    
    def test_prediction_endpoint_validation(self) -> bool:
        """Test prediction endpoint input validation."""
        self.log("Testing /predict endpoint validation...")
        
        try:
            # Test without file
            response = requests.post(
                f"{self.base_url}/predict",
                timeout=self.timeout
            )
            
            # Should return 422 (validation error)
            if response.status_code == 422:
                self.log("✓ Prediction validation passed", "SUCCESS")
                self.passed += 1
                return True
            else:
                self.log(f"✗ Prediction validation failed: expected 422, got {response.status_code}", "ERROR")
                self.failed += 1
                return False
                
        except Exception as e:
            self.log(f"✗ Prediction validation failed: {e}", "ERROR")
            self.failed += 1
            return False
    
    def wait_for_service(self, max_attempts: int = 10, wait_seconds: int = 5) -> bool:
        """
        Wait for service to become available.
        
        Args:
            max_attempts: Maximum number of attempts
            wait_seconds: Seconds to wait between attempts
            
        Returns:
            True if service is available, False otherwise
        """
        self.log(f"Waiting for service at {self.base_url}...")
        
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.get(
                    f"{self.base_url}/health",
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    self.log(f"✓ Service is available (attempt {attempt}/{max_attempts})", "SUCCESS")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            if attempt < max_attempts:
                self.log(f"Service not ready, waiting {wait_seconds}s... (attempt {attempt}/{max_attempts})")
                time.sleep(wait_seconds)
        
        self.log(f"✗ Service did not become available after {max_attempts} attempts", "ERROR")
        return False
    
    def run_all_tests(self, wait_for_service: bool = True) -> bool:
        """
        Run all smoke tests.
        
        Args:
            wait_for_service: Whether to wait for service to be ready
            
        Returns:
            True if all tests passed, False otherwise
        """
        self.log("=" * 60)
        self.log("Starting smoke tests")
        self.log("=" * 60)
        
        # Wait for service if requested
        if wait_for_service:
            if not self.wait_for_service():
                self.log("Service is not available, aborting tests", "ERROR")
                return False
        
        # Run tests
        self.test_root_endpoint()
        self.test_health_endpoint()
        self.test_info_endpoint()
        self.test_metrics_endpoint()
        self.test_prediction_endpoint_validation()
        
        # Summary
        self.log("=" * 60)
        self.log(f"Smoke test summary: {self.passed} passed, {self.failed} failed")
        self.log("=" * 60)
        
        return self.failed == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run smoke tests for deployed API")
    parser.add_argument(
        '--url',
        type=str,
        default='http://localhost:8000',
        help='Base URL of the API (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Request timeout in seconds (default: 30)'
    )
    parser.add_argument(
        '--no-wait',
        action='store_true',
        help='Do not wait for service to be ready'
    )
    
    args = parser.parse_args()
    
    # Run tests
    tester = SmokeTest(args.url, args.timeout)
    success = tester.run_all_tests(wait_for_service=not args.no_wait)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
