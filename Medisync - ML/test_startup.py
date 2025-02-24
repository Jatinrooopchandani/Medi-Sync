#!/usr/bin/env python3
"""
Startup test script for Medisync ML application.
Tests core functionality before the application goes live.
"""
import os
import sys
import requests
import time
import logging
from subprocess import Popen
import signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def wait_for_server(url, timeout=30):
    """Wait for server to become available"""
    start_time = time.time()
    while True:
        try:
            response = requests.get(url + '/health')
            if response.status_code == 200:
                return True
        except requests.RequestException:
            if time.time() - start_time > timeout:
                return False
            time.sleep(1)

def run_tests():
    # Start server in test mode
    os.environ['FLASK_ENV'] = 'production'
    os.environ['PORT'] = '8081'  # Use different port for testing
    server_process = Popen(['python', 'main.py'])
    
    try:
        base_url = 'http://localhost:8081'
        
        # Wait for server to start
        logger.info("Waiting for server to start...")
        if not wait_for_server(base_url):
            logger.error("Server failed to start")
            return False

        # Test 1: Health Check
        logger.info("Testing health endpoint...")
        response = requests.get(f'{base_url}/health')
        if response.status_code != 200:
            logger.error(f"Health check failed: {response.status_code}")
            return False
        
        # Test 2: Home Page
        logger.info("Testing home page...")
        response = requests.get(base_url)
        if response.status_code != 200:
            logger.error(f"Home page failed: {response.status_code}")
            return False

        # Test 3: Rate Limiting
        logger.info("Testing rate limiting...")
        responses = [requests.get(f'{base_url}/health') for _ in range(35)]
        if not any(r.status_code == 429 for r in responses):
            logger.error("Rate limiting test failed")
            return False

        # Test 4: Error Handling
        logger.info("Testing error handling...")
        response = requests.get(f'{base_url}/nonexistent')
        if response.status_code != 404:
            logger.error("Error handling test failed")
            return False

        logger.info("All tests passed successfully!")
        return True

    finally:
        # Clean up
        server_process.send_signal(signal.SIGTERM)
        server_process.wait()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)