#!/usr/bin/env python3
"""
Pre-deployment validation script for Medisync ML service.
Performs comprehensive checks of all components and configurations.
"""
import os
import sys
import json
import subprocess
import importlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentValidator:
    def __init__(self):
        self.required_files = [
            'main.py',
            'requirements.txt',
            'start.sh',
            'system_check.py',
            'monitor.py',
            'medisync.service',
            'medisync.nginx.conf',
            'medisync.logrotate',
            'medisync.crontab',
            'monitoring_config.json',
            '.env.example',
            'DEPLOYMENT.md',
            'README.md'
        ]
        
        self.required_dirs = [
            'logs',
            'logs/archive',
            'datasets',
            'models',
            'templates'
        ]
        
        self.required_permissions = {
            'start.sh': 0o755,
            'system_check.py': 0o755,
            'monitor.py': 0o755,
            'validate_deployment.py': 0o755,
            'main.py': 0o644,
            'requirements.txt': 0o644,
            '.env.example': 0o644
        }
        
        self.issues: List[str] = []

    def check_file_exists(self, filepath: str) -> bool:
        """Check if a required file exists"""
        if not os.path.isfile(filepath):
            self.issues.append(f"Missing required file: {filepath}")
            return False
        return True

    def check_directory_exists(self, dirpath: str) -> bool:
        """Check if a required directory exists"""
        if not os.path.isdir(dirpath):
            self.issues.append(f"Missing required directory: {dirpath}")
            return False
        return True

    def check_file_permissions(self, filepath: str, required_mode: int) -> bool:
        """Check if file has correct permissions"""
        if not os.path.exists(filepath):
            return False
        
        current_mode = os.stat(filepath).st_mode & 0o777
        if current_mode != required_mode:
            self.issues.append(
                f"Incorrect permissions for {filepath}: "
                f"current={oct(current_mode)}, required={oct(required_mode)}"
            )
            return False
        return True

    def check_python_imports(self) -> bool:
        """Verify all required Python packages can be imported"""
        required_packages = [
            'flask',
            'waitress',
            'numpy',
            'pandas',
            'tensorflow',
            'psutil',
            'requests'
        ]
        
        success = True
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError as e:
                self.issues.append(f"Failed to import {package}: {str(e)}")
                success = False
        return success

    def validate_nginx_config(self) -> bool:
        """Validate Nginx configuration syntax"""
        if not self.check_file_exists('medisync.nginx.conf'):
            return False
            
        try:
            # Just check the syntax, don't actually run nginx
            result = subprocess.run(
                ['nginx', '-t', '-c', './medisync.nginx.conf'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                self.issues.append(f"Invalid Nginx configuration: {result.stderr}")
                return False
        except FileNotFoundError:
            logger.warning("Nginx not installed, skipping configuration validation")
        return True

    def validate_systemd_service(self) -> bool:
        """Validate systemd service configuration"""
        if not self.check_file_exists('medisync.service'):
            return False
            
        try:
            result = subprocess.run(
                ['systemd-analyze', 'verify', './medisync.service'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                self.issues.append(f"Invalid systemd service: {result.stderr}")
                return False
        except FileNotFoundError:
            logger.warning("systemd-analyze not found, skipping service validation")
        return True

    def validate_json_configs(self) -> bool:
        """Validate all JSON configuration files"""
        json_files = [
            'monitoring_config.json'
        ]
        
        success = True
        for json_file in json_files:
            if not self.check_file_exists(json_file):
                success = False
                continue
                
            try:
                with open(json_file) as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                self.issues.append(f"Invalid JSON in {json_file}: {str(e)}")
                success = False
        return success

    def validate_env_example(self) -> bool:
        """Validate .env.example contains required variables"""
        required_vars = [
            'FLASK_APP',
            'FLASK_ENV',
            'FLASK_DEBUG',
            'HOST',
            'PORT'
        ]
        
        if not self.check_file_exists('.env.example'):
            return False
            
        with open('.env.example') as f:
            content = f.read()
            
        success = True
        for var in required_vars:
            if var not in content:
                self.issues.append(f"Missing required variable in .env.example: {var}")
                success = False
        return success

    def validate_crontab(self) -> bool:
        """Validate crontab syntax"""
        if not self.check_file_exists('medisync.crontab'):
            return False
            
        try:
            result = subprocess.run(
                ['crontab', '-u', 'root', 'medisync.crontab'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                self.issues.append(f"Invalid crontab: {result.stderr}")
                return False
        except FileNotFoundError:
            logger.warning("crontab not found, skipping validation")
        return True

    def run_validation(self) -> Tuple[bool, List[str]]:
        """Run all validation checks"""
        logger.info("Starting deployment validation...")
        
        # Check required files
        for file in self.required_files:
            self.check_file_exists(file)
            
        # Check required directories
        for directory in self.required_dirs:
            self.check_directory_exists(directory)
            
        # Check file permissions
        for file, mode in self.required_permissions.items():
            self.check_file_permissions(file, mode)
            
        # Run all validations
        self.check_python_imports()
        self.validate_nginx_config()
        self.validate_systemd_service()
        self.validate_json_configs()
        self.validate_env_example()
        self.validate_crontab()
        
        success = len(self.issues) == 0
        status = "PASSED" if success else "FAILED"
        logger.info(f"Validation {status}")
        
        if self.issues:
            logger.info("\nIssues found:")
            for issue in self.issues:
                logger.info(f"- {issue}")
        
        return success, self.issues

def main():
    validator = DeploymentValidator()
    success, issues = validator.run_validation()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()