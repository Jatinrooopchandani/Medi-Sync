#!/usr/bin/env python3
"""
Security verification script for Medisync ML service.
Performs comprehensive security checks of the implementation.
"""
import os
import re
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SecurityChecker:
    def __init__(self):
        self.issues: List[str] = []
        self.warnings: List[str] = []

    def check_sensitive_data(self, file_path: str) -> bool:
        """Check for potential sensitive data in files"""
        sensitive_patterns = [
            r'(?i)password\s*=\s*["\'][^"\']+["\']',
            r'(?i)secret\s*=\s*["\'][^"\']+["\']',
            r'(?i)api[_-]key\s*=\s*["\'][^"\']+["\']',
            r'(?i)token\s*=\s*["\'][^"\']+["\']',
            r'(?i)access[_-]key\s*=\s*["\'][^"\']+["\']',
            r'(?i)private[_-]key\s*=\s*["\'][^"\']+["\']',
            r'(?i)aws[_-]?(secret|key)',
            r'(?i)connectionstring\s*=\s*["\'][^"\']+["\']',
            r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
            r'(?i)BEGIN( RSA)? PRIVATE KEY',
        ]

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for pattern in sensitive_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    if '.example' not in file_path and 'test' not in file_path.lower():
                        self.issues.append(f"Potential sensitive data in {file_path}: {match.group()}")
                    else:
                        self.warnings.append(f"Sensitive pattern in example/test file {file_path}: {match.group()}")
        except Exception as e:
            self.issues.append(f"Error checking {file_path}: {str(e)}")
            return False
        return True

    def check_debug_settings(self) -> bool:
        """Verify debug settings are secure"""
        debug_patterns = [
            (r'debug\s*=\s*True', 'main.py'),
            (r'FLASK_DEBUG\s*=\s*True', '.env'),
            (r'DEBUG\s*=\s*True', '*.py'),
        ]
        
        success = True
        for pattern, file_glob in debug_patterns:
            for file_path in Path('.').glob(file_glob):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if re.search(pattern, content):
                        if 'test' not in str(file_path):
                            self.issues.append(f"Debug enabled in {file_path}")
                            success = False
        return success

    def check_security_headers(self) -> bool:
        """Verify security headers are properly configured"""
        required_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection',
            'Strict-Transport-Security',
            'Content-Security-Policy'
        ]
        
        success = True
        with open('main.py', 'r', encoding='utf-8') as f:
            content = f.read()
            for header in required_headers:
                if header not in content:
                    self.issues.append(f"Missing security header: {header}")
                    success = False
        return success

    def check_error_handling(self) -> bool:
        """Verify proper error handling"""
        required_handlers = [
            '@app.errorhandler(404)',
            '@app.errorhandler(500)',
            '@app.errorhandler(403)',
            'render_template(\'error.html\''
        ]
        
        success = True
        with open('main.py', 'r', encoding='utf-8') as f:
            content = f.read()
            for handler in required_handlers:
                if handler not in content:
                    self.issues.append(f"Missing error handler: {handler}")
                    success = False
        return success

    def check_rate_limiting(self) -> bool:
        """Verify rate limiting is configured"""
        required_patterns = [
            r'@limiter\.limit',
            r'RATE_LIMIT_DEFAULT',
            r'RateLimitExceeded'
        ]
        
        success = True
        with open('main.py', 'r', encoding='utf-8') as f:
            content = f.read()
            for pattern in required_patterns:
                if not re.search(pattern, content):
                    self.issues.append(f"Missing rate limiting pattern: {pattern}")
                    success = False
        return success

    def check_file_permissions(self) -> bool:
        """Verify file permissions are secure"""
        file_permissions = {
            'main.py': 0o644,
            'start.sh': 0o755,
            'monitor.py': 0o755,
            'system_check.py': 0o755,
            '.env.example': 0o644,
            'requirements.txt': 0o644
        }
        
        success = True
        for file_path, required_mode in file_permissions.items():
            if os.path.exists(file_path):
                mode = os.stat(file_path).st_mode & 0o777
                if mode != required_mode:
                    self.issues.append(f"Incorrect permissions for {file_path}: {oct(mode)} should be {oct(required_mode)}")
                    success = False
        return success

    def run_security_check(self) -> Tuple[bool, List[str], List[str]]:
        """Run all security checks"""
        logger.info("Starting security verification...")
        
        # Check all Python files for sensitive data
        for py_file in Path('.').glob('**/*.py'):
            self.check_sensitive_data(str(py_file))
            
        # Check configuration files
        for config_file in Path('.').glob('**/*.json'):
            self.check_sensitive_data(str(config_file))
            
        # Run all security checks
        self.check_debug_settings()
        self.check_security_headers()
        self.check_error_handling()
        self.check_rate_limiting()
        self.check_file_permissions()
        
        success = len(self.issues) == 0
        status = "PASSED" if success else "FAILED"
        logger.info(f"Security verification {status}")
        
        if self.issues:
            logger.error("\nSecurity issues found:")
            for issue in self.issues:
                logger.error(f"- {issue}")
                
        if self.warnings:
            logger.warning("\nWarnings:")
            for warning in self.warnings:
                logger.warning(f"- {warning}")
        
        return success, self.issues, self.warnings

def main():
    checker = SecurityChecker()
    success, issues, warnings = checker.run_security_check()
    sys.exit(0 if success and not warnings else 1)

if __name__ == '__main__':
    main()