"""Security validation and verification utilities"""
import os
import stat
import logging
import platform
import psutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import secrets
import hashlib

logger = logging.getLogger(__name__)

class SecurityValidator:
    """Comprehensive security validation utilities"""
    
    def __init__(self, app_config: Dict[str, Any]):
        """Initialize validator with application configuration."""
        self.config = app_config
        self.security_checks = []
        self._initialize_security_checks()
    
    def _initialize_security_checks(self):
        """Initialize the list of security checks to perform."""
        self.security_checks = [
            self._check_file_permissions,
            self._check_os_security,
            self._check_process_security,
            self._check_network_security,
            self._check_memory_limits,
            self._check_sensitive_data
        ]
    
    def _check_file_permissions(self) -> List[str]:
        """Check file and directory permissions."""
        issues = []
        paths_to_check = [
            self.config.get('UPLOAD_FOLDER'),
            Path(__file__).parent.parent / 'models',
            Path(__file__).parent.parent / 'models/manifest.json'
        ]
        
        for path in paths_to_check:
            if not path:
                continue
                
            path = Path(path)
            try:
                if path.exists():
                    st = os.stat(path)
                    # Check for world-writeable permissions
                    if bool(st.st_mode & stat.S_IWOTH):
                        issues.append(f"Unsafe permissions on {path}: world-writeable")
                    # Check ownership
                    if platform.system() != 'Windows':  # Skip on Windows
                        if st.st_uid == 0:  # root ownership
                            issues.append(f"Unsafe ownership on {path}: owned by root")
            except Exception as e:
                issues.append(f"Error checking {path}: {str(e)}")
        
        return issues
    
    def _check_os_security(self) -> List[str]:
        """Check operating system security settings."""
        issues = []
        
        # Check if running as root
        if os.geteuid() == 0:
            issues.append("Application running as root")
        
        # Check umask
        current_umask = os.umask(0o077)  # Get current umask
        os.umask(current_umask)  # Reset to original
        if current_umask < 0o077:
            issues.append(f"Unsafe umask setting: {oct(current_umask)}")
        
        # Check for sensitive environment variables
        sensitive_vars = ['API_KEY', 'SECRET_KEY', 'PASSWORD', 'TOKEN']
        for var in sensitive_vars:
            if var.lower() in [k.lower() for k in os.environ.keys()]:
                issues.append(f"Sensitive data found in environment: {var}")
        
        return issues
    
    def _check_process_security(self) -> List[str]:
        """Check process security settings."""
        issues = []
        
        try:
            process = psutil.Process()
            
            # Check open files
            if len(process.open_files()) > 1000:
                issues.append("Too many open files")
            
            # Check process connections
            connections = process.connections()
            for conn in connections:
                if conn.status == 'LISTEN' and conn.laddr.ip == '0.0.0.0':
                    issues.append(f"Process listening on all interfaces: {conn.laddr.port}")
            
            # Check process memory
            mem_info = process.memory_info()
            if mem_info.rss > 2 * 1024 * 1024 * 1024:  # 2GB
                issues.append("Process memory usage exceeds 2GB")
        
        except Exception as e:
            issues.append(f"Error checking process security: {str(e)}")
        
        return issues
    
    def _check_network_security(self) -> List[str]:
        """Check network security configuration."""
        issues = []
        
        # Check SSL/TLS configuration
        if not self.config.get('SESSION_COOKIE_SECURE'):
            issues.append("Session cookies not set to secure")
        
        if not self.config.get('SESSION_COOKIE_HTTPONLY'):
            issues.append("Session cookies not set to HttpOnly")
        
        # Check rate limiting
        if not self.config.get('RATELIMIT_ENABLED'):
            issues.append("Rate limiting not enabled")
        
        # Check trusted proxies configuration
        trusted_proxies = self.config.get('TRUSTED_PROXIES', [])
        if '0.0.0.0' in trusted_proxies or '*' in trusted_proxies:
            issues.append("Unsafe trusted proxy configuration")
        
        return issues
    
    def _check_memory_limits(self) -> List[str]:
        """Check memory limits and usage."""
        issues = []
        
        try:
            # Check system memory
            mem = psutil.virtual_memory()
            if mem.percent > 90:
                issues.append(f"High memory usage: {mem.percent}%")
            
            # Check swap usage
            swap = psutil.swap_memory()
            if swap.percent > 80:
                issues.append(f"High swap usage: {swap.percent}%")
            
        except Exception as e:
            issues.append(f"Error checking memory limits: {str(e)}")
        
        return issues
    
    def _check_sensitive_data(self) -> List[str]:
        """Check for sensitive data exposure."""
        issues = []
        
        # Check debug mode
        if self.config.get('DEBUG'):
            issues.append("Debug mode enabled in production")
        
        # Check error handling configuration
        if self.config.get('PROPAGATE_EXCEPTIONS'):
            issues.append("Exception propagation enabled")
        
        # Check temporary files
        temp_dir = self.config.get('UPLOAD_FOLDER')
        if temp_dir and Path(temp_dir).exists():
            for file in Path(temp_dir).glob('*'):
                if file.stat().st_mtime < (time.time() - 3600):  # Older than 1 hour
                    issues.append(f"Stale temporary file: {file}")
        
        return issues
    
    def run_security_checks(self) -> Dict[str, Any]:
        """Run all security checks and return results.
        
        Returns:
            Dict containing check results and overall status
        """
        all_issues = []
        check_results = {}
        
        for check in self.security_checks:
            try:
                issues = check()
                if issues:
                    all_issues.extend(issues)
                check_results[check.__name__] = {
                    'status': 'failed' if issues else 'passed',
                    'issues': issues
                }
            except Exception as e:
                logger.error(f"Error in security check {check.__name__}: {str(e)}")
                all_issues.append(f"Check failed: {check.__name__}")
                check_results[check.__name__] = {
                    'status': 'error',
                    'issues': [str(e)]
                }
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'failed' if all_issues else 'passed',
            'total_issues': len(all_issues),
            'checks': check_results,
            'issues': all_issues
        }
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate a cryptographically secure token.
        
        Args:
            length: Length of token in bytes
            
        Returns:
            str: Secure token
        """
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def secure_hash(data: str) -> str:
        """Generate a secure hash of data.
        
        Args:
            data: String to hash
            
        Returns:
            str: SHA-256 hash
        """
        return hashlib.sha256(data.encode()).hexdigest()
    
    def validate_upload_file(self, file_path: str, allowed_extensions: Optional[List[str]] = None) -> bool:
        """Validate an uploaded file.
        
        Args:
            file_path: Path to file
            allowed_extensions: List of allowed file extensions
            
        Returns:
            bool: True if file is safe
        """
        if allowed_extensions is None:
            allowed_extensions = ['.csv']
            
        path = Path(file_path)
        
        # Check extension
        if path.suffix.lower() not in allowed_extensions:
            return False
            
        # Check file size
        if path.stat().st_size > self.config.get('MAX_CONTENT_LENGTH', 10 * 1024 * 1024):
            return False
            
        # Check file permissions
        if bool(path.stat().st_mode & stat.S_IWOTH):
            return False
            
        return True