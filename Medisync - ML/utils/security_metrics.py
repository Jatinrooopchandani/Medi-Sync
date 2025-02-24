"""Security metrics collection and monitoring utilities"""
import time
from datetime import datetime
import threading
from collections import deque
import logging
from typing import Dict, List, Optional, Deque
import numpy as np

logger = logging.getLogger(__name__)

class SecurityMetricsCollector:
    """Thread-safe collector for security-related metrics"""
    
    def __init__(self, window_size: int = 1000):
        """Initialize the metrics collector.
        
        Args:
            window_size: Number of events to keep in rolling windows
        """
        self._lock = threading.Lock()
        self._window_size = window_size
        
        # Request metrics
        self._request_times: Deque[float] = deque(maxlen=window_size)
        self._failed_requests: Deque[Dict] = deque(maxlen=window_size)
        self._blocked_ips: Dict[str, int] = {}
        
        # Model metrics
        self._prediction_times: Deque[float] = deque(maxlen=window_size)
        self._prediction_values: Deque[float] = deque(maxlen=window_size)
        self._validation_failures: Deque[Dict] = deque(maxlen=window_size)
        
        # System metrics
        self._memory_usage: Deque[float] = deque(maxlen=window_size)
        self._upload_sizes: Deque[int] = deque(maxlen=window_size)
        
        # Last check timestamp
        self._last_check = time.time()
    
    def record_request(self, duration: float, success: bool, ip: str, error: Optional[str] = None):
        """Record metrics for an API request."""
        with self._lock:
            self._request_times.append(duration)
            
            if not success:
                self._failed_requests.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'ip': ip,
                    'error': error
                })
                
                # Track failed attempts by IP
                self._blocked_ips[ip] = self._blocked_ips.get(ip, 0) + 1
    
    def record_prediction(self, duration: float, values: np.ndarray):
        """Record metrics for model predictions."""
        with self._lock:
            self._prediction_times.append(duration)
            self._prediction_values.extend(values.flatten())
    
    def record_validation_failure(self, model: str, error: str):
        """Record a model validation failure."""
        with self._lock:
            self._validation_failures.append({
                'timestamp': datetime.utcnow().isoformat(),
                'model': model,
                'error': error
            })
    
    def record_system_metrics(self, memory_usage: float, upload_size: int):
        """Record system-level metrics."""
        with self._lock:
            self._memory_usage.append(memory_usage)
            self._upload_sizes.append(upload_size)
    
    def get_security_report(self) -> Dict:
        """Generate a security status report.
        
        Returns:
            Dict containing current security metrics
        """
        with self._lock:
            now = time.time()
            time_window = now - self._last_check
            
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'time_window_seconds': time_window,
                
                'requests': {
                    'total': len(self._request_times),
                    'average_duration': np.mean(self._request_times) if self._request_times else 0,
                    'failed_count': len(self._failed_requests),
                    'recent_failures': list(self._failed_requests)[-5:],  # Last 5 failures
                    'blocked_ips_count': len(self._blocked_ips),
                    'top_blocked_ips': dict(sorted(
                        self._blocked_ips.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5])  # Top 5 blocked IPs
                },
                
                'predictions': {
                    'total': len(self._prediction_times),
                    'average_duration': np.mean(self._prediction_times) if self._prediction_times else 0,
                    'value_distribution': {
                        'mean': np.mean(self._prediction_values) if self._prediction_values else 0,
                        'std': np.std(self._prediction_values) if self._prediction_values else 0,
                        'min': np.min(self._prediction_values) if self._prediction_values else 0,
                        'max': np.max(self._prediction_values) if self._prediction_values else 0
                    }
                },
                
                'validation': {
                    'total_failures': len(self._validation_failures),
                    'recent_failures': list(self._validation_failures)[-5:]  # Last 5 validation failures
                },
                
                'system': {
                    'memory_usage': {
                        'current': self._memory_usage[-1] if self._memory_usage else 0,
                        'average': np.mean(self._memory_usage) if self._memory_usage else 0
                    },
                    'upload_sizes': {
                        'total': sum(self._upload_sizes),
                        'average': np.mean(self._upload_sizes) if self._upload_sizes else 0
                    }
                }
            }
            
            # Update last check timestamp
            self._last_check = now
            
            return report
    
    def check_security_status(self) -> bool:
        """Check if current metrics indicate security issues.
        
        Returns:
            bool: True if security status is OK, False if issues detected
        """
        with self._lock:
            # Check for suspicious patterns
            
            # 1. High failure rate
            total_requests = len(self._request_times)
            if total_requests > 0:
                failure_rate = len(self._failed_requests) / total_requests
                if failure_rate > 0.2:  # More than 20% failures
                    logger.warning(f"High request failure rate detected: {failure_rate:.2%}")
                    return False
            
            # 2. Blocked IP concentration
            if self._blocked_ips:
                max_blocks = max(self._blocked_ips.values())
                if max_blocks > 50:  # Single IP blocked more than 50 times
                    logger.warning(f"High concentration of blocks from single IP detected")
                    return False
            
            # 3. Unusual prediction patterns
            if self._prediction_values:
                pred_std = np.std(self._prediction_values)
                if pred_std < 0.01:  # Very low variance might indicate attacks
                    logger.warning(f"Unusual prediction pattern detected: low variance {pred_std}")
                    return False
            
            # 4. System resource usage
            if self._memory_usage:
                current_memory = self._memory_usage[-1]
                if current_memory > 90:  # Over 90% memory usage
                    logger.warning(f"High memory usage detected: {current_memory}%")
                    return False
            
            # 5. Validation failures
            if len(self._validation_failures) > 10:  # More than 10 validation failures in window
                logger.warning("High number of validation failures detected")
                return False
            
            return True  # All checks passed