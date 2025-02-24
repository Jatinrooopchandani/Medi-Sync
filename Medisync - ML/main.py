from flask import Flask, request, render_template, Response, jsonify
from werkzeug.wrappers import Request
from werkzeug.middleware.proxy_fix import ProxyFix
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys
import signal
import psutil
from waitress import serve
from dotenv import load_dotenv
import logging
from datetime import datetime
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
from typing import Dict, List, Union, Optional, Tuple
import re
import hashlib
import secrets
from pathlib import Path
from utils.secure_loader import (
    load_tokenizer_safely, 
    load_drug_encoder_safely, 
    load_model_safely,
    load_keras_model_safely,
    SecurityError
)
from utils.model_validator import (
    validate_prediction_shape,
    validate_model_input,
    validate_model_compatibility,
    verify_prediction_consistency,
    ModelValidationError
)
from utils.model_security import (
    update_model_manifest,
    verify_all_models,
    compute_file_hash
)
from utils.security_metrics import SecurityMetricsCollector
from utils.prediction_processor import (
    process_input_securely,
    generate_predictions,
    PredictionError
)

# Security and monitoring components
from utils.manifest_updater import ModelManifestManager
from utils.security_validator import SecurityValidator
from utils.model_reloader import ModelReloader

# Global model references that will be managed by ModelReloader
rf_model = None
lstm_model = None
tokenizer = None
drug_encoder = None

def update_global_models(models: Dict[str, Any]):
    """Safely update global model references"""
    global rf_model, lstm_model, tokenizer, drug_encoder
    rf_model = models.get('rf')
    lstm_model = models.get('lstm')
    tokenizer = models.get('tokenizer')
    drug_encoder = models.get('encoder')

# Initialize Flask app with security configurations
app = Flask(__name__)
app.config.update({
    'MAX_CONTENT_LENGTH': MAX_FILE_SIZE,
    'UPLOAD_FOLDER': '/tmp/secure_uploads',
    'SESSION_COOKIE_SECURE': True,
    'SESSION_COOKIE_HTTPONLY': True,
    'SESSION_COOKIE_SAMESITE': 'Strict',
    'PERMANENT_SESSION_LIFETIME': 1800,  # 30 minutes
    'DEBUG': False,
    'TESTING': False,
    'PROPAGATE_EXCEPTIONS': False,
    'PREFERRED_URL_SCHEME': 'https',
    'RATELIMIT_ENABLED': True,
    'TRUSTED_PROXIES': os.environ.get('TRUSTED_PROXIES', '127.0.0.1').split(',')
})

# Initialize security and monitoring components
metrics_collector = SecurityMetricsCollector(window_size=1000)
manifest_manager = ModelManifestManager(check_interval=3600)  # Check every hour
audit_trail = ModelAuditTrail(
    audit_dir=str(Path(__file__).parent / "logs" / "audit"),
    max_file_size=10 * 1024 * 1024  # 10MB per audit file
)

def cleanup_prediction_resources(request_id: str, prediction_id: str):
    """Clean up any temporary resources created during prediction"""
    try:
        # Clean up any temporary files
        temp_pattern = f"pred_{prediction_id}_*"
        for temp_file in Path(app.config['UPLOAD_FOLDER']).glob(temp_pattern):
            try:
                # Securely overwrite before deletion
                with open(temp_file, 'wb') as f:
                    f.write(os.urandom(os.path.getsize(temp_file)))
                temp_file.unlink()
                logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up {temp_file}: {str(e)}")
        
        # Record cleanup in audit trail
        audit_trail.record_event(
            event_type='resource_cleanup',
            model_type='system',
            model_version='current',
            operation='cleanup',
            status='complete',
            details={
                'request_id': request_id,
                'prediction_id': prediction_id,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Error during resource cleanup: {str(e)}")
        # Don't raise the exception - cleanup errors shouldn't affect the response

def format_prediction_response(
    predictions: np.ndarray,
    request_id: str,
    prediction_id: str,
    start_time: float,
    metadata: Dict[str, Any]
) -> Tuple[Dict[str, Any], int]:
    """Format the prediction response with consistent structure"""
    duration = time.time() - start_time
    
    try:
        response_data = {
            'success': True,
            'request_id': request_id,
            'prediction_id': prediction_id,
            'predictions': predictions.tolist(),
            'duration': f"{duration:.2f}s",
            'model_versions': {
                'rf': getattr(rf_model, 'version', '1.0.0'),
                'lstm': getattr(lstm_model, 'version', '1.0.0')
            },
            'metadata': {
                **metadata,
                'audit_id': prediction_id,
                'completion_time': datetime.utcnow().isoformat(),
                'system_info': {
                    'memory_usage': f"{psutil.Process().memory_percent():.1f}%",
                    'system_load': f"{os.getloadavg()[0]:.2f}"
                }
            }
        }
        
        # Record successful completion in audit trail
        audit_trail.record_event(
            event_type='response_generation',
            model_type='system',
            model_version='current',
            operation='format',
            status='success',
            details={
                'request_id': request_id,
                'prediction_id': prediction_id,
                'response_size': len(json.dumps(response_data)),
                'duration': f"{duration:.2f}s"
            }
        )
        
        return response_data, 200
        
    except Exception as e:
        logger.error(f"Error formatting prediction response: {str(e)}")
        raise
security_validator = SecurityValidator(app.config)

# Constants for prediction processing
MAX_TEXT_LENGTH = 1000
MAX_SEQUENCE_LENGTH = 100
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

class HealthMonitor:
    """Thread-safe health monitoring system"""
    
    def __init__(self, check_interval: int = 300):  # 5 minutes default
        self._lock = threading.Lock()
        self.check_interval = check_interval
        self.last_check = None
        self.health_status = {'status': 'initializing'}
        self._monitoring = False
        self._monitor_thread = None
        
    def start(self):
        """Start health monitoring"""
        with self._lock:
            if self._monitoring:
                return
            self._monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="HealthMonitor"
            )
            self._monitor_thread.start()
            logger.info("Health monitoring started")
    
    def stop(self):
        """Stop health monitoring"""
        with self._lock:
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5)
                logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                self.check_health()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health monitor: {str(e)}")
    
    def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        with self._lock:
            try:
                process = psutil.Process()
                
                # System health checks
                system_health = {
                    'memory_usage': process.memory_percent(),
                    'cpu_usage': process.cpu_percent(),
                    'open_files': len(process.open_files()),
                    'threads': len(process.threads()),
                    'connections': len(process.connections())
                }
                
                # Model health checks
                model_health = {
                    'rf_loaded': rf_model is not None,
                    'lstm_loaded': lstm_model is not None,
                    'tokenizer_loaded': tokenizer is not None,
                    'encoder_loaded': drug_encoder is not None
                }
                
                # Security checks
                security_status = security_validator.run_security_checks()
                model_integrity = manifest_manager.verify_integrity()
                
                # Storage checks
                upload_dir = Path(app.config['UPLOAD_FOLDER'])
                storage_health = {
                    'upload_dir_exists': upload_dir.exists(),
                    'upload_dir_writable': os.access(upload_dir, os.W_OK),
                    'upload_dir_size': sum(
                        f.stat().st_size for f in upload_dir.glob('**/*')
                        if f.is_file()
                    ) / (1024 * 1024)  # MB
                }
                
                # Determine overall health status
                is_healthy = (
                    all(model_health.values()) and
                    system_health['memory_usage'] < 90 and
                    security_status['status'] == 'passed' and
                    model_integrity and
                    storage_health['upload_dir_exists'] and
                    storage_health['upload_dir_writable']
                )
                
                health_status = {
                    'status': 'healthy' if is_healthy else 'unhealthy',
                    'timestamp': datetime.utcnow().isoformat(),
                    'system_health': system_health,
                    'model_health': model_health,
                    'security_status': security_status['status'],
                    'model_integrity': model_integrity,
                    'storage_health': storage_health
                }
                
                # Record health check in audit trail
                audit_trail.record_event(
                    event_type='health_check',
                    model_type='system',
                    model_version='current',
                    operation='monitor',
                    status=health_status['status'],
                    details=health_status
                )
                
                self.health_status = health_status
                self.last_check = datetime.utcnow()
                
                return health_status
                
            except Exception as e:
                error_status = {
                    'status': 'error',
                    'timestamp': datetime.utcnow().isoformat(),
                    'error': str(e)
                }
                self.health_status = error_status
                logger.error(f"Health check failed: {str(e)}")
                return error_status
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current health status"""
        with self._lock:
            return self.health_status.copy()

# Initialize health monitor
health_monitor = HealthMonitor(check_interval=300)  # 5 minutes

def initialize_security_components():
    """Initialize and verify all security components"""
    try:
        # Run comprehensive security checks
        security_results = security_validator.run_security_checks()
        if security_results['status'] == 'failed':
            logger.error("Security validation failed:")
            for issue in security_results['issues']:
                logger.error(f"- {issue}")
            raise SecurityError("Security validation failed")
            
        # Update model manifest
        manifest_manager.update_manifest()
        
        # Verify model integrity
        if not manifest_manager.verify_integrity():
            logger.error("Model integrity check failed - attempting recovery from backup")
            try:
                # List available backups
                backups = backup_manager.list_backups()
                if not backups:
                    raise SecurityError("No backups available for recovery")
                    
                # Attempt to restore most recent backup
                latest_backup = backups[0]['name']
                logger.info(f"Attempting to restore from backup: {latest_backup}")
                
                # Verify and restore backup
                verification = backup_manager.verify_backup(latest_backup)
                if not verification['success']:
                    raise SecurityError(f"Backup verification failed: {verification['issues']}")
                    
                restore_result = backup_manager.restore_backup(latest_backup, validate=True)
                logger.info(f"Successfully restored from backup: {restore_result}")
                
                # Verify integrity again after restore
                if not manifest_manager.verify_integrity():
                    raise SecurityError("Model integrity check failed after restore")
                    
            except Exception as e:
                logger.critical(f"Recovery from backup failed: {str(e)}")
                raise SecurityError("Failed to recover from backup")
                
        # Create initial backup if none exists
        try:
            backups = backup_manager.list_backups()
            if not backups:
                logger.info("No backups found - creating initial backup")
                backup_result = backup_manager.create_backup()
                logger.info(f"Created initial backup: {backup_result['backup_name']}")
        except Exception as e:
            logger.error(f"Failed to create initial backup: {str(e)}")
            
        # Start background monitoring
        manifest_manager.start_monitoring()
        
        # Initial security status check
        if not metrics_collector.check_security_status():
            raise SecurityError("Initial security status check failed")
            
        # Generate new secure session key
        app.secret_key = security_validator.generate_secure_token(32)
        
        # Set up secure file upload directory
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        if upload_dir.exists():
            # Clean any existing files
            for file in upload_dir.glob('*'):
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove old file {file}: {e}")
        else:
            upload_dir.mkdir(mode=0o750, parents=True, exist_ok=True)
            
        # Verify upload directory permissions
        st = os.stat(upload_dir)
        if bool(st.st_mode & stat.S_IWOTH):
            raise SecurityError("Upload directory has unsafe permissions")
            
        # Initialize rate limiting
        if not app.config['RATELIMIT_ENABLED']:
            raise SecurityError("Rate limiting must be enabled")
            
        # Verify SSL/TLS configuration
        if not app.config['SESSION_COOKIE_SECURE']:
            raise SecurityError("Secure cookies must be enabled")
            
        logger.info("Security components initialized successfully")
        logger.info(f"Security check results: {json.dumps(security_results, indent=2)}")
        return True
        
    except Exception as e:
        logger.critical(f"Failed to initialize security components: {str(e)}")
        return False
        
def secure_file_validation(file) -> bool:
    """Enhanced secure file validation using SecurityValidator"""
    if not file or not file.filename:
        return False
        
    try:
        # Create temporary file for validation
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                               security_validator.generate_secure_token(16) + '.tmp')
        file.save(temp_path)
        
        # Validate file
        is_valid = security_validator.validate_upload_file(
            temp_path,
            allowed_extensions=['.csv']
        )
        
        # Remove temporary file
        os.unlink(temp_path)
        
        return is_valid
        
    except Exception as e:
        logger.error(f"File validation error: {str(e)}")
        # Clean up if temp file exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        return False

def shutdown_security_components():
    """Safely shutdown security components and monitoring systems"""
    shutdown_errors = []
    
    try:
        # Stop manifest monitoring
        manifest_manager.stop_monitoring()
        logger.info("Manifest manager monitoring stopped")
    except Exception as e:
        error_msg = f"Error stopping manifest manager: {str(e)}"
        logger.error(error_msg)
        shutdown_errors.append(error_msg)

    try:
        # Stop health monitoring
        health_monitor.stop()
        logger.info("Health monitor stopped")
    except Exception as e:
        error_msg = f"Error stopping health monitor: {str(e)}"
        logger.error(error_msg)
        shutdown_errors.append(error_msg)

    try:
        # Record final metrics before shutdown
        final_metrics = metrics_collector.get_security_report()
        audit_trail.record_event(
            event_type='shutdown',
            model_type='system',
            model_version='current',
            operation='metrics',
            status='complete',
            details={
                'final_metrics': final_metrics,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    except Exception as e:
        error_msg = f"Error recording final metrics: {str(e)}"
        logger.error(error_msg)
        shutdown_errors.append(error_msg)

    try:
        # Record shutdown in audit trail
        audit_trail.record_event(
            event_type='shutdown',
            model_type='system',
            model_version='current',
            operation='stop',
            status='complete' if not shutdown_errors else 'partial',
            details={
                'uptime': str(datetime.utcnow() - startup_time),
                'errors': shutdown_errors,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Error recording shutdown event: {str(e)}")

    if shutdown_errors:
        logger.warning(f"Shutdown completed with {len(shutdown_errors)} errors")
    else:
        logger.info("Security components shutdown completed successfully")

# Configure logging with UTC timezone
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Set logging to use UTC
logging.Formatter.converter = datetime.utcnow

# Initialize logger
logger = logging.getLogger(__name__)

# Define secure model paths
MODEL_BASE_DIR = Path(__file__).parent / "models"
MODEL_PATHS = {
    'rf': MODEL_BASE_DIR / "rf.json",
    'lstm': MODEL_BASE_DIR / "lstm_drug_model.keras",
    'tokenizer': MODEL_BASE_DIR / "tokenizer.json",
    'encoder': MODEL_BASE_DIR / "drug_encoder.json"
}
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.Formatter.converter = datetime.utcnow
logger = logging.getLogger(__name__)

# Model paths
MODEL_DIR = Path(__file__).parent / "models"
RF_MODEL_PATH = MODEL_DIR / "rf.json"
LSTM_MODEL_PATH = MODEL_DIR / "lstm_drug_model.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
DRUG_ENCODER_PATH = MODEL_DIR / "drug_encoder.json"

def load_all_models():
    """Securely load all required models with integrity checks.
    
    Returns:
        Tuple of loaded models (rf, lstm, tokenizer, encoder)
        
    Raises:
        SecurityError: If any model fails integrity checks
        FileNotFoundError: If any required model file is missing
    """
    try:
        # Load models with security checks
        rf = load_model_safely(str(RF_MODEL_PATH))
        lstm = tf.keras.models.load_model(str(LSTM_MODEL_PATH))
        tokenizer = load_tokenizer_safely(str(TOKENIZER_PATH))
        encoder = load_drug_encoder_safely(str(DRUG_ENCODER_PATH))
        
        logger.info("All models loaded successfully with security checks")
        return rf, lstm, tokenizer, encoder
        
    except (SecurityError, FileNotFoundError) as e:
        logger.error(f"Failed to load models securely: {str(e)}")
        raise

# Security Constants
ALLOWED_EXTENSIONS = {'csv'}  # Removed 'pkl' for security, keras models loaded differently
SECURE_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'SAMEORIGIN',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'Content-Security-Policy': "default-src 'self'",
    'Referrer-Policy': 'strict-origin-when-cross-origin',
    'Feature-Policy': "geolocation 'none'; microphone 'none'; camera 'none'",
    'Server': 'Medisync ML (Production)',
    'Cache-Control': 'no-store, max-age=0',
}

# Load environment variables from .env file if it exists
load_dotenv()

# Load models securely at startup
def load_all_models():
    """Securely load all required models with integrity checks.
    
    Returns:
        dict: Dictionary containing all loaded models
        
    Raises:
        SecurityError: If any model fails integrity checks
        FileNotFoundError: If any required model file is missing
    """
    models = {}
    try:
        # Load all models with security checks
        models['rf'] = load_model_safely(str(MODEL_PATHS['rf']))
        models['lstm'] = load_keras_model_safely(str(MODEL_PATHS['lstm']))
        models['tokenizer'] = load_tokenizer_safely(str(MODEL_PATHS['tokenizer']))
        models['encoder'] = load_drug_encoder_safely(str(MODEL_PATHS['encoder']))
        
        # Verify all models were loaded
        if not all(models.values()):
            raise SecurityError("One or more models failed to load")
            
        logger.info("All models loaded successfully with security checks")
        return models
        
    except Exception as e:
        logger.critical(f"Failed to load models securely: {str(e)}")
        raise

# Load models at startup
try:
    models = load_all_models()
    rf_model = models['rf']
    lstm_model = models['lstm']
    tokenizer = models['tokenizer']
    drug_encoder = models['encoder']
    logger.info("Models loaded and validated successfully at startup")
except Exception as e:
    logger.critical(f"Failed to load models at startup: {str(e)}")
    sys.exit(1)  # Exit if models can't be loaded securely

# Initialize Flask app with security configurations
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp/secure_uploads'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Strict'
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes

# Ensure upload directory exists and has proper permissions
os.makedirs(app.config['UPLOAD_FOLDER'], mode=0o750, exist_ok=True)

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[RATE_LIMIT_DEFAULT],
    storage_uri="memory://"
)

# Apply security middleware
app.wsgi_app = SecurityMiddleware(app.wsgi_app)

@app.before_request
def before_request():
    """Security checks before processing each request"""
    # Force HTTPS
    if not request.is_secure and not request.headers.get('X-Forwarded-Proto', 'http') == 'https':
        return redirect(request.url.replace('http://', 'https://', 1), code=301)
        
    # Add security headers
    response = make_response()
    for header, value in SECURE_HEADERS.items():
        response.headers[header] = value

@app.after_request
def after_request(response):
    """Post-process all responses"""
    # Ensure security headers
    for header, value in SECURE_HEADERS.items():
        response.headers[header] = value
    return response

@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler"""
    logger.error(f"Unhandled error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
@limiter.limit(RATE_LIMIT_HEALTH)
def health_check():
    """Enhanced health check endpoint with security validation"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {}
        }

        # Check models
        models_loaded = all([rf_model, lstm_model, tokenizer, drug_encoder])
        health_status['components']['models'] = {
            'status': 'healthy' if models_loaded else 'unhealthy',
            'details': {
                'rf_version': getattr(rf_model, 'version', 'unknown'),
                'lstm_version': getattr(lstm_model, 'version', 'unknown')
            }
        }

        # Check security components
        security_status = metrics_collector.check_security_status()
        health_status['components']['security'] = {
            'status': 'healthy' if security_status else 'warning',
            'metrics_collector': 'active',
            'manifest_manager': 'active' if manifest_manager._monitoring else 'inactive'
        }

        # Check file system
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        fs_status = upload_dir.exists() and os.access(upload_dir, os.W_OK)
        health_status['components']['filesystem'] = {
            'status': 'healthy' if fs_status else 'unhealthy',
            'upload_dir': str(upload_dir)
        }

        # Quick security validation
        security_results = security_validator.run_security_checks()
        health_status['components']['security_checks'] = {
            'status': security_results['status'],
            'total_issues': security_results['total_issues']
        }

        # Determine overall status
        if not models_loaded or not fs_status:
            health_status['status'] = 'unhealthy'
        elif not security_status or security_results['status'] == 'failed':
            health_status['status'] = 'warning'

        status_code = 200 if health_status['status'] == 'healthy' else 500
        return jsonify(health_status), status_code

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/diagnostic')
@limiter.limit("10 per hour")
def security_diagnostic():
    """Secure diagnostic endpoint for monitoring system security status"""
    # Verify admin authentication
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Unauthorized'}), 401

    token = auth_header.split(' ')[1]
    admin_token = os.environ.get('ADMIN_TOKEN')
    if not admin_token or not secrets.compare_digest(token, admin_token):
        metrics_collector.record_request(
            duration=0,
            success=False,
            ip=request.remote_addr,
            error="Unauthorized diagnostic access attempt"
        )
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        # Collect comprehensive diagnostic information
        diagnostic_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'security_status': security_validator.run_security_checks(),
            'metrics': metrics_collector.get_security_report(),
            'models': {
                'rf': {
                    'version': getattr(rf_model, 'version', 'unknown'),
                    'loaded': rf_model is not None
                },
                'lstm': {
                    'version': getattr(lstm_model, 'version', 'unknown'),
                    'loaded': lstm_model is not None
                }
            },
            'system': {
                'memory_usage': psutil.Process().memory_percent(),
                'cpu_usage': psutil.Process().cpu_percent(),
                'open_files': len(psutil.Process().open_files()),
                'connections': len(psutil.Process().connections()),
                'upload_dir_size': sum(
                    f.stat().st_size for f in Path(app.config['UPLOAD_FOLDER']).glob('*')
                    if f.is_file()
                )
            }
        }

        # Record successful diagnostic check
        metrics_collector.record_request(
            duration=time.time() - request.start_time,
            success=True,
            ip=request.remote_addr
        )

        return jsonify(diagnostic_info), 200

    except Exception as e:
        logger.error(f"Diagnostic check failed: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/metrics', methods=['GET'])
@limiter.limit("10 per minute")  # Strict rate limit for security metrics
def security_metrics():
    """Secure endpoint for getting security metrics - admin only"""
    # Verify admin authentication
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Unauthorized'}), 401
        
    token = auth_header.split(' ')[1]
    admin_token = os.environ.get('ADMIN_TOKEN')
    if not admin_token or not secrets.compare_digest(token, admin_token):
        metrics_collector.record_request(
            duration=0,
            success=False,
            ip=request.remote_addr,
            error="Unauthorized metrics access attempt"
        )
        return jsonify({'error': 'Unauthorized'}), 401

    # Generate security report
    try:
        report = metrics_collector.get_security_report()
        status_ok = metrics_collector.check_security_status()
        
        response = {
            'status': 'ok' if status_ok else 'warning',
            'metrics': report
        }
        
        metrics_collector.record_request(
            duration=time.time() - request.start_time,
            success=True,
            ip=request.remote_addr
        )
        
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Failed to generate security metrics: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.before_request
def before_request_func():
    """Record request start time and perform security checks"""
    request.start_time = time.time()
    
    # Record system metrics
    try:
        memory_usage = psutil.Process().memory_percent()
        upload_size = sum(
            os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], f))
            for f in os.listdir(app.config['UPLOAD_FOLDER'])
            if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))
        )
        metrics_collector.record_system_metrics(memory_usage, upload_size)
    except Exception as e:
        logger.warning(f"Failed to record system metrics: {str(e)}")

@app.route('/predict', methods=['POST'])
@limiter.limit(RATE_LIMIT_DEFAULT)
def predict():
    """Secure endpoint for model predictions with comprehensive validation, metrics, and audit"""
    request_id = secrets.token_hex(8)
    prediction_id = secrets.token_hex(8)
    start_time = time.time()
    logger.info(f"[{request_id}] New prediction request received")
    
    # Record prediction request in audit trail
    audit_trail.record_event(
        event_type='prediction_request',
        model_type='all',
        model_version='current',
        operation='start',
        status='received',
        details={
            'request_id': request_id,
            'prediction_id': prediction_id,
            'client_ip': request.remote_addr,
            'timestamp': datetime.utcnow().isoformat()
        }
    )
    
    def record_failure(error_msg: str, status_code: int = 400, operation: str = 'unknown'):
        """Helper to record failed requests with metrics and audit"""
        duration = time.time() - start_time
        
        # Record metrics
        metrics_collector.record_request(
            duration=duration,
            success=False,
            ip=request.remote_addr,
            error=f"[{request_id}] {error_msg}"
        )
        
        try:
            memory_usage = psutil.Process().memory_percent()
            metrics_collector.record_system_metrics(
                memory_usage=memory_usage,
                upload_size=request.content_length or 0
            )
            
            # Record failure in audit trail
            audit_trail.record_event(
                event_type='prediction_failure',
                model_type='all',
                model_version='current',
                operation=operation,
                status='failed',
                details={
                    'request_id': request_id,
                    'prediction_id': prediction_id,
                    'error': error_msg,
                    'duration': f"{duration:.2f}s",
                    'memory_usage': memory_usage,
                    'client_ip': request.remote_addr,
                    'status_code': status_code
                }
            )
        except Exception as e:
            logger.warning(f"Failed to record metrics/audit during error: {str(e)}")
            
        return jsonify({
            'error': error_msg,
            'request_id': request_id,
            'prediction_id': prediction_id,
            'duration': f"{duration:.2f}s"
        }), status_code

    try:
        # Check security status
        security_status = metrics_collector.check_security_status()
        audit_trail.record_event(
            event_type='security_check',
            model_type='system',
            model_version='current',
            operation='validate',
            status='success' if security_status else 'failed',
            details={
                'request_id': request_id,
                'prediction_id': prediction_id,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        if not security_status:
            logger.error(f"[{request_id}] Security status check failed")
            return record_failure("Service temporarily unavailable due to security concerns", 503, "security_check")

        # Verify models are loaded
        models_ready = all([rf_model, lstm_model, tokenizer, drug_encoder])
        audit_trail.record_event(
            event_type='model_check',
            model_type='all',
            model_version='current',
            operation='availability',
            status='success' if models_ready else 'failed',
            details={
                'request_id': request_id,
                'prediction_id': prediction_id,
                'models_status': {
                    'rf': rf_model is not None,
                    'lstm': lstm_model is not None,
                    'tokenizer': tokenizer is not None,
                    'encoder': drug_encoder is not None
                }
            }
        )
        if not models_ready:
            logger.error(f"[{request_id}] Models not properly loaded")
            return record_failure("Service unavailable - models not ready", 503, "model_check")

        # Periodic model integrity check
        if hash(request_id) % 100 == 0:
            integrity_result = verify_model_integrity()
            audit_trail.record_event(
                event_type='integrity_check',
                model_type='all',
                model_version='current',
                operation='verify',
                status='success' if integrity_result else 'failed',
                details={
                    'request_id': request_id,
                    'prediction_id': prediction_id,
                    'periodic_check': True,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            if not integrity_result:
                logger.error(f"[{request_id}] Model integrity check failed")
                return record_failure("Service unavailable - model integrity check failed", 503, "integrity_check")

        # Validate request and record in audit
        audit_trail.record_event(
            event_type='request_validation',
            model_type='system',
            model_version='current',
            operation='start',
            status='in_progress',
            details={
                'request_id': request_id,
                'prediction_id': prediction_id,
                'content_type': request.content_type,
                'content_length': request.content_length
            }
        )

        if not request.files or 'file' not in request.files:
            audit_trail.record_event(
                event_type='request_validation',
                model_type='system',
                model_version='current',
                operation='validate',
                status='failed',
                details={
                    'request_id': request_id,
                    'prediction_id': prediction_id,
                    'error': 'No file provided',
                    'files_present': list(request.files.keys()) if request.files else []
                }
            )
            return record_failure("No file provided", operation='file_validation')
            
        file = request.files['file']
        if not file.filename:
            audit_trail.record_event(
                event_type='request_validation',
                model_type='system',
                model_version='current',
                operation='validate',
                status='failed',
                details={
                    'request_id': request_id,
                    'prediction_id': prediction_id,
                    'error': 'Invalid file',
                    'filename': str(file.filename)
                }
            )
            return record_failure("Invalid file", operation='file_validation')
            
        # Validate file security
        file_validation_start = time.time()
        is_file_valid = secure_file_validation(file)
        
        audit_trail.record_event(
            event_type='file_validation',
            model_type='system',
            model_version='current',
            operation='validate',
            status='success' if is_file_valid else 'failed',
            details={
                'request_id': request_id,
                'prediction_id': prediction_id,
                'filename': file.filename,
                'file_size': request.content_length,
                'content_type': file.content_type,
                'validation_duration': f"{time.time() - file_validation_start:.2f}s"
            }
        )
        
        if not is_file_valid:
            metrics_collector.record_validation_failure(
                "file_validation",
                f"Failed validation for file: {file.filename}"
            )
            return record_failure("Invalid or unsafe file", operation='file_validation')

        try:
            # File size check with audit
            file.seek(0, 2)
            file_size = file.tell()
            file.seek(0)
            
            audit_trail.record_event(
                event_type='file_processing',
                model_type='system',
                model_version='current',
                operation='size_check',
                status='checking',
                details={
                    'request_id': request_id,
                    'prediction_id': prediction_id,
                    'file_size': file_size,
                    'max_allowed': MAX_FILE_SIZE
                }
            )
            
            if file_size > MAX_FILE_SIZE:
                audit_trail.record_event(
                    event_type='file_processing',
                    model_type='system',
                    model_version='current',
                    operation='size_check',
                    status='failed',
                    details={
                        'request_id': request_id,
                        'prediction_id': prediction_id,
                        'file_size': file_size,
                        'max_allowed': MAX_FILE_SIZE,
                        'error': 'File size exceeds limit'
                    }
                )
                return record_failure(
                    f"File size exceeds maximum limit of {MAX_FILE_SIZE/1024/1024}MB",
                    operation='size_check'
                )

            # Read and process input data with audit
            process_start = time.time()
            try:
                df = pd.read_csv(file, encoding='utf-8', nrows=10000)
                audit_trail.record_event(
                    event_type='data_processing',
                    model_type='system',
                    model_version='current',
                    operation='csv_read',
                    status='success',
                    details={
                        'request_id': request_id,
                        'prediction_id': prediction_id,
                        'rows_read': len(df),
                        'columns': list(df.columns),
                        'duration': f"{time.time() - process_start:.2f}s"
                    }
                )
            except Exception as e:
                audit_trail.record_event(
                    event_type='data_processing',
                    model_type='system',
                    model_version='current',
                    operation='csv_read',
                    status='failed',
                    details={
                        'request_id': request_id,
                        'prediction_id': prediction_id,
                        'error': str(e),
                        'duration': f"{time.time() - process_start:.2f}s"
                    }
                )
                raise

            # Secure processing with audit
            process_start = time.time()
            try:
                df_processed, input_metadata = process_input_securely(
                    df,
                    max_text_length=MAX_TEXT_LENGTH
                )
                audit_trail.record_event(
                    event_type='data_processing',
                    model_type='system',
                    model_version='current',
                    operation='secure_processing',
                    status='success',
                    details={
                        'request_id': request_id,
                        'prediction_id': prediction_id,
                        'input_rows': len(df),
                        'processed_rows': len(df_processed),
                        'metadata': input_metadata,
                        'duration': f"{time.time() - process_start:.2f}s"
                    }
                )
            except Exception as e:
                audit_trail.record_event(
                    event_type='data_processing',
                    model_type='system',
                    model_version='current',
                    operation='secure_processing',
                    status='failed',
                    details={
                        'request_id': request_id,
                        'prediction_id': prediction_id,
                        'error': str(e),
                        'duration': f"{time.time() - process_start:.2f}s"
                    }
                )
                raise

            # Generate predictions with comprehensive audit
            pred_start = time.time()
            try:
                # Record prediction start
                audit_trail.record_event(
                    event_type='model_prediction',
                    model_type='ensemble',
                    model_version=f"rf-{getattr(rf_model, 'version', '1.0.0')}_lstm-{getattr(lstm_model, 'version', '1.0.0')}",
                    operation='start',
                    status='in_progress',
                    details={
                        'request_id': request_id,
                        'prediction_id': prediction_id,
                        'input_rows': len(df_processed),
                        'memory_usage': psutil.Process().memory_percent()
                    }
                )
                
                predictions, pred_metadata = generate_predictions(
                    df_processed,
                    tokenizer,
                    lstm_model,
                    rf_model,
                    max_sequence_length=MAX_SEQUENCE_LENGTH
                )
                
                # Validate prediction output
                if not isinstance(predictions, np.ndarray):
                    raise PredictionError("Invalid prediction format", {
                        'expected': 'numpy.ndarray',
                        'received': type(predictions).__name__
                    })
                
                if len(predictions) != len(df_processed):
                    raise PredictionError("Prediction length mismatch", {
                        'expected_length': len(df_processed),
                        'actual_length': len(predictions)
                    })
                # Record successful prediction completion
                pred_duration = time.time() - pred_start
                audit_trail.record_event(
                    event_type='prediction',
                    model_type='ensemble',
                    model_version=f"rf-{getattr(rf_model, 'version', '1.0.0')}_lstm-{getattr(lstm_model, 'version', '1.0.0')}",
                    operation='predict',
                    status='success',
                    details={
                        'request_id': request_id,
                        'prediction_id': prediction_id,
                        'input_rows': len(df_processed),
                        'output_shape': predictions.shape,
                        'metadata': pred_metadata,
                        'duration': f"{pred_duration:.2f}s",
                        'memory_usage': psutil.Process().memory_percent(),
                        'prediction_stats': {
                            'min': float(np.min(predictions)),
                            'max': float(np.max(predictions)),
                            'mean': float(np.mean(predictions)),
                            'std': float(np.std(predictions))
                        }
                    }
                )
                
                # Record system metrics
                metrics_collector.record_system_metrics(
                    memory_usage=psutil.Process().memory_percent(),
                    upload_size=file_size
                )
                
                # Format and validate final response
                try:
                    response_data, status_code = format_prediction_response(
                        predictions=predictions,
                        request_id=request_id,
                        prediction_id=prediction_id,
                        start_time=start_time,
                        metadata={
                            'processing_metadata': input_metadata,
                            'prediction_metadata': pred_metadata,
                            'system_metadata': {
                                'processing_time': f"{pred_duration:.2f}s",
                                'total_time': f"{time.time() - start_time:.2f}s",
                                'input_rows': len(df_processed),
                                'output_rows': len(predictions)
                            }
                        }
                    )
                    
                    # Record successful completion in audit trail
                    audit_trail.record_event(
                        event_type='request_complete',
                        model_type='system',
                        model_version='current',
                        operation='complete',
                        status='success',
                        details={
                            'request_id': request_id,
                            'prediction_id': prediction_id,
                            'duration': f"{time.time() - start_time:.2f}s",
                            'response_size': len(json.dumps(response_data)),
                            'status_code': status_code
                        }
                    )
                    
                    # Clean up resources before returning
                    cleanup_prediction_resources(request_id, prediction_id)
                    
                    return jsonify(response_data), status_code
                    
                except Exception as e:
                    logger.error(f"Error formatting response: {str(e)}")
                    error_details = {
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'stage': 'response_formatting',
                        'traceback': traceback.format_exc()
                    }
                    
                    # Record response formatting error
                    audit_trail.record_event(
                        event_type='response_error',
                        model_type='system',
                        model_version='current',
                        operation='format_response',
                        status='failed',
                        details={
                            'request_id': request_id,
                            'prediction_id': prediction_id,
                            'error_details': error_details,
                            'memory_usage': psutil.Process().memory_percent()
                        }
                    )
                    
                    # Attempt cleanup even on error
                    try:
                        cleanup_prediction_resources(request_id, prediction_id)
                    except Exception as cleanup_error:
                        logger.error(f"Cleanup error after response formatting failure: {cleanup_error}")
                    
                    raise PredictionError("Failed to format prediction response", error_details)
                    
        except Exception as e:
            # Capture final error state
            error_time = time.time()
            error_duration = error_time - start_time
            
            error_details = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'duration': f"{error_duration:.2f}s",
                'stage': 'prediction_processing',
                'traceback': traceback.format_exc(),
                'memory_state': {
                    'memory_usage': psutil.Process().memory_percent(),
                    'system_load': os.getloadavg()[0]
                }
            }
            
            logger.error(f"[{request_id}] Critical error during prediction: {json.dumps(error_details, indent=2)}")
            
            # Record comprehensive error information in audit trail
            audit_trail.record_event(
                event_type='critical_error',
                model_type='system',
                model_version='current',
                operation='prediction',
                status='failed',
                details={
                    'request_id': request_id,
                    'prediction_id': prediction_id,
                    'error_details': error_details,
                    'input_info': {
                        'file_size': file_size if 'file_size' in locals() else None,
                        'rows_processed': len(df_processed) if 'df_processed' in locals() else None
                    },
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            # Ensure cleanup is attempted
            try:
                cleanup_prediction_resources(request_id, prediction_id)
            except Exception as cleanup_error:
                logger.error(f"Cleanup error after critical failure: {cleanup_error}")
            
            if isinstance(e, PredictionError):
                # Known prediction errors get passed through
                return record_failure(str(e), 400, "prediction_error")
            else:
                # Unknown errors are treated as internal server errors
                return record_failure("Internal server error", 500, "critical_error")
            except Exception as e:
                audit_trail.record_event(
                    event_type='prediction',
                    model_type='ensemble',
                    model_version=f"rf-{getattr(rf_model, 'version', '1.0.0')}_lstm-{getattr(lstm_model, 'version', '1.0.0')}",
                    operation='predict',
                    status='failed',
                    details={
                        'request_id': request_id,
                        'prediction_id': prediction_id,
                        'error': str(e),
                        'duration': f"{time.time() - pred_start:.2f}s"
                    }
                )
                raise

            # Combine metadata
            metadata = {
                'input': input_metadata,
                'prediction': pred_metadata,
                'system': {
                    'memory_usage': psutil.Process().memory_percent(),
                    'processing_time': f"{time.time() - start_time:.2f}s"
                }
            }

            # Record success metrics
            duration = time.time() - start_time
            metrics_collector.record_request(
                duration=duration,
                success=True,
                ip=request.remote_addr
            )
            metrics_collector.record_prediction(
                duration=duration,
                values=predictions
            )

            # Return successful response
            return jsonify({
                'success': True,
                'request_id': request_id,
                'predictions': predictions.tolist(),
                'duration': f"{duration:.2f}s",
                'model_versions': {
                    'rf': getattr(rf_model, 'version', '1.0.0'),
                    'lstm': getattr(lstm_model, 'version', '1.0.0')
                },
                'metadata': metadata
            }), 200

        except PredictionError as e:
            logger.error(f"[{request_id}] Prediction processing failed: {str(e)}")
            return record_failure(str(e))
            
        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error during prediction: {str(e)}")
            return record_failure("Internal server error", 500)

    except Exception as e:
        logger.error(f"[{request_id}] Unhandled error: {str(e)}")
        return record_failure("Internal server error", 500)
    
    try:
        # Check security status
        if not metrics_collector.check_security_status():
            logger.error(f"[{request_id}] Security status check failed")
            return record_failure("Service temporarily unavailable due to security concerns", 503)

        # Verify models are loaded
        if not all([rf_model, lstm_model, tokenizer, drug_encoder]):
            logger.error(f"[{request_id}] Models not properly loaded")
            return record_failure("Service unavailable - models not ready", 503)

        # Verify model integrity periodically (every 100 requests)
        if hash(request_id) % 100 == 0:
            if not verify_model_integrity():
                logger.error(f"[{request_id}] Model integrity check failed")
                return record_failure("Service unavailable - model integrity check failed", 503)
        
        # Validate request
        if not request.files or 'file' not in request.files:
            logger.warning(f"[{request_id}] No file provided in request")
            return record_failure("No file provided")
            
        file = request.files['file']
        if not file.filename:
            logger.warning(f"[{request_id}] Empty filename provided")
            return record_failure("Invalid file")
            
        # Validate file security
        if not secure_file_validation(file):
            logger.warning(f"[{request_id}] File validation failed for {file.filename}")
            metrics_collector.record_validation_failure(
                "file_validation",
                f"Failed validation for file: {file.filename}"
            )
            return record_failure("Invalid or unsafe file")
            
        # Process file securely with size limit
        try:
            df = pd.read_csv(file, encoding='utf-8', nrows=10000)  # Limit rows as security measure
        except Exception as e:
            logger.error(f"[{request_id}] Failed to read CSV: {str(e)}")
            return jsonify({'error': 'Invalid CSV format'}), 400
            
        # Input validation
        required_columns = ['text', 'drug_name']
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"[{request_id}] Missing columns: {[col for col in required_columns if col not in df.columns]}")
            return jsonify({'error': 'Missing required columns: text, drug_name'}), 400
            
        # Check for empty dataframe
        if df.empty:
            logger.warning(f"[{request_id}] Empty dataframe provided")
            return jsonify({'error': 'No data provided in file'}), 400
            
        # Validate data types
        if not all(df[col].dtype == object for col in required_columns):
            logger.warning(f"[{request_id}] Invalid data types in input")
            return jsonify({'error': 'Invalid data types for text or drug_name'}), 400
            
        # Remove any rows with missing values
        df = df.dropna(subset=required_columns)
        if df.empty:
            logger.warning(f"[{request_id}] All rows contained missing values")
            return jsonify({'error': 'All rows contain missing values'}), 400
            
        # Sanitize input
        df['text'] = df['text'].astype(str).apply(lambda x: re.sub(r'[^\w\s-]', '', x))
        df['drug_name'] = df['drug_name'].astype(str).apply(lambda x: re.sub(r'[^\w\s-]', '', x))
        
        logger.info(f"[{request_id}] Processing {len(df)} valid rows")
        
        # Tokenize and prepare input with error handling
        try:
            sequences = tokenizer.texts_to_sequences(df['text'].values)
            padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
        except Exception as e:
            logger.error(f"[{request_id}] Tokenization failed: {str(e)}")
            return jsonify({'error': 'Failed to process text input'}), 500
        
        # Validate model compatibility first
        try:
            rf_version = getattr(rf_model, 'version', '1.0.0')
            lstm_version = getattr(lstm_model, 'version', '1.0.0')
            validate_model_compatibility(rf_version, lstm_version)
        except ModelValidationError as e:
            logger.error(f"[{request_id}] Model compatibility check failed: {str(e)}")
            return jsonify({'error': 'Model version incompatibility detected'}), 500

        # Validate input data
        try:
            validate_model_input(
                text=df['text'].tolist(),
                drug_names=df['drug_name'].tolist(),
                max_text_length=1000,
                max_drug_name_length=100
            )
        except ModelValidationError as e:
            logger.error(f"[{request_id}] Input validation failed: {str(e)}")
            return jsonify({'error': str(e)}), 400

        # Make predictions using both models with timeouts and validation
        try:
            # Convert to numpy array for consistent processing
            drug_features = np.array(df['drug_name'].values).reshape(-1, 1)
            
            # Make predictions
            lstm_preds = lstm_model.predict(padded_sequences, batch_size=32)
            rf_preds = rf_model.predict(drug_features)
            
            # Validate prediction shapes and values
            expected_shape = (len(df), 1)  # Assuming binary classification
            validate_prediction_shape(lstm_preds, expected_shape)
            validate_prediction_shape(rf_preds, expected_shape)
            
            # Verify prediction consistency between models
            verify_prediction_consistency(rf_preds, lstm_preds, threshold=0.5)
            
            # Combine predictions with validation
            final_predictions = np.clip((lstm_preds + rf_preds) / 2, 0, 1)  # Ensure values are in [0,1]
            
            # Final validation of combined predictions
            validate_prediction_shape(final_predictions, expected_shape)
            
        except Exception as e:
            logger.error(f"[{request_id}] Prediction failed: {str(e)}")
            return jsonify({'error': 'Failed to generate predictions'}), 500
        
        # Prepare response with versions and metadata
        response = {
            'success': True,
            'request_id': request_id,
            'predictions': final_predictions.tolist(),
            'model_versions': {
                'rf': getattr(rf_model, 'version', '1.0.0'),
                'lstm': getattr(lstm_model, 'version', '1.0.0')
            },
            'metadata': {
                'processed_rows': len(df),
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        logger.info(f"[{request_id}] Successfully processed prediction request")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error in prediction endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'request_id': request_id
        }), 500

def verify_model_integrity():
    """Verify integrity of all loaded models and update manifest if needed"""
    try:
        # First update the manifest with current hashes if needed
        try:
            manifest = update_model_manifest()
            logger.info("Model manifest updated successfully")
        except Exception as e:
            logger.error(f"Failed to update model manifest: {str(e)}")
            return False

        # Verify all models against manifest
        if not verify_all_models():
            raise SecurityError("One or more models failed integrity check")

        # Additional security checks
        for model_name, model_path in MODEL_PATHS.items():
            if not model_path.exists():
                raise SecurityError(f"Required model file missing: {model_path}")

            # Check file permissions
            import stat
            st = os.stat(model_path)
            if bool(st.st_mode & stat.S_IWOTH):
                raise SecurityError(f"Unsafe file permissions on {model_path}")

            # Check file size limits
            if os.path.getsize(model_path) > 500 * 1024 * 1024:  # 500MB limit
                raise SecurityError(f"Model file too large: {model_path}")

        # Verify model compatibility
        if not all(hasattr(model, 'version') for model in [rf_model, lstm_model]):
            logger.warning("One or more models missing version information")
            
        try:
            validate_model_compatibility(
                getattr(rf_model, 'version', '1.0.0'),
                getattr(lstm_model, 'version', '1.0.0')
            )
        except ModelValidationError as e:
            raise SecurityError(f"Model compatibility check failed: {str(e)}")

        logger.info("All models passed integrity and compatibility verification")
        return True

    except Exception as e:
        logger.error(f"Model verification failed: {str(e)}")
        return False

def cleanup_handler(signum, frame):
    """Handle cleanup when shutting down"""
    logger.info("Received shutdown signal, cleaning up...")
    try:
        # Create final backup before shutdown
        try:
            logger.info("Creating final backup before shutdown...")
            backup_result = backup_manager.create_backup()
            logger.info(f"Final backup created: {backup_result['backup_name']}")
        except Exception as e:
            logger.error(f"Failed to create final backup: {str(e)}")

        # Generate final security report
        try:
            final_report = metrics_collector.get_security_report()
            logger.info(f"Final security report: {json.dumps(final_report, indent=2)}")
            
            # Save final report to file
            report_path = Path(app.config['UPLOAD_FOLDER']) / f"security_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(final_report, f, indent=2)
            logger.info(f"Final security report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate/save final security report: {str(e)}")

        # Verify model integrity one last time
        try:
            if not manifest_manager.verify_integrity():
                logger.warning("Final model integrity check failed")
                # Record the issue but continue shutdown
        except Exception as e:
            logger.error(f"Error during final integrity check: {str(e)}")

        # Shutdown security components
        try:
            shutdown_security_components()
        except Exception as e:
            logger.error(f"Error shutting down security components: {str(e)}")

        # Clean up temporary files
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for file in os.listdir(app.config['UPLOAD_FOLDER']):
                try:
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                    if os.path.isfile(file_path):
                        # Skip final security report
                        if 'security_report_' in file_path:
                            continue
                        # Securely overwrite sensitive files before deletion
                        with open(file_path, 'wb') as f:
                            f.write(os.urandom(os.path.getsize(file_path)))
                        os.remove(file_path)
                except Exception as e:
                    logger.error(f"Failed to remove file {file}: {str(e)}")

        # Verify no sensitive files remain
        try:
            sensitive_patterns = ['*.csv', '*.json', '*.keras', '*.pkl', '*.h5']
            for pattern in sensitive_patterns:
                for filepath in Path(app.config['UPLOAD_FOLDER']).rglob(pattern):
                    if 'security_report_' not in str(filepath) and 'backup' not in str(filepath):
                        logger.warning(f"Found remaining sensitive file: {filepath}")
        except Exception as e:
            logger.error(f"Error checking for remaining files: {str(e)}")

        # Clear sensitive data from memory
        try:
            # Clear model data
            global rf_model, lstm_model, tokenizer, drug_encoder
            rf_model = None
            lstm_model = None
            tokenizer = None
            drug_encoder = None
            
            # Clear other sensitive objects
            model_reloader.models.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear sensitive environment variables
            sensitive_vars = ['ADMIN_TOKEN', 'API_KEY', 'SECRET_KEY', 'DATABASE_URL']
            for key in sensitive_vars:
                if key in os.environ:
                    os.environ[key] = ''
        except Exception as e:
            logger.error(f"Error clearing sensitive data: {str(e)}")

        logger.info("Cleanup completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Critical error during cleanup: {str(e)}")
        # Attempt to save error report before exit
        try:
            error_report = {
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'type': type(e).__name__
            }
            with open('shutdown_error.json', 'w') as f:
                json.dump(error_report, f, indent=2)
        except:
            pass
        sys.exit(1)

def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown"""
    signal.signal(signal.SIGTERM, cleanup_handler)
    signal.signal(signal.SIGINT, cleanup_handler)

def setup_server_security():
    """Configure server security settings"""
    # Set secure file permissions for upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], mode=0o750, exist_ok=True)
    
    # Set secure umask for new files
    os.umask(0o077)
    
    # Verify environment security
    if not os.environ.get('FLASK_ENV') == 'production':
        logger.warning("Not running in production mode - security features may be limited")
    
    # Check if running as root (not recommended)
    if os.geteuid() == 0:
        logger.error("Running as root is not recommended")
        sys.exit(1)
    
    # Set process resource limits
    import resource
    resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 2048))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 * 1024 * 1024, 3 * 1024 * 1024 * 1024))  # 2-3GB memory limit

def periodic_security_check():
    """Perform periodic security checks including model verification"""
    while True:
        try:
            # Verify model integrity
            if not verify_model_integrity():
                logger.critical("Periodic model verification failed")
                # Don't exit immediately, let monitoring system handle it
                
            # Check system resources
            memory_usage = psutil.Process().memory_percent()
            if memory_usage > 90:  # 90% memory usage threshold
                logger.warning(f"High memory usage detected: {memory_usage}%")
                
            # Check upload directory size
            upload_dir_size = sum(
                os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], f))
                for f in os.listdir(app.config['UPLOAD_FOLDER'])
                if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))
            )
            if upload_dir_size > 500 * 1024 * 1024:  # 500MB limit
                logger.warning("Upload directory size exceeds limit, cleaning old files")
                cleanup_old_uploads()
                
        except Exception as e:
            logger.error(f"Error in periodic security check: {str(e)}")
            
        # Sleep for 1 hour before next check
        time.sleep(3600)

def cleanup_old_uploads(max_age_hours: int = 24):
    """Clean up old files from upload directory"""
    try:
        current_time = datetime.now()
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                if (current_time - file_time).total_seconds() > max_age_hours * 3600:
                    os.remove(filepath)
                    logger.info(f"Removed old upload file: {filename}")
    except Exception as e:
        logger.error(f"Error cleaning up uploads: {str(e)}")

# Initialize model reloader
model_reloader = ModelReloader(MODEL_PATHS, manifest_manager)

def initialize_models():
    """Securely initialize and load all required models with audit trail"""
    try:
        # Record initialization start in audit trail
        audit_trail.record_event(
            event_type='initialization',
            model_type='all',
            model_version='current',
            operation='start',
            status='in_progress',
            details={'timestamp': datetime.utcnow().isoformat()}
        )
        
        # Attempt to load all models using the reloader
        reload_status = model_reloader.reload_models(validate=True)
        
        if not reload_status['success']:
            error_messages = '\n'.join(reload_status['messages'])
            # Record failure in audit trail
            audit_trail.record_event(
                event_type='initialization',
                model_type='all',
                model_version='current',
                operation='load',
                status='failed',
                details={
                    'error': error_messages,
                    'reload_status': reload_status
                }
            )
            raise SecurityError(f"Model initialization failed:\n{error_messages}")
            
        # Update global model references
        update_global_models(model_reloader.models)
        
        # Verify all models are properly loaded
        if not all([rf_model, lstm_model, tokenizer, drug_encoder]):
            audit_trail.record_event(
                event_type='initialization',
                model_type='all',
                model_version='current',
                operation='verify',
                status='failed',
                details={'error': 'One or more models missing'}
            )
            raise SecurityError("One or more models failed to load")
            
        # Record individual model versions in audit trail
        for model_name, model in [
            ('rf', rf_model),
            ('lstm', lstm_model),
            ('tokenizer', tokenizer),
            ('encoder', drug_encoder)
        ]:
            audit_trail.record_event(
                event_type='initialization',
                model_type=model_name,
                model_version=getattr(model, 'version', '1.0.0'),
                operation='load',
                status='success',
                details={
                    'version': getattr(model, 'version', '1.0.0'),
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
        
        # Log successful initialization
        logger.info("All models loaded and validated successfully")
        logger.info(f"Model versions: RF={getattr(rf_model, 'version', '1.0.0')}, "
                   f"LSTM={getattr(lstm_model, 'version', '1.0.0')}")
        
        # Record successful initialization completion
        audit_trail.record_event(
            event_type='initialization',
            model_type='all',
            model_version='current',
            operation='complete',
            status='success',
            details={
                'rf_version': getattr(rf_model, 'version', '1.0.0'),
                'lstm_version': getattr(lstm_model, 'version', '1.0.0'),
                'models_loaded': [
                    'rf', 'lstm', 'tokenizer', 'encoder'
                ]
            }
        )
        return True
        
    except Exception as e:
        logger.critical(f"Failed to initialize models: {str(e)}")
        # Record critical failure in audit trail
        audit_trail.record_event(
            event_type='initialization',
            model_type='all',
            model_version='current',
            operation='error',
            status='failed',
            details={
                'error': str(e),
                'error_type': type(e).__name__,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        return False

@app.route('/admin/models/reload', methods=['POST'])
@limiter.limit("2 per minute")  # Strict rate limit for model reloading
def reload_models():
    """Secure endpoint for reloading models"""
    # Verify admin authentication
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Unauthorized'}), 401
        
    token = auth_header.split(' ')[1]
    admin_token = os.environ.get('ADMIN_TOKEN')
    if not admin_token or not secrets.compare_digest(token, admin_token):
        metrics_collector.record_request(
            duration=0,
            success=False,
            ip=request.remote_addr,
            error="Unauthorized model reload attempt"
        )
        return jsonify({'error': 'Unauthorized'}), 401
        
    try:
        # Record start time for metrics
        start_time = time.time()
        
        # Run security checks before reload
        security_status = security_validator.run_security_checks()
        if security_status['status'] == 'failed':
            return jsonify({
                'error': 'Security check failed',
                'details': security_status['issues']
            }), 503
            
        # Verify manifest integrity
        if not manifest_manager.verify_integrity():
            return jsonify({
                'error': 'Model manifest integrity check failed'
            }), 503
            
        # Attempt to reload models
        reload_status = model_reloader.reload_models(validate=True)
        
        if reload_status['success']:
            # Update global model references
            update_global_models(model_reloader.models)
            
            # Validate loaded models
            validation_results = model_reloader.validate_all_models()
            
            if not validation_results['success']:
                # Rollback to previous models if validation fails
                logger.error("Model validation failed after reload")
                return jsonify({
                    'error': 'Model validation failed after reload',
                    'details': validation_results
                }), 500
                
            # Record successful reload
            duration = time.time() - start_time
            metrics_collector.record_request(
                duration=duration,
                success=True,
                ip=request.remote_addr
            )
            
            return jsonify({
                'success': True,
                'duration': f"{duration:.2f}s",
                'reload_status': reload_status,
                'validation_results': validation_results
            }), 200
            
        else:
            logger.error("Model reload failed")
            return jsonify({
                'error': 'Model reload failed',
                'details': reload_status
            }), 500
            
    except Exception as e:
        logger.error(f"Error during model reload: {str(e)}")
        return jsonify({
            'error': 'Internal server error during model reload',
            'details': str(e)
        }), 500

@app.route('/admin/models/backup', methods=['POST'])
@limiter.limit("2 per hour")
def create_model_backup():
    """Secure endpoint for creating model backups"""
    # Verify admin authentication
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Unauthorized'}), 401
        
    token = auth_header.split(' ')[1]
    admin_token = os.environ.get('ADMIN_TOKEN')
    if not admin_token or not secrets.compare_digest(token, admin_token):
        metrics_collector.record_request(
            duration=0,
            success=False,
            ip=request.remote_addr,
            error="Unauthorized backup attempt"
        )
        return jsonify({'error': 'Unauthorized'}), 401
        
    try:
        # Verify security status before backup
        security_status = security_validator.run_security_checks()
        if security_status['status'] == 'failed':
            return jsonify({
                'error': 'Security check failed',
                'details': security_status['issues']
            }), 503
            
        # Create backup
        backup_result = backup_manager.create_backup()
        
        # Record successful backup
        metrics_collector.record_request(
            duration=time.time() - request.start_time,
            success=True,
            ip=request.remote_addr
        )
        
        return jsonify({
            'success': True,
            'backup': backup_result
        }), 200
        
    except Exception as e:
        logger.error(f"Backup creation failed: {str(e)}")
        return jsonify({
            'error': 'Backup creation failed',
            'details': str(e)
        }), 500

@app.route('/admin/models/backup/restore', methods=['POST'])
@limiter.limit("1 per hour")
def restore_model_backup():
    """Secure endpoint for restoring model backups"""
    # Verify admin authentication
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Unauthorized'}), 401
        
    token = auth_header.split(' ')[1]
    admin_token = os.environ.get('ADMIN_TOKEN')
    if not admin_token or not secrets.compare_digest(token, admin_token):
        metrics_collector.record_request(
            duration=0,
            success=False,
            ip=request.remote_addr,
            error="Unauthorized restore attempt"
        )
        return jsonify({'error': 'Unauthorized'}), 401
        
    try:
        # Get backup name from request
        data = request.get_json()
        if not data or 'backup_name' not in data:
            return jsonify({'error': 'Backup name required'}), 400
            
        backup_name = data['backup_name']
        
        # Verify backup before restore
        verification = backup_manager.verify_backup(backup_name)
        if not verification['success']:
            return jsonify({
                'error': 'Backup verification failed',
                'details': verification['issues']
            }), 400
            
        # Restore backup
        restore_result = backup_manager.restore_backup(backup_name, validate=True)
        
        # Reload models after restore
        reload_status = model_reloader.reload_models(validate=True)
        if not reload_status['success']:
            return jsonify({
                'error': 'Model reload failed after restore',
                'details': reload_status
            }), 500
            
        # Update global model references
        update_global_models(model_reloader.models)
        
        # Record successful restore
        metrics_collector.record_request(
            duration=time.time() - request.start_time,
            success=True,
            ip=request.remote_addr
        )
        
        return jsonify({
            'success': True,
            'restore': restore_result,
            'reload': reload_status
        }), 200
        
    except Exception as e:
        logger.error(f"Backup restoration failed: {str(e)}")
        return jsonify({
            'error': 'Backup restoration failed',
            'details': str(e)
        }), 500

@app.route('/admin/models/backup/list', methods=['GET'])
@limiter.limit("10 per minute")
def list_model_backups():
    """Secure endpoint for listing model backups"""
    # Verify admin authentication
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Unauthorized'}), 401
        
    token = auth_header.split(' ')[1]
    admin_token = os.environ.get('ADMIN_TOKEN')
    if not admin_token or not secrets.compare_digest(token, admin_token):
        return jsonify({'error': 'Unauthorized'}), 401
        
    try:
        backups = backup_manager.list_backups()
        return jsonify({
            'success': True,
            'backups': backups
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to list backups: {str(e)}")
        return jsonify({
            'error': 'Failed to list backups',
            'details': str(e)
        }), 500

@app.route('/admin/audit', methods=['GET'])
@limiter.limit("10 per minute")
def get_audit_trail():
    """Secure endpoint for accessing model audit trail"""
    # Verify admin authentication
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Unauthorized'}), 401
        
    token = auth_header.split(' ')[1]
    admin_token = os.environ.get('ADMIN_TOKEN')
    if not admin_token or not secrets.compare_digest(token, admin_token):
        metrics_collector.record_request(
            duration=0,
            success=False,
            ip=request.remote_addr,
            error="Unauthorized audit trail access attempt"
        )
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        # Get query parameters
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        event_type = request.args.get('event_type')
        model_type = request.args.get('model_type')
        status = request.args.get('status')
        
        # Query audit trail
        events = audit_trail.query_events(
            start_time=start_time,
            end_time=end_time,
            event_type=event_type,
            model_type=model_type,
            status=status
        )
        
        # Get recent activity summary
        recent_activity = audit_trail.get_recent_activity(limit=10)
        
        # Get model-specific histories
        model_histories = {
            'rf': audit_trail.get_model_history('rf')[-5:],
            'lstm': audit_trail.get_model_history('lstm')[-5:],
            'tokenizer': audit_trail.get_model_history('tokenizer')[-5:],
            'encoder': audit_trail.get_model_history('encoder')[-5:]
        }
        
        response = {
            'events': events,
            'recent_activity': recent_activity,
            'model_histories': model_histories,
            'query_params': {
                'start_time': start_time,
                'end_time': end_time,
                'event_type': event_type,
                'model_type': model_type,
                'status': status
            }
        }
        
        # Record successful audit access
        metrics_collector.record_request(
            duration=time.time() - request.start_time,
            success=True,
            ip=request.remote_addr
        )
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error accessing audit trail: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/admin/audit/verify', methods=['POST'])
@limiter.limit("10 per minute")
def verify_audit_event():
    """Verify the integrity of a specific audit event"""
    # Verify admin authentication
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Unauthorized'}), 401
        
    token = auth_header.split(' ')[1]
    admin_token = os.environ.get('ADMIN_TOKEN')
    if not admin_token or not secrets.compare_digest(token, admin_token):
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        data = request.get_json()
        if not data or 'event_hash' not in data:
            return jsonify({'error': 'Event hash required'}), 400
            
        event_hash = data['event_hash']
        is_valid = audit_trail.verify_event_integrity(event_hash)
        
        return jsonify({
            'event_hash': event_hash,
            'is_valid': is_valid,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error verifying audit event: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/admin/monitor', methods=['GET'])
@limiter.limit("30 per minute")
def monitor_system():
    """Secure endpoint for monitoring system health and model performance"""
    # Verify admin authentication
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Unauthorized'}), 401
        
    token = auth_header.split(' ')[1]
    admin_token = os.environ.get('ADMIN_TOKEN')
    if not admin_token or not secrets.compare_digest(token, admin_token):
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        # Get current process info
        process = psutil.Process()
        
        # Collect comprehensive system metrics
        system_metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'memory': {
                'percent': process.memory_percent(),
                'rss': process.memory_info().rss / 1024 / 1024,  # MB
                'vms': process.memory_info().vms / 1024 / 1024   # MB
            },
            'cpu': {
                'percent': process.cpu_percent(),
                'system_load': os.getloadavg(),
                'threads': len(process.threads())
            },
            'io': {
                'read_bytes': process.io_counters().read_bytes / 1024 / 1024,  # MB
                'write_bytes': process.io_counters().write_bytes / 1024 / 1024 # MB
            },
            'files': {
                'open_files': len(process.open_files()),
                'connections': len(process.connections())
            }
        }
        
        # Get model metrics from audit trail
        recent_predictions = audit_trail.query_events(
            event_type='prediction',
            start_time=(datetime.utcnow() - timedelta(hours=1)).isoformat()
        )
        
        model_metrics = {
            'predictions_last_hour': len(recent_predictions),
            'success_rate': sum(1 for p in recent_predictions if p['status'] == 'success') / max(len(recent_predictions), 1),
            'average_duration': statistics.mean(
                float(p['details']['duration'].rstrip('s')) 
                for p in recent_predictions 
                if p['status'] == 'success'
            ) if recent_predictions else 0
        }
        
        # Get security metrics
        security_metrics = {
            'security_status': security_validator.run_security_checks(),
            'model_integrity': manifest_manager.verify_integrity(),
            'recent_security_events': audit_trail.query_events(
                event_type='security_error',
                start_time=(datetime.utcnow() - timedelta(hours=24)).isoformat()
            )
        }
        
        # Check upload directory status
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        upload_metrics = {
            'total_size': sum(f.stat().st_size for f in upload_dir.glob('**/*') if f.is_file()) / 1024 / 1024,  # MB
            'file_count': len(list(upload_dir.glob('**/*'))),
            'oldest_file_age': max(
                (time.time() - f.stat().st_mtime for f in upload_dir.glob('**/*') if f.is_file()),
                default=0
            ) / 3600  # hours
        }
        
        response = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'system_metrics': system_metrics,
            'model_metrics': model_metrics,
            'security_metrics': security_metrics,
            'upload_metrics': upload_metrics
        }
        
        # Record monitoring access in audit trail
        audit_trail.record_event(
            event_type='system_monitor',
            model_type='system',
            model_version='current',
            operation='monitor',
            status='success',
            details={
                'client_ip': request.remote_addr,
                'system_status': 'healthy',
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in system monitoring: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/admin/validate/security', methods=['POST'])
@limiter.limit("5 per hour")
def validate_security():
    """Secure endpoint for validating security measures"""
    # Verify admin authentication
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Unauthorized'}), 401
        
    token = auth_header.split(' ')[1]
    admin_token = os.environ.get('ADMIN_TOKEN')
    if not admin_token or not secrets.compare_digest(token, admin_token):
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        validation_id = secrets.token_hex(8)
        
        # Record validation start
        audit_trail.record_event(
            event_type='security_validation',
            model_type='system',
            model_version='current',
            operation='start',
            status='in_progress',
            details={
                'validation_id': validation_id,
                'client_ip': request.remote_addr
            }
        )
        
        # Comprehensive security validation
        validation_results = {
            'validation_id': validation_id,
            'timestamp': datetime.utcnow().isoformat(),
            'security_checks': security_validator.run_security_checks(),
            'model_integrity': manifest_manager.verify_integrity(),
            'audit_trail': {
                'status': 'valid',
                'recent_events': len(audit_trail.get_recent_activity())
            },
            'model_versions': {
                'rf': getattr(rf_model, 'version', 'unknown'),
                'lstm': getattr(lstm_model, 'version', 'unknown')
            },
            'file_permissions': {},
            'system_security': {}
        }
        
        # Check file permissions
        for path in [app.config['UPLOAD_FOLDER'], Path(__file__).parent / 'models']:
            path = Path(path)
            if path.exists():
                validation_results['file_permissions'][str(path)] = {
                    'mode': oct(path.stat().st_mode)[-3:],
                    'owner': path.owner(),
                    'group': path.group()
                }
        
        # Check system security settings
        validation_results['system_security'] = {
            'umask': oct(os.umask(0o077))[2:],  # Get and restore umask
            'process_user': os.getuid(),
            'env_vars_secure': all(
                var not in os.environ for var in ['DEBUG', 'DEVELOPMENT', 'TEST']
            ),
            'ssl_enabled': app.config.get('SESSION_COOKIE_SECURE', False),
            'rate_limiting': app.config.get('RATELIMIT_ENABLED', False)
        }
        
        # Record validation completion
        audit_trail.record_event(
            event_type='security_validation',
            model_type='system',
            model_version='current',
            operation='complete',
            status='success',
            details={
                'validation_id': validation_id,
                'results': validation_results
            }
        )
        
        return jsonify(validation_results), 200
        
    except Exception as e:
        logger.error(f"Error during security validation: {str(e)}")
        return jsonify({
            'error': 'Security validation failed',
            'details': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/admin/models/status', methods=['GET'])
@limiter.limit("10 per minute")
def model_status():
    """Secure endpoint for checking model status"""
    # Verify admin authentication
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Unauthorized'}), 401
        
    token = auth_header.split(' ')[1]
    admin_token = os.environ.get('ADMIN_TOKEN')
    if not admin_token or not secrets.compare_digest(token, admin_token):
        return jsonify({'error': 'Unauthorized'}), 401
        
    try:
        # Get model status
        validation_results = model_reloader.validate_all_models()
        reload_status = model_reloader.get_reload_status()
        
        return jsonify({
            'validation_results': validation_results,
            'last_reload': reload_status,
            'models': {
                'rf': {
                    'loaded': rf_model is not None,
                    'version': getattr(rf_model, 'version', 'unknown'),
                    'last_verified': getattr(rf_model, 'last_verified', 'unknown')
                },
                'lstm': {
                    'loaded': lstm_model is not None,
                    'version': getattr(lstm_model, 'version', 'unknown'),
                    'last_verified': getattr(lstm_model, 'last_verified', 'unknown')
                },
                'tokenizer': {
                    'loaded': tokenizer is not None,
                    'last_verified': getattr(tokenizer, 'last_verified', 'unknown')
                },
                'encoder': {
                    'loaded': drug_encoder is not None,
                    'last_verified': getattr(drug_encoder, 'last_verified', 'unknown')
                }
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

def validate_environment():
    """Validate and sanitize environment configuration"""
    required_vars = {
        'FLASK_ENV': 'production',
        'PORT': '5000',
        'HOST': '127.0.0.1',
        'MAX_THREADS': '4',
        'CONNECTION_LIMIT': '1024',
        'RATE_LIMIT_DEFAULT': '200 per day',
        'RATE_LIMIT_HEALTH': '60 per minute'
    }
    
    # Validate and set defaults for missing variables
    for var, default in required_vars.items():
        if var not in os.environ:
            logger.warning(f"Environment variable {var} not set, using default: {default}")
            os.environ[var] = default
            
    # Validate numeric values
    try:
        port = int(os.environ['PORT'])
        if port < 1024 or port > 65535:
            raise ValueError(f"Invalid port number: {port}")
            
        max_threads = int(os.environ['MAX_THREADS'])
        if max_threads < 1 or max_threads > 32:
            raise ValueError(f"Invalid thread count: {max_threads}")
            
        conn_limit = int(os.environ['CONNECTION_LIMIT'])
        if conn_limit < 1 or conn_limit > 10000:
            raise ValueError(f"Invalid connection limit: {conn_limit}")
    except ValueError as e:
        logger.critical(f"Environment validation failed: {str(e)}")
        return False
        
    return True

@app.route('/health/check', methods=['GET'])
@limiter.limit(RATE_LIMIT_HEALTH)
def health_check():
    """Enhanced health check endpoint with detailed status"""
    try:
        # Get current health status
        health_status = health_monitor.get_status()
        current_time = datetime.utcnow()
        
        # Check if last health check is too old
        if health_monitor.last_check:
            time_since_check = (current_time - health_monitor.last_check).total_seconds()
            if time_since_check > health_monitor.check_interval * 2:
                # Force a new health check if too old
                health_status = health_monitor.check_health()
        
        # Add API-specific health checks
        api_health = {
            'uptime': str(current_time - startup_time),
            'request_count': metrics_collector._request_times.maxlen,
            'recent_errors': len([r for r in metrics_collector._failed_requests 
                                if (current_time - datetime.fromisoformat(r['timestamp'])).total_seconds() < 3600]),
            'rate_limiting': app.config.get('RATELIMIT_ENABLED', False)
        }
        
        # Combine all health information
        response = {
            'timestamp': current_time.isoformat(),
            'status': health_status['status'],
            'components': {
                'api': api_health,
                'system': health_status.get('system_health', {}),
                'models': health_status.get('model_health', {}),
                'security': {
                    'status': health_status.get('security_status'),
                    'model_integrity': health_status.get('model_integrity')
                },
                'storage': health_status.get('storage_health', {})
            }
        }
        
        # Record health check request in audit trail
        audit_trail.record_event(
            event_type='health_request',
            model_type='system',
            model_version='current',
            operation='check',
            status='success',
            details={
                'client_ip': request.remote_addr,
                'status': response['status'],
                'timestamp': response['timestamp']
            }
        )
        
        status_code = 200 if response['status'] == 'healthy' else 503
        return jsonify(response), status_code
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        # Record error in audit trail
        audit_trail.record_event(
            event_type='health_request',
            model_type='system',
            model_version='current',
            operation='check',
            status='error',
            details={
                'client_ip': request.remote_addr,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# Track application startup time
startup_time = datetime.utcnow()

def initialize_server():
    """Initialize all server components with proper error handling"""
    initialization_errors = []
    
    try:
        # Record startup in audit trail
        audit_trail.record_event(
            event_type='startup',
            model_type='system',
            model_version='current',
            operation='start',
            status='in_progress',
            details={
                'timestamp': datetime.utcnow().isoformat(),
                'pid': os.getpid(),
                'python_version': sys.version
            }
        )
        
        # Validate environment
        if not validate_environment():
            raise RuntimeError("Environment validation failed")
            
        # Set up security configurations
        setup_server_security()
        logger.info("Security configuration completed")
        
        # Initialize security components
        if not initialize_security_components():
            raise RuntimeError("Security component initialization failed")
        logger.info("Security components initialized")
        
        # Initialize models
        if not initialize_models():
            raise RuntimeError("Model initialization failed")
        logger.info("Models initialized successfully")
        
        # Start health monitoring
        try:
            health_monitor.start()
            initial_health = health_monitor.check_health()
            if initial_health['status'] != 'healthy':
                initialization_errors.append(f"Initial health check failed: {initial_health['status']}")
        except Exception as e:
            initialization_errors.append(f"Health monitor startup failed: {str(e)}")
        
        # Perform initial security validation
        try:
            security_status = security_validator.run_security_checks()
            if security_status['status'] != 'passed':
                initialization_errors.append(f"Security validation failed: {security_status['issues']}")
        except Exception as e:
            initialization_errors.append(f"Security validation error: {str(e)}")
        
        # Record successful initialization
        audit_trail.record_event(
            event_type='startup',
            model_type='system',
            model_version='current',
            operation='complete',
            status='success' if not initialization_errors else 'warning',
            details={
                'completion_time': datetime.utcnow().isoformat(),
                'initialization_errors': initialization_errors,
                'components_status': {
                    'security': security_status['status'] if 'security_status' in locals() else 'unknown',
                    'health_monitor': 'active' if health_monitor._monitoring else 'inactive',
                    'models': all([rf_model, lstm_model, tokenizer, drug_encoder])
                }
            }
        )
        
        if initialization_errors:
            logger.warning(f"Server initialized with warnings: {initialization_errors}")
        else:
            logger.info("Server initialization completed successfully")
            
        return True
        
    except Exception as e:
        error_msg = f"Critical initialization error: {str(e)}"
        logger.critical(error_msg)
        audit_trail.record_event(
            event_type='startup',
            model_type='system',
            model_version='current',
            operation='error',
            status='failed',
            details={
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        return False

if __name__ == '__main__':
    try:
        # Initialize server components
        if not initialize_server():
            logger.critical("Server initialization failed")
            sys.exit(1)
        
        # Set up signal handlers
        setup_signal_handlers()
        
        # Get and validate server configuration
        try:
            port = int(os.environ['PORT'])
            host = os.environ['HOST']
            max_threads = int(os.environ['MAX_THREADS'])
            connection_limit = int(os.environ['CONNECTION_LIMIT'])
            
            # Additional security settings
            ssl_enabled = os.environ.get('SSL_ENABLED', 'true').lower() == 'true'
            trusted_proxies = os.environ.get('TRUSTED_PROXIES', '127.0.0.1').split(',')
            allowed_hosts = os.environ.get('ALLOWED_HOSTS', host).split(',')
            
            # Validate configuration
            if port < 1024 or port > 65535:
                raise ValueError(f"Invalid port number: {port}")
            if max_threads < 1 or max_threads > 32:
                raise ValueError(f"Invalid thread count: {max_threads}")
            if connection_limit < 1 or connection_limit > 10000:
                raise ValueError(f"Invalid connection limit: {connection_limit}")
                
            # Server configuration
            server_config = {
                # Basic configuration
                'host': host,
                'port': port,
                'threads': max_threads,
                'connection_limit': connection_limit,
                
                # Security settings
                'url_scheme': 'https' if ssl_enabled else 'http',
                'trusted_proxy_headers': ['x-forwarded-for'],
                'trusted_proxy': trusted_proxies,
                'forward_allow_ips': os.environ.get('ALLOWED_IPS', '*'),
                'log_untrusted_proxy_headers': True,
                'clear_untrusted_proxy_headers': True,
                'expose_tracebacks': False,
                
                # Request limits
                'max_request_header_size': 8192,  # 8KB
                'max_request_body_size': MAX_FILE_SIZE,
                'max_request_fields': 100,
                
                # Timeouts
                'channel_timeout': 30,
                'cleanup_interval': 30,
                'shutdown_timeout': 30,
                'graceful_timeout': 15,
                
                # Performance
                'backlog': 2048,
                'connection_high_water': 1000,
                'connection_low_water': 900,
                
                # Identification
                'thread_name_prefix': 'MedisyncML',
                'ident': f"MedisyncML-{os.environ.get('FLASK_ENV', 'production')}"
            }
            
            # SSL/TLS configuration if enabled
            if ssl_enabled:
                import ssl
                server_config.update({
                    'ssl_version': ssl.PROTOCOL_TLS_SERVER,
                    'ciphers': (
                        'ECDHE-ECDSA-AES128-GCM-SHA256:'
                        'ECDHE-RSA-AES128-GCM-SHA256:'
                        'ECDHE-ECDSA-AES256-GCM-SHA384:'
                        'ECDHE-RSA-AES256-GCM-SHA384'
                    ),
                    'ssl_options': (
                        ssl.OP_NO_SSLv2 | 
                        ssl.OP_NO_SSLv3 | 
                        ssl.OP_NO_TLSv1 | 
                        ssl.OP_NO_TLSv1_1
                    )
                })
            
            # Record configuration in audit trail
            audit_trail.record_event(
                event_type='server_config',
                model_type='system',
                model_version='current',
                operation='configure',
                status='success',
                details={
                    'host': host,
                    'port': port,
                    'ssl_enabled': ssl_enabled,
                    'max_threads': max_threads,
                    'connection_limit': connection_limit
                }
            )

            # Initialize server manager
            server_manager = ServerManager(
                app=app,
                config=server_config,
                audit_trail=audit_trail,
                metrics_collector=metrics_collector
            )

            # Update signal handlers to use server manager
            def signal_handler(signum, frame):
                """Handle shutdown signals"""
                logger.info(f"Received signal {signum}")
                try:
                    server_manager.initiate_shutdown()
                    cleanup_prediction_resources(None, None)  # Clean up any remaining resources
                    shutdown_security_components()
                    sys.exit(0)
                except Exception as e:
                    logger.critical(f"Error during shutdown: {str(e)}")
                    sys.exit(1)

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            # Add server status endpoint
            @app.route('/admin/server/status', methods=['GET'])
            @limiter.limit("10 per minute")
            def server_status():
                """Get current server status"""
                # Verify admin authentication
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    return jsonify({'error': 'Unauthorized'}), 401
                    
                token = auth_header.split(' ')[1]
                admin_token = os.environ.get('ADMIN_TOKEN')
                if not admin_token or not secrets.compare_digest(token, admin_token):
                    return jsonify({'error': 'Unauthorized'}), 401

                try:
                    status = server_manager.get_status()
                    return jsonify(status), 200
                except Exception as e:
                    logger.error(f"Error getting server status: {str(e)}")
                    return jsonify({
                        'error': 'Failed to get server status',
                        'details': str(e)
                    }), 500

            return server_manager
            
        except Exception as e:
            logger.critical(f"Server configuration error: {str(e)}")
            audit_trail.record_event(
                event_type='server_config',
                model_type='system',
                model_version='current',
                operation='configure',
                status='failed',
                details={'error': str(e)}
            )
            sys.exit(1)
        
        # Configure server security settings
        server_config = {
            'host': host,
            'port': port,
            'threads': max_threads,
            'url_scheme': 'https',
            'connection_limit': connection_limit,
            'channel_timeout': 30,
            'cleanup_interval': 30,
            'thread_name_prefix': 'MedisyncML',
            'clear_untrusted_proxy_headers': True,
            'log_untrusted_proxy_headers': True,
            'forward_allow_ips': os.environ.get('ALLOWED_IPS', '*'),
            'url_prefix': os.environ.get('URL_PREFIX', ''),
            'trusted_proxy': os.environ.get('TRUSTED_PROXIES', '127.0.0.1').split(','),
            'timeout': 30,
            'max_request_header_size': 8192,  # 8KB max header size
            'max_request_body_size': MAX_FILE_SIZE,
            'expose_tracebacks': False,  # Hide tracebacks in production
            'ident': f"MedisyncML-{os.environ.get('FLASK_ENV', 'production')}"
        }
        
        # Additional security headers
        @app.after_request
        def add_security_headers(response):
            """Add security headers to all responses"""
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
            response.headers['Content-Security-Policy'] = "default-src 'self'"
            response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
            response.headers['Feature-Policy'] = "camera 'none'; microphone 'none'; geolocation 'none'"
            response.headers['Permissions-Policy'] = "camera=(), microphone=(), geolocation=()"
            return response
            
        # Error handlers
        @app.errorhandler(Exception)
        def handle_exception(e):
            """Global exception handler"""
            logger.error(f"Unhandled exception: {str(e)}")
            metrics_collector.record_request(
                duration=0,
                success=False,
                ip=request.remote_addr,
                error=f"Internal error: {str(e)}"
            )
            return jsonify({
                'error': 'Internal server error',
                'request_id': secrets.token_hex(8)
            }), 500
            
        # Log startup configuration (excluding sensitive data)
        safe_config = {k: v for k, v in server_config.items() 
                      if k not in ['trusted_proxy', 'forward_allow_ips']}
        logger.info(f"Starting server with configuration: {json.dumps(safe_config, indent=2)}")
        
        # Start the server
        serve(app, **server_config)
        
        # Additional security settings from environment
        trusted_proxies = os.environ.get('TRUSTED_PROXIES', '127.0.0.1').split(',')
        allowed_hosts = os.environ.get('ALLOWED_HOSTS', host).split(',')
        
        # Validate settings
        if not all(ip.strip() for ip in trusted_proxies):
            raise ValueError("Invalid trusted proxies configuration")
        if not all(host.strip() for host in allowed_hosts):
            raise ValueError("Invalid allowed hosts configuration")
        
        # Configure production server with security settings
        server_config = {
            'host': host,
            'port': port,
            'threads': max_threads,
            'url_scheme': 'https',
            'connection_limit': connection_limit,
            'channel_timeout': 30,
            'cleanup_interval': 30,
            'thread_name_prefix': 'MedisyncML',
            'trusted_proxy_headers': ['x-forwarded-for'],
            'trusted_proxy': trusted_proxies,
            'forward_allow_ips': os.environ.get('ALLOWED_IPS', '*'),
            'log_untrusted_proxy_headers': True,
            'clear_untrusted_proxy_headers': True
        }
        
        logger.info(f"Starting secure server on {host}:{port}")
        logger.info(f"Server configuration: {server_config}")
        serve(app, **server_config)
        
    except Exception as e:
        logger.critical(f"Failed to start server: {str(e)}")
        sys.exit(1)

# Configure rate limits from environment
RATE_LIMIT_DEFAULT = os.environ.get('RATE_LIMIT_DEFAULT', "200 per day")
RATE_LIMIT_HEALTH = os.environ.get('RATE_LIMIT_HEALTH', "60 per minute")

class SecurityMiddleware:
    """Security middleware for request/response processing"""
    
    def __init__(self, app, **kwargs):
        self.app = app
        self.trusted_proxies = kwargs.get('trusted_proxies', {'127.0.0.1'})
        self.request_counter = 0
        self.last_counter_reset = datetime.now()
        self.max_requests_per_minute = kwargs.get('max_requests_per_minute', 60)
        self.blocked_ips = set()
        self.suspicious_patterns = [
            r'\.\./', r'%2e%2e', r'exec\(', r'eval\(',
            r'system\(', r'\\x[0-9a-fA-F]{2}',
            r'(?i)union.*select', r'(?i)insert.*into',
            r'(?i)drop.*table', r'(?i)delete.*from'
        ]
        self.pattern_cache = [re.compile(p) for p in self.suspicious_patterns]
        
    def _check_request_rate(self):
        """Check and enforce request rate limits"""
        now = datetime.now()
        if (now - self.last_counter_reset).total_seconds() >= 60:
            self.request_counter = 0
            self.last_counter_reset = now
        
        self.request_counter += 1
        if self.request_counter > self.max_requests_per_minute:
            raise Exception("Rate limit exceeded")
    
    def _is_valid_ip(self, ip: str) -> bool:
        """Validate IP address and check against blocklist"""
        if ip in self.blocked_ips:
            logger.warning(f"Blocked IP attempt: {ip}")
            return False
            
        if not ip or not isinstance(ip, str):
            return False
            
        # Basic IP format validation
        parts = ip.split('.')
        if len(parts) != 4:
            return False
            
        try:
            return all(0 <= int(p) <= 255 for p in parts)
        except ValueError:
            return False
    
    def _is_suspicious_path(self, path: str) -> bool:
        """Check path for suspicious patterns"""
        if not path:
            return False
            
        # Check for directory traversal and common attack patterns
        for pattern in self.pattern_cache:
            if pattern.search(path):
                logger.warning(f"Suspicious pattern detected in path: {path}")
                return True
                
        return False
    
    def _is_sensitive_header(self, header: str) -> bool:
        """Check if header should be removed for security"""
        sensitive_headers = {
            'server', 'x-powered-by', 'x-aspnet-version',
            'x-runtime', 'x-version', 'x-instance'
        }
        return header.lower() in sensitive_headers
    
    def _reject_request(self, reason: str, status: str, start_response) -> List[bytes]:
        """Handle rejected requests with proper logging"""
        logger.warning(f"Request rejected: {reason}")
        headers = [('Content-Type', 'application/json')]
        headers.extend([(k, v) for k, v in SECURE_HEADERS.items()])
        start_response(status, headers)
        return [json.dumps({'error': reason}).encode()]
    
    def __call__(self, environ, start_response):
        try:
            # Check request rate
            self._check_request_rate()
            
            # Validate client IP
            remote_addr = environ.get('REMOTE_ADDR')
            if not self._is_valid_ip(remote_addr):
                return self._reject_request('Invalid IP address', '403 Forbidden', start_response)
            
            # Check for suspicious patterns
            path_info = environ.get('PATH_INFO', '')
            if self._is_suspicious_path(path_info):
                # Add IP to blocklist after suspicious activity
                self.blocked_ips.add(remote_addr)
                return self._reject_request('Suspicious request pattern', '400 Bad Request', start_response)
            
            # Log request info
            logger.info(f"Processing request from {remote_addr} to {path_info}")
            
            def security_start_response(status, headers, exc_info=None):
                # Add security headers
                secure_headers = [(k, v) for k, v in SECURE_HEADERS.items()]
                headers.extend(secure_headers)
                
                # Remove sensitive headers
                headers = [(k, v) for k, v in headers if not self._is_sensitive_header(k)]
                
                # Log response info
                logger.debug(f"Sending response with status: {status}")
                
                return start_response(status, headers, exc_info)
            
            # Process the request through the WSGI application
            try:
                return self.app(environ, security_start_response)
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                return self._reject_request('Internal server error', '500 Internal Server Error', start_response)
                
        except Exception as e:
            logger.error(f"Security middleware error: {str(e)}")
            return self._reject_request('Security check failed', '403 Forbidden', start_response)

def allowed_file(filename: str) -> bool:
    """Check if a file is allowed based on its extension and basic security checks.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        bool: True if file is allowed, False otherwise
    """
    if not filename or not isinstance(filename, str):
        return False
        
    # Check file extension
    allowed = '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    if not allowed:
        logger.warning(f"Rejected file with unauthorized extension: {filename}")
        return False
        
    # Additional security checks
    # Check for null bytes
    if '\x00' in filename:
        logger.warning(f"Rejected file with null bytes in name: {filename}")
        return False
        
    # Check for suspicious patterns
    suspicious_patterns = [
        r'\.\.', r'~', r'%00', r'\.\w+\.',  # Double extensions
        r'[<>:"|?*]',  # Invalid filename chars
        r'\\x[0-9a-fA-F]{2}'  # Hex encoded chars
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, filename):
            logger.warning(f"Rejected file with suspicious pattern: {filename}")
            return False
            
    return True

def secure_file_validation(file) -> bool:
    """Perform comprehensive security validation on uploaded file.
    
    Args:
        file: File object to validate
        
    Returns:
        bool: True if file is safe, False otherwise
    """
    try:
        if not file or not file.filename:
            return False
            
        # Check file extension
        if not allowed_file(file.filename):
            return False
            
        # Check file size (max 10MB)
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)  # Reset position
        
        if size > 10 * 1024 * 1024:  # 10MB
            logger.warning(f"Rejected file exceeding size limit: {file.filename}")
            return False
            
        # Check file content (first 512 bytes)
        content_start = file.read(512)
        file.seek(0)  # Reset position
        
        # Check for binary content or suspicious patterns
        if bool(content_start.translate(None, bytearray(range(32, 127)) + b'\n\r\t')):
            logger.warning(f"Rejected file with suspicious binary content: {file.filename}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error during file validation: {str(e)}")
        return False
        
        try:
            return self.app(environ, security_start_response)
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            return self._reject_request('Internal server error', '500 Internal Server Error', start_response)
    
    def _check_request_rate(self):
        """Basic rate limiting"""
        now = datetime.now()
        if (now - self.last_counter_reset).seconds > 60:
            self.request_counter = 0
            self.last_counter_reset = now
        self.request_counter += 1
        if self.request_counter > 1000:  # Global rate limit
            raise Exception("Global rate limit exceeded")
    
    def _is_valid_ip(self, ip):
        """Validate IP address"""
        if not ip:
            return False
        # Add your IP validation logic here
        return True
    
    def _is_suspicious_path(self, path):
        """Check for suspicious request patterns"""
        suspicious_patterns = [
            r'\.\./',           # Directory traversal
            r'%00',             # Null byte
            r';\s*(',           # Command injection
            r'(?i)script',      # XSS attempts
            r'(?i)admin',       # Admin access attempts
            r'(?i)login',       # Login attempts
            r'(?i)select.*from' # SQL injection
        ]
        return any(re.search(pattern, path) for pattern in suspicious_patterns)
    
    def _is_sensitive_header(self, header):
        """Check for sensitive headers that should be removed"""
        sensitive_headers = {
            'server',
            'x-powered-by',
            'x-aspnet-version',
            'x-runtime',
            'x-version'
        }
        return header.lower() in sensitive_headers
    
    def _reject_request(self, message, status, start_response):
        """Return a rejection response"""
        response = json.dumps({'error': message}).encode('utf-8')
        headers = [
            ('Content-Type', 'application/json'),
            ('Content-Length', str(len(response)))
        ]
        headers.extend([(k, v) for k, v in SECURE_HEADERS.items()])
        start_response(status, headers)
        return [response]

class RequestValidationMiddleware:
    """Middleware for validating requests"""
    
    def __init__(self, app):
        self.app = app
        self.max_content_length = 1 * 1024 * 1024  # 1MB
    
    def __call__(self, environ, start_response):
        try:
            # Validate request size
            content_length = environ.get('CONTENT_LENGTH')
            if content_length is not None:
                content_length = int(content_length)
                if content_length > self.max_content_length:
                    return self._reject_request('Request too large', '413 Request Entity Too Large', start_response)
            
            # Validate content type
            if environ['REQUEST_METHOD'] == 'POST':
                content_type = environ.get('CONTENT_TYPE', '')
                if not content_type.startswith('application/x-www-form-urlencoded'):
                    return self._reject_request('Invalid content type', '415 Unsupported Media Type', start_response)
            
            return self.app(environ, start_response)
            
        except Exception as e:
            logger.error(f"Request validation error: {e}")
            return self._reject_request('Bad request', '400 Bad Request', start_response)
    
    def _reject_request(self, message, status, start_response):
        response = json.dumps({'error': message}).encode('utf-8')
        headers = [
            ('Content-Type', 'application/json'),
            ('Content-Length', str(len(response)))
        ]
        start_response(status, headers)
        return [response]

def generate_secure_key(length=32):
    """Generate a secure random key"""
    return secrets.token_hex(length)

def create_app():
    """Create Flask app with security configurations"""
    app = Flask(__name__)
    
    # Security configuration
    app.config.update(
        # Basic Flask security
        ENV='production',
        DEBUG=False,
        TESTING=False,
        SECRET_KEY=os.environ.get('SECRET_KEY', generate_secure_key()),
        
        # Session security
        SESSION_COOKIE_NAME='medisync_session',
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Strict',
        PERMANENT_SESSION_LIFETIME=1800,  # 30 minutes
        
        # Request security
        MAX_CONTENT_LENGTH=1 * 1024 * 1024,  # 1MB
        PREFERRED_URL_SCHEME='https',
        
        # CSRF protection
        WTF_CSRF_ENABLED=True,
        WTF_CSRF_SECRET_KEY=os.environ.get('CSRF_SECRET_KEY', generate_secure_key()),
        WTF_CSRF_TIME_LIMIT=3600,  # 1 hour
        
        # Security headers (in addition to middleware)
        SEND_FILE_MAX_AGE_DEFAULT=31556926,  # 1 year
        
        # Logging
        LOG_LEVEL=os.environ.get('LOG_LEVEL', 'INFO'),
        
        # Rate limiting
        RATELIMIT_STORAGE_URL="memory://",
        RATELIMIT_STRATEGY="fixed-window",
        RATELIMIT_HEADERS_ENABLED=True,
        
        # Custom security settings
        TRUSTED_PROXIES=os.environ.get('TRUSTED_PROXIES', '127.0.0.1').split(','),
        ALLOWED_HOSTS=os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(','),
        MIN_PASSWORD_LENGTH=12,
        MAX_LOGIN_ATTEMPTS=5,
        LOGIN_ATTEMPT_TIMEOUT=300,  # 5 minutes
    )
    
    # Apply security middleware
    app.wsgi_app = SecurityMiddleware(app.wsgi_app)
    app.wsgi_app = RequestValidationMiddleware(app.wsgi_app)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
    
    # Initialize CSRF protection
    from flask_wtf.csrf import CSRFProtect
    csrf = CSRFProtect(app)
    
    # Custom error handler for CSRF errors
    @app.errorhandler(CSRFProtect.error_handler)
    def handle_csrf_error(e):
        logger.warning(f"CSRF validation failed: {e}")
        return render_template('error.html',
                             error_title='Security Error',
                             error_message='Security token validation failed. Please try again.',
                             error_code=400), 400
    
    return app

# Create the Flask app with security configurations
app = create_app()

# Configure logging
logging.basicConfig(
    level=os.environ.get('LOG_LEVEL', 'INFO'),
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Security helper functions
def sanitize_input(data: str) -> str:
    """Sanitize user input to prevent XSS and injection attacks"""
    if not isinstance(data, str):
        return str(data)
    return (data.replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#x27;')
                .replace('(', '&#40;')
                .replace(')', '&#41;'))

def validate_input_length(data: str, max_length: int = 256) -> bool:
    """Validate input length to prevent buffer overflow attacks"""
    return len(str(data)) <= max_length

def validate_numeric_range(value: int, min_val: int, max_val: int) -> bool:
    """Validate numeric input ranges"""
    try:
        num = int(value)
        return min_val <= num <= max_val
    except (ValueError, TypeError):
        return False

# Request validation middleware
class RequestValidator:
    def __init__(self, app):
        self.app = app
        
    def __call__(self, environ, start_response):
        request = Request(environ)
        
        # Validate request size
        content_length = request.content_length or 0
        if content_length > app.config['MAX_CONTENT_LENGTH']:
            response = Response('Request too large', status=413)
            return response(environ, start_response)
        
        # Validate content type for POST requests
        if request.method == 'POST':
            content_type = request.content_type or ''
            if not content_type.startswith('application/x-www-form-urlencoded'):
                response = Response('Invalid content type', status=415)
                return response(environ, start_response)
        
        # Continue with request
        return self.app(environ, start_response)

# Initialize Flask app with production settings
app = Flask(__name__)
app.config.update(
    ENV='production',  # Force production environment
    DEBUG=False,       # Ensure debug mode is disabled
    TESTING=False,     # Disable testing mode
    PROPAGATE_EXCEPTIONS=False,  # Don't propagate exceptions in production
    PRESERVE_CONTEXT_ON_EXCEPTION=False,  # Don't preserve context on exceptions
    MAX_CONTENT_LENGTH=1 * 1024 * 1024,  # Limit request size to 1MB
    # Additional security settings
    PREFERRED_URL_SCHEME='https',
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Strict',
    PERMANENT_SESSION_LIFETIME=1800,  # 30 minutes
    # Input validation limits
    MAX_INPUT_LENGTH=256,
    MIN_AGE=0,
    MAX_AGE=120
)

# Apply request validator middleware
app.wsgi_app = RequestValidator(app.wsgi_app)

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[RATE_LIMIT_DEFAULT],
    storage_uri="memory://"
)

# Initialize variables
sym_des = precautions = workout = description = medications = diets = None
rf = lstm = drug_encoder = tokenizer = None

def validate_request_size(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check request size (limit to 1MB)
        if request.content_length and request.content_length > 1 * 1024 * 1024:
            app.logger.warning(f"Request too large from {request.remote_addr}")
            return render_template('error.html',
                                error_title='Request Too Large',
                                error_message='The request exceeds the maximum allowed size.',
                                error_code=413,
                                timestamp=datetime.now().isoformat()), 413
        return f(*args, **kwargs)
    return decorated_function

def load_data():
    """Load all required datasets and models"""
    global sym_des, precautions, workout, description, medications, diets
    global rf, lstm, drug_encoder, tokenizer
    
    try:
        # Load database files
        sym_des = pd.read_csv('datasets/symtoms_df.csv')
        precautions = pd.read_csv('datasets/precautions_df.csv')
        workout = pd.read_csv("datasets/workout_df.csv")
        description = pd.read_csv("datasets/description.csv")
        medications = pd.read_csv("datasets/medications.csv")
        diets = pd.read_csv("datasets/diets.csv")
        
        # Load ML models securely
        rf = load_model_safely('models/rf.json')
        lstm = tf.keras.models.load_model("models/lstm_drug_model.keras")
        drug_encoder = load_drug_encoder_safely('models/drug_encoder.json')
        tokenizer = load_tokenizer_safely('models/tokenizer.json')
        
        return True
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        return False

def validate_models():
    """Validate that the ML models are working correctly with test data"""
    try:
        # Test Random Forest model with a sample input
        test_symptoms = [0] * len(symptoms_dict)
        test_symptoms[symptoms_dict['fever']] = 1 if 'fever' in symptoms_dict else 0
        test_rf_result = predict_val(test_symptoms)
        if not isinstance(test_rf_result, str):
            raise ValueError("Random Forest model validation failed")

        # Test LSTM model with sample input
        test_disease = "Common Cold"
        test_age = 30
        test_gender = 0  # male
        test_lstm_result = predict_from_lstm(test_disease, test_age, test_gender)
        if test_lstm_result is None:
            raise ValueError("LSTM model validation failed")

        logger.info("ML models validated successfully")
        return True
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        return False

# Load data and validate models on startup
load_success = load_data()
if not load_success:
    logger.error("Failed to initialize application data and models")
else:
    validation_success = validate_models()
    if not validation_success:
        logger.error("Model validation failed - application may not function correctly")
        load_success = False

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    logger.error(f'Page not found: {request.url}')
    return render_template('error.html',
                         error_title='Page Not Found',
                         error_message='The requested page could not be found.',
                         error_code=404,
                         timestamp=datetime.now().isoformat()), 404

@app.errorhandler(429)
def ratelimit_handler(error):
    logger.warning(f'Rate limit exceeded for {request.remote_addr}')
    return render_template('error.html',
                         error_title='Too Many Requests',
                         error_message='Please try again later.',
                         error_code=429,
                         timestamp=datetime.now().isoformat()), 429

@app.errorhandler(500)
def internal_error(error):
    logger.error(f'Server Error: {error}')
    return render_template('error.html',
                         error_title='Internal Server Error',
                         error_message='An unexpected error has occurred.',
                         error_code=500,
                         timestamp=datetime.now().isoformat()), 500

@app.errorhandler(403)
def forbidden_error(error):
    logger.error(f'Forbidden access: {request.url}')
    return render_template('error.html',
                         error_title='Access Forbidden',
                         error_message='You do not have permission to access this resource.',
                         error_code=403,
                         timestamp=datetime.now().isoformat()), 403

class RateLimitExceeded(Exception):
    pass

@app.errorhandler(RateLimitExceeded)
def handle_ratelimit_exceeded(error):
    return ratelimit_handler(error)

# Security headers
@app.after_request
def add_security_headers(response):
    """Add security headers to each response"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response

@app.route('/', methods=['GET', 'POST'])
@limiter.limit(RATE_LIMIT_DEFAULT)
@validate_request_size
def home():
    """Handle home route with enhanced security and input validation"""
    # Check if required data is loaded
    if not load_success:
        return render_template('error.html', 
                             error_title='Service Unavailable',
                             error_message='Service temporarily unavailable. Please try again later.',
                             error_code=503,
                             timestamp=datetime.now().isoformat()), 503

    if request.method == 'POST':
        try:
            # Get and validate symptoms
            symptoms = request.form.getlist('symptoms')
            
            # Input validation
            if not symptoms:
                return render_template('index.html', 
                                     error="Please select at least one symptom",
                                     error_code=400)

            # Validate length for each symptom
            if not all(validate_input_length(symptom, app.config['MAX_INPUT_LENGTH']) 
                      for symptom in symptoms):
                return render_template('index.html',
                                     error="Symptom name too long",
                                     error_code=400)

            # Sanitize and validate all symptoms
            sanitized_symptoms = []
            for symptom in symptoms:
                # Sanitize the symptom name
                clean_symptom = sanitize_input(symptom)
                
                # Validate symptom format (only allow letters, underscores, and spaces)
                if not re.match(r'^[A-Za-z\s_]+$', clean_symptom):
                    return render_template('index.html',
                                         error="Invalid symptom format detected",
                                         error_code=400)
                
                # Validate against known symptoms
                if clean_symptom not in symptoms_dict:
                    logger.warning(f"Invalid symptom attempted: {clean_symptom}")
                    return render_template('index.html',
                                         error="Invalid symptom selected",
                                         error_code=400)
                
                sanitized_symptoms.append(clean_symptom)

            # Convert symptoms to model input format
            try:
                user_symptoms = [0] * len(symptoms_dict)
                for symptom in sanitized_symptoms:
                    user_symptoms[symptoms_dict[symptom]] = 1
            except Exception as e:
                logger.error(f"Error processing symptoms: {str(e)}")
                return render_template('index.html',
                                     error="Error processing symptoms",
                                     error_code=500)

            # Get disease prediction with validation
            try:
                predicted_disease = predict_val(user_symptoms)
                if not predicted_disease or not isinstance(predicted_disease, str):
                    logger.error("Invalid prediction result type")
                    return render_template('index.html',
                                         error="Unable to generate prediction",
                                         error_code=500)
                
                # Sanitize the prediction
                predicted_disease = sanitize_input(predicted_disease)
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                return render_template('index.html',
                                     error="Error generating prediction",
                                     error_code=500)

            # Get disease details with validation
            try:
                descr, pre, med, dt, wrk = helper(predicted_disease)
                if all(v is None for v in [descr, pre, med, dt, wrk]):
                    logger.error(f"No data found for disease: {predicted_disease}")
                    return render_template('index.html',
                                         error="Error retrieving disease information",
                                         error_code=500)

                # Convert and sanitize all data for template
                safe_data = {
                    'predicted_disease': sanitize_input(predicted_disease),
                    'dis_desc': sanitize_input(descr),
                    'dis_pre': [sanitize_input(str(p)) for p in (pre[0] if pre and len(pre) > 0 else [])],
                    'dis_med': [sanitize_input(str(m)) for m in (med.tolist() if isinstance(med, pd.Series) else med)],
                    'dis_dt': [sanitize_input(str(d)) for d in (dt.tolist() if isinstance(dt, pd.Series) else dt)],
                    'dis_wrk': [sanitize_input(str(w)) for w in (wrk.tolist() if isinstance(wrk, pd.Series) else wrk)]
                }

                # Validate all data lengths
                if any(not validate_input_length(str(v), app.config['MAX_INPUT_LENGTH'] * 2) 
                      for v in safe_data.values()):
                    logger.error("Output data exceeds maximum allowed length")
                    return render_template('index.html',
                                         error="Error processing results",
                                         error_code=500)

                return render_template('index.html', **safe_data)

            except Exception as e:
                logger.error(f"Error processing disease data: {str(e)}")
                return render_template('index.html',
                                     error="Error processing results",
                                     error_code=500)

        except Exception as e:
            logger.error(f"Unexpected error in home route: {str(e)}")
            return render_template('index.html',
                                 error="An unexpected error occurred",
                                 error_code=500)
    
    # GET request
    return render_template('index.html')

@app.route('/diagnosis', methods=['POST', 'GET'])
@limiter.limit(RATE_LIMIT_DEFAULT)
@validate_request_size
def diagnosis():
    """Handle diagnosis requests with enhanced security validation"""
    # Check if required data is loaded
    if not load_success:
        return render_template('diagnosis.html', 
                             error="Service temporarily unavailable. Please try again later.",
                             error_code=503)

    if request.method == 'POST':
        try:
            # Get and sanitize input data
            diag = sanitize_input(request.form.get('diagnosis', ''))
            gender = sanitize_input(request.form.get('Gender', ''))
            age = request.form.get('Age', '')

            # Validate input presence and length
            if not all([diag, gender, age]):
                return render_template('diagnosis.html', 
                                     error="Please provide all required information",
                                     error_code=400)

            if not all(validate_input_length(x, app.config['MAX_INPUT_LENGTH']) 
                      for x in [diag, gender, str(age)]):
                return render_template('diagnosis.html',
                                     error="Input exceeds maximum allowed length",
                                     error_code=400)

            # Validate and convert gender
            gender_lower = gender.lower()
            if gender_lower == 'male':
                gender_val = 0
            elif gender_lower == 'female':
                gender_val = 1
            else:
                return render_template('diagnosis.html',
                                     error="Invalid gender selected. Please choose 'male' or 'female'",
                                     error_code=400)

            # Validate age
            try:
                age_val = int(age)
                if not validate_numeric_range(age_val, 
                                           app.config['MIN_AGE'],
                                           app.config['MAX_AGE']):
                    raise ValueError(
                        f"Age must be between {app.config['MIN_AGE']} and {app.config['MAX_AGE']}"
                    )
            except ValueError as e:
                return render_template('diagnosis.html',
                                     error=f"Invalid age provided: {str(e)}",
                                     error_code=400)

            # Additional input validation for diagnosis
            if not re.match(r'^[A-Za-z\s\-]+$', diag):
                return render_template('diagnosis.html',
                                     error="Invalid diagnosis format",
                                     error_code=400)

            try:
                # Predict medication using LSTM with validated inputs
                predicted_medication = predict_from_lstm(diag, age_val, gender_val)
                if predicted_medication is None:
                    logger.error(f"Null prediction for diagnosis: {diag}")
                    return render_template('diagnosis.html',
                                         error="Unable to generate prediction. Please try again.",
                                         error_code=500)

                # Sanitize output before rendering
                safe_medication = sanitize_input(predicted_medication)
                return render_template('diagnosis.html',
                                     medication=safe_medication)

            except Exception as e:
                logger.error(f"Prediction error for diagnosis {diag}: {str(e)}")
                return render_template('diagnosis.html',
                                     error="An error occurred while processing your request",
                                     error_code=500)

        except Exception as e:
            logger.error(f"Unexpected error in diagnosis route: {str(e)}")
            return render_template('diagnosis.html',
                                 error="An unexpected error occurred",
                                 error_code=500)

    # GET request
    return render_template('diagnosis.html')

def check_system_resources() -> Dict[str, Union[bool, float, str]]:
    """Check system resource usage"""
    try:
        import psutil
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_ok = memory.percent < app.config.get('MAX_MEMORY_PERCENT', 90)
        
        # Get disk usage
        disk = psutil.disk_usage('.')
        disk_ok = disk.percent < 90
        
        return {
            'status': memory_ok and disk_ok,
            'memory_used_percent': memory.percent,
            'disk_used_percent': disk.percent,
            'memory_available_gb': round(memory.available / (1024**3), 2),
            'disk_free_gb': round(disk.free / (1024**3), 2)
        }
    except Exception as e:
        logger.error(f"Error checking system resources: {e}")
        return {
            'status': False,
            'error': 'Resource check failed'
        }

def check_model_health() -> Dict[str, Union[bool, List[str]]]:
    """Verify ML models are functioning"""
    issues = []
    try:
        # Test Random Forest model
        test_symptoms = [0] * len(symptoms_dict)
        test_symptoms[0] = 1  # Set first symptom
        rf_result = predict_val(test_symptoms)
        if not isinstance(rf_result, str):
            issues.append("RF model validation failed")

        # Test LSTM model
        lstm_result = predict_from_lstm("Common Cold", 30, 0)
        if lstm_result is None:
            issues.append("LSTM model validation failed")

        return {
            'status': len(issues) == 0,
            'issues': issues
        }
    except Exception as e:
        logger.error(f"Error checking model health: {e}")
        return {
            'status': False,
            'issues': [str(e)]
        }

@app.route('/health')
@limiter.limit(RATE_LIMIT_HEALTH)
def health_check():
    """Enhanced health check endpoint with comprehensive system status"""
    try:
        # Basic data load check
        data_status = {
            'data_loaded': all([sym_des is not None, 
                              precautions is not None,
                              workout is not None,
                              description is not None,
                              medications is not None,
                              diets is not None]),
            'models_loaded': all([rf is not None,
                                lstm is not None,
                                drug_encoder is not None,
                                tokenizer is not None])
        }

        # System resources check
        resources = check_system_resources()
        
        # Model health check
        model_health = check_model_health()

        # Determine overall status
        status = 'healthy'
        if not all([data_status['data_loaded'], 
                   data_status['models_loaded'],
                   resources.get('status', False),
                   model_health.get('status', False)]):
            status = 'degraded'
        
        health_status = {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'details': {
                'data_status': data_status,
                'system_resources': resources,
                'model_health': model_health,
                'rate_limits': {
                    'default': RATE_LIMIT_DEFAULT,
                    'health': RATE_LIMIT_HEALTH
                },
                'environment': os.environ.get('FLASK_ENV', 'production'),
                'debug_mode': app.debug
            }
        }
        
        # Set response headers
        response = jsonify(health_status)
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        
        # Determine status code
        status_code = 200 if status == 'healthy' else 503
        
        # Log health check result
        if status != 'healthy':
            logger.warning(f"Health check returned status '{status}': {health_status}")
        
        return response, status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': 'Health check failed'
        }), 500

def initialize_system():
    """Perform system initialization and checks"""
    try:
        # Create required directories
        for directory in ['logs', 'logs/archive']:
            os.makedirs(directory, exist_ok=True)

        # Configure file logging
        file_handler = logging.FileHandler('logs/medisync.log')
        file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
        ))
        logger.addHandler(file_handler)

        # Verify environment
        if os.environ.get('FLASK_ENV') != 'production':
            logger.warning("Non-production environment detected - forcing production mode")
            os.environ['FLASK_ENV'] = 'production'

        # Verify security settings
        security_checks = {
            'debug_mode': not app.debug,
            'testing_mode': not app.testing,
            'secure_cookies': app.config.get('SESSION_COOKIE_SECURE', False),
            'httponly_cookies': app.config.get('SESSION_COOKIE_HTTPONLY', False),
            'rate_limiting': bool(RATE_LIMIT_DEFAULT),
        }

        failed_checks = [check for check, passed in security_checks.items() if not passed]
        if failed_checks:
            raise ValueError(f"Failed security checks: {', '.join(failed_checks)}")

        # Verify required files exist
        required_files = [
            'requirements.txt',
            'monitoring_config.json',
            '.env',
        ]
        missing_files = [f for f in required_files if not os.path.isfile(f)]
        if missing_files:
            raise ValueError(f"Missing required files: {', '.join(missing_files)}")

        return True

    except Exception as e:
        logger.critical(f"System initialization failed: {e}")
        return False

def cleanup():
    """Perform cleanup operations before shutdown"""
    try:
        logger.info("Performing cleanup operations...")
        
        # Close any open file handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        # Archive current log file if it exists
        log_file = 'logs/medisync.log'
        if os.path.exists(log_file):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_file = f'logs/archive/medisync_{timestamp}.log'
            os.rename(log_file, archive_file)

        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    signals = {
        signal.SIGTERM: "SIGTERM",
        signal.SIGINT: "SIGINT"
    }
    signal_name = signals.get(signum, f"Signal {signum}")
    logger.info(f"Received {signal_name} - initiating graceful shutdown")
    cleanup()
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Get configuration from environment variables
    HOST = os.environ.get('HOST', '127.0.0.1')  # Default to localhost for security
    PORT = int(os.environ.get('PORT', 8080))
    
    # Initialize system
    if not initialize_system():
        logger.critical("System initialization failed - unable to start")
        sys.exit(1)

    # Verify critical security settings
    if app.debug or app.testing:
        logger.critical("Debug/Testing mode detected in production - shutting down")
        sys.exit(1)
    
    # Log startup information
    logger.info(f"Starting Medisync ML service on {HOST}:{PORT}")
    logger.info("Running in production mode with security measures enabled")
    logger.info(f"Rate limiting: {RATE_LIMIT_DEFAULT}")
    
    try:
        # Start production server with Waitress
        serve(app, 
              host=HOST, 
              port=PORT, 
              threads=4, 
              url_scheme='https', 
              channel_timeout=30,
              ident='Medisync ML (Production)',
              cleanup_interval=30,
              connection_limit=1024,
              max_request_header_size=32768,
              max_request_body_size=1 * 1024 * 1024,  # 1MB
              retry_startup=False)
    except Exception as e:
        logger.critical(f"Failed to start server: {e}")
        cleanup()
        sys.exit(1)