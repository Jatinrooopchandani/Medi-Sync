"""Secure model reloading and validation utilities"""
import logging
import threading
from typing import Dict, Any, Optional
from pathlib import Path
import time
from datetime import datetime

from .secure_loader import (
    load_model_safely,
    load_keras_model_safely,
    load_tokenizer_safely,
    load_drug_encoder_safely,
    SecurityError
)
from .model_validator import (
    validate_model_compatibility,
    ModelValidationError
)

logger = logging.getLogger(__name__)

class ModelReloader:
    """Thread-safe model reloader with validation"""
    
    def __init__(self, model_paths: Dict[str, Path], manifest_manager: Any):
        """Initialize model reloader.
        
        Args:
            model_paths: Dictionary of model paths
            manifest_manager: Instance of ModelManifestManager
        """
        self._lock = threading.Lock()
        self.model_paths = model_paths
        self.manifest_manager = manifest_manager
        self.models = {}
        self.last_reload = None
        self._reload_status = {'success': False, 'message': 'Not initialized'}
    
    def _load_single_model(self, model_type: str, model_path: Path) -> Any:
        """Safely load a single model with validation.
        
        Args:
            model_type: Type of model to load ('rf', 'lstm', 'tokenizer', 'encoder')
            model_path: Path to model file
            
        Returns:
            Loaded model instance
            
        Raises:
            SecurityError: If model loading or validation fails
        """
        if not model_path.exists():
            raise SecurityError(f"Model file not found: {model_path}")
            
        # Verify model type and path match manifest
        manifest = self.manifest_manager.load_manifest()
        model_info = manifest.get(model_path.name, {})
        if model_info.get('model_type') != model_type:
            raise SecurityError(f"Model type mismatch for {model_path.name}")
            
        # Load model based on type
        try:
            if model_type == 'rf':
                model = load_model_safely(str(model_path))
            elif model_type == 'lstm':
                model = load_keras_model_safely(str(model_path))
            elif model_type == 'tokenizer':
                model = load_tokenizer_safely(str(model_path))
            elif model_type == 'encoder':
                model = load_drug_encoder_safely(str(model_path))
            else:
                raise SecurityError(f"Invalid model type: {model_type}")
                
            # Store version information
            model.version = model_info.get('version', '1.0.0')
            model.last_verified = datetime.utcnow().isoformat()
            
            return model
            
        except Exception as e:
            raise SecurityError(f"Failed to load {model_type} model: {str(e)}")
    
    def reload_models(self, validate: bool = True) -> Dict[str, Any]:
        """Safely reload all models with validation.
        
        Args:
            validate: Whether to perform model validation
            
        Returns:
            Dict containing reload status and messages
        """
        with self._lock:
            start_time = time.time()
            status = {
                'success': False,
                'timestamp': datetime.utcnow().isoformat(),
                'duration': 0,
                'messages': [],
                'loaded_models': []
            }
            
            try:
                # Create temporary model dictionary
                temp_models = {}
                
                # Load each model
                for model_type, model_path in self.model_paths.items():
                    try:
                        temp_models[model_type] = self._load_single_model(model_type, model_path)
                        status['messages'].append(f"Successfully loaded {model_type} model")
                        status['loaded_models'].append(model_type)
                    except Exception as e:
                        status['messages'].append(f"Failed to load {model_type} model: {str(e)}")
                        raise SecurityError(f"Model loading failed: {str(e)}")
                
                # Validate models if requested
                if validate:
                    # Verify RF and LSTM model compatibility
                    if 'rf' in temp_models and 'lstm' in temp_models:
                        validate_model_compatibility(
                            temp_models['rf'].version,
                            temp_models['lstm'].version
                        )
                        status['messages'].append("Model compatibility validated")
                
                # Update models atomically
                self.models = temp_models
                self.last_reload = datetime.utcnow()
                
                status['success'] = True
                status['duration'] = f"{time.time() - start_time:.2f}s"
                self._reload_status = status
                
                logger.info(f"Models reloaded successfully in {status['duration']}")
                return status
                
            except Exception as e:
                error_msg = f"Model reload failed: {str(e)}"
                logger.error(error_msg)
                status['messages'].append(error_msg)
                status['duration'] = f"{time.time() - start_time:.2f}s"
                self._reload_status = status
                raise
    
    def get_model(self, model_type: str) -> Optional[Any]:
        """Safely get a loaded model.
        
        Args:
            model_type: Type of model to get
            
        Returns:
            Model instance or None if not loaded
        """
        with self._lock:
            return self.models.get(model_type)
    
    def get_reload_status(self) -> Dict[str, Any]:
        """Get the status of the last reload attempt.
        
        Returns:
            Dict containing reload status information
        """
        with self._lock:
            return self._reload_status.copy()
    
    def validate_all_models(self) -> Dict[str, Any]:
        """Validate all currently loaded models.
        
        Returns:
            Dict containing validation results
        """
        with self._lock:
            results = {
                'success': False,
                'timestamp': datetime.utcnow().isoformat(),
                'validations': {}
            }
            
            try:
                # Check all models are loaded
                for model_type in self.model_paths.keys():
                    if model_type not in self.models:
                        raise SecurityError(f"Model not loaded: {model_type}")
                
                # Validate model compatibility
                if 'rf' in self.models and 'lstm' in self.models:
                    validate_model_compatibility(
                        self.models['rf'].version,
                        self.models['lstm'].version
                    )
                    results['validations']['compatibility'] = 'passed'
                
                # Check model versions against manifest
                manifest = self.manifest_manager.load_manifest()
                for model_type, model in self.models.items():
                    model_info = manifest.get(self.model_paths[model_type].name, {})
                    if model.version != model_info.get('version'):
                        raise SecurityError(
                            f"Version mismatch for {model_type}: "
                            f"loaded={model.version}, manifest={model_info.get('version')}"
                        )
                    results['validations'][model_type] = {
                        'status': 'valid',
                        'version': model.version,
                        'last_verified': getattr(model, 'last_verified', 'unknown')
                    }
                
                results['success'] = True
                return results
                
            except Exception as e:
                logger.error(f"Model validation failed: {str(e)}")
                results['error'] = str(e)
                return results