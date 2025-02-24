"""Secure model manifest management utilities"""
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, List
import threading
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class ManifestError(Exception):
    """Raised when manifest operations fail"""
    pass

class ModelManifestManager:
    """Thread-safe manager for model manifest updates and verification"""
    
    def __init__(self, manifest_path: Optional[str] = None, check_interval: int = 3600):
        """Initialize the manifest manager.
        
        Args:
            manifest_path: Optional path to manifest file. If None, uses default location
            check_interval: Time between integrity checks in seconds (default: 1 hour)
        """
        self._lock = threading.Lock()
        self.manifest_path = Path(manifest_path) if manifest_path else Path(__file__).parent.parent / "models" / "manifest.json"
        self.check_interval = check_interval
        self.last_check = 0
        self._monitoring = False
        self._monitor_thread = None
    
    def compute_model_hash(self, model_path: str) -> str:
        """Securely compute SHA-256 hash of model file.
        
        Args:
            model_path: Path to model file
            
        Returns:
            str: Hex digest of SHA-256 hash
            
        Raises:
            ManifestError: If hash computation fails
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(model_path, 'rb') as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            raise ManifestError(f"Failed to compute hash for {model_path}: {str(e)}")
    
    def load_manifest(self) -> Dict:
        """Load and validate the current manifest.
        
        Returns:
            Dict containing manifest data
            
        Raises:
            ManifestError: If manifest loading or validation fails
        """
        try:
            if not self.manifest_path.exists():
                raise ManifestError("Manifest file not found")
                
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
                
            # Validate manifest structure
            required_keys = {'version', 'expected_hash', 'model_type'}
            for model, info in manifest.items():
                missing_keys = required_keys - set(info.keys())
                if missing_keys:
                    raise ManifestError(f"Invalid manifest structure for {model}: missing {missing_keys}")
                    
            return manifest
        except Exception as e:
            raise ManifestError(f"Failed to load manifest: {str(e)}")
    
    def update_manifest(self, model_updates: Optional[Dict] = None) -> Dict:
        """Update manifest with new model information.
        
        Args:
            model_updates: Optional dict of model updates to apply
            
        Returns:
            Dict: Updated manifest data
            
        Raises:
            ManifestError: If update fails
        """
        with self._lock:
            try:
                manifest = self.load_manifest()
                
                # Update model information
                if model_updates:
                    manifest.update(model_updates)
                
                # Update hashes for all models
                models_dir = self.manifest_path.parent
                for model_file, info in manifest.items():
                    model_path = models_dir / model_file
                    if model_path.exists():
                        if info['expected_hash'] == 'TO_BE_COMPUTED':
                            info['expected_hash'] = self.compute_model_hash(str(model_path))
                            info['last_updated'] = datetime.utcnow().isoformat()
                            logger.info(f"Updated hash for {model_file}")
                
                # Save updated manifest
                with open(self.manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=4)
                
                logger.info("Manifest updated successfully")
                return manifest
                
            except Exception as e:
                raise ManifestError(f"Failed to update manifest: {str(e)}")
    
    def verify_integrity(self, model_paths: Optional[List[str]] = None) -> bool:
        """Verify integrity of specified or all models.
        
        Args:
            model_paths: Optional list of specific model paths to verify
            
        Returns:
            bool: True if all verifications pass
            
        Raises:
            ManifestError: If verification fails
        """
        try:
            manifest = self.load_manifest()
            models_dir = self.manifest_path.parent
            
            # Determine which models to verify
            if model_paths:
                models_to_check = [Path(p).name for p in model_paths]
            else:
                models_to_check = list(manifest.keys())
            
            # Verify each model
            for model_file in models_to_check:
                if model_file not in manifest:
                    raise ManifestError(f"Model {model_file} not found in manifest")
                    
                model_path = models_dir / model_file
                if not model_path.exists():
                    raise ManifestError(f"Model file missing: {model_file}")
                    
                info = manifest[model_file]
                if info['expected_hash'] == 'TO_BE_COMPUTED':
                    logger.warning(f"Hash not computed for {model_file}")
                    continue
                    
                current_hash = self.compute_model_hash(str(model_path))
                if current_hash != info['expected_hash']:
                    raise ManifestError(
                        f"Hash mismatch for {model_file}\n"
                        f"Expected: {info['expected_hash']}\n"
                        f"Got: {current_hash}"
                    )
                    
            return True
            
        except Exception as e:
            logger.error(f"Integrity verification failed: {str(e)}")
            return False
    
    def start_monitoring(self):
        """Start background thread for periodic integrity checks"""
        if self._monitoring:
            return
            
        def monitor_loop():
            while self._monitoring:
                try:
                    if self.verify_integrity():
                        logger.info("Periodic integrity check passed")
                    else:
                        logger.error("Periodic integrity check failed")
                except Exception as e:
                    logger.error(f"Error in integrity monitoring: {str(e)}")
                    
                time.sleep(self.check_interval)
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=monitor_loop,
            daemon=True,
            name="ManifestMonitor"
        )
        self._monitor_thread.start()
        logger.info("Started manifest integrity monitoring")
    
    def stop_monitoring(self):
        """Stop background integrity monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            logger.info("Stopped manifest integrity monitoring")