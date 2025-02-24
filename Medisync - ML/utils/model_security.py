"""Model security utilities for hash computation and verification"""
import hashlib
import json
from pathlib import Path
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file.
    
    Args:
        filepath: Path to the file
        
    Returns:
        str: Hex digest of SHA-256 hash
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file can't be read
    """
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
        
    sha256_hash = hashlib.sha256()
    
    try:
        with open(filepath, 'rb') as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b''):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except IOError as e:
        logger.error(f"Failed to read file {filepath}: {str(e)}")
        raise

def update_model_manifest(manifest_path: Optional[str] = None) -> Dict:
    """Update model manifest with current file hashes.
    
    Args:
        manifest_path: Optional path to manifest file. If None, uses default location
        
    Returns:
        Dict: Updated manifest data
        
    Raises:
        FileNotFoundError: If manifest or model files don't exist
        IOError: If files can't be read/written
    """
    if manifest_path is None:
        manifest_path = Path(__file__).parent.parent / "models" / "manifest.json"
    else:
        manifest_path = Path(manifest_path)
        
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        
    try:
        # Load current manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        # Update hashes for all models
        models_dir = manifest_path.parent
        for model_file, info in manifest.items():
            model_path = models_dir / model_file
            if model_path.exists():
                new_hash = compute_file_hash(str(model_path))
                if info['expected_hash'] == 'TO_BE_COMPUTED' or info['expected_hash'] != new_hash:
                    info['expected_hash'] = new_hash
                    logger.info(f"Updated hash for {model_file}: {new_hash}")
            else:
                logger.warning(f"Model file not found: {model_file}")
                
        # Save updated manifest
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=4)
            
        logger.info("Model manifest updated successfully")
        return manifest
        
    except Exception as e:
        logger.error(f"Failed to update model manifest: {str(e)}")
        raise

def verify_all_models(manifest_path: Optional[str] = None) -> bool:
    """Verify integrity of all model files against manifest.
    
    Args:
        manifest_path: Optional path to manifest file. If None, uses default location
        
    Returns:
        bool: True if all models verified successfully
        
    Raises:
        FileNotFoundError: If manifest or model files don't exist
    """
    if manifest_path is None:
        manifest_path = Path(__file__).parent.parent / "models" / "manifest.json"
    else:
        manifest_path = Path(manifest_path)
        
    try:
        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        models_dir = manifest_path.parent
        all_valid = True
        
        # Check each model
        for model_file, info in manifest.items():
            model_path = models_dir / model_file
            if not model_path.exists():
                logger.error(f"Model file missing: {model_file}")
                all_valid = False
                continue
                
            if info['expected_hash'] == 'TO_BE_COMPUTED':
                logger.warning(f"Hash not computed for {model_file}")
                continue
                
            current_hash = compute_file_hash(str(model_path))
            if current_hash != info['expected_hash']:
                logger.error(f"Hash mismatch for {model_file}")
                logger.error(f"Expected: {info['expected_hash']}")
                logger.error(f"Got: {current_hash}")
                all_valid = False
            else:
                logger.info(f"Verified {model_file}")
                
        return all_valid
        
    except Exception as e:
        logger.error(f"Model verification failed: {str(e)}")
        return False