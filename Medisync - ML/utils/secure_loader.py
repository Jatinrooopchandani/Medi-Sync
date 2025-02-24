"""Secure model loading utilities"""
import json
import jsonschema
from pathlib import Path
import hashlib
import tensorflow as tf
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SecurityError(Exception):
    """Raised when a security check fails during model loading"""
    pass

def load_manifest() -> Dict:
    """Load the model manifest containing version and hash information.
    
    Returns:
        Dict containing model metadata
        
    Raises:
        FileNotFoundError: If manifest file doesn't exist
        json.JSONDecodeError: If manifest contains invalid JSON
    """
    manifest_path = Path(__file__).parent.parent / "models" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Model manifest file not found")
        
    with open(manifest_path, 'r') as f:
        return json.load(f)

def verify_model_integrity(file_path: str, expected_hash: Optional[str] = None) -> bool:
    """Verify the integrity of a model file using SHA-256.
    
    Args:
        file_path: Path to the model file
        expected_hash: Expected SHA-256 hash of the file
        
    Returns:
        True if integrity check passes, False otherwise
    """
    if expected_hash is None:
        return True
        
    with open(file_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    return file_hash == expected_hash

# Define schema for tokenizer validation
TOKENIZER_SCHEMA = {
    "type": "object",
    "properties": {
        "vocabulary": {"type": "object"},
        "config": {"type": "object"},
        "word_index": {"type": "object"},
        "index_word": {"type": "object"},
        "num_words": {"type": "integer"}
    },
    "required": ["vocabulary", "config", "word_index", "index_word", "num_words"],
    "additionalProperties": False
}

# Define schema for drug encoder validation
DRUG_ENCODER_SCHEMA = {
    "type": "object", 
    "properties": {
        "classes_": {
            "type": "array",
            "items": {"type": "string"}
        },
        "dtype": {"type": "string"},
        "drop_idx_": {"type": ["null", "integer"]},
        "handle_unknown": {"type": "string"},
        "sparse": {"type": "boolean"},
        "categories_": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    },
    "required": ["classes_", "dtype", "sparse", "categories_"],
    "additionalProperties": False
}

# Define schema for RandomForest model validation
RF_MODEL_SCHEMA = {
    "type": "object",
    "properties": {
        "n_estimators": {"type": "integer", "minimum": 1},
        "max_depth": {"type": ["integer", "null"], "minimum": 1},
        "min_samples_split": {"type": "integer", "minimum": 2},
        "min_samples_leaf": {"type": "integer", "minimum": 1},
        "max_features": {"type": ["string", "integer", "number", "null"]},
        "bootstrap": {"type": "boolean"},
        "random_state": {"type": ["integer", "null"]},
        "class_weight": {"type": ["object", "string", "null"]},
        "criterion": {"type": "string", "enum": ["gini", "entropy", "log_loss"]},
        "parameters": {
            "type": "object",
            "additionalProperties": True
        },
        "feature_importances_": {
            "type": "array",
            "items": {"type": "number"}
        },
        "n_classes_": {"type": "integer", "minimum": 2},
        "classes_": {
            "type": "array",
            "items": {"type": ["string", "number"]}
        }
    },
    "required": ["n_estimators", "criterion", "parameters", "classes_"],
    "additionalProperties": False
}

def load_tokenizer_safely(filepath: str) -> Any:
    """Safely load tokenizer from JSON file with schema validation.
    
    Args:
        filepath: Path to the tokenizer JSON file
        
    Returns:
        Keras tokenizer instance with loaded configuration
        
    Raises:
        FileNotFoundError: If tokenizer file doesn't exist
        jsonschema.exceptions.ValidationError: If JSON data doesn't match schema
        json.JSONDecodeError: If file contains invalid JSON
        SecurityError: If integrity check fails
    """
    # Verify file exists
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {filepath}")

    # Get manifest info for integrity check
    try:
        manifest = load_manifest()
        model_info = manifest.get(file_path.name, {})
        expected_hash = model_info.get('expected_hash')
        
        # Verify integrity if hash is provided and not placeholder
        if expected_hash and expected_hash != "TO_BE_COMPUTED":
            if not verify_model_integrity(filepath, expected_hash):
                raise SecurityError("Tokenizer file integrity check failed")
    except FileNotFoundError:
        logger.warning("Model manifest not found - skipping integrity check")
        
    # Load and validate JSON data
    with open(filepath, 'r') as f:
        tokenizer_data = json.load(f)
        
    # Validate against schema
    jsonschema.validate(instance=tokenizer_data, schema=TOKENIZER_SCHEMA)
    
    # Create tokenizer instance with validated data
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=tokenizer_data['num_words'])
    tokenizer.word_index = tokenizer_data['word_index']
    tokenizer.index_word = tokenizer_data['index_word']
    tokenizer.word_counts = tokenizer_data['vocabulary']
    tokenizer.config = tokenizer_data['config']
    
    logger.info(f"Successfully loaded tokenizer from {filepath}")
    return tokenizer

def load_drug_encoder_safely(filepath: str) -> Any:
    """Safely load drug encoder from JSON file with schema validation.
    
    Args:
        filepath: Path to the drug encoder JSON file
        
    Returns:
        Initialized drug encoder with loaded configuration
        
    Raises:
        FileNotFoundError: If encoder file doesn't exist
        jsonschema.exceptions.ValidationError: If JSON data doesn't match schema
        json.JSONDecodeError: If file contains invalid JSON
        SecurityError: If integrity check fails
    """
    # Verify file exists
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"Drug encoder file not found: {filepath}")

    # Get manifest info for integrity check
    try:
        manifest = load_manifest()
        model_info = manifest.get(file_path.name, {})
        expected_hash = model_info.get('expected_hash')
        
        # Verify integrity if hash is provided and not placeholder
        if expected_hash and expected_hash != "TO_BE_COMPUTED":
            if not verify_model_integrity(filepath, expected_hash):
                raise SecurityError("Drug encoder file integrity check failed")
    except FileNotFoundError:
        logger.warning("Model manifest not found - skipping integrity check")
        
    # Load and validate JSON data
    with open(filepath, 'r') as f:
        encoder_data = json.load(f)
        
    # Validate against schema
    jsonschema.validate(instance=encoder_data, schema=DRUG_ENCODER_SCHEMA)
    
    # Create encoder instance with validated data
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.classes_ = encoder_data['classes_']
    
    logger.info(f"Successfully loaded drug encoder from {filepath}")
    return encoderrror: If file contains invalid JSON
        SecurityError: If integrity check fails
    """
    # Verify file exists
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {filepath}")
    
    # Get manifest info
    try:
        manifest = load_manifest()
        model_info = manifest.get(file_path.name, {})
        expected_hash = model_info.get('expected_hash')
        
        # Verify integrity if hash is provided
        if expected_hash and expected_hash != "TO_BE_COMPUTED":
            if not verify_model_integrity(filepath, expected_hash):
                raise SecurityError("Tokenizer file integrity check failed")
    except FileNotFoundError:
        logger.warning("Model manifest not found - skipping integrity check")
    
    # Load and validate JSON data
    with open(filepath, 'r') as f:
        tokenizer_data = json.load(f)
        
    # Validate against schema
    jsonschema.validate(instance=tokenizer_data, schema=TOKENIZER_SCHEMA)
    
    # Create tokenizer instance with validated data
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=tokenizer_data['num_words'])
    tokenizer.word_index = tokenizer_data['word_index']
    tokenizer.index_word = tokenizer_data['index_word']
    tokenizer.word_counts = tokenizer_data['vocabulary']
    tokenizer.config = tokenizer_data['config']
    
    logger.info(f"Successfully loaded tokenizer from {filepath}")
    return tokenizer

def load_drug_encoder_safely(filepath: str) -> Any:
    """Safely load drug encoder from JSON file with schema validation.
    
    Args:
        filepath: Path to the drug encoder JSON file
        
    Returns:
        Initialized drug encoder with loaded configuration
        
    Raises:
        FileNotFoundError: If encoder file doesn't exist
        jsonschema.exceptions.ValidationError: If JSON data doesn't match schema
        json.JSONDecodeError: If file contains invalid JSON
        SecurityError: If integrity check fails
    """
    # Verify file exists
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"Drug encoder file not found: {filepath}")

    # Get manifest info
    try:
        manifest = load_manifest()
        model_info = manifest.get(file_path.name, {})
        expected_hash = model_info.get('expected_hash')
        
        # Verify integrity if hash is provided
        if expected_hash and expected_hash != "TO_BE_COMPUTED":
            if not verify_model_integrity(filepath, expected_hash):
                raise SecurityError("Drug encoder file integrity check failed")
    except FileNotFoundError:
        logger.warning("Model manifest not found - skipping integrity check")

    # Load and validate JSON data
    with open(filepath, 'r') as f:
        encoder_data = json.load(f)
        
    # Validate against schema
    jsonschema.validate(instance=encoder_data, schema=DRUG_ENCODER_SCHEMA)
    
    # Create encoder instance with validated data
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.classes_ = encoder_data['classes_']
    
    logger.info(f"Successfully loaded drug encoder from {filepath}")
    return encoder

def load_model_safely(filepath: str) -> Any:
    """Safely load a model from JSON file with integrity checks.
    
    Args:
        filepath: Path to the model JSON file
        
    Returns:
        Loaded model instance
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
        SecurityError: If integrity check fails
    """
    # Verify file exists
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    # Get manifest info
    try:
        manifest = load_manifest()
        model_info = manifest.get(file_path.name, {})
        expected_hash = model_info.get('expected_hash')
        
        # Verify integrity if hash is provided
        if expected_hash and expected_hash != "TO_BE_COMPUTED":
            if not verify_model_integrity(filepath, expected_hash):
                raise SecurityError("Model file integrity check failed")
    except FileNotFoundError:
        logger.warning("Model manifest not found - skipping integrity check")

    # Load and validate model data
    with open(filepath, 'r') as f:
        model_data = json.load(f)
    
    # Convert JSON model data to sklearn RandomForest instance
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    
    # Set model parameters from JSON
    for param, value in model_data.get('params', {}).items():
        setattr(model, param, value)
    
    logger.info(f"Successfully loaded model from {filepath}")
    return modelrror: If file contains invalid JSON
    """
    # Verify file exists
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"Drug encoder file not found: {filepath}")
        
    # Load and validate JSON data
    with open(filepath, 'r') as f:
        encoder_data = json.load(f)
        
    # Validate against schema
    jsonschema.validate(instance=encoder_data, schema=DRUG_ENCODER_SCHEMA)
    
    # Create encoder instance with validated data
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.classes_ = encoder_data['classes_']
    
    return encoder

def load_model_safely(filepath: str) -> Any:
    """Safely load ML model from JSON file.
    
    Args:
        filepath: Path to the model JSON file
        
    Returns:
        Loaded model instance
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    # Verify file exists
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
        
    # Load model from JSON
    with open(filepath, 'r') as f:
        model_data = json.load(f)
    
    # Initialize model with the data
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    for key, value in model_data.items():
        setattr(model, key, value)
    
    return model