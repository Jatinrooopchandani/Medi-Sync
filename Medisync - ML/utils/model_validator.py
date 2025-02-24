"""Model validation utilities for ensuring safe and valid model predictions"""
import numpy as np
from typing import Any, Dict, List, Union, Optional
import logging

logger = logging.getLogger(__name__)

class ModelValidationError(Exception):
    """Raised when model validation fails"""
    pass

def validate_prediction_shape(predictions: np.ndarray, expected_shape: Optional[tuple] = None) -> bool:
    """Validate shape and basic properties of model predictions.
    
    Args:
        predictions: NumPy array of predictions
        expected_shape: Optional expected shape to validate against
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ModelValidationError: If validation fails
    """
    try:
        # Check if predictions is a numpy array
        if not isinstance(predictions, np.ndarray):
            raise ModelValidationError("Predictions must be a numpy array")
            
        # Check if predictions contain valid numerical values
        if not np.isfinite(predictions).all():
            raise ModelValidationError("Predictions contain invalid values (inf/nan)")
            
        # Check shape if provided
        if expected_shape and predictions.shape != expected_shape:
            raise ModelValidationError(f"Invalid prediction shape: {predictions.shape}, expected: {expected_shape}")
            
        # Check value range (assuming probabilities)
        if not ((predictions >= 0) & (predictions <= 1)).all():
            raise ModelValidationError("Predictions contain values outside [0,1] range")
            
        return True
        
    except Exception as e:
        logger.error(f"Prediction validation failed: {str(e)}")
        raise ModelValidationError(f"Prediction validation failed: {str(e)}")

def validate_model_input(
    text: List[str],
    drug_names: List[str],
    max_text_length: int = 1000,
    max_drug_name_length: int = 100
) -> bool:
    """Validate model input data.
    
    Args:
        text: List of text inputs
        drug_names: List of drug names
        max_text_length: Maximum allowed text length
        max_drug_name_length: Maximum allowed drug name length
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ModelValidationError: If validation fails
    """
    try:
        # Check input types
        if not isinstance(text, list) or not isinstance(drug_names, list):
            raise ModelValidationError("Inputs must be lists")
            
        # Check lengths match
        if len(text) != len(drug_names):
            raise ModelValidationError("Length mismatch between text and drug_names")
            
        # Validate text entries
        for t in text:
            if not isinstance(t, str):
                raise ModelValidationError("Text entries must be strings")
            if len(t) > max_text_length:
                raise ModelValidationError(f"Text exceeds maximum length of {max_text_length}")
            if len(t.strip()) == 0:
                raise ModelValidationError("Empty text entry found")
                
        # Validate drug names
        for drug in drug_names:
            if not isinstance(drug, str):
                raise ModelValidationError("Drug names must be strings")
            if len(drug) > max_drug_name_length:
                raise ModelValidationError(f"Drug name exceeds maximum length of {max_drug_name_length}")
            if len(drug.strip()) == 0:
                raise ModelValidationError("Empty drug name found")
                
        return True
        
    except Exception as e:
        logger.error(f"Input validation failed: {str(e)}")
        raise ModelValidationError(f"Input validation failed: {str(e)}")

def validate_model_compatibility(rf_version: str, lstm_version: str) -> bool:
    """Validate that model versions are compatible.
    
    Args:
        rf_version: Version string of RandomForest model
        lstm_version: Version string of LSTM model
        
    Returns:
        bool: True if models are compatible
        
    Raises:
        ModelValidationError: If validation fails
    """
    try:
        from packaging import version
        
        rf_ver = version.parse(rf_version)
        lstm_ver = version.parse(lstm_version)
        
        # Check major versions match (assuming same major version required for compatibility)
        if rf_ver.major != lstm_ver.major:
            raise ModelValidationError(f"Incompatible model versions: RF {rf_version}, LSTM {lstm_version}")
            
        return True
        
    except Exception as e:
        logger.error(f"Model compatibility check failed: {str(e)}")
        raise ModelValidationError(f"Model compatibility check failed: {str(e)}")

def verify_prediction_consistency(rf_pred: np.ndarray, lstm_pred: np.ndarray, threshold: float = 0.5) -> bool:
    """Verify that predictions from different models are reasonably consistent.
    
    Args:
        rf_pred: Predictions from RandomForest model
        lstm_pred: Predictions from LSTM model
        threshold: Maximum allowed average absolute difference
        
    Returns:
        bool: True if predictions are consistent
        
    Raises:
        ModelValidationError: If validation fails
    """
    try:
        # Verify shapes match
        if rf_pred.shape != lstm_pred.shape:
            raise ModelValidationError("Prediction shape mismatch between models")
            
        # Check average absolute difference
        avg_diff = np.mean(np.abs(rf_pred - lstm_pred))
        if avg_diff > threshold:
            raise ModelValidationError(f"Model predictions differ by {avg_diff:.3f} (threshold: {threshold})")
            
        return True
        
    except Exception as e:
        logger.error(f"Prediction consistency check failed: {str(e)}")
        raise ModelValidationError(f"Prediction consistency check failed: {str(e)}")