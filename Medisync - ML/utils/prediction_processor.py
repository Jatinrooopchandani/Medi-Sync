"""Secure prediction processing utilities"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import logging
from .model_validator import (
    validate_prediction_shape,
    validate_model_input,
    verify_prediction_consistency,
    ModelValidationError
)

logger = logging.getLogger(__name__)

class PredictionError(Exception):
    """Raised when prediction processing fails"""
    pass

def process_input_securely(df: pd.DataFrame, max_text_length: int = 1000) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Process and validate input data securely.
    
    Args:
        df: Input DataFrame
        max_text_length: Maximum allowed text length
        
    Returns:
        Tuple of (processed DataFrame, metadata)
        
    Raises:
        PredictionError: If input processing fails
    """
    metadata = {
        'original_rows': len(df),
        'processed_rows': 0,
        'dropped_rows': 0,
        'warnings': []
    }
    
    try:
        # Validate required columns
        required_columns = ['text', 'drug_name']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise PredictionError(f"Missing required columns: {missing}")
            
        # Remove any rows with missing values
        df_clean = df.dropna(subset=required_columns)
        metadata['dropped_rows'] = len(df) - len(df_clean)
        if metadata['dropped_rows'] > 0:
            metadata['warnings'].append(f"Dropped {metadata['dropped_rows']} rows with missing values")
            
        if df_clean.empty:
            raise PredictionError("No valid data rows after cleaning")
            
        # Validate input content
        validate_model_input(
            text=df_clean['text'].tolist(),
            drug_names=df_clean['drug_name'].tolist(),
            max_text_length=max_text_length
        )
        
        # Sanitize inputs
        df_clean['text'] = df_clean['text'].astype(str).apply(lambda x: x[:max_text_length])
        df_clean['text'] = df_clean['text'].str.replace(r'[^\w\s-]', '', regex=True)
        df_clean['drug_name'] = df_clean['drug_name'].str.replace(r'[^\w\s-]', '', regex=True)
        
        metadata['processed_rows'] = len(df_clean)
        return df_clean, metadata
        
    except Exception as e:
        logger.error(f"Input processing failed: {str(e)}")
        raise PredictionError(f"Input processing failed: {str(e)}")

def generate_predictions(
    df: pd.DataFrame,
    tokenizer: Any,
    lstm_model: Any,
    rf_model: Any,
    max_sequence_length: int = 100
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate predictions using both models securely.
    
    Args:
        df: Processed DataFrame
        tokenizer: Text tokenizer
        lstm_model: LSTM model
        rf_model: RandomForest model
        max_sequence_length: Maximum sequence length for LSTM
        
    Returns:
        Tuple of (predictions array, metadata)
        
    Raises:
        PredictionError: If prediction generation fails
    """
    metadata = {
        'lstm_shape': None,
        'rf_shape': None,
        'warnings': []
    }
    
    try:
        # Prepare text sequences
        sequences = tokenizer.texts_to_sequences(df['text'].values)
        padded_sequences = np.array(sequences)
        if len(padded_sequences.shape) == 1:
            padded_sequences = padded_sequences.reshape(-1, 1)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            padded_sequences,
            maxlen=max_sequence_length,
            padding='post'
        )
        
        # Generate LSTM predictions
        lstm_preds = lstm_model.predict(padded_sequences, batch_size=32)
        metadata['lstm_shape'] = lstm_preds.shape
        
        # Generate RF predictions
        drug_features = np.array(df['drug_name'].values).reshape(-1, 1)
        rf_preds = rf_model.predict(drug_features)
        if len(rf_preds.shape) == 1:
            rf_preds = rf_preds.reshape(-1, 1)
        metadata['rf_shape'] = rf_preds.shape
        
        # Validate predictions
        expected_shape = (len(df), 1)
        validate_prediction_shape(lstm_preds, expected_shape)
        validate_prediction_shape(rf_preds, expected_shape)
        
        # Check prediction consistency
        verify_prediction_consistency(rf_preds, lstm_preds, threshold=0.5)
        
        # Combine predictions
        final_predictions = np.clip((lstm_preds + rf_preds) / 2, 0, 1)
        
        # Final validation
        validate_prediction_shape(final_predictions, expected_shape)
        
        return final_predictions, metadata
        
    except Exception as e:
        logger.error(f"Prediction generation failed: {str(e)}")
        raise PredictionError(f"Prediction generation failed: {str(e)}")