"""Security test script to verify the pickle vulnerability is fixed."""
import os
import pickle
import json
import sys
from pathlib import Path

def test_old_vulnerability():
    """Try to create and load a malicious pickle file."""
    print("Testing old vulnerability...")
    
    class MaliciousPayload:
        def __reduce__(self):
            return (os.system, ('echo "Malicious code executed" > /tmp/hacked.txt',))
    
    # Create malicious pickle files
    print("Creating malicious pickle files...")
    malicious_files = ['tokenizer.pkl', 'drug_encoder.pkl', 'rf.pkl']
    for file in malicious_files:
        with open(f'models/{file}', 'wb') as f:
            pickle.dump(MaliciousPayload(), f)
    
    print("Attempting to trigger vulnerability...")
    try:
        # Try to import main module (should fail since we now use JSON)
        import main
        print("WARNING: Main module loaded without error - this might indicate the fix is not complete")
    except Exception as e:
        if "No such file or directory" in str(e) and ".json" in str(e):
            print("SUCCESS: Pickle files are no longer loaded, system expects JSON files")
        else:
            print(f"ERROR: Unexpected error: {str(e)}")

def test_secure_implementation():
    """Test that the new secure JSON implementation works correctly."""
    print("\nTesting secure implementation...")
    
    # Create test JSON files with valid schemas
    test_data = {
        'tokenizer.json': {
            'vocabulary': {'test': 1},
            'config': {'num_words': 1000},
            'word_index': {'test': 1},
            'index_word': {'1': 'test'},
            'num_words': 1000
        },
        'drug_encoder.json': {
            'classes_': ['drug1', 'drug2'],
            'dtype': 'int64',
            'sparse': False,
            'categories_': [['drug1', 'drug2']]
        },
        'rf.json': {
            'n_estimators': 100,
            'criterion': 'gini',
            'max_depth': None
        }
    }
    
    print("Creating valid JSON test files...")
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    for filename, data in test_data.items():
        with open(models_dir / filename, 'w') as f:
            json.dump(data, f)
    
    print("Testing secure loading...")
    try:
        from utils.secure_loader import load_tokenizer_safely, load_drug_encoder_safely, load_model_safely
        
        # Test each loader
        tokenizer = load_tokenizer_safely('models/tokenizer.json')
        drug_encoder = load_drug_encoder_safely('models/drug_encoder.json')
        rf = load_model_safely('models/rf.json')
        
        print("SUCCESS: All secure loaders working correctly")
    except Exception as e:
        print(f"ERROR: Failed to load secure JSON files: {str(e)}")

def cleanup():
    """Clean up test files."""
    print("\nCleaning up test files...")
    test_files = [
        'models/tokenizer.pkl', 'models/drug_encoder.pkl', 'models/rf.pkl',
        'models/tokenizer.json', 'models/drug_encoder.json', 'models/rf.json',
        '/tmp/hacked.txt'
    ]
    for file in test_files:
        try:
            os.remove(file)
            print(f"Removed {file}")
        except FileNotFoundError:
            pass

if __name__ == '__main__':
    try:
        test_old_vulnerability()
        test_secure_implementation()
    finally:
        cleanup()