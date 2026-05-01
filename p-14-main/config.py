# config.py
"""
Global configuration values for the project.
This helps to keep paths and constants in one place.
"""

# Global random seed for reproducibility
RANDOM_SEED = 42

# Official large synthetic dataset path
DATA_LARGE_PATH = "data/processed/large_dataset.csv"

# Simple demo dataset path (used by run_simple.py)
SIMPLE_DATA_PATH = "data/processed/simple_dataset.csv"

# Trained model file path
MODEL_PATH = "models/ai_detector_model.pkl"

# Main test results json path
RESULTS_PATH = "models/test_results.json"
