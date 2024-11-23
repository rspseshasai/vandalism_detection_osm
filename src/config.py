# src/config.py

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, 'external')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Data file paths
CONTRIBUTIONS_DATA_FILE = os.path.join(RAW_DATA_DIR, 'osm_labelled_contributions.parquet')
FEATURES_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, 'extracted_features_contributions.parquet')

# Other configurations
RANDOM_STATE = 42
