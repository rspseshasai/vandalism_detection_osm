import logging
import os
import sys

import coloredlogs

# Define split types
SPLIT_TYPES = ['random', 'temporal', 'geographic']
SPLIT_METHOD = 'random'  # 'random', 'temporal', or 'geographic'

# === Base Directories ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed', SPLIT_METHOD)
MODELS_DIR = os.path.join(BASE_DIR, 'models', SPLIT_METHOD)

if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
# === Data File Paths ===
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, 'osm_labelled_contributions.parquet')
PROCESSED_FEATURES_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, 'extracted_features_contributions.parquet')

# Paths for saving models and hyperparameters
BEST_PARAMS_PATH_CONTRIBUTION_DATA = os.path.join(MODELS_DIR, 'best_hyperparameters.json')
FINAL_MODEL_PATH_CONTRIBUTION_DATA = os.path.join(MODELS_DIR, 'final_xgboost_model.pkl')

# === Split Configurations ===
TEST_SIZE = 0.4  # Proportion for the temporary test set
VAL_SIZE = 0.2  # Proportion of the temporary set to use as the final test set
RANDOM_STATE = 42

# === Clustering configuration ===
N_CLUSTERS = 100  # Default number of clusters for KMeans

# === Visualization ===
VISUALIZATION_DIR = os.path.join(DATA_DIR, 'visualization', SPLIT_METHOD)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)  # Ensure the directory exists

VISUALIZATION_DATA_PATH = {
    'data_loading': os.path.join(VISUALIZATION_DIR, 'data_loading_sample.parquet'),
    'feature_engineering': os.path.join(VISUALIZATION_DIR, 'feature_engineering_sample.parquet'),
    'preprocessing_X': os.path.join(VISUALIZATION_DIR, 'preprocessing_X_sample.parquet'),
    'preprocessing_y': os.path.join(VISUALIZATION_DIR, 'preprocessing_y_sample.parquet'),
    'data_splitting_X_train': os.path.join(VISUALIZATION_DIR, 'data_splitting_X_train_sample.parquet'),
    'data_splitting_X_val': os.path.join(VISUALIZATION_DIR, 'data_splitting_X_val_sample.parquet'),
    'data_splitting_X_test': os.path.join(VISUALIZATION_DIR, 'data_splitting_X_test_sample.parquet'),
    'clustering_train': os.path.join(VISUALIZATION_DIR, 'clustering_train_sample.parquet'),
    'clustering_val': os.path.join(VISUALIZATION_DIR, 'clustering_val_sample.parquet'),
    'clustering_test': os.path.join(VISUALIZATION_DIR, 'clustering_test_sample.parquet'),
    'evaluation_results': os.path.join(VISUALIZATION_DIR, 'evaluation_results.parquet'),
    'confusion_matrix': os.path.join(VISUALIZATION_DIR, 'confusion_matrix.csv')
}

# === Additional Configurations ===
SAVE_VISUALIZATION_SAMPLES = True
TEST_RUN = True

# Bootstrapping configurations
BOOTSTRAP_ITERATIONS = 1000  # Number of bootstrap iterations
BOOTSTRAP_RESULTS_DIR = os.path.join(PROCESSED_DATA_DIR, 'bootstrap_results')
if not os.path.exists(BOOTSTRAP_RESULTS_DIR):
    os.makedirs(BOOTSTRAP_RESULTS_DIR)

# Geographical evaluation configurations
GEOGRAPHICAL_RESULTS_DIR = os.path.join(PROCESSED_DATA_DIR, 'geographical_evaluation_results')
if not os.path.exists(GEOGRAPHICAL_RESULTS_DIR):
    os.makedirs(GEOGRAPHICAL_RESULTS_DIR)

# === Logger Configuration ===
LOG_FORMAT = '\n%(asctime)s - %(levelname)s - %(filename)s -- %(message)s'
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the log directory exists
LOG_FILE_PATH = os.path.join(LOG_DIR, 'pipeline.log')

# Number of jobs for parallel processing
N_JOBS = 11  # -1 to use all available cores

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),  # Log to a file
        logging.StreamHandler(sys.stdout),  # Log to stdout (console)
    ]
)

# Custom color scheme for coloredlogs
FIELD_STYLES = {
    'asctime': {'color': 'green'},
    'levelname': {'color': 'black', 'bold': True},
    'filename': {'color': 'magenta'},
}
LEVEL_STYLES = {
    'debug': {'color': 'blue'},
    'info': {'color': 'black'},
    'warning': {'color': 'yellow'},
    'error': {'color': 'red'},
    'critical': {'color': 'red', 'bold': True},
}

# Apply coloredlogs
coloredlogs.install(
    level='INFO',
    logger=logging.getLogger(__name__),
    fmt=LOG_FORMAT,
    level_styles=LEVEL_STYLES,
    field_styles=FIELD_STYLES
)

# Create and export the logger instance
logger = logging.getLogger(__name__)
