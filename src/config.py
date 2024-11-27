import logging
import os
import sys

import coloredlogs

# === Dataset Type ===
DATASET_TYPE = 'contribution'  # Options: 'contribution', 'changeset'

# === Split Configurations ===
SPLIT_TYPES = ['random', 'temporal', 'geographic']
SPLIT_METHOD = 'random'  # 'random', 'temporal', or 'geographic'

TEST_SIZE = 0.4  # Proportion for the temporary test set
VAL_SIZE = 0.2  # Proportion of the temporary test set to use as the final test set
RANDOM_STATE = 42

# === Base Directories ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', f"{DATASET_TYPE}_data")
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
VISUALIZATION_DIR = os.path.join(DATA_DIR, 'visualization', SPLIT_METHOD)
MODELS_DIR = os.path.join(BASE_DIR, 'models', f"{DATASET_TYPE}_model")

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# === Clustering Configuration ===
N_CLUSTERS = 100  # Default number of clusters for KMeans

# Number of jobs for parallel processing
N_JOBS = 11  # -1 to use all available cores

# === File Paths ===
CONTRIBUTION_DATA_RAW_FILE_NAME = 'osm_labelled_contributions.parquet'
CHANGESET_DATA_RAW_FILE_NAME = 'osm_labelled_changeset_features_with_user_info.parquet'
if DATASET_TYPE == 'changeset':
    RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, CHANGESET_DATA_RAW_FILE_NAME)
else:
    RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, CONTRIBUTION_DATA_RAW_FILE_NAME)
PROCESSED_FEATURES_FILE = os.path.join(PROCESSED_DATA_DIR, 'processed_features.parquet')

# Paths for models and hyperparameters
BEST_PARAMS_PATH = os.path.join(MODELS_DIR, SPLIT_METHOD, 'best_hyperparameters.json')
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, SPLIT_METHOD, 'final_xgboost_model.pkl')
os.makedirs(os.path.join(MODELS_DIR, SPLIT_METHOD), exist_ok=True)

# === Visualization Paths ===
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
    'confusion_matrix': os.path.join(VISUALIZATION_DIR, 'confusion_matrix.csv'),
}

# === Bootstrapping Configurations ===
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_RESULTS_DIR = os.path.join(PROCESSED_DATA_DIR, SPLIT_METHOD, 'bootstrap_results')
os.makedirs(BOOTSTRAP_RESULTS_DIR, exist_ok=True)

# === Geographical Evaluation Configurations ===
GEOGRAPHICAL_RESULTS_DIR = os.path.join(PROCESSED_DATA_DIR, SPLIT_METHOD, 'geographical_evaluation_results')
os.makedirs(GEOGRAPHICAL_RESULTS_DIR, exist_ok=True)

# === Geographic Split Parameters ===
GEOGRAPHIC_SPLIT_KEY = 'continent'  # 'continent' or 'country'
TRAIN_REGIONS = ['Oceania', 'Europe']
VAL_REGIONS = ['Africa']
TEST_REGIONS = ['North America', 'Asia']

# === Temporal Split Parameters ===
DATE_COLUMN = 'date_created'  # Column name in the DataFrame
TRAIN_YEARS = [2018, 2019]
VAL_YEARS = [2015]
TEST_YEARS = [2017]

# === Additional Configurations ===
SAVE_VISUALIZATION_SAMPLES = True
TEST_RUN = True
FORCE_COMPUTE_FEATURES = True

# === Logging Configuration ===
LOG_FORMAT = '\n%(asctime)s - %(levelname)s - %(filename)s -- %(message)s'
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, f'{DATASET_TYPE}_pipeline.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout),
    ],
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
    field_styles=FIELD_STYLES,
)

# Export logger instance
logger = logging.getLogger(__name__)
