import logging
import os
import sys

import coloredlogs
import pandas as pd

# === Additional Configurations ===
SAVE_VISUALIZATION_SAMPLES = True
TEST_RUN = False
FORCE_COMPUTE_FEATURES = True
SHOULD_BALANCE_DATASET = True
SHOULD_INCLUDE_USERFEATURES = False
SHOULD_INCLUDE_OSM_ELEMENT_FEATURES = False
# === Dataset Type ===
DATASET_TYPE = 'contribution'  # Options: 'contribution', 'changeset'

# === Split Configurations ===
SPLIT_TYPES = ['random', 'temporal', 'geographic']
SPLIT_METHOD = 'random'  # 'random', 'temporal', or 'geographic'

TEST_SIZE = 0.4  # Proportion for the temporary test set
VAL_SIZE = 0.2  # Proportion of the temporary test set to use as the final test set
RANDOM_STATE = 42

if DATASET_TYPE == 'changeset':
    TEST_SIZE = 0.5  # Proportion for the temporary test set
    VAL_SIZE = 0.1  # Proportion of the temporary test set to use as the final test set
    META_TEST_SIZE = 0.45

# === Base Directories ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', f"{DATASET_TYPE}_data")
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
VISUALIZATION_DIR = os.path.join(DATA_DIR, 'visualization', SPLIT_METHOD)
MODELS_DIR = os.path.join(BASE_DIR, 'models', f"{DATASET_TYPE}_model")
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')

# === Hyper-Classifier Paths ===
HYPER_CLASSIFIER_DIR = os.path.join(BASE_DIR, 'models', f'{DATASET_TYPE}_model', SPLIT_METHOD, 'hyper_classifier')
os.makedirs(HYPER_CLASSIFIER_DIR, exist_ok=True)

# === Meta-Classifier Paths ===
META_CLASSIFIER_DIR = os.path.join(BASE_DIR, 'models', f'{DATASET_TYPE}_model', SPLIT_METHOD, 'meta_classifier')
os.makedirs(META_CLASSIFIER_DIR, exist_ok=True)

prefix = ""
if TEST_RUN:
    prefix = 'test'

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Clustering Configuration ===
N_CLUSTERS = 100  # Default number of clusters for KMeans

# Number of jobs for parallel processing
N_JOBS = 11  # -1 to use all available cores

# === File Paths ===
CHANGESET_DATA_RAW_FILE_NAME = 'osm_labelled_changeset_features_with_user_info.parquet'
UNLABELLED_CHANGESET_DATA_RAW_FILE_NAME = 'changesets_unlabelled_data.parquet'

CONTRIBUTION_DATA_RAW_FILE_NAME = 'training_data_osm_contributions_labeled.parquet'
# UNLABELLED_CONTRIBUTIONS_DATA_RAW_FILE_NAME = '2024-02-01.parquet'
UNLABELLED_CONTRIBUTIONS_DATA_RAW_FILE_NAME = '2022-03-01.parquet'

if DATASET_TYPE == 'changeset':
    RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, CHANGESET_DATA_RAW_FILE_NAME)
    UNLABELLED_RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, UNLABELLED_CONTRIBUTIONS_DATA_RAW_FILE_NAME)

else:
    RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, CONTRIBUTION_DATA_RAW_FILE_NAME)
    UNLABELLED_RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, UNLABELLED_CONTRIBUTIONS_DATA_RAW_FILE_NAME)

PROCESSED_FEATURES_FILE = os.path.join(PROCESSED_DATA_DIR, f'{prefix}_processed_features.parquet')
UNLABELLED_PROCESSED_FEATURES_FILE = os.path.join(PROCESSED_DATA_DIR, f'{prefix}_unlabelled_processed_features.parquet')

PROCESSED_ENCODED_FEATURES_FILE = os.path.join(PROCESSED_DATA_DIR, f'{prefix}_processed_encoded_features.parquet')

# Validation dataset paths
VALIDATION_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, 'validation_dataset.parquet')
VALIDATION_LABELS_PATH = os.path.join(PROCESSED_DATA_DIR, 'validation_labels.parquet')

UNLABELLED_PROCESSED_ENCODED_FEATURES_FILE = os.path.join(PROCESSED_DATA_DIR,
                                                          f'{prefix}_unlabelled_processed_encoded_features.parquet')

CHANGESET_LABELS_FILE = os.path.join(os.path.join(os.path.join(BASE_DIR, 'data', "changeset_data"), 'raw'),
                                     'changeset_labels.tsv')

PREDICTIONS_INPUT_DATA_DIR = os.path.join(RAW_DATA_DIR, 'parquet_files_to_be_predicted', '2022_jan_to_july')
HISTORICAL_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'history_files')

# Paths for models and hyperparameters
BEST_PARAMS_PATH = os.path.join(MODELS_DIR, SPLIT_METHOD, f'{prefix}_best_hyperparameters.json')
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, SPLIT_METHOD, f'{prefix}_final_xgboost_model.pkl')
FINAL_TRAINED_FEATURES_PATH = os.path.join(MODELS_DIR, SPLIT_METHOD, f'{prefix}_final_trained_features.pkl')
OPTIMAL_THRESHOLD_FOR_INFERENCE_PATH = os.path.join(MODELS_DIR, SPLIT_METHOD, f'{prefix}_optimal_threshold_for_inference.pkl')

CLUSTER_MODEL_PATH = os.path.join(MODELS_DIR, SPLIT_METHOD, f'{prefix}_final_kmeans_clustering_model.pkl')

HYPER_MODEL_PATH = os.path.join(HYPER_CLASSIFIER_DIR, f'{prefix}_hyper_classifier_model.xgb')

META_MODEL_BEST_PARAMS_PATH = os.path.join(META_CLASSIFIER_DIR, f'{prefix}_best_hyperparameters.json')
META_MODEL_PATH = os.path.join(META_CLASSIFIER_DIR, f'{prefix}_meta_classifier_model.xgb')

# For Hyper Classifier
CONTRIBUTION_FINAL_MODEL_PATH = os.path.join(os.path.join(BASE_DIR, 'models', f"contribution_model"), SPLIT_METHOD,
                                             f'{prefix}_final_xgboost_model.pkl')
CONTRIBUTION_PROCESSED_ENCODED_FEATURES_FILE = os.path.join(
    os.path.join(os.path.join(BASE_DIR, 'data', f"contribution_data"), 'processed'),
    f'{prefix}_processed_encoded_features.parquet')

os.makedirs(os.path.join(MODELS_DIR, SPLIT_METHOD), exist_ok=True)
UNLABELLED_PROCESSED_OUTPUT_CSV_FILE = os.path.join(OUTPUT_DIR, f'{prefix}_unlabelled_predictions.csv')

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
    'evaluation_results_main': os.path.join(VISUALIZATION_DIR, 'evaluation_results_main.parquet'),
    'evaluation_results_meta_classifier': os.path.join(VISUALIZATION_DIR, 'evaluation_results_meta_classifier.csv'),
    'evaluation_results_hyper_classifier': os.path.join(VISUALIZATION_DIR,
                                                        'evaluation_results_hyper_classifier.parquet'),
    'confusion_matrix_main': os.path.join(VISUALIZATION_DIR, 'confusion_matrix_main.csv'),
    'confusion_matrix_hyper_classifier': os.path.join(VISUALIZATION_DIR, 'confusion_matrix_hyper_classifier.csv'),
    'hyper_classifier_features_sample_path': os.path.join(VISUALIZATION_DIR,
                                                          'hyper_classifier_features_sample.parquet'),
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

# Test changeset ids
# Take the first 1000 changeset IDs for testing
TEST_CHANGESET_IDS = pd.read_csv(os.path.join(os.path.join(os.path.join(BASE_DIR, 'data', "changeset_data"), 'raw'),
                                              'test_common_changesets_1000.csv'))['changeset_id']

COMMON_CHANGESET_IDS = pd.read_csv(os.path.join(os.path.join(os.path.join(BASE_DIR, 'data', "changeset_data"), 'raw'),
                                                '_common_changeset_ids.csv'))['changeset_id']

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
