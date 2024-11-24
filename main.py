import os
import sys

# Adjust the path to import modules from src
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_dir, 'src'))

from config import logger
from src import config
from src.data_loading import load_data
from src.feature_engineering import get_or_generate_features
from src.preprocessing import preprocess_features
from src.data_splitting import split_train_test_val, calculate_statistics, log_dataset_shapes
from src.clustering import perform_clustering


# Step 1: Load Data
def data_loading_helper():
    logger.info("Starting data loading...")
    contributions_df = load_data(print_sample_data=False)

    if config.SAVE_VISUALIZATION_SAMPLES:
        sample_path = config.VISUALIZATION_DATA_PATH['data_loading']
        contributions_df.head(100).to_parquet(sample_path)
        logger.info(f"Saved data loading sample to {sample_path}")

    logger.info("Data loading completed.")
    return contributions_df


# Step 2: Feature Engineering
def feature_engineering_helper(contributions_df):
    logger.info("Starting feature engineering...")
    features_df = get_or_generate_features(
        contributions_df,
        force_compute_features=False,
        test_mode=False
    )

    if config.SAVE_VISUALIZATION_SAMPLES:
        sample_path = config.VISUALIZATION_DATA_PATH['feature_engineering']
        features_df.head(100).to_parquet(sample_path)
        logger.info(f"Saved feature engineering sample to {sample_path}")

    logger.info("Feature engineering completed.")
    return features_df


# Step 3: Preprocessing
def preprocessing_helper(features_df):
    logger.info("Starting preprocessing...")
    X_encoded, y = preprocess_features(features_df)

    if config.SAVE_VISUALIZATION_SAMPLES:
        sample_path_X = config.VISUALIZATION_DATA_PATH['preprocessing_X']
        sample_path_y = config.VISUALIZATION_DATA_PATH['preprocessing_y']
        X_encoded.head(100).to_parquet(sample_path_X)
        y.head(100).to_frame(name='vandalism').to_parquet(sample_path_y)
        logger.info(f"Saved preprocessing samples to {sample_path_X} and {sample_path_y}")

    logger.info("Preprocessing completed.")
    return X_encoded, y


# Step 4: Data Splitting
def data_splitting_helper(X_encoded, y):
    logger.info("Starting data splitting...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_test_val(
        X_encoded, y,
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE
    )

    log_dataset_shapes(X_train, X_val, X_test, y_train, y_val, y_test)
    calculate_statistics(y_train, "Train Set")
    calculate_statistics(y_val, "Validation Set")
    calculate_statistics(y_test, "Test Set")

    if config.SAVE_VISUALIZATION_SAMPLES:
        sample_path = config.VISUALIZATION_DATA_PATH['data_splitting']
        X_train.head(100).to_parquet(sample_path)
        logger.info(f"Saved data splitting sample to {sample_path}")

    logger.info("Data splitting completed.")
    return X_train, X_val, X_test, y_train, y_val, y_test


# Step 5: Clustering
def clustering_helper(X_train, X_val, X_test, n_clusters=100):
    logger.info("Starting clustering...")
    X_train, X_val, X_test = perform_clustering(X_train, X_val, X_test, n_clusters=config.N_CLUSTERS)

    if config.SAVE_VISUALIZATION_SAMPLES:
        sample_path_train = config.VISUALIZATION_DATA_PATH['clustering_train']
        sample_path_val = config.VISUALIZATION_DATA_PATH['clustering_val']
        sample_path_test = config.VISUALIZATION_DATA_PATH['clustering_test']
        X_train.head(100).to_parquet(sample_path_train)
        X_val.head(100).to_parquet(sample_path_val)
        X_test.head(100).to_parquet(sample_path_test)
        logger.info(f"Saved clustering samples to {sample_path_train}, {sample_path_val}, and {sample_path_test}")

    logger.info("Clustering completed.")
    return X_train, X_val, X_test


# Main Pipeline
def main():
    logger.info("Starting the ML pipeline...")

    # Define the pipeline steps and their corresponding functions
    pipeline_steps = {
        'data_loading': data_loading_helper,
        'feature_engineering': feature_engineering_helper,
        'preprocessing': preprocessing_helper,
        'data_splitting': data_splitting_helper,
        'clustering': clustering_helper,
        # Add more steps as needed
    }

    # Data containers
    contributions_df = None
    features_df = None
    X_encoded = None
    y = None
    X_train = X_val = X_test = y_train = y_val = y_test = None

    # Execute each step in the defined order
    for step_name, step_function in pipeline_steps.items():
        logger.info(f"\nExecuting pipeline step: {step_name}")
        if step_name == 'data_loading':
            contributions_df = step_function()
        elif step_name == 'feature_engineering':
            features_df = step_function(contributions_df)
        elif step_name == 'preprocessing':
            X_encoded, y = step_function(features_df)
        elif step_name == 'data_splitting':
            X_train, X_val, X_test, y_train, y_val, y_test = step_function(X_encoded, y)
        elif step_name == 'clustering':
            X_train, X_val, X_test = step_function(X_train, X_val, X_test)
        else:
            logger.warning(f"Unknown pipeline step: {step_name}")

    logger.info("ML pipeline completed successfully.")


if __name__ == '__main__':
    main()
