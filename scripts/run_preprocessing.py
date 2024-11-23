# scripts/run_preprocessing.py

import sys
import os

# Adjust the path to import modules from src
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.preprocessing import preprocess_features
from src.logger_config import logger
from src import config
import pandas as pd


def run_preprocessing(features_df=None):
    """
    Perform data preprocessing.

    Parameters:
    - features_df (pd.DataFrame): Features data. If None, load from file.

    Returns:
    - X_encoded (pd.DataFrame): Preprocessed features.
    - y (pd.Series): Target labels.
    """
    if features_df is None:
        # Load the features DataFrame
        features_file = config.FEATURES_FILE_PATH
        features_df = pd.read_parquet(features_file)

    # Preprocess features
    X_encoded, y = preprocess_features(features_df)
    logger.info("Preprocessing completed.")
    return X_encoded, y


if __name__ == '__main__':
    # Run independently
    X_encoded, y = run_preprocessing()
    # # Optionally save preprocessed data to disk
    # if config.SAVE_PREPROCESSED_DATA:
    #     X_file = config.PREPROCESSED_FEATURES_FILE_PATH
    #     y_file = config.TARGET_LABELS_FILE_PATH
    #     X_encoded.to_parquet(X_file)
    #     y.to_parquet(y_file)
    #     logger.info(f"Preprocessed data saved to {X_file} and {y_file}.")
