# scripts/run_feature_engineering.py

import os
import sys
import time

# Adjust the path to import modules from src
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.data_loading import load_data
from src.feature_engineering import get_or_generate_features
from src.logger_config import logger


def run_feature_engineering(contributions_df=None):
    """
    Perform feature engineering.

    Parameters:
    - contributions_df (pd.DataFrame): Contribution data. If None, load from file.

    Returns:
    - features_df (pd.DataFrame): Extracted features.
    """
    if contributions_df is None:
        # Load the contribution data
        contributions_df = load_data(print_sample_data=False)

    # Sleep for 2 seconds to print the statistics of loaded data.
    time.sleep(2)

    # Extract features
    features_df = get_or_generate_features(
        contributions_df,
        force_compute_features=False,
        test_mode=True,
    )
    logger.info("Feature engineering completed.")
    return features_df


if __name__ == '__main__':
    # Run independently
    features_df = run_feature_engineering()
    # # Optionally save features to disk
    # if config.SAVE_FEATURES:
    #     features_file = config.FEATURES_FILE_PATH
    #     features_df.to_parquet(features_file)
    #     logger.info(f"Features saved to {features_file}.")
