# scripts/run_feature_engineering.py

import os
import sys
import time

# Add the parent directory to sys.path to import modules from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loading import load_data
from src.feature_engineering import get_or_generate_features
from src.logger_config import logger

# Load the contribution data
contributions_df = load_data(print_sample_data=False)

# Sleep for 2 seconds to print the statistics of loaded data.
time.sleep(2)

# Extract features for all contributions and store them in a DataFrame
features_df = get_or_generate_features(contributions_df, force_compute_features=False, test_mode=True)

logger.info("Feature extraction completed.")
