# src/data_loading.py

import pandas as pd

from config import logger
from src import config


def load_data(print_sample_data=False):
    data_path = config.CONTRIBUTIONS_DATA_FILE
    logger.info(f"Loading data from {data_path}...")
    contribution_df = pd.read_parquet(data_path, engine='pyarrow')

    # Print sample data if requested
    if print_sample_data:
        logger.info(f"Sample Data:\n{contribution_df.head()}")

    return contribution_df
