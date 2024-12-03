# src/data_loading.py

import pandas as pd

from config import logger
from src import config


def load_data(data_path, print_sample_data=False):
    logger.info(f"Loading data from {data_path}...")
    contribution_df = pd.read_parquet(data_path, engine='pyarrow')

    # Print sample data if requested
    if print_sample_data:
        logger.info(f"Sample Data:\n{contribution_df.head()}")

    return contribution_df
