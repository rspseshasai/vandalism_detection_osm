# src/data_loading.py

import pandas as pd
from tqdm import tqdm

from src import config
from src.logger_config import logger


def load_data(print_sample_data=False):
    """
    Load contribution data from a Parquet file.

    Parameters:
    - print_sample_data (bool): If True, prints a sample of the data.

    Returns:
    - pd.DataFrame: Loaded contribution data.
    """
    tqdm.pandas()
    data_path = config.CONTRIBUTIONS_DATA_FILE
    logger.info(f"Loading data from {data_path}...")

    contribution_df = pd.read_parquet(data_path, engine='pyarrow')

    # Counting the number of True (vandalism) and False (non-vandalism)
    if 'vandalism' in contribution_df.columns:
        vandalism_count = contribution_df['vandalism'].value_counts()
        num_vandalism = vandalism_count.get(True, 0)
        num_non_vandalism = vandalism_count.get(False, 0)
        print(f"Number of vandalism entries: {num_vandalism}")
        print(f"Number of non-vandalism entries: {num_non_vandalism}")

    print(f"Contribution DataFrame shape: {contribution_df.shape}")
    print(f"Columns: {contribution_df.columns.tolist()}")

    # Print a sample of the data
    if print_sample_data:
        sample_data = contribution_df.head(5)
        logger.info(f"Sample Data:\n{sample_data}")

    return contribution_df
