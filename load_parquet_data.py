import pandas as pd
from tqdm import tqdm

from logger_config import logger


def load(path, print_data):
    tqdm.pandas()
    contribution_df = pd.read_parquet(path, engine='pyarrow')
    # contribution_df = pd.read_parquet('data/contri_test_1.parquet', engine='pyarrow')

    logger.info(f"Reading Parquet File Data...")
    # Show progress while processing rows or other operations
    for index, row in tqdm(contribution_df.iterrows(), total=contribution_df.shape[0]):
        pass

    # Print column headers
    logger.info(f"contribution_df.shape: " + contribution_df.shape.__str__())
    logger.info(f"Columns: " + contribution_df.columns.__str__())

    # Print 1 Contribution Data - 10th row
    if print_data:
        for i in range(0, contribution_df.columns.size):
            print(contribution_df.columns[i].__str__() + ": " + contribution_df.iloc[10][i].__str__() + "\n")

    return contribution_df
