import pandas as pd
from tqdm import tqdm

from logger.logger_config import logger


def load(path, print_sample_data):
    tqdm.pandas()
    contribution_df = pd.read_parquet(path, engine='pyarrow')

    logger.info(f"Reading Parquet File Data...")

    # Counting the number of True (vandalism) and False (non-vandalism)
    if contribution_df.columns.__contains__('vandalism'):
        vandalism_count = contribution_df['vandalism'].value_counts()
        num_vandalism = vandalism_count.get(True, 0)
        num_non_vandalism = vandalism_count.get(False, 0)
        logger.info(f"Number of vandalism entries: {num_vandalism}")
        logger.info(f"Number of non-vandalism entries: {num_non_vandalism}")

    logger.info(f"Gathering the data...")

    for index, row in tqdm(contribution_df.iterrows(), total=contribution_df.shape[0], desc="Collecting the data"):
        pass

    logger.info(f"contribution_df.shape: {contribution_df.shape}")
    logger.info(f"Columns: {contribution_df.columns}")

    # Print 1 Contribution Data - 10th row
    if print_sample_data:
        for i in range(0, contribution_df.columns.size):
            logger.info(contribution_df.columns[i].__str__() + ": " + contribution_df.iloc[10][i].__str__() + "\n")

    return contribution_df
