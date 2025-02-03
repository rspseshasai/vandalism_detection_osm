# src/data_loading.py
import numpy as np
import pandas as pd

from config import logger
from src import config


def load_data(data_path, print_sample_data=False):
    logger.info(f"Loading data from {data_path}...")
    contribution_df = pd.read_parquet(data_path, engine='pyarrow')

    # Print sample data if requested
    if print_sample_data:
        logger.info(f"Sample Data:\n{contribution_df.head()}")

    # Normalize 'vandalism' column to 0/1
    if 'vandalism' in contribution_df.columns:
        if contribution_df['vandalism'].dtype == bool:
            logger.info("Normalizing 'vandalism' column from True/False to 0/1.")
            contribution_df['vandalism'] = contribution_df['vandalism'].astype(int)
        elif set(contribution_df['vandalism'].unique()).issubset({0, 1}):
            logger.info("'vandalism' column already in 0/1 format.")
        else:
            logger.error("Unexpected values in 'vandalism' column.")
            raise ValueError("The 'vandalism' column must contain only True/False or 0/1 values.")

    # Check if balancing is required
    if config.DATASET_TYPE == 'contribution' and config.SHOULD_BALANCE_DATASET:
        logger.info("Balancing the dataset as per configuration...")

        # Get class counts
        class_counts = contribution_df['vandalism'].value_counts()
        vandalism_count = class_counts.get(1, 0)
        non_vandalism_count = class_counts.get(0, 0)

        logger.info(f"Original Class Distribution - Vandalism: {vandalism_count}, Non-Vandalism: {non_vandalism_count}")

        # Check for class imbalance and balance the dataset if needed
        if vandalism_count != non_vandalism_count:
            logger.info("Classes are imbalanced. Balancing the dataset...")

            # Determine majority and minority classes
            if vandalism_count > non_vandalism_count:
                majority_class = 1
                minority_class = 0
                majority_count = vandalism_count
                minority_count = non_vandalism_count
            else:
                majority_class = 0
                minority_class = 1
                majority_count = non_vandalism_count
                minority_count = vandalism_count

            # Randomly sample the majority class to match the minority class count
            majority_indices = contribution_df[contribution_df['vandalism'] == majority_class].index
            minority_indices = contribution_df[contribution_df['vandalism'] == minority_class].index

            sampled_majority_indices = np.random.choice(
                majority_indices,
                size=minority_count,
                replace=False
            )

            # Combine the sampled majority and minority indices
            balanced_indices = np.concatenate([sampled_majority_indices, minority_indices])

            # Create the balanced DataFrame
            contribution_df = contribution_df.loc[balanced_indices].reset_index(drop=True)

            logger.info(f"Dataset balanced: {minority_count} entries for each class.")
        else:
            logger.info("Classes are already balanced. No balancing needed.")

    return contribution_df
