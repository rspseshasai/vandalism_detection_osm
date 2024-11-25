# src/data_splitting.py

import pandas as pd
from sklearn.model_selection import train_test_split

from config import logger


def split_train_test_val(X_encoded, y, test_size=0.4, val_size=0.2, random_state=42):
    """
    Splits the data into training, validation, and test sets.

    Parameters:
    - X_encoded: Encoded feature DataFrame.
    - y: Target labels.
    - test_size: Proportion of the dataset to include in the temporary test set.
    - val_size: Proportion of the temporary test set to include in the final test set.
    - random_state: Random seed for reproducibility.

    Returns:
    - X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: Training set and temporary set
    logger.info("Performing train and temporary set split...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: Validation set and test set
    logger.info("Splitting temporary set into validation and test sets...")
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_size), random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_statistics(y, set_name):
    """
    Calculate and log statistics for a given dataset.

    Parameters:
    - y: Target labels.
    - set_name: Name of the dataset (e.g., Train, Validation, Test).
    """
    total = len(y)
    vandalism = sum(y)  # Assuming 'vandalism' is labeled as 1
    non_vandalism = total - vandalism
    ratio = vandalism / total if total > 0 else 0

    logger.info(
        f"{set_name} Statistics:\n"
        f"Total Samples: {total}\n"
        f"Vandalism: {vandalism}\n"
        f"Non-Vandalism: {non_vandalism}\n"
        f"Vandalism Ratio: {ratio:.4f}\n"
    )


def log_dataset_shapes(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Log the shapes of the datasets.

    Parameters:
    - X_train, X_val, X_test: Feature sets.
    - y_train, y_val, y_test: Target labels.
    """
    shapes = {
        'X_train shape': X_train.shape,
        'X_val shape': X_val.shape,
        'X_test shape': X_test.shape,
        'y_train shape': y_train.shape,
        'y_val shape': y_val.shape,
        'y_test shape': y_test.shape
    }

    shapes_df = pd.DataFrame(shapes, index=['Number of Samples', 'Number of Features']).T
    logger.info(f"Dataset Shapes:\n{shapes_df}")
