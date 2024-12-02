# hyper_classifier/data_splitting.py
import pandas as pd

from config import logger
from data_splitting import log_dataset_shapes, calculate_statistics


def split_data(changeset_features, changeset_labels, split_ids):
    """
    Split data into training, validation, and test sets using the provided split IDs,
    and reorder them according to the order in split_ids.
    """
    logger.info("Splitting data into training, validation, and test sets using provided split IDs...")

    # Merge features and labels
    data = changeset_features.merge(changeset_labels, on='changeset_id', how='inner')

    # Set 'changeset_id' as the index
    data.set_index('changeset_id', inplace=True)

    # Prepare features and target
    X = data.drop(columns=['vandalism'])
    y = data['vandalism']

    # Reindex X and y according to split_ids to ensure correct order
    X_train = X.loc[split_ids['train']]
    y_train = y.loc[split_ids['train']]

    X_val = X.loc[split_ids['val']]
    y_val = y.loc[split_ids['val']]

    X_test = X.loc[split_ids['test']]
    y_test = y.loc[split_ids['test']]

    # Reset index to turn 'changeset_id' back into a column
    X_train = X_train.reset_index()
    X_val = X_val.reset_index()
    X_test = X_test.reset_index()

    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    logger.info(f"Data split completed:")
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Validation set size: {len(X_val)}")
    logger.info(f"Test set size: {len(X_test)}")

    # Log dataset shapes and statistics
    log_dataset_shapes(X_train, X_val, X_test, y_train, y_val, y_test)
    calculate_statistics(y_train, "Train Set")
    calculate_statistics(y_val, "Validation Set")
    calculate_statistics(y_test, "Test Set")

    return X_train, X_val, X_test, y_train, y_val, y_test
