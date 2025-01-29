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

    # Function to check for missing changeset_ids
    def check_missing_ids(split_name, split_list, dataset_index):
        missing_ids = set(split_list) - set(dataset_index)
        if missing_ids:
            logger.error(
                f"Mismatch detected: {len(missing_ids)} changeset IDs in the {split_name} set of the changeset dataset "
                f"are missing from the aggregated contribution-based changeset features."
            )
            raise KeyError(
                f"Total {len(missing_ids)} changeset IDs from the {split_name} split of the changeset dataset do not exist in the contribution-based "
                f"changeset feature dataset. This typically happens when the contribution and changeset datasets are not fully "
                f"aligned. Ensure that all changesets present in the changeset dataset have corresponding entries in the contribution "
                f"dataset before aggregation. Missing changeset IDs example: {list(missing_ids)[:10]}"
            )

    # Check for missing changeset IDs before indexing
    check_missing_ids("train", split_ids['train'], X.index)
    check_missing_ids("val", split_ids['val'], X.index)
    check_missing_ids("test", split_ids['test'], X.index)
    check_missing_ids("meta_test", split_ids['meta_test'], X.index)

    # Reindex X and y according to split_ids to ensure correct order
    X_train = X.loc[split_ids['train']]
    y_train = y.loc[split_ids['train']]

    X_val = X.loc[split_ids['val']]
    y_val = y.loc[split_ids['val']]

    X_test = X.loc[split_ids['test']]
    y_test = y.loc[split_ids['test']]

    X_test_meta = X.loc[split_ids['meta_test']]
    y_test_meta = y.loc[split_ids['meta_test']]

    # Reset index to turn 'changeset_id' back into a column
    X_train = X_train.reset_index()
    X_val = X_val.reset_index()
    X_test = X_test.reset_index()
    X_test_meta = X_test_meta.reset_index()

    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_test_meta = y_test_meta.reset_index(drop=True)

    logger.info(f"Data split completed successfully:")
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Validation set size: {len(X_val)}")
    logger.info(f"Test set size: {len(X_test)}")
    logger.info(f"Meta Test set size: {len(X_test_meta)}")

    # Log dataset shapes and statistics
    log_dataset_shapes(X_train, X_val, X_test, X_test_meta, y_train, y_val, y_test, y_test_meta)
    calculate_statistics(y_train, "Train Set")
    calculate_statistics(y_val, "Validation Set")
    calculate_statistics(y_test, "Test Set")
    calculate_statistics(y_test_meta, "Meta Test Set")

    return X_train, X_val, X_test, X_test_meta, y_train, y_val, y_test, y_test_meta
