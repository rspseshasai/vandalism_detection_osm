# hyper_classifier/data_splitting.py

from sklearn.model_selection import train_test_split

from config import logger, VAL_SIZE, TEST_SIZE


def split_data(changeset_features, changeset_labels, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=42):
    """
    Split data into training, validation, and test sets.
    """
    logger.info("Splitting data into training, validation, and test sets...")

    # Merge features and labels
    data = changeset_features.merge(changeset_labels, on='changeset_id', how='inner')

    # Prepare features and target
    X = data.drop(columns=['changeset_id', 'vandalism'])
    y = data['vandalism']

    # First split off the test set
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size),
        random_state=random_state,
        stratify=y_temp
    )
    logger.info(f"Data split completed:")
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Validation set size: {len(X_val)}")
    logger.info(f"Test set size: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test
