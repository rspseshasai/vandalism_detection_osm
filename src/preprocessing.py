# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from config import logger, DATASET_TYPE
from src import config


def encode_multilabel_column(df, column_name, prefix):
    """
    One-hot encodes a multi-label column using MultiLabelBinarizer.

    Parameters:
    - df: The DataFrame containing the column to encode.
    - column_name: The name of the multi-label column to encode.
    - prefix: The prefix for the encoded columns.

    Returns:
    - Updated DataFrame with the original column replaced by one-hot encoded columns.
    """
    logger.info(f"Encoding multi-label column '{column_name}' with prefix '{prefix}'")
    # Initialize the MultiLabelBinarizer
    mlb = MultiLabelBinarizer()

    # Fit and transform the column
    encoded_data = mlb.fit_transform(df[column_name])

    # Create a DataFrame with the encoded features
    encoded_df = pd.DataFrame(encoded_data, columns=[f'{prefix}_{label}' for label in mlb.classes_])

    # Reset index to align with df
    encoded_df.index = df.index

    # Drop the original column
    df = df.drop(column_name, axis=1)

    # Concatenate the new features with the original DataFrame
    df = pd.concat([df, encoded_df], axis=1)

    return df


def preprocess_changeset_features(features_df):
    logger.info("Starting preprocessing of changeset features...")

    # Drop unnecessary columns
    columns_to_drop = ['geometry', 'created_at', 'user', 'comment',
                       'uid', 'changes_count']

    existing_columns_to_drop = [col for col in columns_to_drop if col in features_df.columns]
    features_df = features_df.drop(existing_columns_to_drop, axis=1)
    logger.info(f"Dropped columns: {existing_columns_to_drop}")

    X = features_df.drop('label', axis=1).copy()
    y = features_df['label'].copy()

    X['closed_at'] = pd.to_datetime(X['closed_at']).astype(int) / 10 ** 9
    X['account_created'] = pd.to_datetime(X['account_created']).astype(int) / 10 ** 9

    X_encoded = pd.get_dummies(X, columns=['created_by'])
    logger.info("Performed one-hot encoding on categorical columns.")

    # Ensure no object-type columns remain
    object_columns = X_encoded.select_dtypes(include=['object']).columns
    if object_columns.any():
        logger.error(f"There are still object-type columns: {list(object_columns)}")
        raise ValueError(f"There are still object-type columns: {list(object_columns)}")

    return X_encoded, y


def preprocess_contribution_features(features_df):
    logger.info("Starting preprocessing of contribution features...")

    # Shuffle the data entries
    features_df = features_df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
    logger.info("Shuffled the features DataFrame.")

    # Handle 'xzcode' column
    if 'xzcode' in features_df.columns:
        # Split 'xzcode' column into two separate columns 'code' and 'level'
        xzcode_df = pd.json_normalize(features_df['xzcode'])
        features_df[['code', 'level']] = xzcode_df[['code', 'level']]
        features_df.drop('xzcode', axis=1, inplace=True)
        logger.info("Processed 'xzcode' column.")

    # Drop unnecessary columns
    columns_to_drop = ['geometry', 'osm_id', 'members', 'status', 'editor_used',
                       'source_used', 'grid_cell_id']
    existing_columns_to_drop = [col for col in columns_to_drop if col in features_df.columns]
    features_df.drop(existing_columns_to_drop, axis=1, inplace=True)
    logger.info(f"Dropped columns: {existing_columns_to_drop}")

    # Replace spaces in column names with underscores
    features_df.columns = features_df.columns.str.replace(' ', '_', regex=True)

    # Split into features and target
    X = features_df.drop('vandalism', axis=1).copy()
    y = features_df['vandalism'].copy()

    # One-hot encode 'countries' if it exists
    if 'countries' in X.columns:
        X = encode_multilabel_column(X, 'countries', 'country')

    # One-hot encode 'continents' if it exists
    if 'continents' in X.columns:
        X = encode_multilabel_column(X, 'continents', 'continent')

    # List of categorical columns to one-hot encode
    categorical_columns = ['osm_type', 'contribution_type', 'geometry_type',
                           'edit_frequency_of_osm_element', 'time_of_day']

    # Perform one-hot encoding
    X_encoded = pd.get_dummies(X, columns=categorical_columns)
    logger.info("Performed one-hot encoding on categorical columns.")

    # Ensure no object-type columns remain
    object_columns = X_encoded.select_dtypes(include=['object']).columns
    if object_columns.any():
        logger.error(f"There are still object-type columns: {list(object_columns)}")
        raise ValueError(f"There are still object-type columns: {list(object_columns)}")

    logger.info("Preprocessing completed successfully.")
    return X_encoded, y


def preprocess_features(features_df):
    """
    Preprocess the features DataFrame for ML training.

    Parameters:
    - features_df: The raw features DataFrame.

    Returns:
    - X_encoded: The preprocessed and encoded feature DataFrame.
    - y: The target labels.
    """

    if DATASET_TYPE == 'changeset':
        return preprocess_changeset_features(features_df)

    return preprocess_contribution_features(features_df)
