# src/preprocessing.py

from sklearn.preprocessing import MultiLabelBinarizer

from config import logger, DATASET_TYPE, SHOULD_INCLUDE_USERFEATURES, SHOULD_INCLUDE_OSM_ELEMENT_FEATURES, SPLIT_METHOD
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
    # logger.info(f"Encoding multi-label column '{column_name}' with prefix '{prefix}'")
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


import struct
import pandas as pd


def preprocess_user_features(df):
    """
    Preprocess user-related features in the DataFrame.

    Parameters:
    - df: The input DataFrame.

    Returns:
    - df: The DataFrame with preprocessed user-related features.
    """
    # Convert timestamps to datetime, handling NULL values
    for column in ['user_previous_edit_timestamp']:
        df[column] = pd.to_datetime(df[column], format='%d/%m/%Y %H:%M', errors='coerce')

    # Convert datetime to numerical values (e.g., Unix timestamp)
    df['user_previous_edit_timestamp'] = df['user_previous_edit_timestamp'].apply(
        lambda x: x.timestamp() if not pd.isnull(x) else -1
    )

    return df


def preprocess_osm_element_features(df):
    """
    Preprocess OSM element-related features in the DataFrame.

    Parameters:
    - df: The input DataFrame.

    Returns:
    - df: The DataFrame with preprocessed OSM element-related features.
    """
    # Convert timestamps to datetime, handling NULL values
    df['element_previous_edit_timestamp'] = pd.to_datetime(
        df['element_previous_edit_timestamp'], format='%d/%m/%Y %H:%M', errors='coerce'
    )

    # Convert datetime to numerical values (e.g., Unix timestamp)
    df['element_previous_edit_timestamp'] = df['element_previous_edit_timestamp'].apply(
        lambda x: x.timestamp() if not pd.isnull(x) else -1
    )

    # Function to decode binary time format into total seconds
    def decode_binary_time(binary_data):
        if binary_data is None:  # Handle missing data
            return -1
        try:
            # Unpack the binary data
            months = struct.unpack('<I', binary_data[0:4])[0]
            days = struct.unpack('<I', binary_data[4:8])[0]
            milliseconds = struct.unpack('<I', binary_data[8:12])[0]

            # Convert to total seconds
            total_seconds = (months * 30 * 24 * 3600) + (days * 24 * 3600) + (milliseconds / 1000)
            return total_seconds
        except Exception as e:
            print(f"Error decoding value {binary_data}: {e}")
            return -1

    # Apply the decoding function to both columns
    df['element_time_since_previous_edit'] = df['element_time_since_previous_edit'].apply(decode_binary_time)
    df['user_time_since_previous_edit'] = df['user_time_since_previous_edit'].apply(decode_binary_time)

    return df


def preprocess_contribution_features(features_df, is_training):
    logger.info("Starting preprocessing of contribution features...")

    if SHOULD_INCLUDE_OSM_ELEMENT_FEATURES:
        features_df = preprocess_osm_element_features(features_df)

    if SHOULD_INCLUDE_USERFEATURES:
        features_df = preprocess_user_features(features_df)

    # Shuffle the data entries
    features_df = features_df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)

    # Handle 'xzcode' column
    if 'xzcode' in features_df.columns:
        # Split 'xzcode' column into two separate columns 'code' and 'level'
        xzcode_df = pd.json_normalize(features_df['xzcode'])
        features_df[['code', 'level']] = xzcode_df[['code', 'level']]
        features_df.drop('xzcode', axis=1, inplace=True)

    # if 'changeset_id' in features_df.columns and DATASET_TYPE == 'contribution':
    #     features_df.drop('changeset_id', axis=1, inplace=True)
    #     logger.info(f"Dropped column: changeset_id")

    # Drop other unnecessary columns
    columns_to_drop = ['geometry', 'code', 'osm_id', 'members', 'status', 'editor_used',
                       'source_used', 'grid_cell_id', 'contribution_key']
    if SPLIT_METHOD != 'temporal':
        columns_to_drop.append("date_created")
    existing_columns_to_drop = [col for col in columns_to_drop if col in features_df.columns]
    features_df.drop(existing_columns_to_drop, axis=1, inplace=True)
    logger.info(f"Dropped columns: {existing_columns_to_drop}")

    # Replace spaces in column names with underscores
    features_df.columns = features_df.columns.str.replace(' ', '_', regex=True)

    # Split into features and target
    if is_training:
        X = features_df.drop('vandalism', axis=1).copy()
        y = features_df['vandalism'].copy()
    else:
        X = features_df
        y = None

    # One-hot encode 'countries' if it exists
    if 'countries' in X.columns:
        X = encode_multilabel_column(X, 'countries', 'country')

    # One-hot encode 'continents' if it exists
    if 'continents' in X.columns:
        X = encode_multilabel_column(X, 'continents', 'continent')

    # List of categorical columns to one-hot encode
    categorical_columns = ['osm_type', 'contribution_type', 'geometry_type',
                           'time_of_day']

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


def preprocess_features(features_df, is_training):
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

    return preprocess_contribution_features(features_df, is_training)
