# src/data_splitting.py

# src/data_splitting.py

import pandas as pd
from sklearn.model_selection import train_test_split

import src.config as config
from config import DATASET_TYPE, REAL_VANDAL_RATIO
from src.config import logger


def split_train_test_val(X_encoded, y, split_type='random', **kwargs):
    """
    Splits the data into training, validation, and test sets according to the split_type.

    Parameters:
    - X_encoded: Encoded feature DataFrame.
    - y: Target labels.
    - split_type: Type of split to perform ('random', 'geographic', 'temporal').
    - **kwargs: Additional parameters for specific split types.

    Returns:
    - X_train, X_val, X_test, y_train, y_val, y_test
    """
    if split_type == 'random':
        # return random_split(X_encoded, y, **kwargs)
        return create_realistic_splits(
            X=X_encoded,  # Features DataFrame
            y=y,  # Labels Series
            val_size=50000,  # Total validation set size
            test_size=50000,  # Total test set size
            vandal_ratio=REAL_VANDAL_RATIO,  # Real-world vandalism ratio (0.4%)
            random_state=42  # Random seed for reproducibility
        )
    elif split_type == 'geographic':
        return geographic_split(X_encoded, y, **kwargs)
    elif split_type == 'temporal':
        return temporal_split(X_encoded, y, **kwargs)
    else:
        raise ValueError(f"Unknown split_type: {split_type}")


def create_realistic_splits(X, y,
                            val_size=50000,
                            test_size=50000,
                            vandal_ratio=0.004,
                            random_state=42):
    """
    Create a validation set and test set that mirrors the real-world vandalism ratio (0.4%).
    The rest is returned as the training set.
    """
    # Combine X and y for easier slicing
    data = X.copy()
    data['label'] = y.values

    # Separate vandalism vs non-vandalism indices
    vandal_indices = data[data['label'] == 1].index
    non_vandal_indices = data[data['label'] == 0].index

    # Shuffle them
    vandal_indices = vandal_indices.to_series().sample(frac=1.0, random_state=random_state)
    non_vandal_indices = non_vandal_indices.to_series().sample(frac=1.0, random_state=random_state)

    # Number of vandal vs non-vandal needed
    vandal_val_needed = int(val_size * vandal_ratio)
    vandal_test_needed = int(test_size * vandal_ratio)

    non_vandal_val_needed = val_size - vandal_val_needed
    non_vandal_test_needed = test_size - vandal_test_needed

    # Construct validation set
    val_vandal_idx = vandal_indices[:vandal_val_needed]
    val_non_vandal_idx = non_vandal_indices[:non_vandal_val_needed]
    val_indices = pd.concat([val_vandal_idx, val_non_vandal_idx])

    # Construct test set
    test_vandal_idx = vandal_indices[vandal_val_needed: vandal_val_needed + vandal_test_needed]
    test_non_vandal_idx = non_vandal_indices[non_vandal_val_needed: non_vandal_val_needed + non_vandal_test_needed]
    test_indices = pd.concat([test_vandal_idx, test_non_vandal_idx])

    # The rest is training
    used_indices = pd.concat([val_indices, test_indices])
    train_indices = data.index.difference(used_indices)

    # Build final data splits
    X_val = data.loc[val_indices].drop(columns=['label'])
    y_val = data.loc[val_indices, 'label']

    X_test = data.loc[test_indices].drop(columns=['label'])
    y_test = data.loc[test_indices, 'label']

    X_train = data.loc[train_indices].drop(columns=['label'])
    y_train = data.loc[train_indices, 'label']

    print(f"Val set size: {len(X_val)} (vandal: {y_val.sum()})")
    print(f"Test set size: {len(X_test)} (vandal: {y_test.sum()})")
    print(f"Train set size: {len(X_train)} (vandal: {y_train.sum()})")

    X_test_meta = y_test_meta = None
    return X_train, X_val, X_test, X_test_meta, y_train, y_val, y_test, y_test_meta


def random_split(X_encoded, y, test_size=0.4, val_size=0.2, random_state=42):
    """
    Performs a random split of the data and saves the validation dataset for explainability.

    Returns:
    - X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info("Performing random train and temporary set split...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_encoded, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info("Splitting temporary set into validation and test sets...")
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size),
        random_state=random_state,
        stratify=y_temp
    )

    X_test_meta = y_test_meta = None
    if DATASET_TYPE == 'changeset':
        logger.info("Splitting test set into meta test and test sets...")
        X_test, X_test_meta, y_test, y_test_meta = train_test_split(
            X_temp, y_temp,
            test_size=config.META_TEST_SIZE,
            random_state=random_state,
            stratify=y_temp
        )

    # Save validation dataset for explainability
    logger.info(f"Saving validation dataset to {config.VALIDATION_DATASET_PATH}")
    X_val.to_parquet(config.VALIDATION_DATASET_PATH)
    y_val.to_frame(name='label').to_parquet(config.VALIDATION_LABELS_PATH)  # Save labels as a DataFrame

    logger.info("Validation dataset saved successfully.")

    return X_train, X_val, X_test, X_test_meta, y_train, y_val, y_test, y_test_meta


def temporal_split(X_encoded, y, date_column='timestamp'):
    """
    Splits the data into training, validation, and test sets based on years.

    Uses TRAIN_YEARS, VAL_YEARS, TEST_YEARS from config.py

    Returns:
    - X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Ensure date_column exists
    if date_column not in X_encoded.columns:
        raise ValueError(f"Date column '{date_column}' not found in X_encoded.")

    logger.info(f"Converting '{date_column}' to datetime...")
    X_encoded[date_column] = pd.to_datetime(X_encoded[date_column], errors='coerce')

    # Drop rows with invalid dates
    X_encoded = X_encoded.dropna(subset=[date_column])
    y = y.loc[X_encoded.index]

    # Extract year
    X_encoded['year'] = X_encoded[date_column].dt.year
    # X_encoded.drop('date_column')

    TRAIN_YEARS = config.TRAIN_YEARS
    VAL_YEARS = config.VAL_YEARS
    TEST_YEARS = config.TEST_YEARS

    logger.info("Splitting data based on years...")

    train_mask = X_encoded['year'].isin(TRAIN_YEARS)
    val_mask = X_encoded['year'].isin(VAL_YEARS)
    test_mask = X_encoded['year'].isin(TEST_YEARS)

    X_encoded.drop(date_column, axis=1, inplace=True)
    X_train = X_encoded[train_mask].drop(columns=['year'])
    y_train = y[train_mask]

    X_val = X_encoded[val_mask].drop(columns=['year'])
    y_val = y[val_mask]

    X_test = X_encoded[test_mask].drop(columns=['year'])
    y_test = y[test_mask]

    return X_train, X_val, X_test, None, y_train, y_val, y_test, None


def geographic_split(X_encoded, y, split_key='continent', train_regions=None, val_regions=None, test_regions=None):
    """
    Splits the data into training, validation, and test sets based on geographic regions.

    Parameters:
    - split_key: 'continent' or 'country'
    - train_regions, val_regions, test_regions: Lists of regions (e.g., ['Asia', 'Europe'])

    Returns:
    - X_train, X_val, X_test, y_train, y_val, y_test
    """
    if split_key not in ['continent', 'country']:
        raise ValueError("split_key must be 'continent' or 'country'")

    # Get list of binary columns corresponding to the split_key
    binary_columns = [col for col in X_encoded.columns if col.startswith(f"{split_key}_")]

    if not binary_columns:
        raise ValueError(f"No columns starting with '{split_key}_' found in X_encoded.")

    # Helper function to create mask for regions
    def get_region_mask(regions):
        masks = []
        for region in regions:
            col_name = f"{split_key}_{region}"
            if col_name not in X_encoded.columns:
                logger.warning(f"Column '{col_name}' not found in X_encoded.")
                continue
            masks.append(X_encoded[col_name] == 1)
        if masks:
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask |= mask
            return combined_mask
        else:
            raise ValueError(f"No valid regions found in {regions}")

    # Get masks for train, val, test
    train_mask = get_region_mask(train_regions)
    val_mask = get_region_mask(val_regions)
    test_mask = get_region_mask(test_regions)

    X_train = X_encoded[train_mask]
    y_train = y[train_mask]

    X_val = X_encoded[val_mask]
    y_val = y[val_mask]

    X_test = X_encoded[test_mask]
    y_test = y[test_mask]

    return X_train, X_val, X_test, None, y_train, y_val, y_test, None


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


def log_dataset_shapes(X_train, X_val, X_test, X_test_meta, y_train, y_val, y_test, y_test_meta):
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
        'y_test shape': y_test.shape,
    }

    if DATASET_TYPE == 'changeset':
        shapes['X_test_meta shape'] = X_test_meta.shape
        shapes['y_test_meta shape'] = y_test_meta.shape

    shapes_df = pd.DataFrame(shapes, index=['Number of Samples', 'Number of Features']).T
    logger.info(f"Dataset Shapes:\n{shapes_df}")
