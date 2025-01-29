# hyper_classifier/data_preparation.py

import os

import pandas as pd
from joblib import load

from config import CONTRIBUTION_FINAL_MODEL_PATH, RAW_DATA_DIR, CONTRIBUTION_PROCESSED_ENCODED_FEATURES_FILE, \
    CHANGESET_LABELS_FILE, COMMON_CHANGESET_IDS
from config import logger


# hyper_classifier/data_preparation.py


def get_changeset_labels():
    """
    Load and preprocess changeset-level labels and contribution-level predictions.
    """
    logger.info("Loading changeset-level labels...")

    # Load changeset-level labels from the TSV file
    changeset_labels = pd.read_csv(CHANGESET_LABELS_FILE, sep='\t')

    # Rename columns to match pipeline conventions, if necessary
    changeset_labels.rename(columns={'changeset': 'changeset_id', 'label': 'vandalism'}, inplace=True)

    # Ensure required columns exist
    if 'changeset_id' not in changeset_labels.columns or 'vandalism' not in changeset_labels.columns:
        raise ValueError("Required columns 'changeset_id' or 'vandalism' are missing in changeset labels.")

    logger.info(f"Loaded {len(changeset_labels)} changeset-level labels.")

    return changeset_labels


def get_contribution_predictions():
    """
    Use the trained contribution-level model to predict vandalism probabilities for each contribution.
    """
    logger.info("Obtaining per-contribution predictions...")

    # Load the trained contribution-level model
    contribution_model = load(CONTRIBUTION_FINAL_MODEL_PATH)

    X_encoded = pd.read_parquet(CONTRIBUTION_PROCESSED_ENCODED_FEATURES_FILE)

    # Predict probabilities
    contributions_df = X_encoded.copy()
    contributions_df['predicted_prob'] = contribution_model.predict_proba(X_encoded)[:, 1]

    predictions_for_all_contributions_df = contributions_df[['changeset_id', 'predicted_prob']]

    logger.info("Limiting to contribution entries matching common changeset IDs - to maintain consistent dataset for hyper classifier that matches with changeset data set.")
    predictions_for_all_contributions_df = predictions_for_all_contributions_df[predictions_for_all_contributions_df['changeset_id'].isin(COMMON_CHANGESET_IDS)]

    return predictions_for_all_contributions_df


def data_loading():
    return get_contribution_predictions(), get_changeset_labels()
