# hyper_classifier/data_preparation.py

import os

import pandas as pd
from joblib import load

from config import CONTRIBUTION_FINAL_MODEL_PATH, CONTRIBUTION_PROCESSED_FEATURES_FILE, RAW_DATA_DIR, TEST_RUN
from config import logger
from src.preprocessing import preprocess_contribution_features


# hyper_classifier/data_preparation.py


def get_changeset_labels():
    """
    Load and preprocess changeset-level labels and contribution-level predictions.
    """
    logger.info("Loading changeset-level labels...")

    # Load changeset-level labels from the TSV file
    changeset_labels_file = os.path.join(RAW_DATA_DIR, 'changeset_labels.tsv')
    changeset_labels = pd.read_csv(changeset_labels_file, sep='\t')

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

    contributions_features_df = pd.read_parquet(CONTRIBUTION_PROCESSED_FEATURES_FILE)
    if TEST_RUN:
        logger.info("Test mode enabled: Limiting to 1000 entries.")
        contributions_features_df = contributions_features_df.head(1000)

    X_encoded, _ = preprocess_contribution_features(contributions_features_df)

    # Predict probabilities
    contributions_df = contributions_features_df.copy()
    contributions_df['predicted_prob'] = contribution_model.predict_proba(X_encoded)[:, 1]

    return contributions_df[['changeset_id', 'predicted_prob']]


def data_loading():
    return get_contribution_predictions(), get_changeset_labels()
