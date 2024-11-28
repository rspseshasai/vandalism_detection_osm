import os

from scipy.stats import skew, kurtosis

from config import logger, SAVE_VISUALIZATION_SAMPLES, HYPER_VISUALIZATION_DIR, TEST_RUN


def engineer_features(contributions_df):
    """
    Aggregate contribution-level predictions into changeset-level features.
    """
    logger.info("Aggregating contribution-level predictions into changeset-level features...")

    # Compute aggregated features
    aggregate_features = contributions_df.groupby('changeset_id').agg(
        num_contributions=('predicted_prob', 'count'),
        mean_prediction=('predicted_prob', 'mean'),
        median_prediction=('predicted_prob', 'median'),
        std_prediction=('predicted_prob', 'std'),
        min_prediction=('predicted_prob', 'min'),
        max_prediction=('predicted_prob', 'max'),
        quantile_25=('predicted_prob', lambda x: x.quantile(0.25)),
        quantile_75=('predicted_prob', lambda x: x.quantile(0.75)),
        skewness_prediction=('predicted_prob', lambda x: skew(x) if len(x) > 1 else 0),
        kurtosis_prediction=('predicted_prob', lambda x: kurtosis(x, fisher=True) if len(x) > 1 else 0)
    ).reset_index()

    # Add proportion-based features
    aggregate_features['proportion_vandalism'] = (
        contributions_df.groupby('changeset_id')['predicted_prob']
        .apply(lambda x: (x > 0.5).sum() / len(x))
        .values
    )
    aggregate_features['num_vandalism_predictions'] = (
        contributions_df.groupby('changeset_id')['predicted_prob']
        .apply(lambda x: (x > 0.5).sum())
        .values
    )

    # Add binary indicators
    aggregate_features['any_vandalism_prediction'] = (
        contributions_df.groupby('changeset_id')['predicted_prob']
        .apply(lambda x: int((x > 0.5).any()))
        .values
    )
    aggregate_features['all_vandalism_predictions'] = (
        contributions_df.groupby('changeset_id')['predicted_prob']
        .apply(lambda x: int((x > 0.5).all()))
        .values
    )

    # Handle NaN values
    aggregate_features.fillna(0, inplace=True)

    # Save visualization samples if enabled
    if SAVE_VISUALIZATION_SAMPLES:
        sample_path = os.path.join(HYPER_VISUALIZATION_DIR, 'aggregated_features_sample.parquet')
        aggregate_features.head(100).to_parquet(sample_path)
        logger.info(f"Saved aggregated features sample to {sample_path}")

    logger.info("Feature aggregation completed.")

    return aggregate_features
