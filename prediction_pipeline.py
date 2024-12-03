import os
import sys

import joblib
import pandas as pd

import config
from data_loading import load_data
from training import load_model

# Adjust the path to import modules from src
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_dir, 'src'))

from config import logger, FINAL_MODEL_PATH
from src.feature_engineering import get_or_generate_features
from src.preprocessing import preprocess_features


def main():
    logger.info("Starting prediction on unlabelled contributions data...")

    # Step 1: Load Unlabelled Data
    data_df = load_data(data_path=config.UNLABELLED_RAW_DATA_FILE, print_sample_data=True)

    # Step 2: Feature Engineering
    features_df = get_or_generate_features(
        data_df,
        False,
        config.UNLABELLED_PROCESSED_FEATURES_FILE,
        force_compute_features=False,
        test_mode=False
    )

    # Step 3: Preprocessing
    X_encoded, _ = preprocess_features(features_df, is_training=False)

    # Step 4: Load Trained Model
    model = load_model(FINAL_MODEL_PATH)

    # Step 5: Load Clustering Model and Assign Cluster Labels
    clustering_model = joblib.load('kmeans_clustering_model.pkl')
    logger.info("Loaded clustering model from 'kmeans_clustering_model.pkl'.")

    # Ensure 'centroid_x' and 'centroid_y' are available in X_encoded
    required_columns = ['centroid_x', 'centroid_y']
    for col in required_columns:
        if col not in X_encoded.columns:
            logger.error(f"Column '{col}' not found in features.")
            raise KeyError(f"Column '{col}' not found in features.")

    centroids = X_encoded[required_columns].values
    cluster_labels = clustering_model.predict(centroids)
    X_encoded = X_encoded.copy()
    X_encoded['cluster_label'] = cluster_labels
    logger.info("Cluster labels assigned to unlabelled data.")

    # Step 6: Make Predictions
    y_pred = model.predict(X_encoded)
    y_prob = model.predict_proba(X_encoded)[:, 1]

    # Step 7: Print Statistics
    total_entries = len(y_pred)
    num_vandalism = sum(y_pred)
    num_non_vandalism = total_entries - num_vandalism
    logger.info(f"Total entries: {total_entries}")
    logger.info(f"Predicted Vandalism: {num_vandalism}")
    logger.info(f"Predicted Non-Vandalism: {num_non_vandalism}")

    # Step 8: Save Predictions
    predictions_df = pd.DataFrame({
        'changeset_id': features_df['changeset_id'],  # Assuming 'changeset_id' is available
        'y_pred': y_pred,
        'y_prob': y_prob
    })
    predictions_df.to_csv('unlabelled_predictions.csv', index=False)
    logger.info("Predictions saved to 'unlabelled_predictions.csv'")

    logger.info("Prediction on unlabelled contributions data completed.")


if __name__ == '__main__':
    main()