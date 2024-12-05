import os
import sys

import joblib
import pandas as pd
import pyarrow.dataset as ds  # For reading Parquet in batches

# Adjust the path to import modules from src
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_dir, 'src'))

import config
from config import logger, FINAL_MODEL_PATH
from training import load_model
from src.feature_engineering import get_or_generate_features
from src.preprocessing import preprocess_features


def load_parquet_in_chunks(data_path, batch_size=10000):
    """
    Load data from a Parquet file in batches using PyArrow Dataset.
    Each yielded chunk is a DataFrame with up to 'batch_size' rows.
    """
    dataset = ds.dataset(data_path, format="parquet")
    scanner = dataset.to_batches(batch_size=batch_size)
    for record_batch in scanner:
        yield record_batch.to_pandas()


def main():
    logger.info("Starting prediction on unlabeled contributions data from Parquet in chunks...")

    data_path = config.UNLABELLED_RAW_DATA_FILE  # Parquet file path
    batch_size = 10000  # Adjust based on memory constraints

    # Step 4: Load Trained Model once
    model = load_model(FINAL_MODEL_PATH)

    # Load Clustering Model once
    clustering_model_path = config.CLUSTER_MODEL_PATH
    if not os.path.exists(clustering_model_path):
        logger.error(f"Clustering model file not found at {clustering_model_path}")
        raise FileNotFoundError(f"Clustering model file not found at {clustering_model_path}")

    clustering_model = joblib.load(clustering_model_path)
    logger.info(f"Loaded clustering model from '{clustering_model_path}'.")

    required_columns = ['centroid_x', 'centroid_y']

    # Prepare counters and output file
    total_entries = 0
    num_vandalism = 0

    output_file = config.UNLABELLED_PROCESSED_OUTPUT_CSV_FILE

    # Write header only once
    with open(output_file, 'w') as f:
        f.write("changeset_id,y_pred,y_prob\n")

    # Process data in batches
    for i, chunk in enumerate(load_parquet_in_chunks(data_path, batch_size=batch_size)):
        logger.info(f"Processing chunk {i + 1}...")

        # Step 2: Feature Engineering for this chunk
        features_df = get_or_generate_features(
            chunk,
            False,
            config.UNLABELLED_PROCESSED_FEATURES_FILE,
            force_compute_features=True,
            test_mode=False
        )

        # Step 3: Preprocessing
        X_encoded, _ = preprocess_features(features_df, is_training=False)

        # Check clustering columns
        for col in required_columns:
            if col not in X_encoded.columns:
                logger.error(f"Column '{col}' not found in features.")
                raise KeyError(f"Column '{col}' not found in features.")

        # Assign cluster labels
        centroids = X_encoded[required_columns].values
        cluster_labels = clustering_model.predict(centroids)
        X_encoded = X_encoded.copy()
        X_encoded['cluster_label'] = cluster_labels

        # Step 6: Make Predictions for this chunk
        y_pred = model.predict(X_encoded)
        y_prob = model.predict_proba(X_encoded)[:, 1]

        # Update counters
        total_entries += len(y_pred)
        num_vandalism += sum(y_pred)

        # Step 8: Append Predictions
        predictions_df = pd.DataFrame({
            'changeset_id': features_df['changeset_id'],
            'y_pred': y_pred,
            'y_prob': y_prob
        })
        predictions_df.to_csv(output_file, mode='a', index=False, header=False)

        logger.info(f"Finished processing chunk {i + 1}. Total processed so far: {total_entries}")

    # After processing all batches, print final stats
    num_non_vandalism = total_entries - num_vandalism
    logger.info("Completed predictions on all unlabeled data.")
    logger.info(f"Total entries: {total_entries}")
    logger.info(f"Predicted Vandalism: {num_vandalism}")
    logger.info(f"Predicted Non-Vandalism: {num_non_vandalism}")
    logger.info(f"Predictions saved to '{output_file}'")


if __name__ == '__main__':
    main()
