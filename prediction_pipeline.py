import os
import sys

import joblib
import pandas as pd
import pyarrow.dataset as ds

# Adjust the path to import modules from src
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_dir, 'src'))

from config import logger, FINAL_MODEL_PATH, UNLABELLED_RAW_DATA_FILE, CLUSTER_MODEL_PATH, \
    UNLABELLED_PROCESSED_OUTPUT_CSV_FILE, UNLABELLED_PROCESSED_FEATURES_FILE, FINAL_TRAINED_FEATURES_PATH
from training import load_model

# from src.feature_engineering import get_or_generate_features
from src.feature_engineering_parallel import get_or_generate_features

from src.preprocessing import preprocess_features


def load_parquet_in_chunks(data_path, batch_size=100000):
    dataset = ds.dataset(data_path, format="parquet")
    scanner = dataset.to_batches(batch_size=batch_size)
    for record_batch in scanner:
        yield record_batch.to_pandas()


def main():
    logger.info("Starting prediction on unlabeled contributions data from Parquet in chunks...")

    data_path = UNLABELLED_RAW_DATA_FILE
    batch_size = 100000

    # Load the model
    model = load_model(FINAL_MODEL_PATH)

    # Load the clustering model
    if not os.path.exists(CLUSTER_MODEL_PATH):
        logger.error(f"Clustering model file not found at {CLUSTER_MODEL_PATH}")
        raise FileNotFoundError(f"Clustering model file not found at {CLUSTER_MODEL_PATH}")

    clustering_model = joblib.load(CLUSTER_MODEL_PATH)
    logger.info(f"Loaded clustering model from '{CLUSTER_MODEL_PATH}'.")

    # Load the trained features to align columns
    trained_feature_names = joblib.load(FINAL_TRAINED_FEATURES_PATH)  # You must have saved these at training time

    required_columns = ['centroid_x', 'centroid_y']

    total_entries = 0
    num_vandalism = 0

    output_file = UNLABELLED_PROCESSED_OUTPUT_CSV_FILE
    with open(output_file, 'w') as f:
        f.write("changeset_id,y_pred,y_prob\n")

    for i, chunk in enumerate(load_parquet_in_chunks(data_path, batch_size=batch_size)):
        logger.info(f"Processing chunk {i + 1}...")
        if i == 5:
            break
        features_df = get_or_generate_features(
            chunk,
            False,
            UNLABELLED_PROCESSED_FEATURES_FILE,
            force_compute_features=True,
            test_mode=False
        )

        X_encoded, _ = preprocess_features(features_df, is_training=False)

        for col in required_columns:
            if col not in X_encoded.columns:
                logger.error(f"Column '{col}' not found in features.")
                raise KeyError(f"Column '{col}' not found in features.")

        # Align columns to training columns
        X_encoded_aligned = X_encoded.reindex(columns=trained_feature_names, fill_value=0)

        centroids = X_encoded_aligned[['centroid_x', 'centroid_y']].values
        cluster_labels = clustering_model.predict(centroids)
        X_encoded_aligned['cluster_label'] = cluster_labels

        # Make Predictions
        y_pred = model.predict(X_encoded_aligned)
        y_prob = model.predict_proba(X_encoded_aligned)[:, 1]

        total_entries += len(y_pred)
        num_vandalism += sum(y_pred)

        predictions_df = pd.DataFrame({
            'changeset_id': features_df['changeset_id'],
            'y_pred': y_pred,
            'y_prob': y_prob
        })
        predictions_df.to_csv(output_file, mode='a', index=False, header=False)

        logger.info(f"Finished processing chunk {i + 1}. Total processed so far: {total_entries}")

    num_non_vandalism = total_entries - num_vandalism
    logger.info("Completed predictions on all unlabeled data.")
    logger.info(f"Total entries: {total_entries}")
    logger.info(f"Predicted Vandalism: {num_vandalism}")
    logger.info(f"Predicted Non-Vandalism: {num_non_vandalism}")
    logger.info(f"Predictions saved to '{output_file}'")


if __name__ == '__main__':
    main()
