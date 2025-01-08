import glob
import os
import sys
from datetime import datetime

import joblib
import pandas as pd
import pyarrow.dataset as ds

# Adjust the path to import modules from src
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_dir, 'src'))
TEST_PREDICTION_RUN = False
OUTPUT_FOLDER_SUFFIX = F"no_user_features_branch"

from config import (
    logger,
    FINAL_MODEL_PATH,
    CLUSTER_MODEL_PATH,
    FINAL_TRAINED_FEATURES_PATH,
    PREDICTIONS_INPUT_DATA_DIR,
    OUTPUT_DIR,
    UNLABELLED_PROCESSED_FEATURES_FILE
)
from training import load_model
from src.feature_engineering_parallel import get_or_generate_features
from src.preprocessing import preprocess_features


def load_parquet_in_chunks(data_path, batch_size=1000000):
    dataset = ds.dataset(data_path, format="parquet")
    scanner = dataset.to_batches(batch_size=batch_size)
    for record_batch in scanner:
        yield record_batch.to_pandas()


def process_file(input_file, model, clustering_model, trained_feature_names, predict_output_folder, batch_size=1000000):
    """Process a single Parquet file in chunks and predict vandalism entries."""
    required_columns = ['centroid_x', 'centroid_y']

    # Prepare output file
    base_name = os.path.basename(input_file)
    output_file_name = base_name.replace('.parquet', '_prediction_output.csv')

    os.makedirs(predict_output_folder, exist_ok=True)
    output_file = os.path.join(predict_output_folder, output_file_name)

    # Write the header only once
    with open(output_file, 'w') as f:
        f.write("changeset_id,y_pred,y_prob\n")

    total_entries = 0
    num_vandalism = 0

    logger.info(f"Processing file: {input_file}")

    for i, chunk in enumerate(load_parquet_in_chunks(input_file, batch_size=batch_size)):
        logger.info(f"Processing chunk {i + 1} for file {base_name}...")
        features_df = get_or_generate_features(
            chunk,
            False,
            UNLABELLED_PROCESSED_FEATURES_FILE,
            force_compute_features=True,
            test_mode=False
        )

        X_encoded, _ = preprocess_features(features_df, is_training=False)

        # Check required columns
        for col in required_columns:
            if col not in X_encoded.columns:
                logger.error(f"Column '{col}' not found in features.")
                raise KeyError(f"Column '{col}' not found in features.")

        # Align columns
        X_encoded_aligned = X_encoded.reindex(columns=trained_feature_names, fill_value=0)

        centroids = X_encoded_aligned[['centroid_x', 'centroid_y']].values
        cluster_labels = clustering_model.predict(centroids)
        X_encoded_aligned['cluster_label'] = cluster_labels

        # Predictions
        y_pred = model.predict(X_encoded_aligned)
        y_prob = model.predict_proba(X_encoded_aligned)[:, 1]

        total_entries += len(y_pred)
        vandal_mask = (y_pred == 1)
        num_vandalism += vandal_mask.sum()

        # Save only vandalism entries
        vandalism_df = pd.DataFrame({
            'changeset_id': features_df['changeset_id'][vandal_mask],
            'y_pred': y_pred[vandal_mask],
            'y_prob': y_prob[vandal_mask]
        })

        vandalism_df.to_csv(output_file, mode='a', index=False, header=False)

        logger.info(
            f"Finished chunk {i + 1} for file {base_name}. "
            f"Total processed so far: {total_entries}, Vandalism so far: {num_vandalism}."
        )
        logger.info(f"------------------------------------------------------------------------------------------")
        if TEST_PREDICTION_RUN and i == 10:
            logger.info("Test enabled; Stopping the prediction...")
            break

    num_non_vandalism = total_entries - num_vandalism
    logger.info(f"Completed predictions on file: {input_file}")
    logger.info(
        f"File stats - Total entries: {total_entries}, Predicted Vandalism: {num_vandalism}, Non-Vandalism: {num_non_vandalism}")
    logger.info(f"Predictions saved to '{output_file}'")


def main():
    logger.info("Starting prediction on unlabeled contributions data from multiple Parquet files in chunks...")

    # Load the model
    model = load_model(FINAL_MODEL_PATH)

    # Load the clustering model
    if not os.path.exists(CLUSTER_MODEL_PATH):
        logger.error(f"Clustering model file not found at {CLUSTER_MODEL_PATH}")
        raise FileNotFoundError(f"Clustering model file not found at {CLUSTER_MODEL_PATH}")

    clustering_model = joblib.load(CLUSTER_MODEL_PATH)
    logger.info(f"Loaded clustering model from '{CLUSTER_MODEL_PATH}'.")

    # Load the trained feature names to align columns
    trained_feature_names = joblib.load(FINAL_TRAINED_FEATURES_PATH)

    # Process all files in UNLABELLED_RAW_DATA_FILE_DIR

    input_files = glob.glob(os.path.join(PREDICTIONS_INPUT_DATA_DIR, "*.parquet"))
    logger.info(f"Found {len(input_files)} input files in {PREDICTIONS_INPUT_DATA_DIR}.")

    predict_output_folder = os.path.join(
        OUTPUT_DIR,
        'predictions_output',
        f"{OUTPUT_FOLDER_SUFFIX}__{os.path.basename(PREDICTIONS_INPUT_DATA_DIR)}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    for file_idx, input_file in enumerate(input_files, start=1):
        logger.info(f"===================================================================")
        logger.info(f"Processing file {file_idx}/{len(input_files)}: {input_file}")
        logger.info(f"===================================================================")
        process_file(input_file, model, clustering_model, trained_feature_names, predict_output_folder, batch_size=100000)

    logger.info("All files processed successfully.")


if __name__ == '__main__':
    main()
