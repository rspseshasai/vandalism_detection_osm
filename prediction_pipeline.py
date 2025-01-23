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
OUTPUT_FOLDER_SUFFIX = "nuof_full_dataset_detailed"

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
    """
    Loads a Parquet file in chunks using pyarrow.dataset, returning a Pandas DataFrame per chunk.
    """
    dataset = ds.dataset(data_path, format="parquet")
    scanner = dataset.to_batches(batch_size=batch_size)
    for record_batch in scanner:
        yield record_batch.to_pandas()


def extract_month_year_from_filename(filename: str) -> str:
    """
    Extracts "MonthYear" (e.g., "2022-02") from a filename like "2022-02-01".
    Assumes the file name starts with a date in the format "YYYY-MM-DD".
    """
    base_name = os.path.basename(filename)
    parts = base_name.split('-')  # Split by hyphen
    if len(parts) >= 2:  # Ensure there are enough parts to extract month and year
        year = parts[0]
        month = parts[1]
        return f"{year}-{month}"
    else:
        logger.warning(f"Unexpected filename format: {filename}. Unable to extract month and year.")
        return "unknown-month"


def append_or_update_overall_summary(
        predict_output_folder: str,
        month_year: str,
        total_entries: int,
        num_vandalism: int
) -> None:
    """
    Append or update a row in the 'overall_summary.csv' with columns:
    [MonthYear, Total Entries, Vandalism Count, Vandalism Percentage, UpdatedAt]

    - If a row with the same MonthYear exists, update it.
    - Otherwise, append a new row.
    """
    summary_file = os.path.join(predict_output_folder, "overall_summary.csv")
    vandal_percent = (num_vandalism / total_entries * 100.0) if total_entries > 0 else 0.0

    new_row = {
        "MonthYear": month_year,
        "Total Entries": total_entries,
        "Vandalism Count": num_vandalism,
        "Vandalism Percentage": f"{vandal_percent:.4f}",
        "UpdatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if not os.path.exists(summary_file):
        # Create a new CSV with headers
        df = pd.DataFrame([new_row])
        df.to_csv(summary_file, index=False, mode="w")
        logger.info(f"Created new overall summary CSV and added entry for {month_year}.")
        return

    # If the file exists, read it
    summary_df = pd.read_csv(summary_file)
    mask = (summary_df["MonthYear"] == month_year)

    if mask.any():
        # Update existing row
        idx = summary_df[mask].index[0]
        # You can choose to replace or increment. Here we replace for a single file scenario.
        summary_df.loc[idx, "Total Entries"] = total_entries
        summary_df.loc[idx, "Vandalism Count"] = num_vandalism
        summary_df.loc[idx, "Vandalism Percentage"] = f"{vandal_percent:.4f}"
        summary_df.loc[idx, "UpdatedAt"] = new_row["UpdatedAt"]
        logger.info(f"Updated existing month entry for {month_year} in overall summary CSV.")
    else:
        # Append a new row
        summary_df = pd.concat([summary_df, pd.DataFrame([new_row])], ignore_index=True)
        logger.info(f"Appended new month entry for {month_year} in overall summary CSV.")

    summary_df.to_csv(summary_file, index=False, mode="w")
    logger.info(f"Overall summary CSV updated for {month_year}.")


def process_file(input_file, model, clustering_model, trained_feature_names, predict_output_folder, batch_size=100000):
    """
    Process a single Parquet file in chunks and predict vandalism entries.
    Also updates the overall_summary.csv after completing all chunks for this file.
    """
    required_columns = ['centroid_x', 'centroid_y']

    # Prepare output file
    base_name = os.path.basename(input_file)
    output_file_name = base_name.replace('.parquet', '_prediction_output.csv')

    os.makedirs(predict_output_folder, exist_ok=True)
    output_file = os.path.join(predict_output_folder, output_file_name)

    # Write the header only once for vandalism output
    with open(output_file, 'w') as f:
        f.write("changeset_id,date_created,osm_id,osm_version,centroid_x,centroid_y,y_pred,y_prob\n")

    # Track entries and vandalism count
    total_entries = 0
    num_vandalism = 0

    logger.info(f"Processing file: {input_file}")

    for i, chunk in enumerate(load_parquet_in_chunks(input_file, batch_size=batch_size)):
        logger.info(f"Processing chunk {i + 1} for file {base_name}...")

        # Feature Engineering
        features_df = get_or_generate_features(
            chunk,
            False,
            UNLABELLED_PROCESSED_FEATURES_FILE,
            force_compute_features=True,
            test_mode=False
        )

        # Preprocessing
        X_encoded, _ = preprocess_features(features_df, is_training=False)

        # Check required columns
        for col in required_columns:
            if col not in X_encoded.columns:
                logger.error(f"Column '{col}' not found in features.")
                raise KeyError(f"Column '{col}' not found in features.")

        # Align columns
        X_encoded_aligned = X_encoded.reindex(columns=trained_feature_names, fill_value=0)

        # Clustering
        centroids = X_encoded_aligned[['centroid_x', 'centroid_y']].values
        cluster_labels = clustering_model.predict(centroids)
        X_encoded_aligned['cluster_label'] = cluster_labels

        # Predictions
        y_pred = model.predict(X_encoded_aligned)
        y_prob = model.predict_proba(X_encoded_aligned)[:, 1]

        # Update counts
        total_entries += len(y_pred)
        vandal_mask = (y_pred == 1)
        num_vandalism += vandal_mask.sum()

        # Save only vandalism entries
        vandalism_df = pd.DataFrame({
            'changeset_id': features_df.loc[vandal_mask, "changeset_id"],
            'date_created': features_df.loc[vandal_mask, "date_created"],
            'osm_id': features_df.loc[vandal_mask, "osm_id"],
            'osm_version': features_df.loc[vandal_mask, "osm_version"],
            'centroid_x': features_df.loc[vandal_mask, "centroid_x"],
            'centroid_y': features_df.loc[vandal_mask, "centroid_y"],
            'y_pred': y_pred[vandal_mask],
            'y_prob': y_prob[vandal_mask]
        })
        vandalism_df.to_csv(output_file, mode='a', index=False, header=False)

        logger.info(
            f"Finished chunk {i + 1} for file {base_name}. "
            f"Total processed so far: {total_entries}, Vandalism so far: {num_vandalism}."
        )
        logger.info("------------------------------------------------------------------------------------------")

        # Stop early if test run
        if TEST_PREDICTION_RUN and i == 10:
            logger.info("Test run enabled; stopping the prediction early.")
            break

    # Log final stats for this file
    num_non_vandalism = total_entries - num_vandalism
    logger.info(f"Completed predictions on file: {input_file}")
    logger.info(
        f"File stats - Total entries: {total_entries}, "
        f"Predicted Vandalism: {num_vandalism}, Non-Vandalism: {num_non_vandalism}"
    )
    logger.info(f"Predictions saved to '{output_file}'")

    # Update overall summary CSV
    month_year = extract_month_year_from_filename(input_file)
    append_or_update_overall_summary(predict_output_folder, month_year, total_entries, num_vandalism)


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

    # List all Parquet files to be processed
    input_files = glob.glob(os.path.join(PREDICTIONS_INPUT_DATA_DIR, "*.parquet"))
    logger.info(f"Found {len(input_files)} input files in {PREDICTIONS_INPUT_DATA_DIR}.")

    predict_output_folder = os.path.join(
        OUTPUT_DIR,
        'predictions_output',
        f"{OUTPUT_FOLDER_SUFFIX}__{os.path.basename(PREDICTIONS_INPUT_DATA_DIR)}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    for file_idx, input_file in enumerate(input_files, start=1):
        logger.info("===================================================================")
        logger.info(f"Processing file {file_idx}/{len(input_files)}: {input_file}")
        logger.info("===================================================================")

        process_file(
            input_file,
            model,
            clustering_model,
            trained_feature_names,
            predict_output_folder,
            batch_size=100000
        )

    logger.info("All files processed successfully.")


if __name__ == '__main__':
    main()
