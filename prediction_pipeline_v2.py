import glob
import os
import sys
from datetime import datetime

import joblib
import pandas as pd

# Adjust the path to import modules from src
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_dir, 'src'))

TEST_PREDICTION_RUN = False

from config import (
    logger,
    FINAL_MODEL_PATH,
    CLUSTER_MODEL_PATH,
    FINAL_TRAINED_FEATURES_PATH,
    PREDICTIONS_INPUT_DATA_DIR,
    OUTPUT_DIR,
    UNLABELLED_PROCESSED_FEATURES_FILE, OPTIMAL_THRESHOLD_FOR_INFERENCE_PATH, DEFAULT_THRESHOLD_FOR_EVALUATION
)
from training import load_model
from src.feature_engineering_parallel import get_or_generate_features
from src.preprocessing import preprocess_features

OUTPUT_FOLDER_SUFFIX = F"pre_computed_user_features_branch_all_{DEFAULT_THRESHOLD_FOR_EVALUATION}"


def extract_year_month_from_filename(filename: str) -> str:
    """
    Extract "yyyy-mm" from a filename such as "2022-06-29_2022-06-30_contributions.parquet".
    We'll parse the first date chunk and keep the first 7 chars -> "2022-06".
    """
    base_name = os.path.basename(filename)
    parts = base_name.split("_")
    if not parts:
        return "unknown-month"

    date_part = parts[0]  # e.g. "2022-06-29"
    yyyy_mm = date_part[:7]  # e.g. "2022-06"
    return yyyy_mm


def load_entire_parquet_file(data_path: str) -> pd.DataFrame:
    """
    Load the entire Parquet file at once (no chunking).
    """
    logger.info(f"Loading entire Parquet file: {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    return df


def append_vandalism_to_monthly_csv(
        vandalism_df: pd.DataFrame, month_str: str, predict_output_folder: str
) -> None:
    """
    Append vandalism predictions to a single monthly CSV file named {month_str}_vandalism_predictions.csv
    in the 'predict_output_folder'. If it doesn't exist, create it with headers.
    Otherwise, append with no headers.
    """
    monthly_csv_file = os.path.join(predict_output_folder, f"{month_str}_vandalism_predictions.csv")

    # If file does not exist, create with headers
    write_header = not os.path.exists(monthly_csv_file)
    mode = "w" if write_header else "a"
    logger.info(f"Saving vandalism predictions to monthly CSV: {monthly_csv_file} (mode={mode})")

    vandalism_df.to_csv(monthly_csv_file, index=False, header=write_header, mode=mode)


def update_overall_summary_csv(
        month_str: str,
        total_entries: int,
        vandalism_count: int,
        overall_summary_csv_path: str
) -> None:
    """
    Keep a single overall summary CSV with columns:
        [MonthYear, Total Entries, Vandalism Count, Vandalism Percentage, UpdatedAt]
    1) If file doesn't exist, create with one row.
    2) If file exists:
       - read it
       - if month_str already exists, update that row (increment totals, or replace with new totals)
       - else append a new row
       - overwrite the CSV with updated data
    """
    logger.info(f"Updating overall summary CSV at {overall_summary_csv_path}...")

    vandalism_percent = (vandalism_count / total_entries * 100.0) if total_entries > 0 else 0.0
    new_row = {
        "MonthYear": month_str,
        "Total Entries": total_entries,
        "Vandalism Count": vandalism_count,
        "Vandalism Percentage": f"{vandalism_percent:.4f}",
        "UpdatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if not os.path.exists(overall_summary_csv_path):
        # Create new CSV with the single row
        summary_df = pd.DataFrame([new_row])
        summary_df.to_csv(overall_summary_csv_path, index=False, mode="w")
        logger.info(f"Created new summary CSV with initial entry for {month_str}.")
        return

    # File exists; read it
    summary_df = pd.read_csv(overall_summary_csv_path)

    # Check if month_str is present
    mask = (summary_df["MonthYear"] == month_str)

    if mask.any():
        # If month row exists, update row
        idx = summary_df[mask].index[0]

        # If you want to *add* to existing totals, you might do:
        summary_df.loc[idx, "Total Entries"] += total_entries
        summary_df.loc[idx, "Vandalism Count"] += vandalism_count

        vandalism_percent = (summary_df.loc[idx, "Vandalism Count"] / summary_df.loc[idx, "Total Entries"] * 100.0) if \
            summary_df.loc[idx, "Total Entries"] > 0 else 0.0

        summary_df.loc[idx, "Vandalism Percentage"] = f"{vandalism_percent:.4f}"
        summary_df.loc[idx, "UpdatedAt"] = new_row["UpdatedAt"]

        logger.info(f"Updated existing month entry for {month_str}.")
    else:
        # Otherwise, append a new row
        summary_df = pd.concat([summary_df, pd.DataFrame([new_row])], ignore_index=True)
        logger.info(f"Appended new month entry for {month_str}.")

    summary_df.to_csv(overall_summary_csv_path, index=False, mode="w")
    logger.info(f"Overall summary CSV updated for {month_str}.")


def process_file(input_file: str, model, clustering_model, trained_feature_names, predict_output_folder) -> None:
    base_name = os.path.basename(input_file)
    # Extract the "yyyy-mm" from the filename
    month_str = extract_year_month_from_filename(base_name)

    # Overall summary CSV (inside the same predictions_output folder)
    overall_summary_csv = os.path.join(predict_output_folder, "overall_summary.csv")

    logger.info(f"Processing file: {input_file} for month: {month_str}")

    # 1) Load entire data
    df = load_entire_parquet_file(input_file)
    total_entries = len(df)
    if total_entries == 0:
        logger.warning(f"No data found in {input_file}. Skipping.")
        return

    # 2) Feature Engineering
    logger.info(f"Extracting features for file: {base_name}")
    features_df = get_or_generate_features(
        df,
        is_training=False,
        processed_features_file_path=UNLABELLED_PROCESSED_FEATURES_FILE,
        force_compute_features=True,
        test_mode=False
    )

    # 3) Preprocessing
    required_columns = ["centroid_x", "centroid_y"]
    X_encoded, _ = preprocess_features(features_df, is_training=False)

    # Check required columns
    for col in required_columns:
        if col not in X_encoded.columns:
            logger.error(f"Column '{col}' not found in features for {base_name}")
            raise KeyError(f"Column '{col}' not found in features.")

    # 4) Align columns with the trained model
    X_encoded_aligned = X_encoded.reindex(columns=trained_feature_names, fill_value=0)

    # 5) Clustering
    centroids = X_encoded_aligned[["centroid_x", "centroid_y"]].values
    cluster_labels = clustering_model.predict(centroids)
    X_encoded_aligned["cluster_label"] = cluster_labels

    # 6) Model Predictions
    if os.path.exists(OPTIMAL_THRESHOLD_FOR_INFERENCE_PATH):
        threshold = joblib.load(OPTIMAL_THRESHOLD_FOR_INFERENCE_PATH)
        logger.info(f"Loaded custom threshold: {threshold:.4f}")
    else:
        threshold = DEFAULT_THRESHOLD_FOR_EVALUATION  # fallback
        logger.warning(f"No custom threshold found. Using default {threshold}")

    y_prob = model.predict_proba(X_encoded_aligned)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # 7) Filter vandalism
    vandal_mask = (y_pred == 1)
    vandal_count = vandal_mask.sum()

    logger.info(
        f"For file {base_name}: total {total_entries}, predicted vandalism {vandal_count}."
    )

    # Create DataFrame of only vandalism
    vandalism_df = pd.DataFrame({
        "changeset_id": features_df["changeset_id"][vandal_mask],
        "y_pred": y_pred[vandal_mask],
        "y_prob": y_prob[vandal_mask],
    })

    # 8) Append vandalism predictions to a single monthly CSV
    append_vandalism_to_monthly_csv(vandalism_df, month_str, predict_output_folder)

    # 9) Update the overall summary for that month
    #    If we want to preserve an existing summary from previous runs, do not overwrite;
    #    we only append if it doesn't exist, or we update the row for that month in the same CSV.
    update_overall_summary_csv(
        month_str=month_str,
        total_entries=total_entries,
        vandalism_count=vandal_count,
        overall_summary_csv_path=overall_summary_csv
    )

    logger.info(f"Done processing {base_name}. Vandalism count: {vandal_count} (month {month_str})")


def main():
    logger.info("Starting prediction on unlabeled contributions data for daily files (entire load).")

    # 1) Load XGBoost model
    model = load_model(FINAL_MODEL_PATH)

    # 2) Load clustering model
    if not os.path.exists(CLUSTER_MODEL_PATH):
        logger.error(f"Clustering model file not found at {CLUSTER_MODEL_PATH}")
        raise FileNotFoundError(f"Clustering model file not found at {CLUSTER_MODEL_PATH}")
    clustering_model = joblib.load(CLUSTER_MODEL_PATH)
    logger.info(f"Loaded clustering model from '{CLUSTER_MODEL_PATH}'.")

    # 3) Load trained feature names
    trained_feature_names = joblib.load(FINAL_TRAINED_FEATURES_PATH)
    logger.info(f"Loaded trained feature names from {FINAL_TRAINED_FEATURES_PATH}.")

    # 4) Gather all daily parquet files
    input_files = glob.glob(os.path.join(PREDICTIONS_INPUT_DATA_DIR, "*.parquet"))
    logger.info(f"Found {len(input_files)} daily input files in {PREDICTIONS_INPUT_DATA_DIR}.")

    # 5) Create output dir
    # Prepare a unique output folder with timestamp for storing predictions
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    predict_output_folder = os.path.join(
        OUTPUT_DIR,
        "predictions_output",
        f"{os.path.basename(PREDICTIONS_INPUT_DATA_DIR)}_{timestamp_str}_{OUTPUT_FOLDER_SUFFIX}"
    )
    os.makedirs(predict_output_folder, exist_ok=True)

    # 6) Process each file
    for file_idx, input_file in enumerate(input_files, start=1):
        logger.info("--------------------------------------------------------------------")
        logger.info(f"Processing file {file_idx}/{len(input_files)}: {input_file}")
        logger.info("--------------------------------------------------------------------")

        process_file(input_file, model, clustering_model, trained_feature_names, predict_output_folder)

        if TEST_PREDICTION_RUN and file_idx >= 5:
            logger.info("Test run enabled. Stopping after 5 files.")
            break

    logger.info("All daily files processed. Completed inference.")


if __name__ == "__main__":
    main()
