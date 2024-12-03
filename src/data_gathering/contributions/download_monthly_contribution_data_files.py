import logging
import os

import pandas as pd
from tqdm import tqdm

from data_gathering.s3clientmanager import S3ClientManager


def download_s3_files_to_local(manager, local_directory, start_month, end_month):
    """
    Downloads parquet files from S3 corresponding to months in the given range.
    Renames files with month names and saves them in the specified local directory.

    Parameters:
        manager (S3ClientManager): Instance of S3ClientManager to interact with S3.
        local_directory (str): Directory to save the files locally.
        start_month (str): Start month in 'YYYY-MM' format.
        end_month (str): End month in 'YYYY-MM' format.
    """
    os.makedirs(local_directory, exist_ok=True)

    current_month = start_month
    with tqdm(total=(pd.date_range(start=start_month, end=end_month, freq='MS')).size,
              desc="Downloading files") as pbar:
        while current_month <= end_month:
            month_prefix = f"{manager.base_prefix}month={current_month}/data_0.parquet"
            local_filename = os.path.join(local_directory, f"{current_month}.parquet")

            try:
                manager.s3_client.download_file(
                    manager.bucket_name,
                    month_prefix,
                    local_filename
                )
                logger.info(f"Successfully downloaded: {local_filename}")
            except Exception as e:
                logger.error(f"Failed to download file for {current_month}: {e}")

            # Move to the next month
            current_month = (pd.Timestamp(current_month) + pd.offsets.MonthBegin(1)).strftime('%Y-%m-%d')
            pbar.update(1)


if __name__ == "__main__":
    # Setup logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    try:
        # Initialize the S3 client manager
        s3_manager = S3ClientManager()

        # Local directory to save the files
        local_dir = "../../../data/contribution_data/raw/"

        # Define the start and end months
        start_month = "2024-02-01"
        end_month = "2024-08-01"

        # Download files
        logger.info("Starting the download process.")
        download_s3_files_to_local(s3_manager, local_dir, start_month, end_month)
        logger.info("Download process completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
