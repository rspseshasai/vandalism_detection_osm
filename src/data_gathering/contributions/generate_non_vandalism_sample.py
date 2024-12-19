import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to the directory containing the .parquet files
input_directory = '../../../data/contribution_data/raw/non_vandalism_files/'

# Output file path for the sampled DataFrame
sampled_output_file = '../../../data/contribution_data/output/osm_sampled_contributions_with_no_vandalism.parquet'

# Total number of entries required
total_entries = 149994

# List all .parquet files in the directory
parquet_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.parquet')]

if len(parquet_files) == 0:
    logger.error("No .parquet files found in the directory.")
    raise ValueError("No .parquet files to process.")

# Calculate the sample size per file dynamically
sample_size_per_file = total_entries // len(parquet_files)
logger.info(f"Number of files: {len(parquet_files)}")
logger.info(f"Sample size per file: {sample_size_per_file}")


# Function to sample contributions randomly from files and add 'vandalism' column
def sample_and_add_vandalism(parquet_files, output_file, sample_size_per_file):
    sampled_dataframes = []

    for parquet_file in parquet_files:
        try:
            # Read the parquet file
            df = pd.read_parquet(parquet_file)
            logger.info(f"File {parquet_file} contains {len(df)} rows.")

            # Skip the first few entries and randomly sample
            if len(df) > sample_size_per_file:
                sampled_df = df.iloc[sample_size_per_file:].sample(n=sample_size_per_file, random_state=42)
            else:
                sampled_df = df.sample(n=min(len(df), sample_size_per_file), random_state=42)
            logger.info(f"Sampled {len(sampled_df)} rows from {parquet_file}.")

            # Add 'vandalism' column
            sampled_df['vandalism'] = 0

            # Append sampled DataFrame
            sampled_dataframes.append(sampled_df)
        except Exception as e:
            logger.error(f"Error processing {parquet_file}: {e}")
            continue

    # Concatenate all sampled DataFrames into a single DataFrame
    try:
        sampled_merged_df = pd.concat(sampled_dataframes, ignore_index=True)
        logger.info(f"Successfully merged sampled DataFrames. Total rows: {sampled_merged_df.shape[0]}")
    except ValueError as e:
        logger.error(f"Error concatenating sampled DataFrames: {e}")
        raise

    # Convert the sampled DataFrame to PyArrow Table
    arrow_table = pa.Table.from_pandas(sampled_merged_df, preserve_index=False)

    # Save the Arrow Table as a single Parquet file
    try:
        pq.write_table(arrow_table, output_file)
        logger.info(f"Sampled contributions with 'vandalism' column saved to {output_file}")
    except Exception as e:
        logger.error(f"Error writing sampled Parquet file: {e}")
        raise


# Run the sampling function
sample_and_add_vandalism(parquet_files, sampled_output_file, sample_size_per_file)
