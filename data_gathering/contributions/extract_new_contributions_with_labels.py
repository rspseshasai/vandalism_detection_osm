import os
import re
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from contribution_schema import get_osm_contribution_schema
from logger_config import logger
from data_gathering.s3clientmanager import S3ClientManager

# DataFrame to hold all matching contributions
filtered_contributions_df = pd.DataFrame()

# Save threshold (number of S3 objects processed to trigger saving)
threshold_to_save_filtered_contributions = 4
processed_objects_count = 0
file_counter = 0

# Define the schema according to the given structure
schema = get_osm_contribution_schema()


def save_filtered_contributions(filtered_df, file_counter, schema):
    """
    Save the filtered contributions to a Parquet file with the defined schema.
    """
    if filtered_df.__len__() > 0:
        output_file = f'../data/contribution_data/filtered_contributions_revert_part_{file_counter}.parquet'
        table = pa.Table.from_pandas(filtered_df, schema=schema, preserve_index=False)
        pq.write_table(table, output_file)
        logger.info(f"Filtered contributions saved to {output_file}")


def load_and_filter_contributions_with_vandalism(path, s3_client, bucket_name):
    """
    Load a Parquet file from S3 and filter contributions based on the presence of
    'vandalism' or 'revert' in the comments.
    """
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp_file:
        temp_file_path = temp_file.name

    s3_client.download_file(bucket_name, path, temp_file_path)
    contribution_df = pd.read_parquet(temp_file_path, engine='pyarrow')
    os.remove(temp_file_path)

    # Filter based on comments containing "vandalism" or "revert"
    if 'comment' in contribution_df.columns:
        # Extract the original changeset ID if it's mentioned in the comment
        def extract_revert_info(comment):
            original_changeset_id = None
            if re.search(r'vandalism|revert', comment, re.IGNORECASE):
                # Try to extract a changeset ID from the comment
                match = re.search(r'\b\d{6,}\b', comment)
                if match:
                    original_changeset_id = match.group(0)
                return original_changeset_id
            return None

        # Filter contributions and create relevant columns
        filtered_df = contribution_df[contribution_df['comment'].apply(
            lambda x: re.search(r'vandalism|revert', str(x), re.IGNORECASE) is not None)]

        # Add the 'original_changeset_id' and 'label' (always 'vandalism' for filtered contributions)
        if not filtered_df.empty:
            filtered_df['original_changeset_id'] = filtered_df['comment'].apply(extract_revert_info)
            filtered_df['label'] = 'vandalism'

        # Keep only the necessary columns: contribution ID, comment, original changeset ID, label
        filtered_df = filtered_df[['contribution_id', 'comment', 'original_changeset_id', 'label']]
        return filtered_df

    return pd.DataFrame()


continuation_token = None

s3_client_manager = S3ClientManager()

while True:
    # List objects in the S3 bucket with pagination
    if continuation_token:
        response = s3_client_manager.s3_client.list_objects_v2(
            Bucket=s3_client_manager.bucket_name, Prefix=s3_client_manager.prefix, ContinuationToken=continuation_token
        )
    else:
        response = s3_client_manager.s3_client.list_objects_v2(Bucket=s3_client_manager.bucket_name,
                                                               Prefix=s3_client_manager.prefix)

    if 'Contents' in response:
        for obj in tqdm(response['Contents']):
            obj_key = obj['Key']
            if obj_key.endswith('.parquet'):
                filtered_df = load_and_filter_contributions_with_vandalism(obj_key,
                                                                           s3_client_manager.s3_client,
                                                                           s3_client_manager.bucket_name)

                # Add to the collected filtered contributions
                if not filtered_df.empty:
                    filtered_contributions_df = pd.concat([filtered_contributions_df, filtered_df], ignore_index=True)
                processed_objects_count += 1

                # Save filtered contributions once the threshold is reached
                if processed_objects_count >= threshold_to_save_filtered_contributions:
                    save_filtered_contributions(filtered_contributions_df, file_counter, schema)
                    file_counter += 1
                    processed_objects_count = 0
                    filtered_contributions_df = pd.DataFrame()

    if response.get('IsTruncated'):
        continuation_token = response['NextContinuationToken']
    else:
        break

# Save any remaining filtered contributions that haven't been saved yet
if not filtered_contributions_df.empty:
    save_filtered_contributions(filtered_contributions_df, file_counter, schema)

logger.info(f"Total number of matching contributions saved in multiple files.")
