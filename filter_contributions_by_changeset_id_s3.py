import os
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from contribution_schema import get_osm_contribution_schema
from logger_config import logger
from s3clientmanager import S3ClientManager

vandalism_labels_with_changeset_ids_file = 'data/ovid_labels.tsv'
labels_df = pd.read_csv(vandalism_labels_with_changeset_ids_file, sep='\t')

changeset_labels = dict(zip(labels_df['changeset'], labels_df['label']))
labelled_changeset_ids_to_filter = list(labels_df['changeset'])

# DataFrame to hold all matching contributions
labelled_contributions_df = pd.DataFrame()

# Save threshold (number of S3 objects processed to trigger saving)
s3_object_threshold = 4
processed_objects_count = 0
file_counter = 0

# Define the schema according to the given structure
schema = get_osm_contribution_schema()


def save_filtered_contributions(filtered_df, file_counter, schema):
    """
    Save the filtered contributions to a Parquet file with the defined schema.
    """
    output_file = f'output/filtered_contributions_part_{file_counter}.parquet'
    table = pa.Table.from_pandas(filtered_df, schema=schema, preserve_index=False)
    pq.write_table(table, output_file)
    logger.info(f"Filtered contributions saved to {output_file}")


def load_and_filter_contributions_with_changeset_id(path, changeset_ids, s3_client, bucket_name):
    """
    Load a Parquet file from S3 and return rows where `changeset.id` is in the specified list.
    """
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp_file:
        temp_file_path = temp_file.name

    s3_client.download_file(bucket_name, path, temp_file_path)
    contribution_df = pd.read_parquet(temp_file_path, engine='pyarrow')
    os.remove(temp_file_path)

    # Filter rows where the changeset.id is in the provided list
    if 'changeset' in contribution_df.columns:
        filtered_df = contribution_df[
            contribution_df['changeset'].apply(lambda x: x['id'] if isinstance(x, dict) else None).isin(changeset_ids)
        ]
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
                filtered_df = load_and_filter_contributions_with_changeset_id(obj_key, labelled_changeset_ids_to_filter,
                                                                              s3_client_manager.s3_client,
                                                                              s3_client_manager.bucket_name)

                # Add the corresponding label to the filtered contributions
                if not filtered_df.empty:
                    filtered_df['vandalism'] = filtered_df['changeset'].apply(
                        lambda x: changeset_labels.get(x['id'], 'unknown')
                    )
                    labelled_contributions_df = pd.concat([labelled_contributions_df, filtered_df], ignore_index=True)
                processed_objects_count += 1

                if processed_objects_count >= s3_object_threshold:
                    save_filtered_contributions(labelled_contributions_df, file_counter, schema)
                    file_counter += 1
                    processed_objects_count = 0
                    labelled_contributions_df = pd.DataFrame()

    if response.get('IsTruncated'):
        continuation_token = response['NextContinuationToken']
    else:
        break

if not labelled_contributions_df.empty:
    save_filtered_contributions(labelled_contributions_df, file_counter, schema)

logger.info(f"Total number of matching contributions saved in multiple files.")
