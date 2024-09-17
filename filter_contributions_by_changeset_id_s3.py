import os
import tempfile

import boto3
import pandas as pd
from botocore.client import Config
from tqdm import tqdm

from logger_config import logger

# Initialize the S3 client
s3_client = boto3.client(
    's3',
    endpoint_url='https://sotm2024.minio.heigit.org',
    aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
    config=Config(signature_version='s3v4'),
    region_name='eu-central-1'
)

bucket_name = 'heigit-ohsome-sotm24'
prefix = 'data/geo_sort_ext/contributions/status=invalid/geometry_type=LineString/'  # Base path in S3

labels_file = 'data/ovid_labels.tsv'
labels_df = pd.read_csv(labels_file, sep='\t')

# Manually add three items for testing
new_items = pd.DataFrame({
    'changeset': [421295, 67890, 420453],
    'label': ['true', 'false', 'false']
})
labels_df = pd.concat([labels_df, new_items], ignore_index=True)
changeset_labels = dict(zip(labels_df['changeset'], labels_df['label']))
changeset_ids_to_filter = list(labels_df['changeset'])

# DataFrame to hold all matching contributions
filtered_contributions = pd.DataFrame()


def load_and_filter(path, changeset_ids, s3_client, bucket_name):
    """
    Load a Parquet file from S3 and return rows where `changeset.id` is in the specified list.
    """

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp_file:
        temp_file_path = temp_file.name

    s3_client.download_file(bucket_name, path, temp_file_path)
    contribution_df = pd.read_parquet(temp_file_path, engine='pyarrow')
    os.remove(temp_file_path)

    if 'changeset' in contribution_df.columns:
        filtered_df = contribution_df[contribution_df['changeset'].apply(lambda x: x['id']).isin(changeset_ids)]
        return filtered_df

    return pd.DataFrame()


continuation_token = None

while True:
    # List objects in the S3 bucket with pagination
    if continuation_token:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name, Prefix=prefix, ContinuationToken=continuation_token
        )
    else:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if 'Contents' in response:
        for obj in tqdm(response['Contents']):
            obj_key = obj['Key']
            if obj_key.endswith('.parquet'):
                filtered_df = load_and_filter(obj_key, changeset_ids_to_filter, s3_client, bucket_name)

                # Add the corresponding label to the filtered contributions
                if not filtered_df.empty:
                    filtered_df['label'] = filtered_df['changeset'].apply(
                        lambda x: changeset_labels.get(x['id'], 'unknown'))
                    filtered_contributions = pd.concat([filtered_contributions, filtered_df], ignore_index=True)

    # Check if there are more objects to process
    if response.get('IsTruncated'):
        continuation_token = response['NextContinuationToken']
    else:
        break

logger.info(f"Total number of matching contributions: {filtered_contributions.shape[0]}")

output_file = 'output/filtered_contributions__' + labels_file.split("/")[1] + '.parquet'
filtered_contributions.to_parquet(output_file, index=False)

logger.info(f"Filtered contributions with labels saved to {output_file}")
