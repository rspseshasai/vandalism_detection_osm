import os

import boto3
from botocore.client import Config

# Initialize the S3 client with your custom endpoint (MinIO server)
s3_client = boto3.client(
    's3',
    endpoint_url='https://sotm2024.minio.heigit.org',  # Custom MinIO endpoint
    aws_access_key_id=os.getenv("S3_ACCESS_KEY"),  # Replace with your Access Key
    aws_secret_access_key=os.getenv("S3_SECRET_KEY"),  # Replace with your Secret Key
    config=Config(signature_version='s3v4'),
    region_name='eu-central-1'  # Set your region if required
)

bucket_name = 'heigit-ohsome-sotm24'
prefix = 'data/geo_sort_ext/'

# List objects in the specific folder
response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

# Check if 'Contents' key is in the response
if 'Contents' in response:
    for obj in response['Contents']:
        print(obj['Key'])
else:
    print(f"No objects found in {bucket_name}/{prefix}")
