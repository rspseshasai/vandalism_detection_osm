import os

import boto3
from botocore.client import Config


class S3ClientManager:
    def __init__(self):
        # Initialize the S3 client inside the constructor
        self.s3_client = boto3.client(
            's3',
            endpoint_url='https://sotm2024.minio.heigit.org',
            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
            config=Config(signature_version='s3v4'),
            region_name='eu-central-1'
        )

        # Bucket name and prefix
        self.bucket_name = 'heigit-ohsome-sotm24'
        # Base path in S3
        # self.prefix = 'data/geo_sort_ext/contributions/'
        self.prefix = 'data/geo_sort_ext/contributions/status=invalid/geometry_type=LineString/'  # Path with lesser contributions for testing
