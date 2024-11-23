import gc
import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from logger_config import logger

# True if you need to filter the changesets using a file.
FILTER = False

# Path to the merged parquet file
parquet_file_path = "../../data/changeset_data/output/quarterly_data/osm_contributions_2022_Q2.parquet"

# Path to the TSV file containing the original changeset data (labels)
tsv_file_path = "../../data/ovid_data/ovid_labels.tsv"


# Function to fetch detailed changeset data from OSM API (for nodes/ways/relations)
def fetch_changeset_details(changeset_id):
    try:
        api_url = f"https://api.openstreetmap.org/api/0.6/changeset/{changeset_id}/download"
        response = requests.get(api_url, timeout=10)  # 10 second timeout
        response.raise_for_status()  # Raise error for bad HTTP response
        return changeset_id, response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching changeset {changeset_id}: {e}")
        return changeset_id, None


# Function to parse the changeset data and count nodes, ways, relations, and operations
def parse_changeset_data(xml_data):
    root = ET.fromstring(xml_data)
    nodes = {'create': 0, 'modify': 0, 'delete': 0}
    ways = {'create': 0, 'modify': 0, 'delete': 0}
    relations = {'create': 0, 'modify': 0, 'delete': 0}

    for element in root:
        if element.tag == 'create':
            for item in element:
                if item.tag == 'node':
                    nodes['create'] += 1
                elif item.tag == 'way':
                    ways['create'] += 1
                elif item.tag == 'relation':
                    relations['create'] += 1
        elif element.tag == 'modify':
            for item in element:
                if item.tag == 'node':
                    nodes['modify'] += 1
                elif item.tag == 'way':
                    ways['modify'] += 1
                elif item.tag == 'relation':
                    relations['modify'] += 1
        elif element.tag == 'delete':
            for item in element:
                if item.tag == 'node':
                    nodes['delete'] += 1
                elif item.tag == 'way':
                    ways['delete'] += 1
                elif item.tag == 'relation':
                    relations['delete'] += 1

    return nodes, ways, relations


# Function to fetch user info from OSM API using uid
def fetch_user_info(uid):
    api_url = f"https://api.openstreetmap.org/api/0.6/user/{uid}"
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        xml_data = response.text

        # Parse the XML data
        root = ET.fromstring(xml_data)
        user_info = root.find('user')

        # Extract relevant fields
        account_created = user_info.get('account_created')
        changes_count = user_info.find('changesets').get('count') if user_info.find('changesets') else 0

        return {
            'uid': uid,
            'account_created': account_created,
            'changes_count': changes_count  # This is basic changesets, contributions data
        }
    except requests.exceptions.RequestException as e:
        # logger.error(f"Error fetching user {uid} info: {e}")
        return {
            'uid': uid,
            'account_created': None,
            'changes_count': 0
        }


# Read the Parquet file containing changeset metadata
df_changesets = pd.read_parquet(parquet_file_path)

# Read the TSV file containing changeset IDs and labels if it exists
if os.path.exists(tsv_file_path):
    changeset_data = pd.read_csv(tsv_file_path, sep='\t')
    changeset_ids = changeset_data['changeset_id'].tolist()

    logger.info("Filtering the changesets which are present in the given list!")
    # Create a DataFrame to hold the features of the changesets
    df_features = df_changesets[df_changesets['changeset_id'].isin(changeset_ids)]

    # Delete the large DataFrame to free up memory
    del df_changesets  # Remove the DataFrame from memory
    gc.collect()  # Force garbage collection to free memory immediately

    # Merge the changeset metadata with labels
    df_merged = pd.merge(df_features, changeset_data, on='changeset_id', how='inner')
else:
    logger.info("No labels file found. Querying all changesets without any filters.")
    df_merged = df_changesets


# Function to process each changeset and update the row with parsed data
def process_changeset(row):
    changeset_id = row['changeset_id']
    uid = row['user_id']

    # Fetch detailed changeset data (for counting nodes/ways/relations)
    details = fetch_changeset_details(changeset_id)
    if details[1]:
        nodes, ways, relations = parse_changeset_data(details[1])
        # Convert the row to a dictionary and update it with parsed API data
        row_data = row.to_dict()
        row_data.update({
            "no_nodes": sum(nodes.values()),
            "no_ways": sum(ways.values()),
            "no_relations": sum(relations.values()),
            "no_creates": nodes['create'] + ways['create'] + relations['create'],
            "no_modifications": nodes['modify'] + ways['modify'] + relations['modify'],
            "no_deletions": nodes['delete'] + ways['delete'] + relations['delete'],
        })
    else:
        # If fetching details failed, set all counts to zero
        row_data = row.to_dict()  # Retain existing fields and update with zero counts
        row_data.update({
            "no_nodes": 0,
            "no_ways": 0,
            "no_relations": 0,
            "no_creates": 0,
            "no_modifications": 0,
            "no_deletions": 0,
        })

    # Fetch user info using the uid
    user_info = fetch_user_info(uid)
    row_data.update(user_info)
    # if changeset_id % 5000 == 0:
    #     logger.info(f"Fetched changeset data for changeset {changeset_id}")
    # Return the updated row data
    return row_data


# Running the multithreading process using ThreadPoolExecutor
def fetch_and_process_changesets_in_parallel(df_merged, batch_size=5000):
    all_changeset_data = []
    total_rows = len(df_merged)
    batch_count = total_rows // batch_size + (1 if total_rows % batch_size > 0 else 0)

    for batch_idx in range(batch_count):
        # Get the current batch of rows (10,000 changesets at a time)
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, total_rows)
        df_batch = df_merged.iloc[batch_start:batch_end]

        logger.info(f"Processing batch {batch_idx + 1}/{batch_count}...")

        # Using ThreadPoolExecutor to parallelize the process for the current batch
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = {executor.submit(process_changeset, row): row['changeset_id'] for _, row in df_batch.iterrows()}

            # Collecting results as they finish
            for future in futures:
                result = future.result()  # Wait for the task to complete
                all_changeset_data.append(result)

        logger.info(f"Batch {batch_idx + 1}/{batch_count} completed. Sleeping for 1 minute.")
        time.sleep(60)  # Sleep for 1 minute after processing each batch

    return all_changeset_data


# Running the multi-threaded fetch and process
logger.info("Fetching detailed changeset data and user info from API...")
all_changeset_data = fetch_and_process_changesets_in_parallel(df_merged)

# Convert the final data to a DataFrame
df_final = pd.DataFrame(all_changeset_data)

# Optionally save to a CSV or Parquet file
df_final.to_parquet("../../data/changeset_data/output/osm_unlabelled_changeset_features_with_user_info_2022_Q2.parquet",
                    index=False)

# Log the head of the final DataFrame
logger.info(df_final.head())
