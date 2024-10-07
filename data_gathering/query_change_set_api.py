import gc
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from tqdm import tqdm  # For progress bar

from logger_config import logger

# Read the TSV file containing changeset IDs and labels
file_path = "../data/ovid_data/ovid_labels.tsv"
changeset_data = pd.read_csv(file_path, sep='\t')

# Extracting changeset IDs from the DataFrame
changeset_ids = changeset_data['changeset_id'].tolist()

# Path to the Parquet file containing changeset information
parquet_file_path = "../data/changeset_data/osm_changesets_ovid.parquet"

logger.info("Reading OSM history files which has entries for OVID changesets...")
# Read the Parquet file containing changeset metadata
df_changesets = pd.read_parquet(parquet_file_path)


# Function to fetch detailed changeset data from OSM API (for nodes/ways/relations)
def fetch_changeset_details(changeset_id):
    try:
        api_url = f"https://api.openstreetmap.org/api/0.6/changeset/{changeset_id}/download"
        response = requests.get(api_url, timeout=10)  # 10 second timeout
        response.raise_for_status()  # Raise error for bad HTTP response
        return response.text
    except requests.exceptions.RequestException as e:
        logger.info(f"Error fetching changeset {changeset_id}: {e}")
        return None


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
        logger.info(f"Error fetching user {uid} info: {e}")
        return {
            'uid': uid,
            'account_created': None,
            'changes_count': 0
        }


# Initialize progress bar for API calls to fetch detailed changeset data
all_changeset_data = []

# Create a DataFrame to hold the features of the changesets
df_features = df_changesets[df_changesets['changeset_id'].isin(changeset_ids)]

# Delete the large DataFrame to free up memory
del df_changesets  # Remove the DataFrame from memory
gc.collect()  # Force garbage collection to free memory immediately

# Merge the changeset metadata with labels
df_merged = pd.merge(df_features, changeset_data, on='changeset_id', how='inner')

with tqdm(total=len(df_merged), desc="Fetching detailed changeset data and user info from API") as pbar:
    for index, row in df_merged.iterrows():
        changeset_id = row['changeset_id']
        uid = row['user_id']

        # Fetch detailed changeset data (for counting nodes/ways/relations)
        details = fetch_changeset_details(changeset_id)
        if details:
            nodes, ways, relations = parse_changeset_data(details)
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

        all_changeset_data.append(row_data)
        pbar.update(1)

# Convert the final data to a DataFrame
df_final = pd.DataFrame(all_changeset_data)

# Optionally save to a CSV or Parquet file
df_final.to_parquet("../data/changeset_data/osm_labelled_changeset_features_with_user_info1.parquet", index=False)

# logger.info the head of the final DataFrame
logger.info(df_final.head())
