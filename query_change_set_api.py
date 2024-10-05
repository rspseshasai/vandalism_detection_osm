import requests
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm  # For progress bar

# List of all changeset IDs to be processed
changeset_ids = [
    66282770, 45575075, 53053690, 26894116, 25445701, 24945451,
    28075756, 63077866, 52357523, 44733212, 56346301, 57652917,
    32586122, 38025732, 66522179, 59026400, 53101846, 73008776,
    54425893, 41371274, 60128676, 59654024, 74770907, 54511384,
    22253435, 45937299, 56106706, 42436654, 45490766, 51029766,
    64268009, 44497155, 73258044, 30664954, 54374138, 22294041,
    34657625, 65028008, 57629554, 41539987, 51893931, 43239174,
    59227852, 44145101, 35493795, 42483432, 63276912, 19947683,
    54574357, 75160933, 31320139, 77173920
]


# Function to fetch detailed changeset data from OSM API
def fetch_changeset_details(changeset_id):
    try:
        api_url = f"https://api.openstreetmap.org/api/0.6/changeset/{changeset_id}/download"
        response = requests.get(api_url, timeout=10)  # 10 second timeout
        response.raise_for_status()  # Raise error for bad HTTP response
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching changeset {changeset_id}: {e}")
        return None


# Function to fetch additional metadata for changeset
def fetch_changeset_metadata(changeset_id):
    try:
        api_url = f"https://api.openstreetmap.org/api/0.6/changeset/{changeset_id}"
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses
        root = ET.fromstring(response.text)
        changeset = root.find("changeset")
        if changeset is not None:
            metadata = {
                "changeset_id": changeset_id,
                "user": changeset.attrib.get("user"),
                "created_at": changeset.attrib.get("created_at"),
                "closed_at": changeset.attrib.get("closed_at"),
                "num_changes": changeset.attrib.get("changes_count"),
                "min_lat": changeset.attrib.get("min_lat"),
                "min_lon": changeset.attrib.get("min_lon"),
                "max_lat": changeset.attrib.get("max_lat"),
                "max_lon": changeset.attrib.get("max_lon"),
            }
            return metadata
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching metadata for changeset {changeset_id}: {e}")
        return None


# Function to fetch user info for a specific user ID
def fetch_user_info(user_id):
    try:
        api_url = f"https://api.openstreetmap.org/api/0.6/user/{user_id}.json"
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses
        user_data = response.json()["user"]

        user_info = {
            "user_id": user_data["id"],
            "display_name": user_data["display_name"],
            "account_created": user_data["account_created"],
            "changesets_count": user_data["changesets"]["count"],
            "traces_count": user_data["traces"]["count"],
            "blocks_received_count": user_data["blocks"]["received"]["count"],
            "blocks_issued_count": user_data["blocks"]["issued"]["count"],
            "roles": user_data.get("roles", []),
            "description": user_data.get("description", ""),
        }
        return user_info
    except requests.exceptions.RequestException as e:
        print(f"Error fetching user info for user ID {user_id}: {e}")
        return None
    except KeyError:
        print(f"No user found for user ID {user_id}")
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


# Fetch and process changesets, combining metadata and API details
all_changeset_data = []

for changeset_id in tqdm(changeset_ids, desc="Processing changesets"):
    # Fetch metadata for the changeset
    metadata = fetch_changeset_metadata(changeset_id)
    if metadata:
        # Fetch additional details using the changeset ID
        details = fetch_changeset_details(changeset_id)
        if details:
            nodes, ways, relations = parse_changeset_data(details)
            metadata.update({
                "no_nodes": sum(nodes.values()),
                "no_ways": sum(ways.values()),
                "no_relations": sum(relations.values()),
                "no_creates": nodes['create'] + ways['create'] + relations['create'],
                "no_modifications": nodes['modify'] + ways['modify'] + relations['modify'],
                "no_deletions": nodes['delete'] + ways['delete'] + relations['delete'],
            })

        # Fetch user information
        user_info = fetch_user_info(metadata['user'])
        if user_info:
            metadata.update(user_info)

        all_changeset_data.append(metadata)

# Convert changesets to DataFrame
df = pd.DataFrame(all_changeset_data)

# Print the head of the DataFrame
print(df.head())
