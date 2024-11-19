import math
import os

import pandas as pd
from tqdm import tqdm

from logger.logger_config import logger

historical_edits = {}

# Path to save and load extracted features in Parquet format
FEATURES_FILE = "data/contribution_data/extracted_features_v3.parquet"


# TODO: Compare the features of other contributions with same changeset id and analyse the difference.

def calculate_user_edit_frequency(contributions):
    """
    Calculate user edit frequencies over different time windows.

    Returns a dictionary with user_id as keys and a dictionary of frequencies as values.
    Frequencies include:
    - 'edit_frequency_7d': Edits per day over the last 7 days.
    - 'edit_frequency_30d': Edits per day over the last 30 days.
    - 'edit_frequency_all': Edits per week over the entire history.
    """
    from datetime import timedelta

    user_edit_frequencies = {}
    logger.info("Calculating user edit frequencies over different time windows...")

    # Get the maximum timestamp in the dataset to define the end point of the time windows
    max_timestamp = contributions['valid_from'].max()

    # Define time windows
    windows = {
        '7d': timedelta(days=7),
        '30d': timedelta(days=30),
        '14d': timedelta(days=14),
        '60d': timedelta(days=60),
        '180d': timedelta(days=180),
        '365d': timedelta(days=365)
    }

    # Iterate over users
    for user_id, group in tqdm(contributions.groupby('user_id'), desc="User Edit Frequency"):
        user_contributions = group.sort_values('valid_from')

        # Initialize frequency dictionary for the user
        frequencies = {}

        # Calculate frequencies for each window
        for window_name, window_timedelta in windows.items():
            window_start = max_timestamp - window_timedelta
            # Filter contributions within the time window
            window_contributions = user_contributions[user_contributions['valid_from'] >= window_start]
            total_edits = len(window_contributions)
            days_in_window = window_timedelta.days
            # Calculate edits per day
            edit_frequency = total_edits / days_in_window
            frequencies[f'edit_frequency_{window_name}'] = edit_frequency

        user_edit_frequencies[user_id] = frequencies

    return user_edit_frequencies


def get_continents_for_bbox(xmin, xmax, ymin, ymax):
    """
    Determine all continents covered by a bounding box.

    Parameters:
    - xmin, xmax: Longitude boundaries of the bounding box.
    - ymin, ymax: Latitude boundaries of the bounding box.

    Returns:
    - continents: List of continents that fall within the bounding box.
    """
    continents = []

    # Define approximate boundaries for each continent
    continent_boundaries = {
        'Africa': (-34.83333, 37.09222, -17.62500, 51.20833),  # (min_lat, max_lat, min_lon, max_lon)
        'Antarctica': (-90, -60, -180, 180),
        'Asia': (11.16056, 77.71917, 25.06667, 168.95833),
        'Europe': (34.50500, 71.18528, -24.95583, 68.93500),
        'North America': (5.49955, 83.16210, -168.17625, -52.23304),
        'South America': (-56.10273, 12.45777, -81.76056, -34.72999),
        'Oceania': (-47.28639, -8.23306, 110.95167, 179.85917),
    }

    # Check intersection of the bounding box with each continent's boundaries
    for continent, (lat_min, lat_max, lon_min, lon_max) in continent_boundaries.items():
        if not (xmax < lon_min or xmin > lon_max or ymax < lat_min or ymin > lat_max):
            continents.append(continent)

    # If no continents match, assign 'Other'
    return continents if continents else ['Other']


def calculate_time_since_last_edit(contribution, contribution_df):
    """
    Calculate the time since the last edit for a specific contribution.
    If the user has no previous edits, return a large placeholder value (e.g., a predefined max value).

    :param contribution: A row (Series) from the DataFrame representing the current contribution.
    :param contribution_df: The full DataFrame of contributions, used to find previous contributions by the same user.
    :return: Time since the last edit in hours or a placeholder for first-time contributors.
    """
    user_id = contribution['user_id']

    # Filter contributions to get only previous contributions made by the same user
    user_contributions = contribution_df[(contribution_df['user_id'] == user_id) &
                                         (contribution_df['valid_from'] < contribution['valid_from'])]

    # Check if the user has previous contributions
    if not user_contributions.empty:
        last_edit = user_contributions['valid_from'].max()  # Get the most recent previous contribution
        # Calculate time since last edit in hours
        time_since_last_edit = (contribution['valid_from'] - last_edit).total_seconds() / 3600.0
    else:
        # Assign a large placeholder value (e.g., 10 years in hours) for first-time contributors
        time_since_last_edit = 10 * 365 * 24  # 10 years in hours

    return time_since_last_edit


def calculate_historical_validity(contribution_df, edit_threshold=10):
    """
    Calculate historical validity based on the number of edits made to each feature (osm_id).

    :param contribution_df: DataFrame containing contribution data, including 'osm_id' and 'valid_from' columns.
    :param edit_threshold: The number of edits that define "frequent modifications".
    :return: A dictionary with osm_id as the key and a historical validity score (1 or 0) as the value.
    """
    logger.info("Calculating historical validity for all features...")

    # Initialize an empty dictionary for historical validity
    historical_validity = {}

    # Use tqdm to add a progress bar
    for osm_id, group in tqdm(contribution_df.groupby('osm_id'), desc="Historical Validity Calculation"):
        feature_edit_counts = len(group)
        historical_validity[osm_id] = 0 if feature_edit_counts > edit_threshold else 1

    return historical_validity


def calculate_perimeter(geometry):
    """
    Calculate the perimeter of the polygon geometry.

    :param geometry: POLYGON geometry as a list of coordinate tuples.
    :return: Perimeter of the polygon.
    """
    coordinates = geometry['coordinates'][0]  # Assuming first polygon ring
    perimeter = 0.0
    for i in range(1, len(coordinates)):
        # Calculate distance between consecutive points
        x1, y1 = coordinates[i - 1]
        x2, y2 = coordinates[i]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        perimeter += distance
    return perimeter


# TODO: Use this funciton somewhere
def calculate_num_vertices(geometry):
    """
    Calculate the number of vertices of the polygon geometry.

    :param geometry: POLYGON geometry as a list of coordinate tuples.
    :return: Number of vertices (points).
    """
    coordinates = geometry['coordinates'][0]  # Assuming first polygon ring
    return len(coordinates) - 1  # Exclude closing point which is a repeat of the first


def get_grid_cell_id(lat, lon, grid_size=0.1):
    """
    Assigns a grid cell ID to the given latitude and longitude.

    Parameters:
    - lat: Latitude value.
    - lon: Longitude value.
    - grid_size: Size of the grid cells in degrees.

    Returns:
    - grid_cell_id: A string representing the grid cell.
    """
    x_index = int((lon + 180) / grid_size)
    y_index = int((lat + 90) / grid_size)
    grid_cell_id = f"{x_index}_{y_index}"
    return grid_cell_id


def extract_features(contribution_df):
    feature_list = []
    user_edit_frequencies = calculate_user_edit_frequency(contribution_df)
    historical_validity = calculate_historical_validity(contribution_df, 8)

    logger.info(f"Extracting the features...")

    for index, contribution in tqdm(contribution_df.iterrows(), total=len(contribution_df), desc="Feature Extraction"):
        features = {}

        # 1. User Behavior Features
        user_id = contribution['user_id']
        features['user_id'] = contribution['user_id']
        # features['user_edit_frequency'] = user_edit_frequencies.get(user_id, 0)
        # Add user edit frequencies for different time windows
        user_frequency = user_edit_frequencies.get(user_id, {})
        features['user_edit_frequency_7d'] = user_frequency.get('edit_frequency_7d', 0)
        features['user_edit_frequency_14d'] = user_frequency.get('edit_frequency_14d', 0)
        features['user_edit_frequency_30d'] = user_frequency.get('edit_frequency_30d', 0)
        features['user_edit_frequency_60d'] = user_frequency.get('edit_frequency_60d', 0)
        features['user_edit_frequency_180d'] = user_frequency.get('edit_frequency_180d', 0)
        features['user_edit_frequency_365d'] = user_frequency.get('edit_frequency_365d', 0)
        features['user_edit_frequency_all'] = user_frequency.get('edit_frequency_all', 0)

        editor = contribution['changeset']['editor'].split('/')[0]
        features['editor_used'] = editor

        # 2. Geometric Features
        features['area_delta'] = contribution['area_delta']
        features['length_delta'] = contribution['length_delta']
        features['area'] = contribution['area']
        features['length'] = contribution['length']
        bbox_size = (contribution['xmax'] - contribution['xmin']) * (contribution['ymax'] - contribution['ymin'])
        features['bounding_box_size'] = bbox_size
        features['geometry_valid'] = int(contribution['geometry_valid'])  # 1 if True, 0 if False

        features['xmax'] = contribution['xmax']
        features['xmin'] = contribution['xmin']
        features['ymax'] = contribution['ymax']
        features['ymin'] = contribution['ymin']

        # 3. Temporal Features
        features['time_since_last_edit'] = calculate_time_since_last_edit(contribution, contribution_df)
        timestamp = contribution['changeset']['timestamp']
        hour = timestamp.hour

        # Convert hour into categorical time segments
        if 5 <= hour < 12:
            features['time_of_day'] = 'morning'
        elif 12 <= hour < 17:
            features['time_of_day'] = 'afternoon'
        elif 17 <= hour < 21:
            features['time_of_day'] = 'evening'
        else:
            features['time_of_day'] = 'night'

        # Add day_of_week and is_weekend features
        features['day_of_week'] = timestamp.weekday()  # 0 = Monday, 6 = Sunday
        features['is_weekend'] = int(timestamp.weekday() >= 5)  # 1 if Saturday or Sunday, else 0

        # 4. Contribution Content Features
        tags_before = dict(contribution['tags_before'])
        tags_after = dict(contribution['tags'])

        # Calculate tags added, removed, and modified
        tags_added_set = set(tags_after.keys()) - set(tags_before.keys())
        tags_removed_set = set(tags_before.keys()) - set(tags_after.keys())
        tags_modified_set = set(tag for tag in tags_before if tag in tags_after and tags_before[tag] != tags_after[tag])

        tags_added = len(tags_added_set)
        tags_removed = len(tags_removed_set)
        tags_modified = len(tags_modified_set)
        tags_changed = tags_added + tags_removed + tags_modified

        features['tags_added'] = tags_added
        features['tags_removed'] = tags_removed
        features['tags_modified'] = tags_modified
        features['tags_changed'] = tags_changed
        features['total_tags'] = len(tags_after)

        # Expand the list of key tags based on OSM domain knowledge
        key_tags = ['name', 'boundary', 'population', 'highway', 'building', 'landuse',
                    'amenity', 'natural', 'waterway', 'place', 'railway', 'shop', 'leisure']

        # For each key tag, check if it was added, removed, or modified
        for tag in key_tags:
            tag_before = tags_before.get(tag)
            tag_after = tags_after.get(tag)
            features[f'{tag}_added'] = int(tag in tags_added_set)
            features[f'{tag}_removed'] = int(tag in tags_removed_set)
            features[f'{tag}_modified'] = int(tag in tags_modified_set)
            features[f'{tag}_changed'] = int(
                (tag_before is not None or tag_after is not None) and tag_before != tag_after)

        features['osm_type'] = contribution["osm_type"]
        features['osm_id'] = contribution["osm_id"]
        features['osm_version'] = contribution["osm_version"]
        features['contribution_type'] = contribution["contrib_type"]
        features['members'] = contribution["members"]
        features['status'] = contribution["status"]

        # 5. Spatial Features
        xmin, xmax = contribution['xmin'], contribution['xmax']
        ymin, ymax = contribution['ymin'], contribution['ymax']
        features['bbox_x_range'] = xmax - xmin
        features['bbox_y_range'] = ymax - ymin

        latitude = contribution['centroid']['y']
        longitude = contribution['centroid']['x']

        # Compute grid cell ID
        grid_cell_id = get_grid_cell_id(latitude, longitude, grid_size=0.1)
        features['grid_cell_id'] = grid_cell_id

        features['centroid_x'] = longitude
        features['centroid_y'] = latitude

        country_iso = contribution['country_iso_a3']
        features['country_count'] = len(country_iso)
        features['countries'] = country_iso
        features['xzcode'] = contribution["xzcode"]
        features['continents'] = get_continents_for_bbox(xmin, xmax, ymin, ymax)

        # 6. Contextual and Historical Features
        features['historical_validity'] = historical_validity.get(contribution['osm_id'])

        # 7. Derived Features
        features['tag_density'] = len(contribution['tags']) / contribution['area'] if contribution['area'] > 0 else 0
        previous_area = contribution.get('previous_area', contribution['area'] - contribution['area_delta'])
        change_ratio = contribution['area_delta'] / previous_area if previous_area > 0 else contribution['area_delta']
        features['change_ratio'] = change_ratio

        # 8. Changeset Features
        comment = next((tag[1] for tag in contribution['changeset']['tags'] if tag[0] == 'comment'), "")
        features['changeset_comment_length'] = len(comment)
        source = next((tag[1] for tag in contribution['changeset']['tags'] if tag[0] == 'source'), "")
        reliable_sources = ['Bing Aerial Imagery', 'Esri World Imagery']
        features['source_reliability'] = 1 if any(s in source for s in reliable_sources) else 0
        features['changeset_id'] = contribution['changeset']['id']

        # 9. Map Features
        map_features = contribution['map_features']  # assuming this is the field name in the DataFrame
        # map_features = json.loads(map_features_str)  # Convert string to dictionary

        # Add each map feature as a separate column
        for feature_name, feature_value in map_features.items():
            features[feature_name] = int(feature_value)  # Store as 0 or 1

        # Exclude the 'geometry' field from the DataFrame to avoid using it directly.
        # features['geometry'] = contribution['geometry']
        features['vandalism'] = contribution['vandalism']  # This is the target label

        feature_list.append(features)

    # Convert to DataFrame
    features_df = pd.DataFrame(feature_list)
    return features_df


# Check if the feature file exists, load it, otherwise generate the features
def get_or_generate_features(contribution_df, force_compute_features):
    if os.path.exists(FEATURES_FILE) and not force_compute_features:
        logger.info(f"Loading features from {FEATURES_FILE}...")
        features_df = pd.read_parquet(FEATURES_FILE)
    else:
        logger.info("Extracting features...")
        features_df = extract_features(contribution_df)
        logger.info(f"Saving features to {FEATURES_FILE}...")
        features_df.to_parquet(FEATURES_FILE)

    logger.info(f"features_df.shape: {features_df.shape}")
    return features_df
