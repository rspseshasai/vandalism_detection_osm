import math
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from config import SPLIT_METHOD, DATASET_TYPE, TEST_CHANGESET_IDS
from config import logger


def calculate_user_edit_frequency(contributions):
    """
    Calculate user edit frequencies over different time windows.

    Returns a dictionary with user_id as keys and a dictionary of frequencies as values.
    Frequencies include:
    - 'edit_frequency_7d': Edits per day over the last 7 days.
    - 'edit_frequency_14d': Edits per day over the last 14 days.
    - 'edit_frequency_30d': Edits per day over the last 30 days.
    - 'edit_frequency_60d': Edits per day over the last 60 days.
    - 'edit_frequency_180d': Edits per day over the last 180 days.
    - 'edit_frequency_365d': Edits per day over the last 365 days.
    - 'edit_frequency_all': Edits per day over the entire history.
    """
    user_edit_frequencies = {}
    logger.info("Calculating user edit frequencies over different time windows...")

    # Get the maximum timestamp in the dataset to define the end point of the time windows
    max_timestamp = contributions['valid_from'].max()

    # Define time windows
    windows = {
        '7d': timedelta(days=7),
        '14d': timedelta(days=14),
        '30d': timedelta(days=30),
        '60d': timedelta(days=60),
        '180d': timedelta(days=180),
        '365d': timedelta(days=365),
        'all': max_timestamp - contributions['valid_from'].min(),
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
            days_in_window = window_timedelta.days if window_timedelta.days > 0 else 1
            # Calculate edits per day
            edit_frequency = total_edits / days_in_window
            frequencies[f'edit_frequency_{window_name}'] = edit_frequency

        user_edit_frequencies[user_id] = frequencies

    return user_edit_frequencies


def get_continents_for_bbox(xmin, xmax, ymin, ymax):
    """
    Determine all continents covered by a bounding box.

    Parameters:
    - xmin (float): Minimum longitude of the bounding box.
    - xmax (float): Maximum longitude of the bounding box.
    - ymin (float): Minimum latitude of the bounding box.
    - ymax (float): Maximum latitude of the bounding box.

    Returns:
    - List[str]: Continents that fall within the bounding box.
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
    If the user has no previous edits, return a large placeholder value.

    Parameters:
    - contribution (pd.Series): The current contribution.
    - contribution_df (pd.DataFrame): DataFrame containing all contributions.

    Returns:
    - float: Time since the last edit in hours.
    """
    user_id = contribution['user_id']

    # Filter contributions to get only previous contributions made by the same user
    user_contributions = contribution_df[
        (contribution_df['user_id'] == user_id) &
        (contribution_df['valid_from'] < contribution['valid_from'])
        ]

    # Check if the user has previous contributions
    if not user_contributions.empty:
        last_edit = user_contributions['valid_from'].max()  # Most recent previous contribution
        # Calculate time since last edit in hours
        time_since_last_edit = (contribution['valid_from'] - last_edit).total_seconds() / 3600.0
    else:
        # Assign a large placeholder value (e.g., 10 years in hours) for first-time contributors
        time_since_last_edit = 10 * 365 * 24  # 10 years in hours

    return time_since_last_edit


import logging
from multiprocessing import Pool, cpu_count


def _process_chunk(df_chunk):
    """
    Process a chunk of the DataFrame that is guaranteed to contain whole osm_id groups.
    Assumes df_chunk is already sorted by osm_id and valid_from.
    """
    edit_frequency = {}
    number_of_past_edits = {}
    last_edit_time = {}

    current_osm_id = None
    current_indices = []
    current_timestamps = []

    # Helper function to finalize a group once we reach the end of that osm_id
    def finalize_group(osm_id, indices, timestamps):
        if not indices:
            return
        num_edits = len(indices)

        # Determine validity
        if num_edits <= 5:
            validity = 'rarely_edited'
        elif num_edits <= 20:
            validity = 'moderately_edited'
        else:
            validity = 'frequently_edited'

        edit_frequency[osm_id] = validity
        number_of_past_edits[osm_id] = num_edits

        # Compute last edit times
        last_times = [0]  # First edit has no previous edit
        for i in range(1, num_edits):
            time_since_last = (timestamps[i] - timestamps[i - 1]) / np.timedelta64(1, 'h')
            last_times.append(time_since_last)

        for idx, lt in zip(indices, last_times):
            last_edit_time[idx] = lt

    # Iterate over chunk rows
    for row in df_chunk.itertuples():
        osm_id = getattr(row, 'osm_id')
        valid_from = getattr(row, 'valid_from')
        idx = row.Index

        # If we moved to a new osm_id, finalize the previous one
        if osm_id != current_osm_id and current_osm_id is not None:
            finalize_group(current_osm_id, current_indices, current_timestamps)
            current_indices = []
            current_timestamps = []

        current_osm_id = osm_id
        current_indices.append(idx)
        current_timestamps.append(valid_from)

    # Finalize the last group in the chunk
    if current_osm_id is not None:
        finalize_group(current_osm_id, current_indices, current_timestamps)

    return edit_frequency, number_of_past_edits, last_edit_time


def calculate_edit_history_features(contribution_df, num_partitions=None):
    """
    Calculate historical features for each OSM element in parallel.

    Parameters:
    - contribution_df (pd.DataFrame): DataFrame containing contribution data.
    - num_partitions (int, optional): Number of parallel partitions. Defaults to number of CPUs.

    Returns:
    - Tuple[dict, dict, dict]: Dictionaries for edit frequency, number of past edits,
      and last edit time per osm_id.
    """
    logger.info("Calculating historical features for OSM elements...")

    if num_partitions is None:
        num_partitions = cpu_count()

    # Sort the entire DataFrame once by osm_id and valid_from
    logger.info("Sorting the DataFrame...")
    contribution_df = contribution_df.sort_values(['osm_id', 'valid_from'])

    # Extract series for detecting group boundaries
    osm_ids = contribution_df['osm_id'].values
    n = len(osm_ids)

    # Find indices where osm_id changes, so we can partition by osm_id boundaries
    logger.info("Determining partitions...")
    change_points = np.where(osm_ids[:-1] != osm_ids[1:])[0] + 1
    # Add start and end boundaries
    boundaries = np.concatenate(([0], change_points, [n]))

    # We'll aim for num_partitions large chunks by dividing the data evenly by count of rows
    # But we must ensure chunks start and end at osm_id boundaries
    desired_chunk_size = n // num_partitions
    chunk_starts = [0]
    for i in range(1, num_partitions):
        # Find the boundary close to i * desired_chunk_size
        target = i * desired_chunk_size
        # Closest boundary to target
        boundary_idx = np.searchsorted(boundaries, target, side='right') - 1
        chunk_starts.append(boundaries[boundary_idx])
    chunk_starts.append(n)

    # Create chunk slices
    chunk_slices = []
    for i in range(len(chunk_starts)-1):
        start = chunk_starts[i]
        end = chunk_starts[i+1]
        chunk_slices.append((start, end))

    # Extract chunks
    # We'll store DataFrame views for each chunk
    logger.info("Creating chunks for parallel processing...")
    df_chunks = [contribution_df.iloc[slice(*slc)] for slc in chunk_slices]

    # Process in parallel
    logger.info("Starting parallel computation...")
    with Pool(processes=num_partitions) as pool:
        results = list(tqdm(pool.imap(_process_chunk, df_chunks), total=len(df_chunks)))

    # Combine results
    logger.info("Combining results from all chunks...")
    edit_frequency = {}
    number_of_past_edits = {}
    last_edit_time = {}

    for ef, npe, let in results:
        edit_frequency.update(ef)
        number_of_past_edits.update(npe)
        last_edit_time.update(let)

    return edit_frequency, number_of_past_edits, last_edit_time


def get_grid_cell_id(lat, lon, grid_size=0.1):
    """
    Assign a grid cell ID to the given latitude and longitude.

    Parameters:
    - lat (float): Latitude value.
    - lon (float): Longitude value.
    - grid_size (float): Size of the grid cells in degrees.

    Returns:
    - str: A string representing the grid cell.
    """
    x_index = int((lon + 180) / grid_size)
    y_index = int((lat + 90) / grid_size)
    return f"{x_index}_{y_index}"


def extract_user_behavior_features(contribution, user_edit_frequencies):
    features = {}
    user_id = contribution['user_id']
    features['user_id'] = user_id
    user_frequency = user_edit_frequencies.get(user_id, {})
    features.update(user_frequency)

    editor = contribution['changeset']['editor'].split('/')[0]
    features['editor_used'] = editor
    return features


def extract_geometric_features(contribution):
    features = {}
    features['area_delta'] = contribution['area_delta']
    features['length_delta'] = contribution['length_delta']
    features['area'] = contribution['area']
    features['length'] = contribution['length']
    features['geometry_type'] = contribution['geometry_type']
    bbox_size = (contribution['xmax'] - contribution['xmin']) * (contribution['ymax'] - contribution['ymin'])
    features['bounding_box_size'] = bbox_size
    features['geometry_valid'] = int(contribution['geometry_valid'])  # 1 if True, 0 if False
    features['xmax'] = contribution['xmax']
    features['xmin'] = contribution['xmin']
    features['ymax'] = contribution['ymax']
    features['ymin'] = contribution['ymin']
    return features


def extract_temporal_features(contribution, contribution_df):
    features = {}
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
    features['is_weekend'] = int(timestamp.weekday() >= 5)  # 1 if Saturday or Sunday

    # Time since last edit
    features['time_since_last_edit'] = calculate_time_since_last_edit(contribution, contribution_df)

    # For Temporal Evaluation
    if SPLIT_METHOD == 'temporal':
        features['date_created'] = contribution['valid_from']
    return features


def extract_content_features(contribution):
    features = {}
    tags_before = dict(contribution['tags_before'])
    tags_after = dict(contribution['tags'])

    # Calculate tags added, removed, and modified
    tags_added_set = set(tags_after.keys()) - set(tags_before.keys())
    tags_removed_set = set(tags_before.keys()) - set(tags_after.keys())
    tags_modified_set = {
        tag for tag in tags_before if tag in tags_after and tags_before[tag] != tags_after[tag]
    }

    features['tags_added'] = len(tags_added_set)
    features['tags_removed'] = len(tags_removed_set)
    features['tags_modified'] = len(tags_modified_set)
    features['tags_changed'] = features['tags_added'] + features['tags_removed'] + features['tags_modified']
    features['total_tags'] = len(tags_after)

    # Key tags based on OSM domain knowledge
    key_tags = [
        'name', 'boundary', 'population', 'highway', 'building', 'landuse',
        'amenity', 'natural', 'waterway', 'place', 'railway', 'shop', 'leisure'
    ]

    # Check if each key tag was added, removed, or modified
    for tag in key_tags:
        tag_before = tags_before.get(tag)
        tag_after = tags_after.get(tag)
        features[f'{tag}_added'] = int(tag in tags_added_set)
        features[f'{tag}_removed'] = int(tag in tags_removed_set)
        features[f'{tag}_modified'] = int(tag in tags_modified_set)
        features[f'{tag}_changed'] = int(
            (tag_before is not None or tag_after is not None) and tag_before != tag_after
        )

    features['osm_type'] = contribution['osm_type']
    features['osm_id'] = contribution['osm_id']
    features['osm_version'] = contribution['osm_version']
    features['contribution_type'] = contribution['contrib_type']
    features['members'] = contribution['members']
    features['status'] = contribution['status']

    return features


def extract_spatial_features(contribution):
    features = {}
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
    features['xzcode'] = contribution['xzcode']
    features['continents'] = get_continents_for_bbox(xmin, xmax, ymin, ymax)

    return features


def extract_historical_features(
        contribution, edit_frequency, num_past_edits, last_edit_time
):
    features = {}
    osm_id = contribution['osm_id']
    features['edit_frequency_of_osm_element'] = edit_frequency.get(osm_id, 'unknown')
    features['number_of_past_edits_of_osm_element'] = num_past_edits.get(osm_id, 0)
    features['last_edit_time_of_osm_element'] = last_edit_time.get(contribution.name, 0)
    return features


def extract_derived_features(contribution):
    features = {}
    # Tag Density: Tags per unit area
    if contribution['area'] > 0:
        raw_tag_density = len(contribution['tags']) / contribution['area']
        features['tag_density'] = min(math.log1p(raw_tag_density), 10)  # Log transform and cap at 10
    else:
        features['tag_density'] = 0  # Default for zero or invalid area

    # Relative Area Change: Ratio of area_delta to the previous area
    previous_area = contribution.get('previous_area', contribution['area'] - contribution['area_delta'])
    if previous_area > 0:
        raw_change_ratio = contribution['area_delta'] / previous_area
        features['relative_area_change'] = min(max(raw_change_ratio, -10), 10)  # Cap between -10 and 10
    else:
        features['relative_area_change'] = min(max(contribution['area_delta'], -10), 10)

    return features


def extract_changeset_features(contribution):
    features = {}
    # Changeset Comment
    comment = next((tag[1] for tag in contribution['changeset']['tags'] if tag[0] == 'comment'), "")
    features['changeset_comment_length'] = len(comment)

    # Source Reliability
    source = next((tag[1] for tag in contribution['changeset']['tags'] if tag[0] == 'source'), "")
    reliable_sources = [
        'bing aerial imagery',
        'esri world imagery',
        'mapbox',
        'digitalglobe',
        'here maps',
        'maxar technologies',
        'google satellite',
    ]
    reliable_sources_lower = [s.lower() for s in reliable_sources]
    source_lower = source.lower()
    features['source_reliability'] = int(any(s in source_lower for s in reliable_sources_lower))
    features['changeset_id'] = contribution['changeset']['id']
    features['source_used'] = source if source else 'unknown'

    return features


def extract_map_features(contribution):
    features = {}
    map_features = contribution['map_features']  # Assuming this is a dict

    # Add each map feature as a separate column
    for feature_name, feature_value in map_features.items():
        if feature_value == None:
            feature_value = False
        features[feature_name] = int(feature_value)  # Store as 0 or 1
    return features


def extract_features(contribution_df, is_training):
    feature_list = []

    user_edit_frequencies = calculate_user_edit_frequency(contribution_df)
    edit_freq, num_past_edits, last_edit_time = calculate_edit_history_features(contribution_df)

    logger.info("Extracting features...")

    for index, contribution in tqdm(
            contribution_df.iterrows(), total=len(contribution_df), desc="Feature Extraction"
    ):
        features = {}

        # 1. User Behavior Features
        features.update(extract_user_behavior_features(contribution, user_edit_frequencies))

        # 2. Geometric Features
        features.update(extract_geometric_features(contribution))

        # 3. Temporal Features
        features.update(extract_temporal_features(contribution, contribution_df))

        # 4. Contribution Content Features
        features.update(extract_content_features(contribution))

        # 5. Spatial Features
        features.update(extract_spatial_features(contribution))

        # 6. Contextual and Historical Features
        features.update(extract_historical_features(contribution, edit_freq, num_past_edits, last_edit_time))

        # 7. Derived Features
        features.update(extract_derived_features(contribution))

        # 8. Changeset Features
        features.update(extract_changeset_features(contribution))

        # 9. Map Features
        features.update(extract_map_features(contribution))

        # Target label
        if is_training:
            features['vandalism'] = contribution['vandalism']

        feature_list.append(features)

    # Convert to DataFrame
    features_df = pd.DataFrame(feature_list)
    return features_df


def extract_features_changeset(data_df):
    # Ensure you're working on the actual DataFrame, not a slice
    data_df = data_df.copy()

    # Handle missing values in the min/max latitude and longitude columns
    data_df["min_lon"] = data_df["min_lon"].fillna(0)
    data_df["max_lon"] = data_df["max_lon"].fillna(0)
    data_df["min_lat"] = data_df["min_lat"].fillna(0)
    data_df["max_lat"] = data_df["max_lat"].fillna(0)

    # Calculate centroid for each row
    data_df.loc[:, "centroid_x"] = (data_df["min_lon"] + data_df["max_lon"]) / 2
    data_df.loc[:, "centroid_y"] = (data_df["min_lat"] + data_df["max_lat"]) / 2

    # Calculate the length of the changeset comment
    data_df['changeset_comment_length'] = data_df['comment'].fillna("").apply(len)

    # For Temporal Evaluation
    if SPLIT_METHOD == 'temporal':
        data_df.loc[:, 'date_created'] = data_df['created_at']

    return data_df


def get_or_generate_features(data_df, is_training, processed_features_file_path, force_compute_features=False,
                             test_mode=False):
    """
    Load existing features or generate them if not available.

    Parameters:
    - data_df (pd.DataFrame): DataFrame containing contribution or changeset data.
    - force_compute_features (bool): If True, forces re-computation of features.
    - test_mode (bool): If True, limit to 100 entries for testing purposes.

    Returns:
    - pd.DataFrame: DataFrame containing extracted features.
    """
    if os.path.exists(processed_features_file_path) and not force_compute_features:
        logger.info(f"Loading features from {processed_features_file_path}...")
        features_df = pd.read_parquet(processed_features_file_path)

    else:
        logger.info("Extracting features...")
        if DATASET_TYPE == 'changeset':
            features_df = extract_features_changeset(data_df)
        else:
            features_df = extract_features(data_df, is_training)
        logger.info(f"Saving features to {processed_features_file_path}...")
        features_df.to_parquet(processed_features_file_path)

    if is_training:
        if test_mode:
            logger.info("Test mode enabled: Limiting to entries matching Test changeset IDs.")
            features_df = features_df[features_df['changeset_id'].isin(TEST_CHANGESET_IDS)]
        else:
            pass
            # logger.info("Limiting to entries matching common changeset IDs.")
            # features_df = features_df[features_df['changeset_id'].isin(COMMON_CHANGESET_IDS)]

    logger.info(f"Features DataFrame Shape: {features_df.shape}")
    return features_df
