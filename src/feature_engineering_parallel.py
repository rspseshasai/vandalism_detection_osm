import math
import os
from datetime import timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count

from config import SPLIT_METHOD, DATASET_TYPE, TEST_CHANGESET_IDS
from config import logger


def calculate_user_edit_frequency(contributions):
    logger.info("Calculating user edit frequencies over different time windows...")
    user_edit_frequencies = {}

    max_timestamp = contributions['valid_from'].max()

    windows = {
        '7d': timedelta(days=7),
        '14d': timedelta(days=14),
        '30d': timedelta(days=30),
        '60d': timedelta(days=60),
        '180d': timedelta(days=180),
        '365d': timedelta(days=365),
        'all': max_timestamp - contributions['valid_from'].min(),
    }

    for user_id, group in tqdm(contributions.groupby('user_id'), desc="User Edit Frequency"):
        user_contributions = group.sort_values('valid_from')
        frequencies = {}
        for window_name, window_timedelta in windows.items():
            window_start = max_timestamp - window_timedelta
            window_contributions = user_contributions[user_contributions['valid_from'] >= window_start]
            total_edits = len(window_contributions)
            days_in_window = max(window_timedelta.days, 1)
            edit_frequency = total_edits / days_in_window
            frequencies[f'edit_frequency_{window_name}'] = edit_frequency
        user_edit_frequencies[user_id] = frequencies

    return user_edit_frequencies


def get_continents_for_bbox(xmin, xmax, ymin, ymax):
    continents = []

    continent_boundaries = {
        'Africa': (-34.83333, 37.09222, -17.62500, 51.20833),
        'Antarctica': (-90, -60, -180, 180),
        'Asia': (11.16056, 77.71917, 25.06667, 168.95833),
        'Europe': (34.50500, 71.18528, -24.95583, 68.93500),
        'North America': (5.49955, 83.16210, -168.17625, -52.23304),
        'South America': (-56.10273, 12.45777, -81.76056, -34.72999),
        'Oceania': (-47.28639, -8.23306, 110.95167, 179.85917),
    }

    for continent, (lat_min, lat_max, lon_min, lon_max) in continent_boundaries.items():
        if not (xmax < lon_min or xmin > lon_max or ymax < lat_min or ymin > lat_max):
            continents.append(continent)

    return continents if continents else ['Other']


def calculate_all_time_since_last_edit(contribution_df):
    """
    Precompute the time_since_last_edit for each row to avoid global DataFrame lookups later.
    We mimic the logic in `calculate_time_since_last_edit`, but do it in one pass for all rows.
    """

    # Sort by user_id and valid_from so previous contributions for a user are contiguous
    df = contribution_df.sort_values(['user_id', 'valid_from'])
    time_since_last = []
    last_edit_per_user = {}

    for idx, row in df.iterrows():
        user_id = row['user_id']
        current_time = row['valid_from']
        if user_id in last_edit_per_user:
            last_time = last_edit_per_user[user_id]
            delta_hours = (current_time - last_time).total_seconds() / 3600.0
            time_since_last.append((idx, delta_hours))
        else:
            # No previous edit: 10 years in hours
            time_since_last.append((idx, 10 * 365 * 24))
        last_edit_per_user[user_id] = current_time

    # Convert to a dictionary
    ts_dict = {i: t for i, t in time_since_last}

    return ts_dict


def _process_chunk(df_chunk):
    edit_frequency = {}
    number_of_past_edits = {}
    last_edit_time = {}

    current_osm_id = None
    current_indices = []
    current_timestamps = []

    def finalize_group(osm_id, indices, timestamps):
        if not indices:
            return
        num_edits = len(indices)
        if num_edits <= 5:
            validity = 'rarely_edited'
        elif num_edits <= 20:
            validity = 'moderately_edited'
        else:
            validity = 'frequently_edited'

        edit_frequency[osm_id] = validity
        number_of_past_edits[osm_id] = num_edits

        last_times = [0]
        for i in range(1, num_edits):
            time_since_last = (timestamps[i] - timestamps[i - 1]) / np.timedelta64(1, 'h')
            last_times.append(time_since_last)

        for idx, lt in zip(indices, last_times):
            last_edit_time[idx] = lt

    for row in df_chunk.itertuples():
        osm_id = getattr(row, 'osm_id')
        valid_from = getattr(row, 'valid_from')
        idx = row.Index
        if osm_id != current_osm_id and current_osm_id is not None:
            finalize_group(current_osm_id, current_indices, current_timestamps)
            current_indices = []
            current_timestamps = []

        current_osm_id = osm_id
        current_indices.append(idx)
        current_timestamps.append(valid_from)

    if current_osm_id is not None:
        finalize_group(current_osm_id, current_indices, current_timestamps)

    return edit_frequency, number_of_past_edits, last_edit_time


def calculate_edit_history_features(contribution_df, num_partitions=None):
    logger.info("Calculating historical features for OSM elements...")

    if num_partitions is None:
        num_partitions = cpu_count()

    contribution_df = contribution_df.sort_values(['osm_id', 'valid_from'])

    osm_ids = contribution_df['osm_id'].values
    n = len(osm_ids)

    change_points = np.where(osm_ids[:-1] != osm_ids[1:])[0] + 1
    boundaries = np.concatenate(([0], change_points, [n]))

    desired_chunk_size = n // num_partitions
    chunk_starts = [0]
    for i in range(1, num_partitions):
        target = i * desired_chunk_size
        boundary_idx = np.searchsorted(boundaries, target, side='right') - 1
        chunk_starts.append(boundaries[boundary_idx])
    chunk_starts.append(n)

    chunk_slices = []
    for i in range(len(chunk_starts) - 1):
        start = chunk_starts[i]
        end = chunk_starts[i + 1]
        chunk_slices.append((start, end))

    df_chunks = [contribution_df.iloc[slice(*slc)] for slc in chunk_slices]

    with Pool(processes=num_partitions) as pool:
        results = list(tqdm(pool.imap(_process_chunk, df_chunks), total=len(df_chunks)))

    edit_frequency = {}
    number_of_past_edits = {}
    last_edit_time = {}

    for ef, npe, let in results:
        edit_frequency.update(ef)
        number_of_past_edits.update(npe)
        last_edit_time.update(let)

    return edit_frequency, number_of_past_edits, last_edit_time


def get_grid_cell_id(lat, lon, grid_size=0.1):
    x_index = int((lon + 180) / grid_size)
    y_index = int((lat + 90) / grid_size)
    return f"{x_index}_{y_index}"


def extract_user_behavior_features(contribution, user_edit_frequencies):
    features = {}
    user_id = contribution['user_id']
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
    features['geometry_valid'] = int(contribution['geometry_valid'])
    features['xmax'] = contribution['xmax']
    features['xmin'] = contribution['xmin']
    features['ymax'] = contribution['ymax']
    features['ymin'] = contribution['ymin']
    return features


def extract_temporal_features(contribution):
    features = {}
    timestamp = contribution['changeset']['timestamp']
    hour = timestamp.hour

    if 5 <= hour < 12:
        features['time_of_day'] = 'morning'
    elif 12 <= hour < 17:
        features['time_of_day'] = 'afternoon'
    elif 17 <= hour < 21:
        features['time_of_day'] = 'evening'
    else:
        features['time_of_day'] = 'night'

    features['day_of_week'] = timestamp.weekday()
    features['is_weekend'] = int(timestamp.weekday() >= 5)

    # Now we use precomputed time_since_last_edit directly from contribution
    features['time_since_last_edit'] = contribution['time_since_last_edit']

    if SPLIT_METHOD == 'temporal':
        features['date_created'] = contribution['valid_from']
    return features


def extract_content_features(contribution):
    features = {}
    tags_before = dict(contribution['tags_before'])
    tags_after = dict(contribution['tags'])

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

    key_tags = [
        'name', 'boundary', 'population', 'highway', 'building', 'landuse',
        'amenity', 'natural', 'waterway', 'place', 'railway', 'shop', 'leisure'
    ]

    for tag in key_tags:
        features[f'{tag}_added'] = int(tag in tags_added_set)
        features[f'{tag}_removed'] = int(tag in tags_removed_set)
        features[f'{tag}_modified'] = int(tag in tags_modified_set)
        before_val = tags_before.get(tag)
        after_val = tags_after.get(tag)
        features[f'{tag}_changed'] = int((before_val is not None or after_val is not None) and before_val != after_val)

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


def extract_historical_features(contribution, edit_frequency, num_past_edits, last_edit_time):
    features = {}
    osm_id = contribution['osm_id']
    features['edit_frequency_of_osm_element'] = edit_frequency.get(osm_id, 'unknown')
    features['number_of_past_edits_of_osm_element'] = num_past_edits.get(osm_id, 0)
    # Use contribution's index (name) to get last_edit_time_of_osm_element
    features['last_edit_time_of_osm_element'] = last_edit_time.get(contribution['original_index'], 0)
    return features


def extract_derived_features(contribution):
    features = {}
    if contribution['area'] > 0:
        raw_tag_density = len(contribution['tags']) / contribution['area']
        features['tag_density'] = min(math.log1p(raw_tag_density), 10)
    else:
        features['tag_density'] = 0

    previous_area = contribution.get('previous_area', contribution['area'] - contribution['area_delta'])
    if previous_area > 0:
        raw_change_ratio = contribution['area_delta'] / previous_area
        features['relative_area_change'] = min(max(raw_change_ratio, -10), 10)
    else:
        features['relative_area_change'] = min(max(contribution['area_delta'], -10), 10)

    return features


def extract_changeset_features(contribution):
    features = {}
    comment = next((tag[1] for tag in contribution['changeset']['tags'] if tag[0] == 'comment'), "")
    features['changeset_comment_length'] = len(comment)

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
    source_lower = source.lower() if source else ''
    features['source_reliability'] = int(any(s in source_lower for s in reliable_sources))
    features['changeset_id'] = contribution['changeset']['id']
    features['source_used'] = source if source else 'unknown'

    return features


def extract_map_features(contribution):
    features = {}
    map_features = contribution['map_features']
    for feature_name, feature_value in map_features.items():
        if feature_value is None:
            feature_value = False
        features[feature_name] = int(feature_value)
    return features


# ----------- PARALLEL EXTRACTION SETUP ----------- #
GLOBAL_EDIT_FREQ = None
GLOBAL_NUM_PAST_EDITS = None
GLOBAL_LAST_EDIT_TIME = None
GLOBAL_USER_EDIT_FREQ = None
GLOBAL_IS_TRAINING = None


def _init_worker(user_edit_frequencies, edit_freq, num_past_edits, last_edit_time, is_training):
    global GLOBAL_USER_EDIT_FREQ
    global GLOBAL_EDIT_FREQ
    global GLOBAL_NUM_PAST_EDITS
    global GLOBAL_LAST_EDIT_TIME
    global GLOBAL_IS_TRAINING

    GLOBAL_USER_EDIT_FREQ = user_edit_frequencies
    GLOBAL_EDIT_FREQ = edit_freq
    GLOBAL_NUM_PAST_EDITS = num_past_edits
    GLOBAL_LAST_EDIT_TIME = last_edit_time
    GLOBAL_IS_TRAINING = is_training


def _process_records(records):
    # records is a list of row dicts
    # Global vars are set
    results = []
    for contribution in records:
        features = {}
        # 1. User Behavior
        features.update(extract_user_behavior_features(contribution, GLOBAL_USER_EDIT_FREQ))
        # 2. Geometric
        features.update(extract_geometric_features(contribution))
        # 3. Temporal (no df needed now)
        features.update(extract_temporal_features(contribution))
        # 4. Content
        features.update(extract_content_features(contribution))
        # 5. Spatial
        features.update(extract_spatial_features(contribution))
        # 6. Historical
        features.update(
            extract_historical_features(contribution, GLOBAL_EDIT_FREQ, GLOBAL_NUM_PAST_EDITS, GLOBAL_LAST_EDIT_TIME))
        # 7. Derived
        features.update(extract_derived_features(contribution))
        # 8. Changeset
        features.update(extract_changeset_features(contribution))
        # 9. Map
        features.update(extract_map_features(contribution))

        if GLOBAL_IS_TRAINING:
            features['vandalism'] = contribution['vandalism']

        results.append(features)

    return pd.DataFrame(results)


def extract_features(contribution_df, is_training):

    user_edit_frequencies = calculate_user_edit_frequency(contribution_df)
    edit_freq, num_past_edits, last_edit_time = calculate_edit_history_features(contribution_df)

    # Precompute time_since_last_edit for each contribution
    tsle_dict = calculate_all_time_since_last_edit(contribution_df)
    # Add this as a column so we don't need the df in extract_temporal_features
    contribution_df = contribution_df.assign(time_since_last_edit=contribution_df.index.map(tsle_dict))

    # Also store original index for historical features
    contribution_df = contribution_df.assign(original_index=contribution_df.index)
    records = contribution_df.to_dict('records')

    num_partitions = cpu_count()
    chunk_size = len(records) // num_partitions if num_partitions > 1 else len(records)
    index_chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]

    logger.info("Starting parallel feature extraction...")
    with Pool(processes=num_partitions,
              initializer=_init_worker,
              initargs=(user_edit_frequencies, edit_freq, num_past_edits, last_edit_time, is_training)) as pool:
        results_iter = pool.imap(_process_records, index_chunks)
        results = []
        for res in tqdm(results_iter, total=len(index_chunks), desc="Parallel Feature Extraction"):
            results.append(res)

    features_df = pd.concat(results, ignore_index=True)
    return features_df


def extract_features_changeset(data_df):
    data_df = data_df.copy()
    data_df["min_lon"] = data_df["min_lon"].fillna(0)
    data_df["max_lon"] = data_df["max_lon"].fillna(0)
    data_df["min_lat"] = data_df["min_lat"].fillna(0)
    data_df["max_lat"] = data_df["max_lat"].fillna(0)
    data_df["centroid_x"] = (data_df["min_lon"] + data_df["max_lon"]) / 2
    data_df["centroid_y"] = (data_df["min_lat"] + data_df["max_lat"]) / 2
    data_df['changeset_comment_length'] = data_df['comment'].fillna("").apply(len)

    if SPLIT_METHOD == 'temporal':
        data_df['date_created'] = data_df['created_at']

    return data_df


def get_or_generate_features(data_df, is_training, processed_features_file_path, force_compute_features=False,
                             test_mode=False):
    if os.path.exists(processed_features_file_path) and not force_compute_features:
        logger.info(f"Loading features from {processed_features_file_path}...")
        features_df = pd.read_parquet(processed_features_file_path)
    else:
        logger.info("Extracting features...")
        if DATASET_TYPE == 'changeset':
            features_df = extract_features_changeset(data_df)
        else:
            features_df = extract_features(data_df, is_training)
        if is_training:
            logger.info(f"Saving features to {processed_features_file_path}...")
            features_df.to_parquet(processed_features_file_path)

    if is_training:
        if test_mode:
            logger.info("Test mode enabled: Limiting to entries matching Test changeset IDs.")
            features_df = features_df[features_df['changeset_id'].isin(TEST_CHANGESET_IDS)]

    logger.info(f"Features DataFrame Shape: {features_df.shape}")
    return features_df
