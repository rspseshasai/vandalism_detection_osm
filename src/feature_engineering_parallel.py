import logging
import math
import os
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm

from config import SPLIT_METHOD, DATASET_TYPE, TEST_CHANGESET_IDS, SHOULD_INCLUDE_USERFEATURES, \
    SHOULD_INCLUDE_OSM_ELEMENT_FEATURES
from config import logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    # Previously we used precomputed time_since_last_edit, but now removed
    # so we won't set 'time_since_last_edit' here anymore.

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

    # We no longer use user_edit_frequencies or historical features,
    # so no need for complex logic here.
    # Just compute spatial features as before.

    # grid cell
    def get_grid_cell_id(lat, lon, grid_size=0.1):
        x_index = int((lon + 180) / grid_size)
        y_index = int((lat + 90) / grid_size)
        return f"{x_index}_{y_index}"

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
    features['contribution_key'] = str(contribution['valid_from']) + "__" + str(contribution['osm_id']) + "__" + str(contribution['osm_version'])
    features['source_used'] = source if source else 'unknown'
    # features['changeset_timestamp'] = contribution['changeset_timestamp']

    return features


def extract_map_features(contribution):
    features = {}
    map_features = contribution['map_features']
    for feature_name, feature_value in map_features.items():
        if feature_value is None:
            feature_value = False
        features[feature_name] = int(feature_value)
    return features


# Previously we had GLOBAL_* vars and complex logic for historical and user frequencies.
# Now we remove them since we are not using these complicated features.

GLOBAL_IS_TRAINING = None


def _init_worker(is_training):
    global GLOBAL_IS_TRAINING
    GLOBAL_IS_TRAINING = is_training


def extract_osm_element_features(contribution):
    features = {}
    features['element_n_users_cum'] = contribution['element_n_users_cum']
    features['element_n_versions'] = contribution['element_n_versions']
    try:
        features['element_previous_edit_timestamp'] = contribution['element_previous_edit_timestamp']
    except KeyError:
        features['element_previous_edit_timestamp'] = contribution['element_previous_edit_valid']
    features['element_time_since_previous_edit'] = contribution['element_time_since_previous_edit']
    return features


def extract_user_features(contribution):
    features = {}
    features['n_edits'] = contribution['n_edits']
    features['user_n_changesets_cum'] = contribution['user_n_changesets_cum']
    features['user_n_edits_cum'] = contribution['user_n_edits_cum']
    features['user_n_edit_days_cum'] = contribution['user_n_edit_days_cum']
    features['user_previous_edit_timestamp'] = contribution['user_previous_edit_timestamp']
    features['user_time_since_previous_edit'] = contribution['user_time_since_previous_edit']
    return features


def _process_records(records):
    results = []
    for contribution in records:
        features = {}

        if SHOULD_INCLUDE_OSM_ELEMENT_FEATURES:
            # 1. OSM Element features
            features.update(extract_osm_element_features(contribution))
        if SHOULD_INCLUDE_USERFEATURES:
            # 2. User  features
            features.update(extract_user_features(contribution))
        # 3. Geometric
        features.update(extract_geometric_features(contribution))
        # 4. Temporal
        features.update(extract_temporal_features(contribution))
        # 5. Content
        features.update(extract_content_features(contribution))
        # 6. Spatial
        features.update(extract_spatial_features(contribution))
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


def extract_features_contributions(contribution_df, is_training):
    # We no longer calculate user_edit_frequencies, historical features, or time_since_last_edit dict.
    # Just directly extract features based on data.

    records = contribution_df.to_dict('records')

    num_partitions = cpu_count()
    if not is_training:
        num_partitions = min(4, os.cpu_count() // 2)
    record_count = len(records)
    if record_count == 0:
        return pd.DataFrame()

    chunk_size = record_count // num_partitions if num_partitions > 1 else record_count
    if chunk_size == 0 and record_count > 0:
        chunk_size = 1

    index_chunks = [records[i:i + chunk_size] for i in range(0, record_count, chunk_size)]

    logger.info(
        "Starting parallel feature extraction...")
    with Pool(processes=num_partitions,
              initializer=_init_worker,
              initargs=(is_training,)) as pool:
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
            features_df = extract_features_contributions(data_df, is_training)
        if is_training:
            logger.info(f"Saving features to {processed_features_file_path}...")
            features_df.to_parquet(processed_features_file_path)

    if is_training:
        if test_mode:
            logger.info("Test mode enabled: Limiting to entries matching Test changeset IDs.")
            features_df = features_df[features_df['changeset_id'].isin(TEST_CHANGESET_IDS)]

    logger.info(f"Features DataFrame Shape: {features_df.shape}")
    return features_df
