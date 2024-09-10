import pandas as pd  # Import pandas for DataFrame handling

from logger_config import logger

# Placeholder for historical data
historical_edits = {}


def calculate_user_edit_frequency(contributions):
    # Step 1: Group contributions by user_id and calculate user-specific statistics
    user_edit_frequencies = {}

    # Iterate through all contributions and calculate user edit frequency
    for user_id, group in contributions.groupby('user_id'):
        # Sort the user's contributions by 'valid_from' timestamp
        user_contributions = group.sort_values('valid_from')

        # Get the total number of contributions made by the user
        total_edits = len(user_contributions)

        # Calculate the time range (in weeks) between the first and last edit
        first_edit = user_contributions['valid_from'].min()
        last_edit = user_contributions['valid_from'].max()

        # Calculate the number of weeks between the first and last edit (at least 1 week to avoid division by zero)
        time_span_in_weeks = max((last_edit - first_edit).days / 7.0, 1)

        # Calculate average edit frequency (edits per week)
        edit_frequency = total_edits / time_span_in_weeks
        user_edit_frequencies[user_id] = edit_frequency

    return user_edit_frequencies


def calculate_time_since_last_edit(contribution, contribution_df):
    """
    Calculate the time since the last edit for a specific contribution.

    :param contribution: A row (Series) from the DataFrame representing the current contribution.
    :param contribution_df: The full DataFrame of contributions, used to find previous contributions by the same user.
    :return: Time since the last edit in hours.
    """

    # TODO: Also can add contextual features. To detect if similar changes happened in this area!

    user_id = contribution['user_id']

    # Filter contributions to get only previous contributions made by the same user
    user_contributions = contribution_df[(contribution_df['user_id'] == user_id) &
                                         (contribution_df['valid_from'] < contribution['valid_from'])]

    # Check if the user has previous contributions, if not, set the current contribution's timestamp
    if not user_contributions.empty:
        last_edit = user_contributions['valid_from'].max()  # Get the most recent previous contribution
    else:
        last_edit = contribution['valid_from']

    # Calculate time since last edit in hours
    time_since_last_edit = (contribution['valid_from'] - last_edit).total_seconds() / 3600.0
    return time_since_last_edit


# TODO: Check and update the edit_threshold
def calculate_historical_validity(contribution_df, edit_threshold=0):
    # TODO: Work more on the contextual and Historical features.
    #  TODO: Put advance logic here. May be Time Series Analysis? At least more number of features.

    feature_edit_counts = contribution_df.groupby('osm_id').size()
    historical_validity = feature_edit_counts.apply(lambda count: 0 if count > edit_threshold else 1)
    return historical_validity


# Updated function to handle a DataFrame and return a DataFrame of features
def extract_features(contribution_df):
    feature_list = []  # Initialize a list to collect feature dictionaries

    user_edit_frequencies = calculate_user_edit_frequency(contribution_df)
    historical_validity = calculate_historical_validity(contribution_df)

    # Iterate over each row in the DataFrame
    for index, contribution in contribution_df.iterrows():
        features = {}

        # 1. User Behavior Features
        user_id = contribution['user_id']

        # 1.1 User Edit Frequency (Average number of edits per week)
        features['user_edit_frequency'] = user_edit_frequencies.get(user_id, 0)

        # 1.2 Editor Used (one-hot encoding for JOSM, iD, etc.)
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

        # 3. Temporal Features
        features['time_since_last_edit'] = calculate_time_since_last_edit(contribution, contribution_df)
        features['edit_time_of_day'] = contribution['changeset']['timestamp'].hour

        # 4. Contribution Content Features
        tags_before = dict(contribution['tags_before'])
        tags_after = dict(contribution['tags'])
        tags_added = len(set(tags_after.keys()) - set(tags_before.keys()))
        tags_removed = len(set(tags_before.keys()) - set(tags_after.keys()))
        tags_modified = sum(1 for tag in tags_before if tag in tags_after and tags_before[tag] != tags_after[tag])
        features['tags_added'] = tags_added
        features['tags_removed'] = tags_removed
        features['tags_modified'] = tags_modified

        key_tags = ['name', 'boundary', 'population']
        for tag in key_tags:
            features[f'{tag}_changed'] = int(tags_before.get(tag) != tags_after.get(tag))

        # 5. Spatial Features
        features['bbox_x_range'] = contribution['xmax'] - contribution['xmin']
        features['bbox_y_range'] = contribution['ymax'] - contribution['ymin']
        features['centroid_x'] = contribution['centroid']['x']
        features['centroid_y'] = contribution['centroid']['y']
        country_iso = contribution['country_iso_a3']
        features['country_count'] = len(country_iso)

        # 6. Contextual and Historical Features
        # Calculate historical validity once for all contributions

        features['historical_validity'] = historical_validity.get(contribution['osm_id'], 10)

        # 7. Derived Features
        features['tag_density'] = len(contribution['tags']) / contribution['area'] if contribution['area'] > 0 else 0
        previous_area = contribution.get('previous_area', contribution['area'] - contribution['area_delta'])
        change_ratio = contribution['area_delta'] / previous_area if previous_area > 0 else contribution['area_delta']
        features['change_ratio'] = change_ratio

        # 8. Changeset Features
        comment = next((tag[1] for tag in contribution['changeset']['tags'] if tag[0] == 'comment'), "")
        features['changeset_comment_length'] = len(comment)
        source = next((tag[1] for tag in contribution['changeset']['tags'] if tag[0] == 'source'), "")
        # TODO: Check and add more reliable sources
        reliable_sources = ['Bing Aerial Imagery', 'Esri World Imagery']
        features['source_reliability'] = 1 if any(s in source for s in reliable_sources) else 0

        feature_list.append(features)  # Append the features for this contribution

    # Convert list of feature dictionaries to a DataFrame
    features_df = pd.DataFrame(feature_list)
    logger.info(f"features_df.shape: {features_df.shape}")
    return features_df
