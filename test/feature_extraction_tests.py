from datetime import datetime

import pandas as pd

from feature_extraction import calculate_time_since_last_edit, calculate_historical_validity
from feature_extraction import calculate_user_edit_frequency


def test_calculate_user_edit_frequency():
    from datetime import datetime

    # Create test data with datetime objects spanning multiple time windows
    test_data = [
        {'user_id': 1, 'valid_from': datetime(2024, 6, 1, 12, 0, 0)},  # Falls into 365d, 180d, 60d, 30d
        {'user_id': 1, 'valid_from': datetime(2024, 6, 20, 12, 0, 0)},  # Falls into 365d, 180d, 60d, 30d
        {'user_id': 1, 'valid_from': datetime(2024, 6, 28, 12, 0, 0)},  # Falls into 365d, 180d, 60d, 30d, 14d
        {'user_id': 1, 'valid_from': datetime(2024, 6, 30, 12, 0, 0)},  # Falls into all windows

        {'user_id': 4, 'valid_from': datetime(2023, 6, 1, 12, 0, 0)},  # Only in 365d
    ]

    # Convert the test data into a DataFrame
    contributions_df = pd.DataFrame(test_data)

    # Call the function to calculate user edit frequencies
    actual_frequencies = calculate_user_edit_frequency(contributions_df)

    # Expected results for user frequencies across different windows
    expected_frequencies = {
        1: {
            'edit_frequency_7d': 2.0 / 7,  # 2 edits in the last 7 days
            'edit_frequency_14d': 3.0 / 14,  # 3 edits in the last 14 days
            'edit_frequency_30d': 4.0 / 30,  # 4 edits in the last 30 days
            'edit_frequency_60d': 4.0 / 60,  # 4 edits in the last 60 days
            'edit_frequency_180d': 4.0 / 180,  # 4 edits in the last 180 days
            'edit_frequency_365d': 4.0 / 365  # 4 edits in the last 365 days
        },

        4: {
            'edit_frequency_7d': 0.0,  # No edits in the last 7 days
            'edit_frequency_14d': 0.0,  # No edits in the last 14 days
            'edit_frequency_30d': 0.0,  # No edits in the last 30 days
            'edit_frequency_60d': 0.0,  # No edits in the last 60 days
            'edit_frequency_180d': 0.0,  # No edits in the last 180 days
            'edit_frequency_365d': 1.0 / 365  # 1 edit in the last 365 days
        },

    }

    # Compare the actual results with expected frequencies
    for user_id, expected_values in expected_frequencies.items():
        actual_values = actual_frequencies.get(user_id, {})
        for key, expected_value in expected_values.items():
            assert round(actual_values.get(key, 0), 2) == round(expected_value, 2), \
                f"Test failed for user {user_id}, {key}: expected {expected_value}, got {actual_values.get(key, 0)}"


def test_calculate_time_since_last_edit():
    # Create test data with datetime objects
    test_data = [
        {'user_id': 1, 'valid_from': datetime(2024, 6, 1, 12, 0, 0)},
        {'user_id': 1, 'valid_from': datetime(2024, 6, 5, 12, 0, 0)},
        {'user_id': 1, 'valid_from': datetime(2024, 6, 10, 12, 0, 0)},
        {'user_id': 1, 'valid_from': datetime(2024, 6, 15, 12, 0, 0)},
        {'user_id': 2, 'valid_from': datetime(2024, 6, 1, 12, 0, 0)},
        {'user_id': 2, 'valid_from': datetime(2024, 6, 8, 12, 0, 0)},
        {'user_id': 2, 'valid_from': datetime(2024, 6, 15, 12, 0, 0)},
    ]

    # Convert the test data into a DataFrame
    contributions_df = pd.DataFrame(test_data)

    # Placeholder for first-time contributors
    first_time_placeholder = 10 * 365 * 24  # 10 years in hours

    # Test case 1: User 1, 4th contribution (2024-06-15)
    contribution = {'user_id': 1, 'valid_from': datetime(2024, 6, 15, 12, 0, 0)}
    expected_hours = (datetime(2024, 6, 15, 12, 0, 0) - datetime(2024, 6, 10, 12, 0, 0)).total_seconds() / 3600.0
    actual_hours = calculate_time_since_last_edit(contribution, contributions_df)
    assert actual_hours == expected_hours, \
        f"Test failed: expected {expected_hours} hours, got {actual_hours} hours"

    # Test case 2: User 2, 3rd contribution (2024-06-15)
    contribution = {'user_id': 2, 'valid_from': datetime(2024, 6, 15, 12, 0, 0)}
    expected_hours = (datetime(2024, 6, 15, 12, 0, 0) - datetime(2024, 6, 8, 12, 0, 0)).total_seconds() / 3600.0
    actual_hours = calculate_time_since_last_edit(contribution, contributions_df)
    assert actual_hours == expected_hours, \
        f"Test failed: expected {expected_hours} hours, got {actual_hours} hours"

    # Test case 3: User 2, first contribution (2024-06-01)
    contribution = {'user_id': 4, 'valid_from': datetime(2024, 6, 1, 12, 0, 0)}
    expected_hours = first_time_placeholder  # Placeholder for first-time contributors
    actual_hours = calculate_time_since_last_edit(contribution, contributions_df)
    assert actual_hours == expected_hours, \
        f"Test failed: expected {expected_hours} hours, got {actual_hours} hours"

    # Test case 4: User 3, first contribution in dataset (new user)
    contribution = {'user_id': 3, 'valid_from': datetime(2024, 6, 25, 12, 0, 0)}
    expected_hours = first_time_placeholder  # Placeholder for first-time contributors
    actual_hours = calculate_time_since_last_edit(contribution, contributions_df)
    assert actual_hours == expected_hours, \
        f"Test failed: expected {expected_hours} hours, got {actual_hours} hours"


def test_calculate_historical_validity():
    # Create sample contribution data
    contribution_data = [
        {'osm_id': 101, 'valid_from': datetime(2024, 6, 1, 12, 0, 0)},
        {'osm_id': 101, 'valid_from': datetime(2024, 6, 2, 12, 0, 0)},
        {'osm_id': 102, 'valid_from': datetime(2024, 6, 3, 12, 0, 0)},
        {'osm_id': 101, 'valid_from': datetime(2024, 6, 4, 12, 0, 0)},
        {'osm_id': 103, 'valid_from': datetime(2024, 6, 5, 12, 0, 0)},
        {'osm_id': 103, 'valid_from': datetime(2024, 6, 6, 12, 0, 0)},
        {'osm_id': 101, 'valid_from': datetime(2024, 6, 7, 12, 0, 0)},
    ]
    contribution_df = pd.DataFrame(contribution_data)

    # Set a threshold of 3 edits to mark features as historically invalid
    historical_validity = calculate_historical_validity(contribution_df, edit_threshold=3)

    # Check the historical validity for each osm_id
    assert historical_validity[101] == 0, "osm_id 101 should be marked as historically invalid"
    assert historical_validity[102] == 1, "osm_id 102 should be valid"
    assert historical_validity[103] == 1, "osm_id 103 should be valid"
