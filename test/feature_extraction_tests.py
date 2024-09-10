from datetime import datetime

import pandas as pd

from feature_extraction import calculate_time_since_last_edit, calculate_historical_validity
from feature_extraction import calculate_user_edit_frequency


def test_calculate_user_edit_frequency():
    # Create test data with datetime objects
    test_data = [
        {'user_id': 1, 'valid_from': datetime(2024, 6, 1, 12, 0, 0)},
        {'user_id': 1, 'valid_from': datetime(2024, 6, 5, 12, 0, 0)},
        {'user_id': 1, 'valid_from': datetime(2024, 6, 10, 12, 0, 0)},
        {'user_id': 1, 'valid_from': datetime(2024, 6, 15, 12, 0, 0)},

        {'user_id': 2, 'valid_from': datetime(2024, 6, 1, 12, 0, 0)},
        {'user_id': 2, 'valid_from': datetime(2024, 6, 8, 12, 0, 0)},
        {'user_id': 2, 'valid_from': datetime(2024, 6, 15, 12, 0, 0)},
        {'user_id': 2, 'valid_from': datetime(2024, 6, 22, 12, 0, 0)},
        {'user_id': 2, 'valid_from': datetime(2024, 6, 29, 12, 0, 0)},

        {'user_id': 3, 'valid_from': datetime(2024, 6, 1, 12, 0, 0)},

        {'user_id': 4, 'valid_from': datetime(2024, 6, 1, 12, 0, 0)},
        {'user_id': 4, 'valid_from': datetime(2024, 6, 21, 12, 0, 0)},
    ]
    # Convert the test data into a DataFrame
    contributions_df = pd.DataFrame(test_data)

    # Expected Results:
    expected_frequencies = {
        1: 2.0,  # 4 edits in 2 weeks
        2: 1.25,  # 5 edits in 4 weeks
        3: 1.0,  # 1 edit in 1 week
        4: 0.7,  # 2 edits in 3 weeks (rounded to 2 decimal places)
    }

    # Calculate the actual user edit frequencies using the function
    actual_frequencies = calculate_user_edit_frequency(contributions_df)

    # Check if the actual frequencies match the expected values
    for user_id, expected_frequency in expected_frequencies.items():
        assert round(actual_frequencies.get(user_id, 0), 2) == expected_frequency, \
            f"Test failed for user {user_id}: expected {expected_frequency}, got {actual_frequencies.get(user_id, 0)}"


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
    contribution = {'user_id': 2, 'valid_from': datetime(2024, 6, 1, 12, 0, 0)}
    expected_hours = 0  # First contribution, so no previous edit
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
