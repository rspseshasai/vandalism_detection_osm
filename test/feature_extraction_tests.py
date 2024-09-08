from datetime import datetime

import pandas as pd

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
