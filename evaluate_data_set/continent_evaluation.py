# continent_evaluation.py

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)


def calculate_center_coordinates(df, xmin_col='xmin', xmax_col='xmax', ymin_col='ymin', ymax_col='ymax'):
    """
    Calculate the center latitude and longitude from bounding box coordinates.

    Parameters:
    - df: DataFrame containing bounding box columns.
    - xmin_col, xmax_col, ymin_col, ymax_col: Column names for bounding box coordinates.

    Returns:
    - df: DataFrame with added 'latitude' and 'longitude' columns.
    """
    df['longitude'] = (df[xmin_col] + df[xmax_col]) / 2
    df['latitude'] = (df[ymin_col] + df[ymax_col]) / 2
    return df


def get_continent(lat, lon):
    """
    Map latitude and longitude to a continent based on approximate boundaries.

    Parameters:
    - lat: Latitude value.
    - lon: Longitude value.

    Returns:
    - continent: Name of the continent.
    """
    if -34.83333 <= lat <= 37.09222 and -17.62500 <= lon <= 51.20833:
        return 'Africa'
    elif -90 <= lat <= -60:
        return 'Antarctica'
    elif 11.16056 <= lat <= 77.71917 and 25.06667 <= lon <= 168.95833:
        return 'Asia'
    elif 34.50500 <= lat <= 71.18528 and -24.95583 <= lon <= 68.93500:
        return 'Europe'
    elif 5.49955 <= lat <= 83.16210 and -168.17625 <= lon <= -52.23304:
        return 'North America'
    elif -56.10273 <= lat <= 12.45777 and -81.76056 <= lon <= -34.72999:
        return 'South America'
    elif -47.28639 <= lat <= -8.23306 and 110.95167 <= lon <= 179.85917:
        return 'Oceania'
    else:
        return 'Other'


def assign_continents(df):
    """
    Assign continents to each row in the DataFrame based on latitude and longitude.

    Parameters:
    - df: DataFrame with 'latitude' and 'longitude' columns.

    Returns:
    - df: DataFrame with an added 'continent' column.
    """
    df['continent'] = df.apply(lambda row: get_continent(row['latitude'], row['longitude']), axis=1)
    return df


def split_test_set_by_continent(X_test, y_test):
    """
    Split the test set into subsets based on continents.

    Parameters:
    - X_test: Test features DataFrame.
    - y_test: Test labels Series.

    Returns:
    - continent_data: Dictionary with continent names as keys and DataFrames as values.
    """
    # Reset indices to ensure alignment
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Combine features and labels
    test_data = X_test.copy()
    test_data['label'] = y_test

    # Get list of continents
    continents = test_data['continent'].unique()

    # Dictionary to store data per continent
    continent_data = {}

    for continent in continents:
        continent_df = test_data[test_data['continent'] == continent]
        continent_data[continent] = continent_df

    return continent_data


def evaluate_model_on_continents(continent_data, model):
    """
    Evaluate the model on each continental subset and compute confusion matrices.

    Parameters:
    - continent_data: Dictionary with continent names as keys and DataFrames as values.
    - model: Trained model to evaluate.

    Returns:
    - results: Dictionary with continent names as keys and evaluation metrics as values.
    """
    results = {}

    for continent, data in continent_data.items():
        # Prepare features and labels
        features_to_drop = ['continent', 'label', 'latitude', 'longitude']
        X_continent = data.drop(columns=features_to_drop, errors='ignore')
        y_continent = data['label']

        # Ensure feature alignment with the model
        X_continent = X_continent[model.get_booster().feature_names]

        # Make predictions
        y_pred = model.predict(X_continent)
        y_prob = model.predict_proba(X_continent)[:, 1]

        # Calculate performance metrics
        accuracy = accuracy_score(y_continent, y_pred)
        precision = precision_score(y_continent, y_pred, zero_division=0)
        recall = recall_score(y_continent, y_pred, zero_division=0)
        f1 = f1_score(y_continent, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_continent, y_prob)
        auc_pr = average_precision_score(y_continent, y_prob)

        # Compute confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_continent, y_pred)
        TN, FP, FN, TP = cm.ravel()

        # Total correct predictions
        total_correct = TP + TN

        # Store results
        results[continent] = {
            'Total Samples': len(data),
            'Total Correct Predictions': total_correct,
            'True Positives (TP)': TP,
            'True Negatives (TN)': TN,
            'False Positives (FP)': FP,
            'False Negatives (FN)': FN,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'AUC-ROC': auc_roc,
            'AUC-PR': auc_pr
        }

    return results
