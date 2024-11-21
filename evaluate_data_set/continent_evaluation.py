from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.metrics import confusion_matrix


def split_test_set_by_continent(X_test, y_test, continent_columns):
    """
    Split the test set into subsets based on continents using binary continent columns.

    Parameters:
    - X_test: Test features DataFrame.
    - y_test: Test labels Series.
    - continent_columns: List of binary continent columns (e.g., 'continent_Asia', 'continent_Africa').

    Returns:
    - continent_data: Dictionary with continent names as keys and DataFrames as values.
    """
    # Reset indices to ensure alignment
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Combine features and labels
    test_data = X_test.copy()
    test_data['label'] = y_test

    # Dictionary to store data per continent
    continent_data = {}

    for continent_col in continent_columns:
        continent_name = continent_col.replace("continent_", "")
        continent_df = test_data[test_data[continent_col] == 1]
        if not continent_df.empty:
            continent_data[continent_name] = continent_df

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
        features_to_drop = ['label']
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
        cm = confusion_matrix(y_continent, y_pred)
        TN, FP, FN, TP = cm.ravel()

        # Total correct predictions
        total_correct = TP + TN

        # Store results
        results[continent] = {
            'Total Samples': len(data),
            'Total Correct Predictions': total_correct,
            'Total Incorrect Predictions': len(data) - total_correct,
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
