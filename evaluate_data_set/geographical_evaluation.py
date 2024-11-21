from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)


def split_test_set_by_key(X_test, y_test, binary_columns, split_key):
    """
    Split the test set into subsets based on a generic key (e.g., continents or countries).

    Parameters:
    - X_test: Test features DataFrame.
    - y_test: Test labels Series.
    - binary_columns: List of binary columns representing the split groups (e.g., 'continent_Asia', 'country_US').
    - split_key: Key indicating the grouping basis (e.g., "continent" or "country").

    Returns:
    - split_data: Dictionary with group names as keys and DataFrames as values.
    """
    # Reset indices to ensure alignment
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Combine features and labels
    test_data = X_test.copy()
    test_data['label'] = y_test

    # Dictionary to store data per group
    split_data = {}

    for binary_col in binary_columns:
        group_name = binary_col.replace(f"{split_key}_", "")
        group_df = test_data[test_data[binary_col] == 1]
        if not group_df.empty:
            split_data[group_name] = group_df

    return split_data


def evaluate_model_on_split_groups(split_data, model):
    """
    Evaluate the model on each group (continent or country) and compute metrics.

    Parameters:
    - split_data: Dictionary with group names as keys and DataFrames as values.
    - model: Trained model to evaluate.
    - split_key: Key indicating the grouping basis (e.g., "continent" or "country").

    Returns:
    - results: Dictionary with group names as keys and evaluation metrics as values.
    """
    results = {}

    for group, data in split_data.items():
        # Prepare features and labels
        features_to_drop = ['label']
        X_group = data.drop(columns=features_to_drop, errors='ignore')
        y_group = data['label']

        # Ensure feature alignment with the model
        X_group = X_group[model.get_booster().feature_names]

        # Make predictions
        y_pred = model.predict(X_group)
        y_prob = model.predict_proba(X_group)[:, 1]

        # Initialize metrics
        accuracy = accuracy_score(y_group, y_pred)
        precision = precision_score(y_group, y_pred, zero_division=0)
        recall = recall_score(y_group, y_pred, zero_division=0)
        f1 = f1_score(y_group, y_pred, zero_division=0)

        # Check if y_group has at least two classes
        if len(y_group.unique()) > 1:
            auc_roc = roc_auc_score(y_group, y_prob)
            auc_pr = average_precision_score(y_group, y_prob)
        else:
            auc_roc = None
            auc_pr = None

        # Compute confusion matrix
        cm = confusion_matrix(y_group, y_pred)
        if cm.shape == (2, 2):  # Regular case
            TN, FP, FN, TP = cm.ravel()
        else:  # Single-class case
            TN, FP, FN, TP = (cm[0, 0], 0, 0, 0) if y_group.iloc[0] == 0 else (0, 0, 0, cm[0, 0])

        # Total correct predictions
        total_correct = TP + TN

        # Store results
        results[group] = {
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
            'AUC-ROC': auc_roc if auc_roc is not None else "Not defined (single class)",
            'AUC-PR': auc_pr if auc_pr is not None else "Not defined (single class)"
        }

    return results
