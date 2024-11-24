# src/geographical_evaluation.py

import warnings

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# Suppress specific UserWarning about single label in y_true and y_pred
warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    message="A single label was found in 'y_true' and 'y_pred'.*"
)


def split_test_set_by_key(X_test, y_test, binary_columns, split_key):
    """
    Split the test set into subsets based on a generic key (e.g., continents or countries).
    """
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    test_data = X_test.copy()
    test_data['label'] = y_test
    split_data = {}
    for binary_col in binary_columns:
        group_name = binary_col.replace(f"{split_key}_", "")
        group_df = test_data[test_data[binary_col] == 1]
        if not group_df.empty:
            split_data[group_name] = group_df
    return split_data


def evaluate_model_on_split_groups(split_data, model):
    """
    Evaluate the model on each group and compute metrics.
    """
    results = {}
    for group, data in split_data.items():
        features_to_drop = ['label']
        X_group = data.drop(columns=features_to_drop, errors='ignore')
        y_group = data['label']
        X_group = X_group[model.get_booster().feature_names]
        y_pred = model.predict(X_group)
        y_prob = model.predict_proba(X_group)[:, 1]
        accuracy = accuracy_score(y_group, y_pred)
        precision = precision_score(y_group, y_pred, zero_division=0)
        recall = recall_score(y_group, y_pred, zero_division=0)
        f1 = f1_score(y_group, y_pred, zero_division=0)
        if len(y_group.unique()) > 1:
            auc_roc = roc_auc_score(y_group, y_prob)
            auc_pr = average_precision_score(y_group, y_prob)
        else:
            auc_roc = None
            auc_pr = None
        cm = confusion_matrix(y_group, y_pred)
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
        else:
            TN = FP = FN = TP = 0
            if y_group.iloc[0] == 0:
                TN = cm[0, 0]
            else:
                TP = cm[0, 0]
        total_correct = TP + TN
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
