# Define class names for the confusion matrix heatmap
class_names = ['Not Vandalism', 'Vandalism']

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

from config import logger, VISUALIZATION_DATA_PATH


def save_evaluation_results(evaluation_results_main_model, cm, model_type):
    """
    Save evaluation results and confusion matrix to disk.
    """
    evaluation_results_path = VISUALIZATION_DATA_PATH[f'evaluation_results_{model_type}']
    confusion_matrix_path = VISUALIZATION_DATA_PATH[f'confusion_matrix_{model_type}']

    # Save evaluation results
    evaluation_results_main_model.to_parquet(evaluation_results_path)

    # Save confusion matrix
    np.savetxt(confusion_matrix_path, cm, delimiter=",")

    logger.info(f"Saved evaluation results to {evaluation_results_path}")
    logger.info(f"Saved confusion matrix to {confusion_matrix_path}")


def print_metrics(true, pred, prob):
    print("Accuracy:", accuracy_score(true, pred))
    print("AUC-ROC:", roc_auc_score(true, prob))
    print("\nClassification Report:\n", classification_report(true, pred, target_names=class_names))


def evaluate_train_test_metrics(model, X_train, y_train, X_test=None, y_test=None, threshold=None):

    # ---------------------
    # 1) Training Metrics
    # ---------------------
    # Predict probabilities on train set
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_train_pred = (y_train_prob >= threshold).astype(int)

    # Print train set evaluation
    print("\nTrain Set Evaluation\n--------------------\n")
    print_metrics(y_train, y_train_pred, y_train_prob)

    # ---------------------
    # 2) Test Metrics (if test set exists)
    # ---------------------
    if X_test is not None and y_test is not None and not X_test.empty and not y_test.empty:
        # Predict probabilities on test set
        y_test_prob = model.predict_proba(X_test)[:, 1]

        if threshold is None:
            # Use model's default .predict() with 0.5 cutoff
            y_test_pred = model.predict(X_test)
        else:
            # Apply custom threshold to probabilities
            y_test_pred = (y_test_prob >= threshold).astype(int)

        # Print test set evaluation
        print("\nTest Set Evaluation\n--------------------\n")
        print_metrics(y_test, y_test_pred, y_test_prob)

        return y_test_pred, y_test_prob
    else:
        print("\nNo test set provided. Skipping test evaluation.\n")
        return None, None


def calculate_auc_scores(y_test, y_test_pred, y_test_prob):
    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )

    """Calculate and print AUC-PR and ROC-AUC scores for the test set."""

    aucpr = average_precision_score(y_test, y_test_prob)
    roc_auc = roc_auc_score(y_test, y_test_prob)

    print(f"\nAUC-PR Score on Test Set: {aucpr}")
    print(f"ROC-AUC Score on Test Set: {roc_auc}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)

    # Calculate additional statistics
    TN, FP, FN, TP = cm.ravel()  # Unpack the confusion matrix
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    # Print statistics
    print("\nStatistics:\n-----------")
    print(f"True Negatives (TN, correctly identified non-vandalism): {TN}")
    print(f"False Positives (FP, non-vandalism incorrectly classified as vandalism): {FP}")
    print(f"False Negatives (FN, vandalism incorrectly classified as non-vandalism): {FN}")
    print(f"True Positives (TP, correctly identified vandalism): {TP}")
    print("\nPerformance Metrics:")
    print(f"Accuracy (overall correctness of the model): {accuracy:.4f}")
    print(f"Precision (proportion of predicted vandalism that is actually vandalism): {precision:.4f}")
    print(f"Recall (sensitivity, proportion of actual vandalism correctly identified): {recall:.4f}")
    print(f"F1 Score (harmonic mean of precision and recall): {f1:.4f}")

    return cm


def plot_confusion_matrix(model_type):
    """
    Plot confusion matrix using saved data.
    """
    confusion_matrix_path = VISUALIZATION_DATA_PATH[f'confusion_matrix_{model_type}']
    cm = np.loadtxt(confusion_matrix_path, delimiter=",")
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm / cm.sum(), annot=True, fmt='.3%', cmap='Blues',
                xticklabels=['Not Vandalism', 'Vandalism'],
                yticklabels=['Not Vandalism', 'Vandalism'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.show()


def plot_roc_pr_curves(model_type):
    """
    Plot ROC and Precision-Recall curves using saved evaluation data.
    """
    evaluation_results_path = VISUALIZATION_DATA_PATH[f'evaluation_results_{model_type}']
    evaluation_results = pd.read_parquet(evaluation_results_path)

    y_test = evaluation_results[f'y_true']
    y_test_prob = evaluation_results[f'y_prob_{model_type}']

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()


def evaluate_model_with_cv(X, y, best_params, cv=5):

    print("\nPerforming 5-fold Cross-Validation on the training data...")

    # Initialize the models with the best hyperparameters
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        n_jobs=1,  # Prevent nested parallelism
        **best_params
    )

    # Perform cross-validation on the entire dataset
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X, y,
        cv=skf,
        scoring='average_precision',
        n_jobs=11  # -1 to use all available cores
    )

    # Print out the performance metrics
    print(f"Cross-Validation AUC Scores: {cv_scores}")
    print(f"Mean AUC Score: {np.mean(cv_scores)}")
    print(f"Standard Deviation of AUC Scores: {np.std(cv_scores)}")

    return cv_scores
