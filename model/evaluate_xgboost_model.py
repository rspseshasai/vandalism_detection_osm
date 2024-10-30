import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (classification_report, accuracy_score,
                             roc_auc_score, average_precision_score)
from sklearn.model_selection import cross_val_score

# Define class names for the confusion matrix heatmap
class_names = ['Not Vandalism', 'Vandalism']


def evaluate_train_test_metrics(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance on both training and test datasets."""
    print("Evaluating model performance on both training and test datasets...")
    # Predictions and probabilities for both sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # Training set metrics
    print("\nTrain Set Evaluation")
    print("Accuracy:", accuracy_score(y_train, y_train_pred))
    print("AUC-ROC:", roc_auc_score(y_train, y_train_prob))
    print("Classification Report (Train):\n", classification_report(y_train, y_train_pred))

    # Test set metrics
    print("\nTest Set Evaluation")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_test_prob))
    print("Classification Report (Test):\n", classification_report(y_test, y_test_pred))

    return y_test_pred, y_test_prob


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
    print(f"\nStatistics:")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    print(f"True Positives (TP): {TP}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return cm


def plot_confusion_matrix(cm):
    """Plot the confusion matrix as a heatmap."""

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.show()


def evaluate_model_with_cv(X, y, best_params, cv=5):
    print("\nPerforming 5-fold Cross-Validation on the entire data...")

    # Initialize the model with the best hyperparameters
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        **best_params
    )

    # Perform cross-validation on the entire dataset
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

    # print out the performance metrics
    print(f"Cross-Validation AUC Scores: {cv_scores}")
    print(f"Mean AUC Score: {np.mean(cv_scores)}")
    print(f"Standard Deviation of AUC Scores: {np.std(cv_scores)}")

    return cv_scores
