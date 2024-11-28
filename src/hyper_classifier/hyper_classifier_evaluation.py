# hyper_classifier/hyper_classifier_evaluation.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from config import logger, SAVE_VISUALIZATION_SAMPLES, HYPER_VISUALIZATION_DIR


def evaluate_hyper_classifier(hyper_model, X_test, y_test):
    logger.info("Evaluating hyper-classifier model...")

    # Prepare DMatrix for XGBoost
    dtest = xgb.DMatrix(X_test)

    # Predictions
    logger.info("Generating predictions on test data...")
    y_pred_prob = hyper_model.predict(dtest)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Evaluation metrics
    logger.info("Calculating evaluation metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_pred_prob)
    auc_pr = average_precision_score(y_test, y_pred_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print evaluation metrics
    logger.info("\nEvaluation Metrics:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"AUC-ROC Score: {auc_score:.4f}")
    logger.info(f"AUC-PR Score: {auc_pr:.4f}")
    logger.info(f"\nConfusion Matrix:\n{conf_matrix}")

    # Detailed classification report
    report = classification_report(y_test, y_pred, target_names=['Non-Vandalism', 'Vandalism'], zero_division=0)
    logger.info(f"\nClassification Report:\n{report}")

    # Save evaluation results for visualization
    if SAVE_VISUALIZATION_SAMPLES:
        logger.info("Saving evaluation results...")
        evaluation_results = pd.DataFrame({
            'y_true': y_test,
            'y_pred': y_pred,
            'y_prob': y_pred_prob
        })
        results_path = os.path.join(HYPER_VISUALIZATION_DIR, 'evaluation_results.parquet')
        evaluation_results.to_parquet(results_path)
        logger.info(f"Saved evaluation results to {results_path}")

        confusion_matrix_path = os.path.join(HYPER_VISUALIZATION_DIR, 'confusion_matrix.npy')
        np.save(confusion_matrix_path, conf_matrix)
        logger.info(f"Saved confusion matrix to {confusion_matrix_path}")

        # # Plot and save ROC and Precision-Recall curves
        # plot_roc_curve(y_test, y_pred_prob, HYPER_VISUALIZATION_DIR)
        # plot_pr_curve(y_test, y_pred_prob, HYPER_VISUALIZATION_DIR)

    return accuracy, precision, recall, f1, auc_score, auc_pr


def plot_roc_curve(y_test, y_pred_prob, save_dir):
    """Plot ROC curve and save."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.4f})'.format(roc_auc_score(y_test, y_pred_prob)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid()
    roc_curve_path = os.path.join(save_dir, 'roc_curve.png')
    plt.savefig(roc_curve_path)
    logger.info(f"Saved ROC curve to {roc_curve_path}")
    plt.close()


def plot_pr_curve(y_test, y_pred_prob, save_dir):
    """Plot Precision-Recall curve and save."""
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall Curve (AP = {:.4f})'.format(average_precision_score(y_test, y_pred_prob)))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid()
    pr_curve_path = os.path.join(save_dir, 'pr_curve.png')
    plt.savefig(pr_curve_path)
    logger.info(f"Saved Precision-Recall curve to {pr_curve_path}")
    plt.close()
