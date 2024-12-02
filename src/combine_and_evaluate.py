# combine_and_evaluate.py

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from config import logger
from evaluation import calculate_auc_scores, save_evaluation_results, print_metrics


def check_for_data_consistency(evaluation_results_main_model, evaluation_results_hyper_classifier_model):
    # Extract labels and changeset_ids from both evaluation results
    main_labels = evaluation_results_main_model[['changeset_id', 'y_true']]
    hyper_labels = evaluation_results_hyper_classifier_model[['changeset_id', 'y_true']]

    # Merge on changeset_id to compare labels
    label_comparison = main_labels.merge(hyper_labels, on='changeset_id', suffixes=('_main', '_hyper'))

    # Identify changeset_ids where the labels differ
    differences = label_comparison[label_comparison['y_true_main'] != label_comparison['y_true_hyper']]
    if len(differences) == 0:
        logger.info("Data and labels are consistent in both main and hyper classifier models")
        return

    logger.info(f"Number of changeset_ids with different labels: {len(differences)}")
    logger.info(differences)

    raise Exception("Data and labels are not consistent in both main and hyper classifier models")


def combine_and_evaluate(evaluation_results_main_model, evaluation_results_hyper_classifier_model):
    """
    Combine predictions from the main model and hyper-classifier,
    evaluate the ensemble, and generate comparison statistics.
    """
    logger.info("Combining predictions and evaluating the ensemble model...")

    check_for_data_consistency(evaluation_results_main_model, evaluation_results_hyper_classifier_model)

    # Merge on 'changeset_id' only
    merged_results = pd.merge(
        evaluation_results_main_model,
        evaluation_results_hyper_classifier_model,
        on=['changeset_id']
    )

    # Check if 'y_true' columns are identical
    if not (merged_results['y_true_x'] == merged_results['y_true_y']).all():
        # If they differ, raise an error
        raise ValueError("Mismatch in 'y_true' between main model and hyper-classifier model.")
    else:
        # If they are the same, consolidate into a single 'y_true' column
        merged_results['y_true'] = merged_results['y_true_x']
        merged_results.drop(columns=['y_true_x', 'y_true_y'], inplace=True)

    # Check if the number of samples matches the original test set size
    logger.info(f"Number of samples after merging: {len(merged_results)}")

    # Compute the average of the predicted probabilities
    merged_results['y_prob_combined'] = merged_results[['y_prob_main', 'y_prob_hyper_classifier']].mean(axis=1)

    # Threshold the averaged probabilities to get final predictions
    merged_results['y_pred_combined'] = (merged_results['y_prob_combined'] >= 0.5).astype(int)

    # Evaluate and print metrics for the ensemble model
    print("\nCombined Ensemble Evaluation\n--------------------\n")
    print_metrics(merged_results['y_true'], merged_results['y_pred_combined'], merged_results['y_prob_combined'])

    # Calculate additional metrics and confusion matrix
    cm = calculate_auc_scores(merged_results['y_true'], merged_results['y_pred_combined'],
                              merged_results['y_prob_combined'])

    # Save evaluation data for visualization
    save_evaluation_results(merged_results, cm, 'ensemble')

    # -------------- New Code to Compare Metrics of All Three Models --------------

    # Evaluate and collect metrics for the main model
    metrics_main_model = get_metrics(
        evaluation_results_main_model['y_true'],
        evaluation_results_main_model['y_pred_main'],
        evaluation_results_main_model['y_prob_main']
    )

    # Evaluate and collect metrics for the hyper-classifier model
    metrics_hyper_classifier_model = get_metrics(
        evaluation_results_hyper_classifier_model['y_true'],
        evaluation_results_hyper_classifier_model['y_pred_hyper_classifier'],
        evaluation_results_hyper_classifier_model['y_prob_hyper_classifier']
    )

    # Evaluate and collect metrics for the ensemble model
    metrics_ensemble_model = get_metrics(
        merged_results['y_true'],
        merged_results['y_pred_combined'],
        merged_results['y_prob_combined']
    )

    # Combine all metrics into a DataFrame
    metrics_df = pd.DataFrame({
        'Main Model': metrics_main_model,
        'Hyper-Classifier Model': metrics_hyper_classifier_model,
        'Ensemble Model': metrics_ensemble_model
    })

    # Transpose the DataFrame to have models as rows
    metrics_df = metrics_df.transpose()

    # Print the comparison table
    print("\nComparison of Evaluation Metrics:\n")
    print(metrics_df.to_string(float_format='{:,.4f}'.format))

    logger.info("Ensemble model evaluation completed.")


def get_metrics(y_true, y_pred, y_prob):
    """
    Calculate evaluation metrics and return them in a dictionary.
    """
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['F1-Score'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob)
    return metrics
