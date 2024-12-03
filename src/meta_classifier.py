# meta_classifier.py

import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.model_selection import GridSearchCV
from config import logger


def check_for_data_consistency(evaluation_results_main_model, evaluation_results_hyper_classifier_model):
    """
    Check for consistency in 'y_true' between main and hyper-classifier models.
    """
    # Extract labels and changeset_ids from both evaluation results
    main_labels = evaluation_results_main_model[['changeset_id', 'y_true']]
    hyper_labels = evaluation_results_hyper_classifier_model[['changeset_id', 'y_true']]

    # Merge on changeset_id to compare labels
    label_comparison = main_labels.merge(hyper_labels, on='changeset_id', suffixes=('_main', '_hyper'))

    # Identify changeset_ids where the labels differ
    differences = label_comparison[label_comparison['y_true_main'] != label_comparison['y_true_hyper']]
    if len(differences) == 0:
        logger.info("Data and labels are consistent in both main and hyper-classifier models")
        return

    logger.info(f"Number of changeset_ids with different labels: {len(differences)}")
    logger.info(differences)

    raise Exception("Data and labels are not consistent in both main and hyper-classifier models")


def meta_classifier(evaluation_results_main_model, evaluation_results_hyper_classifier_model, main_model, hyper_model,
                    X_test_meta_main, X_test_meta_hyper, y_test_meta):
    """
    Combine predictions from the main model and hyper-classifier,
    evaluate the ensemble using a meta-classifier, and generate comparison statistics.

    Parameters:
    - evaluation_results_main_model: DataFrame containing main model's evaluation results on validation set.
    - evaluation_results_hyper_classifier_model: DataFrame containing hyper-classifier's evaluation results on validation set.
    - main_model: Trained main model.
    - hyper_model: Trained hyper-classifier model.
    - X_test_meta_main: Features for the main model on the meta-test set.
    - X_test_meta_hyper: Features for the hyper-classifier on the meta-test set.
    - y_test_meta: True labels for the meta-test set.
    """
    logger.info("Combining predictions and evaluating the meta model...")

    # Ensure data consistency (if necessary)
    check_for_data_consistency(evaluation_results_main_model, evaluation_results_hyper_classifier_model)

    # Prepare training data for meta-model
    # Use the existing merged_results from validation data
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

    # Now, we will evaluate all three models on the meta-test set
    metrics_df = meta_classifier_pipeline(
        merged_results,
        main_model,
        hyper_model,
        X_test_meta_main,
        X_test_meta_hyper,
        y_test_meta,
        model_type='xgboost'  # or 'logistic_regression'
    )

    # Print the comparison table
    print("\nComparison of Evaluation Metrics on Meta-Test Set:\n")
    print(metrics_df.to_string(float_format='{:,.4f}'.format))

    logger.info("Meta-classifier model evaluation completed.")


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


def train_meta_classifier(X_meta_train, y_meta_train, model_type='logistic_regression'):
    """
    Train a meta-classifier model.

    Parameters:
    - X_meta_train: DataFrame with meta features (probabilities from base models)
    - y_meta_train: Series with true labels
    - model_type: 'logistic_regression' or 'xgboost'

    Returns:
    - trained meta-classifier model
    """
    if model_type == 'logistic_regression':
        meta_model = LogisticRegression()
        meta_model.fit(X_meta_train, y_meta_train)
    elif model_type == 'xgboost':
        # Define the parameter grid to search
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [2, 3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [1, 1.5, 2]
        }

        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42
        )

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=5,
            n_jobs=-1,
            verbose=1
        )

        # Perform the grid search
        grid_search.fit(X_meta_train, y_meta_train)

        # Get the best model
        meta_model = grid_search.best_estimator_

        print("Best parameters found for XGBoost meta-classifier:")
        print(grid_search.best_params_)
        print(f"Best AUC-ROC score from cross-validation: {grid_search.best_score_:.4f}")
    else:
        raise ValueError("Invalid model_type. Choose 'logistic_regression' or 'xgboost'.")

    return meta_model


def get_meta_predictions(meta_model, X_meta_test):
    """
    Get predictions and probabilities from the meta-classifier.

    Parameters:
    - meta_model: trained meta-classifier model
    - X_meta_test: DataFrame with meta features

    Returns:
    - y_meta_pred: predicted labels
    - y_meta_prob: predicted probabilities
    """
    y_meta_pred = meta_model.predict(X_meta_test)
    y_meta_prob = meta_model.predict_proba(X_meta_test)[:, 1]
    return y_meta_pred, y_meta_prob


def meta_classifier_pipeline(merged_results, main_model, hyper_classifier_model, X_test_meta_main, X_test_meta_hyper,
                             y_test_meta, model_type='xgboost'):
    """
    Train the meta-classifier and evaluate it along with base models on the meta-test set.

    Parameters:
    - merged_results: DataFrame with merged predictions from base models on validation set
    - main_model: trained main model
    - hyper_classifier_model: trained hyper-classifier model
    - X_test_meta_main: features for the main model on the meta-test set
    - X_test_meta_hyper: features for the hyper-classifier on the meta-test set
    - y_test_meta: true labels for the meta-test set
    - model_type: 'logistic_regression' or 'xgboost'

    Returns:
    - metrics_df: DataFrame containing evaluation metrics for all three models
    """
    # Prepare training data for meta-model
    X_meta_train = merged_results[['y_prob_main', 'y_prob_hyper_classifier']]
    y_meta_train = merged_results['y_true']

    # Train the meta-classifier with hyperparameter tuning
    meta_model = train_meta_classifier(X_meta_train, y_meta_train, model_type=model_type)

    # Generate predictions from base models on meta-test set

    # Generate predictions from main model on X_test_meta_main
    y_prob_main_meta = main_model.predict_proba(X_test_meta_main)[:, 1]
    y_pred_main_meta = main_model.predict(X_test_meta_main)

    # Generate predictions from hyper-classifier on X_test_meta_hyper
    y_prob_hyper_meta = hyper_classifier_model.predict_proba(X_test_meta_hyper)[:, 1]
    y_pred_hyper_meta = hyper_classifier_model.predict(X_test_meta_hyper)

    # Prepare test data for meta-model
    X_meta_test = pd.DataFrame({
        'y_prob_main': y_prob_main_meta,
        'y_prob_hyper_classifier': y_prob_hyper_meta
    })

    y_meta_test = y_test_meta.reset_index(drop=True)

    # Get predictions from meta-classifier
    y_meta_pred, y_meta_prob = get_meta_predictions(meta_model, X_meta_test)

    # Evaluate and collect metrics for the main model
    metrics_main_model = get_metrics(
        y_meta_test,
        y_pred_main_meta,
        y_prob_main_meta
    )

    # Evaluate and collect metrics for the hyper-classifier model
    metrics_hyper_classifier_model = get_metrics(
        y_meta_test,
        y_pred_hyper_meta,
        y_prob_hyper_meta
    )

    # Evaluate and collect metrics for the meta-classifier model
    metrics_meta_classifier_model = get_metrics(
        y_meta_test,
        y_meta_pred,
        y_meta_prob
    )

    # Combine all metrics into a DataFrame
    metrics_df = pd.DataFrame({
        'Main Model': metrics_main_model,
        'Hyper-Classifier Model': metrics_hyper_classifier_model,
        'Meta-Classifier Model': metrics_meta_classifier_model
    })

    # Transpose the DataFrame to have models as rows
    metrics_df = metrics_df.transpose()

    # Print evaluation metrics for meta-classifier
    print(f"\nMeta-Model ({model_type.replace('_', ' ').title()}) Evaluation Metrics:")
    print(f"Accuracy: {metrics_meta_classifier_model['Accuracy']:.4f}")
    print(f"Precision: {metrics_meta_classifier_model['Precision']:.4f}")
    print(f"Recall: {metrics_meta_classifier_model['Recall']:.4f}")
    print(f"F1 Score: {metrics_meta_classifier_model['F1-Score']:.4f}")
    print(f"AUC-ROC Score: {metrics_meta_classifier_model['AUC-ROC']:.4f}")

    # Confusion matrix for meta-classifier
    conf_matrix = confusion_matrix(y_meta_test, y_meta_pred)
    print(f"Confusion Matrix:\n{conf_matrix}")

    TN, FP, FN, TP = conf_matrix.ravel()
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    print(f"True Positives (TP): {TP}")

    # Classification report for meta-classifier
    report = classification_report(y_meta_test, y_meta_pred, target_names=['Non-Vandalism', 'Vandalism'],
                                   zero_division=0)
    print(f"\nClassification Report:\n{report}")

    # Print best parameters if XGBoost
    if model_type == 'xgboost':
        print("\nBest parameters used for XGBoost meta-classifier:")
        print(meta_model.get_params())

    return metrics_df
