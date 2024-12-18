# hyper_classifier/hyper_classifier_main.py

from config import logger
from hyper_classifier.hyper_classfier_data_collection import data_loading
from hyper_classifier.hyper_classifier_data_splitting import split_data
from hyper_classifier.hyper_classifier_feature_engineering import engineer_features
from hyper_classifier.hyper_classifier_training import train_hyper_classifier
from ml_training_and_eval_pipeline import evaluation_helper


def run_hyper_classifier_pipeline(split_ids):
    logger.info("Starting hyper-classifier pipeline...")

    # Step 1: Obtain Per-Contribution Predictions and changeset labels
    changeset_per_contribution_pred_df, changeset_labels = data_loading()

    # Step 2: Feature Engineering
    changeset_features = engineer_features(changeset_per_contribution_pred_df)

    # Step 3: Data Splitting
    X_train, X_val, X_test, X_test_meta, y_train, y_val, y_test, y_test_meta = split_data(changeset_features, changeset_labels, split_ids)

    # Step 4: Model Training
    hyper_model = train_hyper_classifier(X_train, y_train, X_val, y_val)

    # Step 5: Model Evaluation
    evaluation_results_hyper_classifier_model = evaluation_helper(hyper_model, X_train, y_train, X_test, y_test, split_ids['test'], 'hyper_classifier')

    logger.info("Hyper-classifier pipeline completed.")
    return hyper_model, evaluation_results_hyper_classifier_model, X_test_meta
