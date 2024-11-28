# hyper_classifier/hyper_classifier_main.py

from config import logger
from hyper_classifier.hyper_classfier_data_collection import data_loading
from hyper_classifier.hyper_classifier_data_splitting import split_data
from hyper_classifier.hyper_classifier_evaluation import evaluate_hyper_classifier
from hyper_classifier.hyper_classifier_feature_engineering import engineer_features
from hyper_classifier.hyper_classifier_training import train_hyper_classifier


def run_hyper_classifier_pipeline():
    logger.info("Starting hyper-classifier pipeline...")

    # Step 1: Obtain Per-Contribution Predictions and changeset labels
    contributions_df, changeset_labels = data_loading()

    # Step 2: Feature Engineering
    changeset_features = engineer_features(contributions_df)

    # Step 3: Data Splitting
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(changeset_features, changeset_labels)

    # Step 4: Model Training
    hyper_model = train_hyper_classifier(X_train, y_train, X_val, y_val)

    # Step 5: Model Evaluation
    evaluate_hyper_classifier(hyper_model, X_test, y_test)

    logger.info("Hyper-classifier pipeline completed.")
