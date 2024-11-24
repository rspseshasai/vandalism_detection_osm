import os
import sys

import pandas as pd

from evaluation import save_evaluation_results
from geographical_evaluation import split_test_set_by_key, evaluate_model_on_split_groups
from hyper_parameter_search import randomized_search_cv, load_best_hyperparameters

# Adjust the path to import modules from src
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_dir, 'src'))

from config import logger, BEST_PARAMS_PATH_CONTRIBUTION_DATA, TEST_RUN
from src import config
from src.data_loading import load_data
from src.feature_engineering import get_or_generate_features
from src.preprocessing import preprocess_features
from src.data_splitting import split_train_test_val, calculate_statistics, log_dataset_shapes
from src.clustering import perform_clustering
from src.training import train_final_model, save_model
from src.evaluation import calculate_auc_scores, evaluate_train_test_metrics, \
    evaluate_model_with_cv

from src.evaluation import (
    evaluate_train_test_metrics,
    calculate_auc_scores,
    save_evaluation_results
)
from src.bootstrap_evaluation import (
    perform_bootstrap_evaluation,
    calculate_bootstrap_statistics,
    save_bootstrap_results, compute_additional_statistics
)


# Step 1: Load Data
def data_loading_helper():
    logger.info("Starting data loading...")
    contributions_df = load_data(print_sample_data=False)

    if config.SAVE_VISUALIZATION_SAMPLES:
        sample_path = config.VISUALIZATION_DATA_PATH['data_loading']
        contributions_df.head(100).to_parquet(sample_path)
        logger.info(f"Saved data loading sample to {sample_path}")

    logger.info("Data loading completed.")
    return contributions_df


# Step 2: Feature Engineering
def feature_engineering_helper(contributions_df):
    logger.info("Starting feature engineering...")
    features_df = get_or_generate_features(
        contributions_df,
        force_compute_features=False,
        test_mode=TEST_RUN
    )

    if config.SAVE_VISUALIZATION_SAMPLES:
        sample_path = config.VISUALIZATION_DATA_PATH['feature_engineering']
        features_df.head(100).to_parquet(sample_path)
        logger.info(f"Saved feature engineering sample to {sample_path}")

    logger.info("Feature engineering completed.")
    return features_df


# Step 3: Preprocessing
def preprocessing_helper(features_df):
    logger.info("Starting preprocessing...")
    X_encoded, y = preprocess_features(features_df)

    if config.SAVE_VISUALIZATION_SAMPLES:
        sample_path_X = config.VISUALIZATION_DATA_PATH['preprocessing_X']
        sample_path_y = config.VISUALIZATION_DATA_PATH['preprocessing_y']
        X_encoded.head(100).to_parquet(sample_path_X)
        y.head(100).to_frame(name='vandalism').to_parquet(sample_path_y)
        logger.info(f"Saved preprocessing samples to {sample_path_X} and {sample_path_y}")

    logger.info("Preprocessing completed.")
    return X_encoded, y


# Step 4: Data Splitting
def data_splitting_helper(X_encoded, y):
    logger.info("Starting data splitting...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_test_val(
        X_encoded, y,
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE
    )

    log_dataset_shapes(X_train, X_val, X_test, y_train, y_val, y_test)
    calculate_statistics(y_train, "Train Set")
    calculate_statistics(y_val, "Validation Set")
    calculate_statistics(y_test, "Test Set")

    if config.SAVE_VISUALIZATION_SAMPLES:
        X_train.head(100).to_parquet(config.VISUALIZATION_DATA_PATH['data_splitting_X_train'])
        X_val.head(100).to_parquet(config.VISUALIZATION_DATA_PATH['data_splitting_X_val'])
        X_test.head(100).to_parquet(config.VISUALIZATION_DATA_PATH['data_splitting_X_test'])
        logger.info("Saved all data splits for visualization.")

    logger.info("Data splitting completed.")
    return X_train, X_val, X_test, y_train, y_val, y_test


# Step 5: Clustering
def clustering_helper(X_train, X_val, X_test):
    logger.info("Starting clustering...")
    X_train, X_val, X_test = perform_clustering(X_train, X_val, X_test, n_clusters=config.N_CLUSTERS)

    if config.SAVE_VISUALIZATION_SAMPLES:
        sample_path_train = config.VISUALIZATION_DATA_PATH['clustering_train']
        sample_path_val = config.VISUALIZATION_DATA_PATH['clustering_val']
        sample_path_test = config.VISUALIZATION_DATA_PATH['clustering_test']
        X_train.head(100).to_parquet(sample_path_train)
        X_val.head(100).to_parquet(sample_path_val)
        X_test.head(100).to_parquet(sample_path_test)
        logger.info(f"Saved clustering samples to {sample_path_train}, {sample_path_val}, and {sample_path_test}")

    logger.info("Clustering completed.")
    return X_train, X_val, X_test


# Step 6: Model Training
def training_helper(X_train, y_train, X_val, y_val):
    # Generate random hyperparameters or load from a saved file
    best_params = randomized_search_cv(
        X_train, y_train, BEST_PARAMS_PATH_CONTRIBUTION_DATA
    )

    logger.info("Starting model training...")
    # Train final model
    final_model = train_final_model(X_train, y_train, X_val, y_val, best_params)

    # Save the model
    save_model(final_model, config.FINAL_MODEL_PATH_CONTRIBUTION_DATA)

    logger.info("Model training completed.")
    return final_model


# Step 7: Model Evaluation
def evaluation_helper(model, X_train, y_train, X_test, y_test):
    logger.info("Starting evaluation...")

    # Evaluate train and test metrics
    y_test_pred, y_test_prob = evaluate_train_test_metrics(model, X_train, y_train, X_test, y_test)

    # Calculate additional metrics and confusion matrix
    cm = calculate_auc_scores(y_test, y_test_pred, y_test_prob)

    # Save evaluation data for visualization
    save_evaluation_results(y_test, y_test_pred, y_test_prob, cm)

    # (OPTIONAL): Perform Cross-Validation on Training Data
    evaluate_model_with_cv(
        X_train, y_train, load_best_hyperparameters(BEST_PARAMS_PATH_CONTRIBUTION_DATA)
    )

    logger.info("Evaluation completed.")


# Step 8: Bootstrapping Evaluation
def bootstrapping_evaluation_helper(model, X_test, y_test):
    logger.info("Starting bootstrapping evaluation...")

    # Perform bootstrapping
    metrics_df = perform_bootstrap_evaluation(
        model=model,
        X_test=X_test,
        y_test=y_test,
        n_iterations=config.BOOTSTRAP_ITERATIONS,
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS
    )

    # Calculate statistics
    results_df = calculate_bootstrap_statistics(metrics_df)
    stats_df = compute_additional_statistics(metrics_df)

    # Save results
    save_bootstrap_results(
        metrics_df,
        results_df,
        stats_df,
        folder_to_save_bootstrap_results=config.BOOTSTRAP_RESULTS_DIR,
        prefix='bootstrap_test_set'
    )

    logger.info("Bootstrapping evaluation completed.")


# Step 9: Geographical Evaluation
def geographical_evaluation_helper(model, X_test, y_test):
    logger.info("Starting geographical evaluation...")

    # Evaluate on continents
    geographical_evaluation(model, X_test, y_test, split_key='continent')

    # Evaluate on countries
    geographical_evaluation(model, X_test, y_test, split_key='country')

    logger.info("Geographical evaluation completed.")


def geographical_evaluation(model, X_test, y_test, split_key):
    # Get binary columns for the split key
    binary_columns = [col for col in X_test.columns if col.startswith(f"{split_key}_")]

    # Remove any unwanted columns
    if 'country_count' in binary_columns:
        binary_columns.remove('country_count')

    # Split test set by the specified key
    split_data = split_test_set_by_key(X_test, y_test, binary_columns, split_key)

    # Evaluate model on each split group
    results = evaluate_model_on_split_groups(split_data, model)

    # Convert results to DataFrame
    stats_columns = [
        'Total Samples', 'Total Correct Predictions', 'Total Incorrect Predictions', 'True Positives (TP)',
        'True Negatives (TN)', 'False Positives (FP)', 'False Negatives (FN)',
        'Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC', 'AUC-PR'
    ]

    stats_df = pd.DataFrame.from_dict(results, orient='index')[stats_columns]
    stats_df.index.name = f'{split_key.capitalize()} Name'
    stats_df.reset_index(inplace=True)

    # Save the geographical evaluation results
    geo_results_path = os.path.join(config.GEOGRAPHICAL_RESULTS_DIR, f'{split_key}_evaluation_results.csv')
    stats_df.to_csv(geo_results_path, index=False)
    logger.info(f"Saved geographical evaluation results to {geo_results_path}")


# Main Pipeline
def main():
    logger.info("Starting the ML pipeline...")

    # Define the pipeline steps and their corresponding functions
    pipeline_steps = [
        ('data_loading', data_loading_helper),
        ('feature_engineering', feature_engineering_helper),
        ('preprocessing', preprocessing_helper),
        ('data_splitting', data_splitting_helper),
        ('clustering', clustering_helper),
        ('training', training_helper),
        ('evaluation', evaluation_helper),
        ('bootstrapping_evaluation', bootstrapping_evaluation_helper),
        ('geographical_evaluation', geographical_evaluation_helper),
    ]

    # Data containers
    contributions_df = features_df = None
    X_encoded = y = None
    X_train = X_val = X_test = y_train = y_val = y_test = None
    model = None

    # Execute each step in the defined order
    for step_name, step_function in pipeline_steps:
        logger.info(f"\nExecuting pipeline step: {step_name}")
        if step_name == 'data_loading':
            contributions_df = step_function()
        elif step_name == 'feature_engineering':
            features_df = step_function(contributions_df)
        elif step_name == 'preprocessing':
            X_encoded, y = step_function(features_df)
        elif step_name == 'data_splitting':
            X_train, X_val, X_test, y_train, y_val, y_test = step_function(X_encoded, y)
        elif step_name == 'clustering':
            X_train, X_val, X_test = step_function(X_train, X_val, X_test)
        elif step_name == 'training':
            model = step_function(X_train, y_train, X_val, y_val)
        elif step_name == 'evaluation':
            step_function(model, X_train, y_train, X_test, y_test)
        elif step_name == 'bootstrapping_evaluation':
            step_function(model, X_test, y_test)
        elif step_name == 'geographical_evaluation':
            step_function(model, X_test, y_test)
        else:
            logger.warning(f"Unknown pipeline step: {step_name}")

    logger.info("ML pipeline completed successfully.")


if __name__ == '__main__':
    main()
