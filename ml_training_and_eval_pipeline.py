import os
import sys
from sqlite3 import NotSupportedError

import joblib
import pandas as pd

# Adjust the path to import modules from src
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_dir, 'src'))

from config import logger, BEST_PARAMS_PATH, TEST_RUN, SPLIT_METHOD, FORCE_COMPUTE_FEATURES, DATASET_TYPE, \
    PROCESSED_ENCODED_FEATURES_FILE, PROCESSED_FEATURES_FILE, CLUSTER_MODEL_PATH, OPTIMAL_THRESHOLD_FOR_INFERENCE_PATH, \
    DEFAULT_THRESHOLD_FOR_EVALUATION, SHOULD_PERFORM_BOOTSTRAP_EVALUATION, COMMON_CHANGESET_IDS, \
    SHOULD_PERFORM_CROSS_VALIDATION
from src import config
from src.data_loading import load_data

from src.feature_engineering import get_or_generate_features

from src.preprocessing import preprocess_features
from src.data_splitting import split_train_test_val, calculate_statistics, log_dataset_shapes
from src.clustering import perform_clustering
from src.training import train_final_model, save_model

from geographical_evaluation import geographical_evaluation
from hyper_parameter_search import randomized_search_cv, load_best_hyperparameters

from src.evaluation import (
    evaluate_train_test_metrics,
    calculate_auc_scores,
    save_evaluation_results, evaluate_model_with_cv
)
from src.bootstrap_evaluation import (
    perform_bootstrap_evaluation,
    calculate_bootstrap_statistics,
    save_bootstrap_results,
    compute_additional_statistics
)


# Step 1: Load Data
def data_loading_helper():
    logger.info("Starting data loading...")

    data_df = load_data(data_path=config.RAW_DATA_FILE, print_sample_data=False)

    if config.SAVE_VISUALIZATION_SAMPLES:
        sample_path = config.VISUALIZATION_DATA_PATH['data_loading']
        data_df.head(100).to_parquet(sample_path)
        logger.info(f"Saved data loading sample to {sample_path}")

    try:
        counts = data_df['vandalism'].value_counts()
    except KeyError:
        counts = data_df['label'].value_counts()

    logger.info(f"Number of vandalism contributions in the data set: {counts.get(1)}")
    logger.info(f"Number of non-vandalism contributions in the data set: {counts.get(0)}")

    logger.info("Data loading completed.")
    if DATASET_TYPE == 'changeset':
        logger.info(
            "Limiting to the changeset entries matching common changeset IDs - to maintain consistent dataset for hyper classifier that matches with contribution data set.")
        data_df = data_df[data_df['changeset_id'].isin(COMMON_CHANGESET_IDS)]
    return data_df


# Step 2: Feature Engineering
def feature_engineering_helper(data_df):
    logger.info("Starting feature engineering...")
    features_df = get_or_generate_features(
        data_df,
        True,
        PROCESSED_FEATURES_FILE,
        force_compute_features=FORCE_COMPUTE_FEATURES,
        test_mode=TEST_RUN,
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
    X_encoded, y = preprocess_features(features_df, True)

    if config.SAVE_VISUALIZATION_SAMPLES:
        sample_path_X = config.VISUALIZATION_DATA_PATH['preprocessing_X']
        sample_path_y = config.VISUALIZATION_DATA_PATH['preprocessing_y']
        X_encoded.head(100).to_parquet(sample_path_X)
        y.head(100).to_frame(name='vandalism').to_parquet(sample_path_y)
        logger.info(f"Saved preprocessing samples to {sample_path_X} and {sample_path_y}")

    logger.info("Preprocessing completed.")
    return X_encoded, y


# Step 4: Data Splitting
def data_splitting_helper(X_encoded, y, split_type, train_regions, val_regions, test_regions):
    logger.info(f"Starting data splitting with method: {split_type}")

    if split_type == 'random':
        split_params = {
            'test_size': config.TEST_SIZE,
            'val_size': config.VAL_SIZE,
            'random_state': config.RANDOM_STATE
        }
    elif split_type == 'geographic':
        if DATASET_TYPE != 'contribution':
            raise NotSupportedError(f"Split type '{split_type}' is only supported with contribution dataset")
        split_params = {
            'split_key': config.GEOGRAPHIC_SPLIT_KEY,
            'train_regions': train_regions,
            'val_regions': val_regions,
            'test_regions': test_regions
        }
    elif split_type == 'temporal':
        if DATASET_TYPE != 'contribution':
            raise NotSupportedError(f"Split type '{split_type}' is only supported with contribution dataset")
        split_params = {
            'date_column': config.DATE_COLUMN
        }
    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    # Ensure 'changeset_id' is retained
    if 'changeset_id' not in X_encoded.columns and DATASET_TYPE == 'changeset':
        raise ValueError("Column 'changeset_id' is missing in X_encoded.")

    # Split the data
    X_train, X_val, X_test, X_test_meta, y_train, y_val, y_test, y_test_meta = split_train_test_val(
        X_encoded, y, split_type=split_type, **split_params
    )

    # Extract the split IDs
    split_ids = {}
    if DATASET_TYPE == 'changeset':
        split_ids['train'] = X_train['changeset_id'].copy()
        split_ids['val'] = X_val['changeset_id'].copy()
        split_ids['test'] = X_test['changeset_id'].copy()
        split_ids['meta_test'] = X_test_meta['changeset_id'].copy()

    log_dataset_shapes(X_train, X_val, X_test, X_test_meta, y_train, y_val, y_test, y_test_meta)
    calculate_statistics(y_train, "Train Set")
    calculate_statistics(y_val, "Validation Set")
    calculate_statistics(y_test, "Test Set")
    if DATASET_TYPE == 'changeset':
        calculate_statistics(y_test_meta, "Meta Test Set")

    if config.SAVE_VISUALIZATION_SAMPLES:
        X_train.head(100).to_parquet(config.VISUALIZATION_DATA_PATH['data_splitting_X_train'])
        X_val.head(100).to_parquet(config.VISUALIZATION_DATA_PATH['data_splitting_X_val'])
        X_test.head(100).to_parquet(config.VISUALIZATION_DATA_PATH['data_splitting_X_test'])
        logger.info("Saved all data splits for visualization.")

    logger.info("Data splitting completed.")
    return X_train, X_val, X_test, X_test_meta, y_train, y_val, y_test, y_test_meta, split_ids


# Step 5: Clustering
def clustering_helper(X_train, X_val, X_test, X_test_meta):
    logger.info("Starting clustering...")
    X_train, X_val, X_test, X_test_meta, clustering_model = perform_clustering(
        X_train, X_val, X_test, X_test_meta, n_clusters=config.N_CLUSTERS
    )

    # Save the clustering model (already done in perform_clustering)
    joblib.dump(clustering_model, CLUSTER_MODEL_PATH)
    logger.info(f"Clustering model saved to '{CLUSTER_MODEL_PATH}'.")

    X_combined = pd.concat([X_train, X_val, X_test, X_test_meta])
    # Save to a Parquet file for hyper classifier
    X_combined.to_parquet(PROCESSED_ENCODED_FEATURES_FILE)
    logger.info(f"Combined data saved to {PROCESSED_ENCODED_FEATURES_FILE}")

    if config.SAVE_VISUALIZATION_SAMPLES:
        sample_path_train = config.VISUALIZATION_DATA_PATH['clustering_train']
        sample_path_val = config.VISUALIZATION_DATA_PATH['clustering_val']
        sample_path_test = config.VISUALIZATION_DATA_PATH['clustering_test']
        X_train.head(100).to_parquet(sample_path_train)
        X_val.head(100).to_parquet(sample_path_val)
        X_test.head(100).to_parquet(sample_path_test)
        logger.info(f"Saved clustering samples to {sample_path_train}, {sample_path_val}, and {sample_path_test}")

    logger.info("Clustering completed.")
    return X_train, X_val, X_test, X_test_meta


# Step 6: Model Training
def training_helper(X_train, y_train, X_val, y_val):
    best_params = randomized_search_cv(
        X_train, y_train, BEST_PARAMS_PATH
    )

    logger.info("Starting model training...")
    final_model = train_final_model(X_train, y_train, X_val, y_val, best_params)
    # compute_optimal_threshold(final_model, X_val, y_val, OPTIMAL_THRESHOLD_FOR_INFERENCE_PATH)
    save_model(final_model, config.FINAL_MODEL_PATH)
    logger.info("Model training completed.")
    return final_model


# Step 7: Model Evaluation
def evaluation_helper(model, X_train, y_train, X_test=None, y_test=None, X_test_ids=None, model_type="model"):
    logger.info(f"Starting evaluation for {model_type} model...")

    # Load or use default threshold
    if os.path.exists(OPTIMAL_THRESHOLD_FOR_INFERENCE_PATH):
        threshold = joblib.load(OPTIMAL_THRESHOLD_FOR_INFERENCE_PATH)
        logger.info(f"Loaded custom threshold: {threshold:.4f}")
    else:
        threshold = DEFAULT_THRESHOLD_FOR_EVALUATION  # fallback
        logger.warning(f"No custom threshold found. Using default {threshold}")

    # Evaluate train and test metrics
    y_test_pred, y_test_prob = evaluate_train_test_metrics(model, X_train, y_train, X_test, y_test, threshold)

    # Initialize evaluation results
    evaluation_results_main_model = None

    # If test set exists, calculate metrics and save results
    if X_test is not None and y_test is not None and not X_test.empty and not y_test.empty:
        logger.info("Calculating additional metrics for test set...")
        cm = calculate_auc_scores(y_test, y_test_pred, y_test_prob)

        if DATASET_TYPE == 'contribution':
            evaluation_results_main_model = pd.DataFrame({
                'y_true': y_test.reset_index(drop=True),
                f'y_pred_{model_type}': y_test_pred,
                f'y_prob_{model_type}': y_test_prob
            })
        else:  # Changeset-specific case
            evaluation_results_main_model = pd.DataFrame({
                'changeset_id': X_test_ids.reset_index(drop=True),
                'y_true': y_test.reset_index(drop=True),
                f'y_pred_{model_type}': y_test_pred,
                f'y_prob_{model_type}': y_test_prob
            })

        # Save evaluation data for visualization
        save_evaluation_results(evaluation_results_main_model, cm, model_type)

    else:
        logger.warning("No test set provided. Skipping test evaluation.")

    # Optional: Perform Cross-Validation on Training Data (if applicable)
    if SHOULD_PERFORM_CROSS_VALIDATION:
        evaluate_model_with_cv(X_train, y_train, load_best_hyperparameters(BEST_PARAMS_PATH))

    logger.info(f"Evaluation completed for {model_type} model.")
    return evaluation_results_main_model


# Step 8: Bootstrapping Evaluation
def bootstrapping_evaluation_helper(model, X_test, y_test):
    if SHOULD_PERFORM_BOOTSTRAP_EVALUATION:
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
    else:
        logger.info("Bootstrapping evaluation skipped.")


# Step 9: Geographical Evaluation
def geographical_evaluation_helper(model, X_test, y_test):
    if DATASET_TYPE != 'contribution':
        logger.info("Skipping geographical evaluation as it's only supported with contribution dataset")
        return

    logger.info("Starting geographical evaluation...")

    # Evaluate on continents
    geographical_evaluation(model, X_test, y_test, split_key='continent')

    # Evaluate on countries
    geographical_evaluation(model, X_test, y_test, split_key='country')
    logger.info("Geographical evaluation completed.")


# Step 10: Hyper-Classifier
def hyper_classifier_helper(split_ids):
    if config.DATASET_TYPE != 'changeset':
        logger.info("Skipping hyper-classifier as it's only applicable for changeset data.")
        return None, None, None
    logger.info("Starting hyper-classifier pipeline...")
    from hyper_classifier.hyper_classifier_main import run_hyper_classifier_pipeline
    hyper_model, evaluation_results_hyper, X_test_meta_hyper = run_hyper_classifier_pipeline(split_ids)
    logger.info("Hyper-classifier pipeline completed.")
    return hyper_model, evaluation_results_hyper, X_test_meta_hyper


def meta_classifier_helper(evaluation_results_main_model, evaluation_results_hyper, main_model, hyper_model,
                           X_test_meta, X_test_meta_hyper, y_test_meta):
    if config.DATASET_TYPE != 'changeset':
        logger.info("Skipping meta_classifier as it's only applicable for changeset data.")
        return
    from meta_classifier import meta_classifier
    meta_classifier(evaluation_results_main_model, evaluation_results_hyper, main_model, hyper_model, X_test_meta,
                    X_test_meta_hyper, y_test_meta)


# Main Pipeline
def pipeline(train_regions, val_regions, test_regions):
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
        ('hyper_classifier', hyper_classifier_helper),
        ('meta_classifier', meta_classifier_helper),
    ]

    data_df = features_df = None
    X_encoded = y = None
    X_train = X_val = X_test = y_train = y_val = y_test = None
    main_model = None
    split_ids = None

    # Initialize variables to capture evaluation results
    evaluation_results_main_model = None
    evaluation_results_hyper_classifier_model = None

    # Execute each step in the defined order
    for step_name, step_function in pipeline_steps:
        logger.info(f"{('_' * 30)} Executing pipeline step: {str(step_name).upper()} {('_' * 30)}")
        if step_name == 'data_loading':
            data_df = step_function()
        elif step_name == 'feature_engineering':
            features_df = step_function(data_df)
        elif step_name == 'preprocessing':
            X_encoded, y = step_function(features_df)
        elif step_name == 'data_splitting':
            X_train, X_val, X_test, X_test_meta, y_train, y_val, y_test, y_test_meta, split_ids = step_function(
                X_encoded, y, SPLIT_METHOD, train_regions, val_regions, test_regions)
        elif step_name == 'clustering':
            X_train, X_val, X_test, X_test_meta = step_function(X_train, X_val, X_test, X_test_meta)
        elif step_name == 'training':
            main_model = step_function(X_train, y_train, X_val, y_val)
        elif step_name == 'evaluation':
            split_ids_temp = {}
            if DATASET_TYPE == 'changeset':
                split_ids_temp = split_ids['test']
            evaluation_results_main_model = step_function(main_model, X_train, y_train, X_test, y_test,
                                                          split_ids_temp,
                                                          'main')
        elif step_name == 'bootstrapping_evaluation':
            if X_test is not None and y_test is not None and not X_test.empty and not y_test.empty:
                step_function(main_model, X_test, y_test)
            else:
                logger.warning("No test set provided. Skipping bootstrap evaluation.")
        elif step_name == 'geographical_evaluation':
            if X_test is not None and y_test is not None and not X_test.empty and not y_test.empty:
                step_function(main_model, X_test, y_test)
            else:
                logger.warning("No test set provided. Skipping geographic evaluation.")
        elif step_name == 'hyper_classifier':
            hyper_model, evaluation_results_hyper_classifier_model, X_test_meta_hyper = step_function(split_ids)
        elif step_name == 'meta_classifier':
            step_function(evaluation_results_main_model, evaluation_results_hyper_classifier_model, main_model,
                          hyper_model, X_test_meta, X_test_meta_hyper, y_test_meta)
        else:
            logger.warning(f"Unknown pipeline step: {step_name}")

    logger.info("ML pipeline completed successfully.")
    return evaluation_results_main_model


if __name__ == '__main__':
    pipeline(config.TRAIN_REGIONS, config.VAL_REGIONS, config.TEST_REGIONS)
