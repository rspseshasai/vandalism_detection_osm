import os

import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
from joblib import dump
from sklearn.metrics import precision_recall_curve

from config import logger, FINAL_TRAINED_FEATURES_PATH, PLOTS_OUTPUT_DIR


def train_final_model(X_train, y_train, X_val, y_val, best_params):
    """
    Train the final XGBoost model using the best hyperparameters
    while tracking training and validation metrics (loss, error, accuracy, AUC, etc.).
    """
    # 1) Save feature names for future alignment
    trained_feature_names = X_train.columns.tolist()
    joblib.dump(trained_feature_names, FINAL_TRAINED_FEATURES_PATH)

    logger.info("Training final model with best hyperparameters...")

    # 2) Ensure eval_metric includes 'logloss','error','auc' so we can track them
    #    If your best_params already has 'eval_metric', update or extend it.
    if 'eval_metric' not in best_params:
        best_params['eval_metric'] = ['logloss', 'error', 'auc']
    else:
        # Ensure these metrics are present
        em = best_params['eval_metric']
        if isinstance(em, str):
            em = [em]
        required_m = {'logloss', 'error', 'auc'}
        best_params['eval_metric'] = list(set(em) | required_m)

    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        early_stopping_rounds=20,
        **best_params
    )

    eval_set = [(X_train, y_train), (X_val, y_val)]
    final_model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )

    logger.info("Final model training completed.")

    # 3) Retrieve evaluation results from the model
    evals_result = final_model.evals_result()

    # 4) Plot training progress
    plot_training_progress(evals_result)

    return final_model


def plot_training_progress(evals_result):
    """
    Plot the training and validation metrics during the training process,
    including logloss, error, AUC, and derived accuracy from error.
    All plots are saved to PLOTS_OUTPUT_DIR.
    """
    logger.info("Plotting training progress...")

    # Create directory if not exists
    os.makedirs(os.path.join(PLOTS_OUTPUT_DIR, 'training_progress'), exist_ok=True)

    # For each metric in validation_0
    for metric in evals_result['validation_0'].keys():
        # e.g. metric could be 'logloss','error','auc'
        train_metric = evals_result['validation_0'][metric]
        val_metric = evals_result['validation_1'][metric]

        # Figure for the metric
        plt.figure(figsize=(10, 6))
        epochs = range(len(train_metric))

        plt.plot(epochs, train_metric, label=f'Train {metric}', color='blue')
        plt.plot(epochs, val_metric, label=f'Validation {metric}', color='orange')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.title(f'Training Progress - {metric.capitalize()}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        plot_path = os.path.join(PLOTS_OUTPUT_DIR, f"training_progress_{metric}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"Saved training progress plot for {metric}: {plot_path}")

        # If metric is 'error', also derive & plot accuracy
        if metric == 'error':
            logger.info("Deriving accuracy from error metric and plotting accuracy graph...")
            train_acc = [1.0 - e for e in train_metric]
            val_acc = [1.0 - e for e in val_metric]

            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_acc, label='Train Accuracy', color='green')
            plt.plot(epochs, val_acc, label='Validation Accuracy', color='red')
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.title('Training Progress - Accuracy (Derived from Error)', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

            acc_plot_path = os.path.join(PLOTS_OUTPUT_DIR, "training_progress_accuracy.png")
            plt.savefig(acc_plot_path, dpi=150)
            plt.close()
            logger.info(f"Saved training progress plot for accuracy: {acc_plot_path}")


def compute_optimal_threshold(model, X_val, y_val, threshold_file_path):
    """
    Use the validation set to compute the optimal threshold that maximizes F1,
    then save it to threshold_file_path.
    """
    y_val_prob = model.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_val, y_val_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)

    best_index = f1_scores.argmax()
    if best_index < len(thresholds):
        best_threshold = thresholds[best_index]
    else:
        logger.warning("Optimal Threshold Calculation: FALLBACK - Best Threshold 0.5")
        best_threshold = 0.5  # Fallback

    joblib.dump(best_threshold, threshold_file_path)
    logger.info(f"Optimal threshold computed: {best_threshold:.4f}")
    logger.info(f"Saved threshold to {threshold_file_path}")
    return best_threshold


def save_model(model, model_path):
    """
    Save the trained model to a file.
    """
    logger.info(f"Saving model to {model_path}...")
    dump(model, model_path)
    logger.info("Model saved successfully.")


def load_model(model_path):
    """
    Load a trained machine learning model from the specified path.

    Parameters:
    - model_path: str, path to the saved model file.

    Returns:
    - model: The loaded model.
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")

    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise
