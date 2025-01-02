import os
import joblib
import xgboost as xgb
from joblib import dump

from config import logger, FINAL_TRAINED_FEATURES_PATH
from sklearn.metrics import precision_recall_curve


def train_final_model(X_train, y_train, X_val, y_val, best_params):
    """
    Train the final XGBoost model using the best hyperparameters.
    """
    # Save the feature names for future alignment
    trained_feature_names = X_train.columns.tolist()
    joblib.dump(trained_feature_names, FINAL_TRAINED_FEATURES_PATH)

    logger.info("Training final model with best hyperparameters...")
    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        early_stopping_rounds=20,
        **best_params
    )

    # Define evaluation set (validation data only)
    eval_set = [(X_val, y_val)]

    # Fit the model and track evaluation metrics
    final_model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False
    )
    logger.info("Final model training completed.")
    return final_model


def compute_optimal_threshold(model, X_val, y_val, threshold_file_path):
    """
    Compute the optimal threshold based on the validation set
    and save it to a .pkl file.

    This example finds the threshold that maximizes the F1 score
    from the precision-recall curve.

    :param model: Trained XGBoost model
    :param X_val: Validation features
    :param y_val: Validation labels
    :param threshold_file_path: Where to save the best threshold
    :return: best_threshold used for classification
    """
    # Get predicted probabilities on validation set
    y_val_prob = model.predict_proba(X_val)[:, 1]

    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_val, y_val_prob)

    # Avoid division by zero
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_index = f1_scores.argmax()
    best_threshold = thresholds[best_index] if best_index < len(thresholds) else 0.5

    # Save the best threshold
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
