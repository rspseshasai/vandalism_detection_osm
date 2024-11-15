import joblib
import xgboost as xgb

from logger.logger_config import logger


def train_final_model(X_train, y_train, X_val, y_val, best_params):
    # Initialize the final model with the best hyperparameters
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

    logger.info("Model training complete.")

    # Log the best iteration and score
    logger.info(f"Best iteration: {final_model.best_iteration + 1}")
    logger.info(f"Best score (on validation set): {final_model.best_score:.4f}")

    # Check if early stopping occurred
    if final_model.best_iteration + 1 < final_model.n_estimators:
        logger.info(f"Early stopping occurred. Model stopped after {final_model.best_iteration + 1} iterations.")
    else:
        logger.info(f"No early stopping. Model trained for all {final_model.n_estimators} iterations.")

    return final_model


def save_model(model, file_path):
    """Save the model to a specified file path."""

    joblib.dump(model, file_path)
    logger.info(f"Model saved to {file_path}")


def load_model(file_path):
    """Load a model from the specified file path."""

    model = joblib.load(file_path)
    logger.info(f"Model loaded from {file_path}")
    return model
