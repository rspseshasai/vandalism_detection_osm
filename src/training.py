# src/training.py

import xgboost as xgb
from joblib import dump

from src.config import logger


def train_final_model(X_train, y_train, X_val, y_val, best_params):
    """
    Train the final XGBoost model using the best hyperparameters.
    """
    logger.info("Training final model with best hyperparameters...")
    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        early_stopping_rounds=20,
        **best_params
    )

    # Define evaluation set (validation data only)
    eval_set = [(X_val, y_val)]

    # Fit the models and track evaluation metrics
    final_model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False
    )
    logger.info("Final model training completed.")
    return final_model


def save_model(model, model_path):
    """
    Save the trained model to a file.
    """
    logger.info(f"Saving model to {model_path}...")
    dump(model, model_path)
    logger.info("Model saved successfully.")
