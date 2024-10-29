import joblib
import xgboost as xgb

from logger.logger_config import logger


def train_final_model(X_train, y_train, X_test, y_test,
                      best_params):
    # Initialize the final model with the best hyperparameters
    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        **best_params
    )

    # Define evaluation sets (train and validation data)
    eval_set = [(X_train, y_train), (X_test, y_test)]

    # Fit the model and track evaluation metrics
    final_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    logger.info("Model training complete.")

    return final_model


def save_model(model, file_path='../saved_parameters/final_xgboost_model.pkl'):
    """Save the model to a specified file path."""

    joblib.dump(model, file_path)
    logger.info(f"Model saved to {file_path}")


def load_model(file_path='../saved_parameters/final_xgboost_model.pkl'):
    """Load a model from the specified file path."""

    model = joblib.load(file_path)
    logger.info(f"Model loaded from {file_path}")
    return model
