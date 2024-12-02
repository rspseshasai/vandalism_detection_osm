# hyper_classifier/hyper_classifier_training.py

import xgboost as xgb

from config import logger, HYPER_MODEL_PATH


def train_hyper_classifier(X_train, y_train, X_val, y_val):
    logger.info("Training hyper-classifier model...")

    eval_set = [(X_val, y_val)]

    hyper_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        num_boost_round=1000,
        evals=eval_set,
        early_stopping_rounds=20,
        verbose_eval=10
    )

    hyper_model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False
    )
    # Save model
    hyper_model.save_model(HYPER_MODEL_PATH)
    logger.info(f"Hyper-classifier model saved to {HYPER_MODEL_PATH}")

    return hyper_model
