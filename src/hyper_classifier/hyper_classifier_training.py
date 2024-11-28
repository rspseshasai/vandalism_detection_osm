# hyper_classifier/hyper_classifier_training.py

import xgboost as xgb

from config import logger, HYPER_MODEL_PATH


def train_hyper_classifier(X_train, y_train, X_val, y_val):
    logger.info("Training hyper-classifier model...")

    # Prepare DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Define parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'seed': 42,
        'use_label_encoder': False
    }

    # Training with evaluation on validation set
    evals = [(dtrain, 'train'), (dval, 'eval')]
    hyper_model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=10
    )

    # Save model
    hyper_model.save_model(HYPER_MODEL_PATH)
    logger.info(f"Hyper-classifier model saved to {HYPER_MODEL_PATH}")

    return hyper_model
