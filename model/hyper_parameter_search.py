import json
import os
import random

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

from logger.logger_config import logger
from model.load_hyper_parameters import load_best_hyperparameters


def get_random_parameters():
    """Generate random hyperparameters from predefined ranges."""
    # Define the parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'lambda': [0, 1, 3, 5, 10],
        'alpha': [0, 1, 2, 3, 5],
        'min_child_weight': [1, 3, 5, 7, 10],
        'gamma': [0, 0.1, 0.3, 0.5, 1, 2, 3],
        'n_estimators': [50, 60, 80, 100]
    }

    # Generate random hyperparameters
    random_params = {
        'learning_rate': random.choice(param_grid['learning_rate']),
        'max_depth': random.choice(param_grid['max_depth']),
        'subsample': random.choice(param_grid['subsample']),
        'colsample_bytree': random.choice(param_grid['colsample_bytree']),
        'lambda': random.choice(param_grid['lambda']),
        'alpha': random.choice(param_grid['alpha']),
        'min_child_weight': random.choice(param_grid['min_child_weight']),
        'gamma': random.choice(param_grid['gamma']),
        'n_estimators': random.choice(param_grid['n_estimators'])
    }

    logger.info("Random Hyperparameters Generated: " + str(random_params))
    return random_params


def randomized_search_cv(X_train, y_train, hyperparams_file='../saved_parameters/best_hyperparameters.json'):
    # Define the parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'lambda': [0, 1, 3, 5, 10],
        'alpha': [0, 1, 2, 3, 5],
        'min_child_weight': [1, 3, 5, 7, 10],
        'gamma': [0, 0.1, 0.3, 0.5, 1, 2, 3],
        'n_estimators': [50, 60, 80, 100]
    }

    # Check if the hyperparameters file exists
    if os.path.exists(hyperparams_file):
        logger.info("Hyperparameters file already exists.")
    else:
        # Initialize the model
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='aucpr',
        )

        # Set up and run RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=50,
            scoring='roc_auc',
            cv=5,
            verbose=2,
            n_jobs=-1,
            random_state=42
        )
        random_search.fit(X_train, y_train)

        # Get the best parameters
        best_params = random_search.best_params_
        logger.info("Best Hyperparameters:", best_params)

        # Save best parameters to file
        with open(hyperparams_file, 'w') as f:
            json.dump(best_params, f)
            logger.info(f"Best hyperparameters saved to {hyperparams_file}")

    # Load the best hyperparameters
    best_params = load_best_hyperparameters(hyperparams_file)
    logger.info("best hyper parameters: " + str(best_params))

    return best_params
