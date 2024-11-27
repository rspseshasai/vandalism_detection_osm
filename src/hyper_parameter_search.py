import json
import os
import random

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

from config import logger


def load_best_hyperparameters(hyperparams_file):
    if os.path.exists(hyperparams_file):
        with open(hyperparams_file, 'r') as f:
            best_params = json.load(f)
            logger.info(f"Loaded hyperparameters from {hyperparams_file}")
        return best_params
    else:
        raise FileNotFoundError(f"No hyperparameters file found at {hyperparams_file}")


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

    print("Random Hyperparameters Generated: \n")
    print(random_params)
    return random_params


def randomized_search_cv(X_train, y_train, hyperparams_file):
    # Define the parameter grid
    # param_grid = {
    #     'learning_rate': [0.005, 0.01, 0.05, 0.1],
    #     'max_depth': [3, 5, 7, 9, 11],
    #     'subsample': [0.6, 0.8, 1.0],
    #     'colsample_bytree': [0.6, 0.8, 1.0],
    #     'lambda': [0, 1, 3],
    #     'alpha': [0, 1, 2],
    #     'min_child_weight': [1, 5, 10],
    #     'gamma': [0, 0.1, 0.5],
    #     'n_estimators': [100, 200, 500],
    #     'scale_pos_weight': [1, 10, 25, 50]  # Adjust based on class imbalance
    # }

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
        # Initialize the models
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='aucpr',
        )

        # TODO: Move CV, n_jobs, n_iter params to config file
        # Set up and run RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=50,
            scoring='roc_auc',
            cv=5,
            verbose=False,
            n_jobs=-1,
            random_state=42
        )
        random_search.fit(X_train, y_train)

        # Get the best parameters
        best_params = random_search.best_params_

        # Save best parameters to file
        with open(hyperparams_file, 'w') as f:
            json.dump(best_params, f)
            logger.info(f"Best hyperparameters saved to {hyperparams_file}")

    # Load the best hyperparameters
    best_params = load_best_hyperparameters(hyperparams_file)
    logger.info("best hyper parameters: " + str(best_params))

    return best_params
