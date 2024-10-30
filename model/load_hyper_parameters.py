import json
import os

from logger.logger_config import logger


def load_best_hyperparameters(hyperparams_file):
    if os.path.exists(hyperparams_file):
        with open(hyperparams_file, 'r') as f:
            best_params = json.load(f)
            logger.info(f"Loaded hyperparameters from {hyperparams_file}")
        return best_params
    else:
        raise FileNotFoundError(f"No hyperparameters file found at {hyperparams_file}")
