# main.py

import os
import sys

# Adjust the path to import modules from src and scripts
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_dir, 'src'))
sys.path.append(os.path.join(project_dir, 'scripts'))

from src.logger_config import logger
from scripts.run_feature_engineering import run_feature_engineering
from scripts.run_preprocessing import run_preprocessing
from src.data_loading import load_data
# from src.data_splitting import split_data


def main():
    logger.info("Starting the ML pipeline...")

    # Step 1: Load Contribution Data
    contributions_df = load_data(print_sample_data=False)
    logger.info("Contribution data loaded.")

    # Step 2: Feature Engineering
    features_df = run_feature_engineering(contributions_df)
    logger.info("Feature engineering completed.")

    # Step 3: Preprocessing
    #TODO: Less columns after encoding. Test this
    X_encoded, y = run_preprocessing(features_df)
    logger.info("Preprocessing completed.")

    # Step 4: Data Splitting
    # X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    #     X_encoded, y, strategy='random', test_size=0.2, val_size=0.1
    # )
    logger.info("Data splitting completed.")

    # Step 5: Model Training (Example)
    # from scripts.run_training import run_training
    # model = run_training(X_train, y_train)

    # Step 6: Evaluation (Example)
    # from scripts.run_evaluation import run_evaluation
    # run_evaluation(model, X_test, y_test)

    logger.info("ML pipeline execution completed.")


if __name__ == '__main__':
    main()
