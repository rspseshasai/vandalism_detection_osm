# bootstrap_evaluation.py
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import skew, kurtosis, normaltest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)
from sklearn.utils import resample
from tqdm import tqdm

from logger_config import logger


def perform_bootstrap_evaluation(
        model, X_test, y_test, n_iterations=1000, random_state=None, n_jobs=-1
):
    """
    Perform bootstrapping_results on the test set to evaluate the models's performance.

    Parameters:
    - models: Trained models to evaluate.
    - X_test: Features of the test set.
    - y_test: Labels of the test set.
    - n_iterations: Number of bootstrap samples to generate (default: 1000).
    - random_state: Random state for reproducibility (default: None).
    - n_jobs: Number of CPU cores to use for parallel processing (default: -1).

    Returns:
    - metrics_df: DataFrame containing all metrics from each iteration.
    """

    def bootstrap_iteration(i):
        # Adjust random state for reproducibility
        rs = i if random_state is None else random_state + i

        # Generate a bootstrap sample from the test set
        X_bootstrap, y_bootstrap = resample(
            X_test, y_test,
            replace=True,
            n_samples=len(y_test),
            random_state=rs
        )

        # Make predictions on the bootstrap sample
        y_pred = model.predict(X_bootstrap)
        y_prob = model.predict_proba(X_bootstrap)[:, 1]

        # Calculate performance metrics
        accuracy = accuracy_score(y_bootstrap, y_pred)
        precision = precision_score(y_bootstrap, y_pred, zero_division=0)
        recall = recall_score(y_bootstrap, y_pred, zero_division=0)
        f1 = f1_score(y_bootstrap, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_bootstrap, y_prob)
        auc_pr = average_precision_score(y_bootstrap, y_prob)

        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'AUC-ROC': auc_roc,
            'AUC-PR': auc_pr
        }

    # Run bootstrap iterations in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(bootstrap_iteration)(i) for i in tqdm(range(n_iterations))
    )

    # Create a DataFrame of the metrics
    metrics_df = pd.DataFrame(results)

    return metrics_df


def calculate_bootstrap_statistics(metrics_df):
    """
    Calculate mean, standard deviation, and 95% confidence intervals for each metric.

    Parameters:
    - metrics_df: DataFrame containing metrics from bootstrapping_results.

    Returns:
    - results_df: DataFrame with statistical summary of each metric.
    """
    results = []
    for metric_name in metrics_df.columns:
        metric_values = metrics_df[metric_name]
        mean = metric_values.mean()
        std_dev = metric_values.std(ddof=1)
        lower_ci = metric_values.quantile(0.025)
        upper_ci = metric_values.quantile(0.975)

        results.append({
            'Metric': metric_name,
            'Mean': mean,
            'Std Dev': std_dev,
            '95% CI Lower': lower_ci,
            '95% CI Upper': upper_ci
        })

    results_df = pd.DataFrame(results)
    return results_df


def plot_metric_distributions(metrics_df):
    """
    Plot histograms and box plots for each performance metric.

    Parameters:
    - metrics_df: DataFrame containing metrics from bootstrapping_results.
    """
    for metric_name in metrics_df.columns:
        plt.figure(figsize=(10, 4))

        # Histogram with KDE
        plt.subplot(1, 2, 1)
        sns.histplot(metrics_df[metric_name], bins=30, kde=True)
        plt.title(f'Distribution of {metric_name}')
        plt.xlabel(metric_name)
        plt.ylabel('Frequency')

        # Box plot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=metrics_df[metric_name])
        plt.title(f'Box Plot of {metric_name}')
        plt.xlabel(metric_name)

        plt.tight_layout()
        plt.show()


def compute_additional_statistics(metrics_df):
    """
    Compute skewness, kurtosis, and normality test for each metric.

    Parameters:
    - metrics_df: DataFrame containing metrics from bootstrapping_results.

    Returns:
    - stats_df: DataFrame containing additional statistics for each metric.
    """
    stats = []
    for metric_name in metrics_df.columns:
        metric_values = metrics_df[metric_name]
        skewness = skew(metric_values)
        kurt = kurtosis(metric_values)
        stat, p_value = normaltest(metric_values)
        normality = 'Yes' if p_value > 0.05 else 'No'

        stats.append({
            'Metric': metric_name,
            'Skewness': skewness,
            'Kurtosis': kurt,
            'Normal Distribution': normality,
            'Normality p-value': p_value
        })

    stats_df = pd.DataFrame(stats)
    return stats_df


def save_bootstrap_results(metrics_df, results_df, stats_df, folder_to_save_bootstrap_results, prefix='bootstrap'):
    """
    Save bootstrap metrics and statistics to CSV files.

    Parameters:
    - metrics_df: DataFrame containing metrics from bootstrapping_results.
    - results_df: DataFrame with statistical summary of each metric.
    - stats_df: DataFrame containing additional statistics for each metric.
    - prefix: Prefix for the saved files (default: 'bootstrap').
    """
    # Check if the directory exists, else create it
    if not os.path.exists(folder_to_save_bootstrap_results):
        os.makedirs(folder_to_save_bootstrap_results)
        logger.info(f"Directory created at: {folder_to_save_bootstrap_results}")
    else:
        logger.info(f"Directory already exists at: {folder_to_save_bootstrap_results}")

    metrics_df.to_csv(f'{folder_to_save_bootstrap_results}/{prefix}_metrics_all_iterations_v2.csv', index=False)
    results_df.to_csv(f'{folder_to_save_bootstrap_results}/{prefix}_results_summary_v2.csv', index=False)
    stats_df.to_csv(f'{folder_to_save_bootstrap_results}/{prefix}_additional_statistics_v2.csv', index=False)
    logger.info(f'Results saved in saved_parameters with prefix "{folder_to_save_bootstrap_results}/{prefix}"')
