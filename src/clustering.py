# src/clustering.py

import os

from sklearn.cluster import KMeans

from src import config

def perform_clustering(X_train, X_val, X_test, X_test_meta, n_clusters=100):
    """
    Fits KMeans clustering on training data and assigns cluster labels to training, validation, test,
    and optionally meta-test data based on the DATASET_TYPE.
    """
    os.environ["LOKY_MAX_CPU_COUNT"] = "4"

    # Ensure 'centroid_x' and 'centroid_y' are available
    required_columns = ['centroid_x', 'centroid_y']
    for col in required_columns:
        if col not in X_train.columns:
            logger.error(f"Column '{col}' not found in X_train.")
            raise KeyError(f"Column '{col}' not found in X_train.")

    # Extract centroids from training data
    centroids_train = X_train[required_columns].values

    # Fit clustering model on training data
    if centroids_train.shape[0] < n_clusters:
        n_clusters = centroids_train.shape[0]
    clustering_model = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_STATE)
    clustering_model.fit(centroids_train)

    # Assign cluster labels to training data
    X_train = X_train.copy()
    X_train['cluster_label'] = clustering_model.labels_
    logger.info("Cluster labels assigned to training data.")

    # Assign cluster labels to validation and test data
    for dataset, name in zip([X_val, X_test], ["Validation", "Test"]):
        for col in required_columns:
            if col not in dataset.columns:
                logger.error(f"Column '{col}' not found in {name} data.")
                raise KeyError(f"Column '{col}' not found in {name} data.")

        centroids = dataset[required_columns].values
        cluster_labels = clustering_model.predict(centroids)
        dataset = dataset.copy()
        dataset['cluster_label'] = cluster_labels
        logger.info(f"Cluster labels assigned to {name} data.")

        if name == "Validation":
            X_val = dataset
        elif name == "Test":
            X_test = dataset

    # Assign cluster labels to meta-test data only if DATASET_TYPE is 'changeset'
    if DATASET_TYPE == "changeset" and X_test_meta is not None:
        for col in required_columns:
            if col not in X_test_meta.columns:
                logger.error(f"Column '{col}' not found in Meta-Test data.")
                raise KeyError(f"Column '{col}' not found in Meta-Test data.")

        centroids = X_test_meta[required_columns].values
        cluster_labels = clustering_model.predict(centroids)
        X_test_meta = X_test_meta.copy()
        X_test_meta['cluster_label'] = cluster_labels
        logger.info("Cluster labels assigned to Meta-Test data.")
    elif DATASET_TYPE != "changeset":
        logger.info("Clustering not required for Meta-Test data as DATASET_TYPE is not 'changeset'.")

    return X_train, X_val, X_test, X_test_meta


import matplotlib.pyplot as plt
import pandas as pd
from config import VISUALIZATION_DATA_PATH, logger, DATASET_TYPE


def load_clustered_data():
    """
    Load the clustered data from saved visualization files.

    Returns:
    - X_train: Training dataset with cluster labels.
    - X_val: Validation dataset with cluster labels.
    - X_test: Test dataset with cluster labels.
    """
    try:
        X_train = pd.read_parquet(VISUALIZATION_DATA_PATH['clustering_train'])
        X_val = pd.read_parquet(VISUALIZATION_DATA_PATH['clustering_val'])
        X_test = pd.read_parquet(VISUALIZATION_DATA_PATH['clustering_test'])
        logger.info("Clustered data loaded successfully.")
        return X_train, X_val, X_test
    except FileNotFoundError as e:
        logger.error(f"Error loading clustered data: {e}")
        raise


def plot_clusters(data, title):
    """
    Helper function to plot clustering results.

    Parameters:
    - data: DataFrame containing clustering results.
    - title: Title for the plot.
    """
    if not {'cluster_label', 'centroid_x', 'centroid_y'}.issubset(data.columns):
        logger.warning(f"Required columns missing in data for {title}. Ensure clustering was performed.")
        return

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        data['centroid_x'],
        data['centroid_y'],
        c=data['cluster_label'],
        cmap='tab10',
        alpha=0.7
    )
    plt.colorbar(scatter, label="Cluster Label")
    plt.title(title)
    plt.xlabel("Centroid X")
    plt.ylabel("Centroid Y")
    plt.grid(True)
    plt.show()


def visualize_clustering():
    """
    Visualize the clustering results for training, validation, and test datasets.
    """
    try:
        # Load clustered data
        X_train, X_val, X_test = load_clustered_data()

        # Visualize each dataset
        logger.info("Visualizing clustering results...")
        plot_clusters(X_train, "Clustering Visualization - Training Set")
        plot_clusters(X_val, "Clustering Visualization - Validation Set")
        plot_clusters(X_test, "Clustering Visualization - Test Set")
    except Exception as e:
        logger.error(f"Error during clustering visualization: {e}")
