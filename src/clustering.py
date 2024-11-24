# src/clustering.py

from sklearn.cluster import KMeans

from src import config
from src.config import logger


def perform_clustering(X_train, X_val, X_test, n_clusters=100):
    """
    Fits KMeans clustering on training data and assigns cluster labels to training, validation, and test data.
    """
    logger.info("Starting clustering...")

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
    logger.info("Clustering model fitted on training data.")

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
        else:
            X_test = dataset

    logger.info("Clustering completed.")

    return X_train, X_val, X_test
