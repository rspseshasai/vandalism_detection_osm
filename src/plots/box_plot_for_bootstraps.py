import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUT_DIR

# File paths (Update these paths accordingly)
FILE_WITH_FEATURES = os.path.join(
    OUTPUT_DIR,
    'geo_bootstrap_results',
    "bootstrap_geo_split_results_with_user_and_osm_element_features_spw_4.5.parquet"
)

FILE_WITHOUT_FEATURES = os.path.join(
    OUTPUT_DIR,
    'geo_bootstrap_results',
    "bootstrap_geo_split_results_no_user_osm_element_features.parquet"
)

# Metrics to plot
METRICS = ["accuracy", "precision", "recall", "f1_score"]

# Load parquet files
df_with_features = pd.read_parquet(FILE_WITH_FEATURES)
df_without_features = pd.read_parquet(FILE_WITHOUT_FEATURES)

# Convert comma decimal separator to dot for numeric conversion
for metric in METRICS:
    df_with_features[metric] = df_with_features[metric].astype(str).str.replace(',', '.').astype(float)
    df_without_features[metric] = df_without_features[metric].astype(str).str.replace(',', '.').astype(float)

# Add a column to indicate the model type
df_with_features["Model"] = "With User & OSM Features"
df_without_features["Model"] = "Without User & OSM Features"

# Concatenate both datasets
df_combined = pd.concat([df_with_features, df_without_features], ignore_index=True)

# Melt dataframe to long format for Seaborn
df_melted = df_combined.melt(id_vars=["Model"], value_vars=METRICS, var_name="Metric", value_name="Score")

# Update font size for all plot elements
plt.figure(figsize=(16, 10))  # Larger figure size to fill space
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.4)  # Increase font scale for better readability

# Plot with updated font and layout
sns.boxplot(
    data=df_melted,
    x="Metric",
    y="Score",
    hue="Model",
    width=0.4,
    palette=["#81C784", "#E57373"],  # Balanced subtle green and red
    dodge=True
)

# Update labels and title with larger fonts
plt.xlabel("Metric", fontsize=16)
plt.ylabel("Score", fontsize=16)
plt.title("Comparison of Bootstrap Performance Metrics (Geo-Split Models)", fontsize=18)
plt.legend(title="Model", loc="lower right", fontsize=14)  # Bigger legend font
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Adjust layout to use space more efficiently
plt.tight_layout()

# Show plot
plt.show()
