import glob
import os
import sys
from datetime import datetime

import folium
import matplotlib.pyplot as plt
import pandas as pd
from folium.plugins import HeatMapWithTime

from config import logger
OUTPUT_DIR_NAME= 'nuof'

# Paths
folder_path = r"D:\PycharmProjects\vandalism_detection_osm\data\contribution_data\output\predictions_output\pcuf_full_dataset_detailed_2022_to_2024_monthly"
plots_dir = os.path.join(
    r"D:\PycharmProjects\vandalism_detection_osm\data\contribution_data\output\plots",
    f"{OUTPUT_DIR_NAME}__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
)

output_path_bar_graph = os.path.join(plots_dir, "vandalism_predictions_2022_to_2024_bar_graph.png")
output_path_heatmap = os.path.join(plots_dir, "vandalism_predictions_2022_to_2024_heat_map.html")
output_path_line_graph = os.path.join(plots_dir, "vandalism_predictions_2022_to_2024_line_graph.png")


# ----------------------------------------------------------------------------
# BAR & LINE PLOTS CODE (MERGED FROM plots_for_predictions.py)
# ----------------------------------------------------------------------------

def aggregate_vandalism_counts(folder_path):
    """
    Aggregates vandalism counts for each file that ends with '_prediction_output.csv'.
    Logic remains unchanged.
    """
    vandalism_counts = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith("_prediction_output.csv"):
            month = file_name.split("_")[0]  # e.g. "2022-01"
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            # Filter only vandalism
            df_vandal = df[df['y_pred'] == 1]
            vandalism_count = df_vandal.shape[0]
            vandalism_counts[month] = vandalism_count

    # Sort by month
    vandalism_counts = dict(sorted(vandalism_counts.items()))
    return vandalism_counts


def plot_bar_graph(months, values, adjusted_colors, threshold):
    """
    Creates the bar graph with a broken x-axis, as in your code.
    """
    logger.info("Creating bar graph with broken axis...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8),
                                   gridspec_kw={'width_ratios': [3, 1]})

    # Left subplot: Normal range
    ax1.barh(months, [val / 1000 for val in values], color=adjusted_colors, height=0.6)
    ax1.set_xlim(40, 230)  # Scale in thousands
    ax1.set_ylim(-0.5, len(months) - 0.5)
    ax1.set_xticks(range(40, 231, 25))
    ax1.set_xticklabels([f"{x}" for x in range(40, 231, 25)])

    # Right subplot: Outlier range
    ax2.barh(months, [val / 1000 for val in values], color=adjusted_colors, height=0.6)
    ax2.set_xlim(230, max(values) / 1000)  # Scale in thousands
    ax2.set_ylim(-0.5, len(months) - 0.5)

    # Reduced ticks in the outlier range
    outlier_ticks = [230, int(max(values) / 1000)]
    ax2.set_xticks(outlier_ticks)
    ax2.set_xticklabels([f"{x}" for x in outlier_ticks])

    # Hide spines
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.tick_params(labelright=False)
    ax2.tick_params(labelleft=False)

    # Diagonal lines for the broken axis
    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    # Add value labels
    for i, val in enumerate(values):
        if val <= threshold:
            ax1.text(val / 1000 + 5, i, f"{val / 1000:.1f}k", ha="left", va="center", fontsize=10)
        else:
            ax2.text(val / 1000 + 5, i, f"{val / 1000:.1f}k", ha="left", va="center", fontsize=10)

    # Single unified x-axis label
    fig.text(0.5, 0.04, 'Vandalism Entries (in Thousands)', ha='center', fontsize=12)
    fig.text(0.01, 0.5, 'Months', va='center', rotation='vertical', fontsize=12)
    plt.subplots_adjust(left=0.25, right=0.85)

    # Title
    plt.suptitle('Monthly Vandalism Predictions in OSM (2022-2024)', fontsize=16)
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    # Save and show plot
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(output_path_bar_graph)
    plt.show()
    logger.info(f"Bar graph saved to {output_path_bar_graph}")


def plot_line_graph(months, values, output_path):
    """
    Plots line graph with Y-axis break, as in your original code.
    """
    logger.info("Creating line graph with broken axis...")
    values_in_thousands = [val / 1000 for val in values]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                   gridspec_kw={'height_ratios': [1, 2]})

    # Top subplot: focus on outlier range
    ax1.plot(range(len(values_in_thousands)), values_in_thousands, marker='o',
             linestyle='-', color='blue', label='Vandalism Predictions')
    ax1.scatter(range(len(values_in_thousands)), values_in_thousands,
                c=['red' if val == max(values_in_thousands) else 'blue' for val in values_in_thousands],
                s=60, edgecolor='black', label='Outlier')
    ax1.set_ylim(1315.5, 1320)  # Outlier range
    ax1.set_xticks([])
    ax1.set_yticks([1250, 1305, 1310, 1315, 1320])
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Bottom subplot: normal range
    ax2.plot(range(len(values_in_thousands)), values_in_thousands, marker='o',
             linestyle='-', color='blue', label='Vandalism Predictions')
    ax2.set_ylim(40, 230)
    ax2.axhline(y=100, color='red', linestyle='--', linewidth=0.5, label='100k Threshold')
    ax2.set_xticks(range(0, len(months), 2))
    ax2.set_xticklabels(months[::2], rotation=45, fontsize=10)
    ax2.set_yticks(range(40, 231, 10))
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlabel("Months", fontsize=12)

    # Unified Y-axis label
    fig.text(0.02, 0.5, 'Vandalism Entries (in Thousands)', va='center',
             rotation='vertical', fontsize=12)

    # Title and layout
    plt.suptitle('Monthly Vandalism Predictions in OSM (2022-2024)', fontsize=16)
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    fig.subplots_adjust(left=0.1)

    # Broken axis markers
    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    plt.savefig(output_path)
    plt.show()
    logger.info(f"Line graph saved to {output_path}")


# ----------------------------------------------------------------------------
# HEATMAP CODE (MERGED FROM heatmaps.py)
# ----------------------------------------------------------------------------

def create_heatmap_folium(folder_path, output_dir, output_map_name):
    """
    Creates a monthly-based heatmap with manual slider using Folium + HeatMapWithTime.
    Logic from your original `heatmaps.py`.
    """
    logger.info("Creating monthly-based folium heatmap...")

    csv_files = glob.glob(os.path.join(folder_path, "*_prediction_output.csv"))
    csv_files.sort()

    time_index = []
    all_month_data = []

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        month_label = filename.split("_")[0]  # e.g. "2022-01"
        df = pd.read_csv(file_path)

        # If you only want vandalism:
        # df = df[df['y_pred'] == 1]

        df['centroid_x'] = pd.to_numeric(df['centroid_x'], errors='coerce')
        df['centroid_y'] = pd.to_numeric(df['centroid_y'], errors='coerce')
        df.dropna(subset=['centroid_x', 'centroid_y'], inplace=True)

        # Potential sub-sampling to avoid huge HTML
        # df = df.sample(frac=0.01, random_state=42) if len(df) > 100000 else df

        lat_lon_list = df[['centroid_y', 'centroid_x']].values.tolist()
        time_index.append(month_label)
        all_month_data.append(lat_lon_list)

    m = folium.Map(location=[0, 0], zoom_start=2, tiles="OpenStreetMap")

    custom_gradient = {
        0.2: 'lavenderblush',
        0.4: 'plum',
        0.6: 'mediumorchid',
        0.8: 'darkorchid',
        1.0: 'indigo'
    }

    hm = HeatMapWithTime(
        data=all_month_data,
        index=time_index,
        auto_play=False,  # manual slider
        radius=2,
        max_opacity=0.8,
        gradient=custom_gradient
    )
    hm.add_to(m)

    os.makedirs(output_dir, exist_ok=True)
    m.save(output_path_heatmap)
    logger.info(f"Heatmap saved to: {output_path_heatmap}")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting script for bar/line plots and heatmaps.")
    os.makedirs(plots_dir, exist_ok=True)

    # 1) Aggregate data for bar/line plots
    vandalism_counts = aggregate_vandalism_counts(folder_path)
    if not vandalism_counts:
        logger.info("No vandalism CSV files found or no data in them. Exiting.")
        sys.exit(0)

    # Prepare months & values
    months = list(vandalism_counts.keys())
    values = list(vandalism_counts.values())
    threshold = 230000

    # Adjust bar colors
    non_outlier_values = [val for val in values if val <= threshold]
    if len(non_outlier_values) > 0:
        non_outlier_max = max(non_outlier_values)
    else:
        non_outlier_max = 1

    adjusted_colors = [
        (0.5, 0, 0.5, val / non_outlier_max) if val <= threshold else (0.3, 0, 0.3, 1.0)
        for val in values
    ]

    # 2) Plot bar graph with a broken axis
    plot_bar_graph(months, values, adjusted_colors, threshold)

    # 3) Plot line graph
    plot_line_graph(months, values, output_path_line_graph)

    # 4) Create Folium heatmap with monthly slider
    create_heatmap_folium(folder_path, plots_dir, output_path_heatmap)

    logger.info("All plots and heatmap creation completed successfully.")
