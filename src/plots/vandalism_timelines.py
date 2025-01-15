import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

from config import OUTPUT_DIR, PREDICTIONS_INPUT_DATA_DIR, logger

OUTPUT_FOLDER_NAME_SUFFIX = '2022_jan_to_july_2025-01-13_20-52-59_pcuf__balanced__spw_4.5__real_vandal_0.2__threshold_0.5'

import os
import glob
import gc
import pandas as pd


def build_contribution_center_map(raw_input_dir):
    """
    Build a map from contribution_key -> (lon, lat) based on bounding box columns.
    Memory-Efficient Version:
      - Reads only required columns from each file.
      - Drops columns as soon as they're no longer needed.
      - Calls garbage collection after processing each file.

    Required columns in each file:
        valid_from, osm_id, osm_version,
        xmin, ymin, xmax, ymax
    """
    # Columns we must read
    required_cols = ["valid_from", "osm_id", "osm_version", "xmin", "ymin", "xmax", "ymax"]
    center_map = {}

    # Match all CSV/Parquet files in the directory
    pattern = os.path.join(raw_input_dir, "*.*")
    all_files = glob.glob(pattern)

    for fpath in all_files:
        # Determine file extension
        if fpath.endswith(".parquet"):
            # Read only the needed columns
            if not os.path.exists(fpath):
                continue
            df_raw = pd.read_parquet(fpath, columns=required_cols)

        elif fpath.endswith(".csv"):
            # Use usecols to limit columns read
            if not os.path.exists(fpath):
                continue
            df_raw = pd.read_csv(fpath, usecols=required_cols)

        else:
            # Skip unsupported file formats
            continue

        # In case the file doesn't contain all required columns, skip it
        if not set(required_cols).issubset(df_raw.columns):
            del df_raw
            gc.collect()
            continue

        # Create 'contribution_key' on the fly
        # Then drop columns used for creating it
        df_raw["contribution_key"] = (
                df_raw["valid_from"].astype(str) + "__" +
                df_raw["osm_id"].astype(str) + "__" +
                df_raw["osm_version"].astype(str)
        )
        df_raw.drop(["valid_from", "osm_id", "osm_version"], axis=1, inplace=True)

        # Compute centroid from bounding box, then drop bounding box columns
        df_raw["center_lon"] = (df_raw["xmin"] + df_raw["xmax"]) / 2
        df_raw["center_lat"] = (df_raw["ymin"] + df_raw["ymax"]) / 2
        df_raw.drop(["xmin", "xmax", "ymin", "ymax"], axis=1, inplace=True)

        # Add to our dictionary: {contribution_key: (center_lon, center_lat)}
        # Use itertuples (memory-friendly) to iterate
        for row in df_raw[["contribution_key", "center_lon", "center_lat"]].itertuples(index=False):
            # row is a namedtuple => (contribution_key, center_lon, center_lat)
            center_map[row.contribution_key] = (row.center_lon, row.center_lat)

        # Remove intermediate DataFrame and force garbage collection
        del df_raw
        gc.collect()

    return center_map


def plot_global_vandalism_heatmap(vandalism_df, center_map, method="hexbin"):
    """
    Plot a global heatmap of vandalism. We have a DataFrame vandalism_df with 'contribution_key'
    that indicates vandal entries, and a dictionary center_map {contribution_key -> (lon, lat)}.

    method: "hexbin" or "scatter"
    """
    # Extract (lon, lat) for each vandalism entry
    lons = []
    lats = []

    for ckey in vandalism_df['contribution_key']:
        if ckey in center_map:
            lon, lat = center_map[ckey]
            lons.append(lon)
            lats.append(lat)
    if not lons:
        logger.warn("No matching contribution keys for vandalism. Cannot plot heatmap.")
        return

    # Convert to numpy arrays
    lons = np.array(lons)
    lats = np.array(lats)

    # Setup Cartopy figure
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())  # PlateCarree = lat/lon map
    ax.set_global()  # Show entire world
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Depending on method, choose hexbin or scatter
    if method == "hexbin":
        # Create hexbin
        plt.hexbin(lons, lats, gridsize=100, transform=ccrs.PlateCarree(),
                   cmap='Reds', mincnt=1)
        plt.colorbar(label='Number of Vandalism Points')
    else:
        # Simple scatter plot with alpha for density
        plt.scatter(lons, lats, s=4, c='red', alpha=0.3, transform=ccrs.PlateCarree())

    plt.title("Global Vandalism Heatmap")
    plt.show()


def plot_world_heatmap_of_vandalism(vandalism_df):
    """
    Demonstrates how to:
    1) Build a center_map from raw bounding box data.
    2) Load vandalism predictions (across months).
    3) Plot a global heatmap (hexbin) of vandalism.
    """
    # 1) Build center_map from raw bounding box input files
    raw_input_dir = PREDICTIONS_INPUT_DATA_DIR  # adapt
    center_map = build_contribution_center_map(raw_input_dir)

    if vandalism_df.empty:
        logger.warn("No vandalism data found.")
        return

    logger.info(f"Loaded {len(vandalism_df)} vandalism rows for heatmap plotting...")

    # 3) Plot with hexbin
    plot_global_vandalism_heatmap(vandalism_df, center_map, method="hexbin")

    # 4) If you want a scatter plot instead:
    # plot_global_vandalism_heatmap(vandalism_df, center_map, method="scatter")
    logger.info("Heat map plotted")

def parse_date_from_key(contribution_key: str):
    """Extract the date portion from the contribution_key."""
    parts = contribution_key.split('__')
    if not parts:
        return None
    date_str = parts[0]
    try:
        date_val = pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S')
        return date_val
    except ValueError:
        return None


def load_all_vandalism_files(folder_path: str):
    """Load all CSV files ending with '_vandalism_predictions.csv' and parse the contribution dates."""
    pattern = os.path.join(folder_path, '*_vandalism_predictions.csv')
    all_files = glob.glob(pattern)
    dfs = []
    for file_path in all_files:
        df = pd.read_csv(file_path)
        df['contribution_datetime'] = df['contribution_key'].apply(parse_date_from_key)
        df = df.dropna(subset=['contribution_datetime'])
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def plot_bar_with_labels(x, y, title, xlabel, ylabel, rotation=0):
    """Helper function to plot bar graphs with labels on top."""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, y, color='skyblue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=rotation)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 str(int(bar.get_height())), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_line_graph(x, y, title, xlabel, ylabel, rotation=0):
    """Helper function to plot line graphs."""
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', color='salmon', linestyle='-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()


def plot_vandalism_by_month(df: pd.DataFrame):
    """Plot vandalism contributions by month (bar and line)."""
    df['year_month'] = df['contribution_datetime'].dt.to_period('M').astype(str)
    monthly_counts = df.groupby('year_month')['contribution_key'].count().reset_index()
    monthly_counts.rename(columns={'contribution_key': 'vandalism_count'}, inplace=True)
    # Bar plot
    plot_bar_with_labels(
        monthly_counts['year_month'],
        monthly_counts['vandalism_count'],
        "Vandalism Contributions by Month (Bar)",
        "Month (YYYY-MM)",
        "Number of Vandalism Contributions",
        rotation=45
    )
    # Line plot
    plot_line_graph(
        monthly_counts['year_month'],
        monthly_counts['vandalism_count'],
        "Vandalism Contributions by Month (Line)",
        "Month (YYYY-MM)",
        "Number of Vandalism Contributions",
        rotation=45
    )


def plot_vandalism_by_week(df: pd.DataFrame):
    """Plot vandalism contributions by week (bar and line)."""
    df['year_week'] = df['contribution_datetime'].dt.isocalendar().week
    df['year'] = df['contribution_datetime'].dt.year
    df['year_week_str'] = df.apply(lambda row: f"{row['year']}-W{int(row['year_week']):02d}", axis=1)
    weekly_counts = df.groupby('year_week_str')['contribution_key'].count().reset_index()
    weekly_counts.rename(columns={'contribution_key': 'vandalism_count'}, inplace=True)
    # Bar plot
    plot_bar_with_labels(
        weekly_counts['year_week_str'],
        weekly_counts['vandalism_count'],
        "Vandalism Contributions by Week (Bar)",
        "Week (Year-Week)",
        "Number of Vandalism Contributions",
        rotation=90
    )
    # Line plot
    plot_line_graph(
        weekly_counts['year_week_str'],
        weekly_counts['vandalism_count'],
        "Vandalism Contributions by Week (Line)",
        "Week (Year-Week)",
        "Number of Vandalism Contributions",
        rotation=90
    )


def main():
    folder_path = os.path.join(
        OUTPUT_DIR,
        "predictions_output", OUTPUT_FOLDER_NAME_SUFFIX
    )
    df = load_all_vandalism_files(folder_path)
    if df.empty:
        logger.warn("No vandalism files found or no data after parsing. Exiting.")
        return

    logger.info(f"Loaded {len(df)} vandalism rows from {folder_path}.")
    # plot_vandalism_by_month(df)
    # plot_vandalism_by_week(df)
    plot_world_heatmap_of_vandalism(df)


if __name__ == "__main__":
    main()
