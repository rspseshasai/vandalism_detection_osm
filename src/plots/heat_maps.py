import os
import glob
import folium
import pandas as pd
from folium.plugins import HeatMapWithTime

# Paths (modify as needed)
folder_path = r"D:\PycharmProjects\vandalism_detection_osm\data\contribution_data\output\predictions_output\pcuf_full_dataset_detailed_2022_to_2024_monthly"
output_dir = r"D:\PycharmProjects\vandalism_detection_osm\data\contribution_data\output\plots"
output_map_name = "vandalism_heatmap.html"

# Grab all CSV files ending with '_prediction_output.csv'
csv_files = glob.glob(os.path.join(folder_path, "*_prediction_output.csv"))

# Sort the files so that they're in chronological order (assuming filenames start with YYYY-MM)
csv_files.sort()

# Lists to hold time labels and data for HeatMapWithTime
time_index = []  # e.g., ["2022-01", "2022-02", ...]
all_month_data = []  # list of [ [lat, lon], [lat, lon], ... ] for each month

for file_path in csv_files:
    # Extract a label from the filename, e.g. "2022-01" from "2022-01_prediction_output.csv"
    filename = os.path.basename(file_path)
    month_label = filename.split("_")[0]  # or filename[:7] if your files are exactly "YYYY-MM_..."

    # Read CSV
    df = pd.read_csv(file_path)

    # (Optional) Filter for only vandalism predictions if needed:
    # df = df[df['y_pred'] == 1]

    # Convert columns to numeric if needed (just in case)
    df['centroid_x'] = pd.to_numeric(df['centroid_x'], errors='coerce')
    df['centroid_y'] = pd.to_numeric(df['centroid_y'], errors='coerce')

    # Drop rows with invalid coords
    df.dropna(subset=['centroid_x', 'centroid_y'], inplace=True)

    # OPTIONAL (Performance): Subsample if the data is too large
    # e.g. Keep only 1% or a fixed number of points. Uncomment if needed.
    # df = df.sample(frac=0.01, random_state=42)
    # or
    # df = df.sample(n=50000, random_state=42)  # sample 50k points if extremely large

    # Prepare data for HeatMapWithTime: a list of [lat, lon]
    lat_lon_list = df[['centroid_y', 'centroid_x']].values.tolist()

    # Collect in lists
    all_month_data.append(lat_lon_list)
    time_index.append(month_label)

# Create a Folium map centered on [0, 0] with an initial zoom.
# You can adjust zoom_start as you like. 2 or 3 is usually good for a world view.
m = folium.Map(location=[0, 0], zoom_start=2)

# Create HeatMapWithTime layer
# - auto_play=False ensures no automatic playback
# - radius=1 or 2 keeps the points small
# - We set a 'purple' gradient by specifying the same color at different points in the scale.
#   This effectively keeps the heatmap in purple shades.
HeatMapWithTime(
    data=all_month_data,
    index=time_index,
    auto_play=False,
    radius=1,
    max_opacity=0.8,
    gradient={0.0: 'purple', 1.0: 'purple'},  # all purple
).add_to(m)

# Save to HTML
output_path = os.path.join(output_dir, output_map_name)
m.save(output_path)

print(f"Heatmap saved to: {output_path}")
