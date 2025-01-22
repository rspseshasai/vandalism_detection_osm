import os
import glob
import logging
import pandas as pd
from datetime import datetime
from calendar import monthrange
import plotly.express as px

from config import logger

# ------------------------------------------------------------------------
# 1. CONFIGURATION AND LOGGING
# ------------------------------------------------------------------------
folder_path = r"D:\PycharmProjects\vandalism_detection_osm\data\contribution_data\output\predictions_output\pcuf_full_dataset_detailed_2022_to_2024_monthly"
output_dir = r"D:\PycharmProjects\vandalism_detection_osm\data\contribution_data\output\plots"

os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------------------
# 2. READ AND COMBINE ALL CSV FILES
# ------------------------------------------------------------------------
logger.info(f"Looking for CSV files in: {folder_path}")
csv_files = glob.glob(os.path.join(folder_path, "*_prediction_output.csv"))

if not csv_files:
    logger.warning(f"No CSV files found in: {folder_path}")
    raise SystemExit("No input files. Exiting.")

logger.info(f"Found {len(csv_files)} CSV file(s). Reading them all...")

dfs = []
for file in csv_files:
    temp_df = pd.read_csv(file)
    dfs.append(temp_df)

df = pd.concat(dfs, ignore_index=True)
logger.info(f"Combined dataframe shape: {df.shape}")

# ------------------------------------------------------------------------
# 3. PREPARE THE DATA
# ------------------------------------------------------------------------
logger.info("Converting 'date_created' to datetime...")
df["date_created"] = pd.to_datetime(df["date_created"], errors='coerce')

# (Optional) Filter only vandalism predictions
# df = df[df["y_pred"] == 1]

logger.info("Creating 'year_month' (YYYY-MM).")
df["year_month"] = df["date_created"].dt.to_period("M").astype(str)
df = df.sort_values("date_created").reset_index(drop=True)

logger.info(f"After sorting, final dataframe shape: {df.shape}")

# ------------------------------------------------------------------------
# 4. BUILD MONTHLY-ONLY DATAFRAME
# ------------------------------------------------------------------------
logger.info("Preparing monthly-only (each month) dataframe...")
df_monthly_only = df.copy()

# ------------------------------------------------------------------------
# 5. REDUCE DATA SIZE TO SPEED UP ANIMATION AND FIX MONTH ORDERING
# ------------------------------------------------------------------------
# 5a. Limit the number of frames (months)
unique_months = sorted(df_monthly_only["year_month"].unique(), key=lambda x: pd.to_datetime(x))
max_frames = 24  # keep at most 24 frames
if len(unique_months) > max_frames:
    logger.info(f"Too many unique months ({len(unique_months)}). Keeping only {max_frames} frames.")
    step = len(unique_months) // max_frames
    months_to_keep = unique_months[::step]
    df_monthly_only = df_monthly_only[df_monthly_only["year_month"].isin(months_to_keep)]

# Ensure the 'year_month' column is sorted in chronological order
df_monthly_only["year_month"] = pd.Categorical(
    df_monthly_only["year_month"],
    categories=unique_months,  # Ensure ordering matches the sorted unique_months
    ordered=True
)

# 5b. Sample the data if there are too many rows
max_points = 100_0000
if df_monthly_only.shape[0] > max_points:
    logger.info(f"Data too large ({df_monthly_only.shape[0]} rows). Sampling down to {max_points} rows.")
    df_monthly_only = df_monthly_only.sample(n=max_points, random_state=42).reset_index(drop=True)

logger.info(f"Final monthly-only dataframe shape after reductions: {df_monthly_only.shape}")

# ------------------------------------------------------------------------
# 6. CREATE A HELPER FUNCTION TO MAKE THE DENSITY MAP
# ------------------------------------------------------------------------
def create_density_map(dataframe, frame_col, title):
    """
    Create a Plotly density map with an animated slider
    using the column 'frame_col' to define each animation frame.
    """
    logger.info(f"Building figure for '{title}' using frame column: {frame_col}")

    fig = px.density_map(
        dataframe,
        lat="centroid_y",
        lon="centroid_x",
        radius=5,  # controls how "spread out" the heat is
        zoom=1,
        center={"lat": 20, "lon": 0},  # roughly center on the globe
        animation_frame=frame_col,
        hover_data=["osm_id", "date_created"],
        color_continuous_scale=[
            (0, "rgba(128,0,128,0)"),
            (1, "rgba(128,0,128,1)")
        ],  # Transparent purple => solid purple
    )

    fig.update_layout(
        title=title,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_showscale=False,  # Hide the color scale if not needed
        updatemenus=[  # Add play/pause buttons for better control
            {
                "buttons": [
                    {"args": [None, {"frame": {"duration": 1000, "redraw": True}, "mode": "immediate", "transition": {"duration": 300}}],
                     "label": "Play",
                     "method": "animate"},
                    {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                     "label": "Pause",
                     "method": "animate"},
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }
        ]
    )

    return fig

# ------------------------------------------------------------------------
# 7. BUILD AND SAVE THE FIGURE (MONTHLY ONLY)
# ------------------------------------------------------------------------
logger.info("Creating monthly-only heatmap figure...")
fig_monthly_only = create_density_map(
    df_monthly_only,
    frame_col="year_month",
    title="(1) Monthly-Only Vandalism Heatmap (Use Slider to Pick Month)"
)

# Update animation speed
fig_monthly_only.update_layout(
    sliders=[{
        "steps": [
            {"label": m, "method": "animate", "args": [[m], {"frame": {"duration": 100, "redraw": True}, "mode": "immediate"}]}
            for m in sorted(df_monthly_only["year_month"].unique(), key=lambda x: pd.to_datetime(x))
        ],
        "currentvalue": {"prefix": "Month: "}
    }],
    updatemenus=[{
        "buttons": [
            {"args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}], "label": "Play", "method": "animate"},
            {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}], "label": "Pause", "method": "animate"}
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }]
)

output_file_monthly = os.path.join(output_dir, "vandalism_heatmap_monthly_only.html")
logger.info(f"Saving monthly-only figure to: {output_file_monthly}")

# Use include_plotlyjs='cdn' to reduce HTML size
fig_monthly_only.write_html(output_file_monthly, include_plotlyjs='cdn')

logger.info("All done! Open the HTML file in a web browser to see the animation.")

