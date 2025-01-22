import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def load_vandalism_files(folder_path: str):
    """
    Loads all CSV files ending with '_prediction_output.csv', each containing vandalism entries.
    Extracts a 'month_year' string (e.g., '2022-02') from the filename.
    Returns a single DataFrame with columns:
      [changeset_id, date_created, osm_id, osm_version, centroid_x, centroid_y, y_pred, y_prob, month_year]
    """
    all_files = glob.glob(os.path.join(folder_path, "*_prediction_output.csv"))
    dfs = []

    for file_path in all_files:
        base_name = os.path.basename(file_path)
        # Example: '2022-02_contributions_prediction_output.csv' -> '2022-02'
        month_year = base_name.split('_')[0]  # 'YYYY-MM'

        df = pd.read_csv(file_path)
        # If your CSV has non-vandal entries, filter them out: df = df[df['y_pred'] == 1]

        df['month_year'] = month_year
        if 'date_created' in df.columns:
            df['date_created'] = pd.to_datetime(df['date_created'], errors='coerce')
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def prepare_monthly_counts(df):
    """
    Groups by 'month_year' and counts vandalism entries,
    then parses year/month as integers for sorting.
    Returns a DataFrame with columns:
        ['month_year', 'vandalism_count', 'year', 'month', 'month_year_label']
    """
    monthly_counts = df.groupby('month_year')['changeset_id'].count().reset_index()
    monthly_counts.rename(columns={'changeset_id': 'vandalism_count'}, inplace=True)

    # Extract numeric year/month for sorting
    monthly_counts['year'] = monthly_counts['month_year'].apply(lambda x: int(x.split('-')[0]))
    monthly_counts['month'] = monthly_counts['month_year'].apply(lambda x: int(x.split('-')[1]))
    monthly_counts = monthly_counts.sort_values(by=['year', 'month'])

    # We can create a label for plotting
    monthly_counts['month_year_label'] = monthly_counts['month_year']
    return monthly_counts


def plot_bar_line_overall(df, output_dir):
    """
    Plots an overall bar and line chart for all data across multiple years.
    Less cluttered bar labels: we reduce font size or skip labeling if it's too many bars.
    """
    monthly_counts = prepare_monthly_counts(df)

    # Overall Bar Plot
    plt.figure(figsize=(10, 5))
    bars = plt.bar(monthly_counts['month_year_label'],
                   monthly_counts['vandalism_count'], color='skyblue')
    plt.xlabel('Month (YYYY-MM)')
    plt.ylabel('Number of Vandalism Contributions')
    plt.title('Vandalism Contributions by Month (Bar) - Overall')

    # Conditionally place labels if # of bars < threshold
    if len(bars) < 40:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2,
                     height,
                     str(height),
                     ha='center', va='bottom', fontsize=8, rotation=45)

    plt.xticks(rotation=45)
    plt.tight_layout()
    bar_output = os.path.join(output_dir, "vandalism_bar_by_month_overall.png")
    plt.savefig(bar_output, dpi=150)
    plt.close()

    # Overall Line Plot
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_counts['month_year_label'],
             monthly_counts['vandalism_count'],
             marker='o', linestyle='-', color='salmon')
    plt.xlabel('Month (YYYY-MM)')
    plt.ylabel('Number of Vandalism Contributions')
    plt.title('Vandalism Contributions by Month (Line) - Overall')
    plt.xticks(rotation=45)
    plt.tight_layout()
    line_output = os.path.join(output_dir, "vandalism_line_by_month_overall.png")
    plt.savefig(line_output, dpi=150)
    plt.close()


def plot_bar_line_by_year(df, output_dir):
    """
    For each unique year, create separate bar and line plots of monthly vandalism counts.
    """
    monthly_counts = prepare_monthly_counts(df)

    unique_years = monthly_counts['year'].unique()

    for yr in unique_years:
        yr_df = monthly_counts[monthly_counts['year'] == yr].copy()

        # Bar Plot for this year
        plt.figure(figsize=(8, 4))
        bars = plt.bar(yr_df['month_year_label'], yr_df['vandalism_count'], color='skyblue')
        plt.xlabel(f'Month of {yr}')
        plt.ylabel('Vandalism Count')
        plt.title(f'Vandalism by Month (Bar) - {yr}')

        # Label bars if not too many
        if len(bars) < 20:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2,
                         height,
                         str(height),
                         ha='center', va='bottom', fontsize=8, rotation=45)

        plt.xticks(rotation=45)
        plt.tight_layout()
        bar_file = os.path.join(output_dir, f"vandalism_bar_{yr}.png")
        plt.savefig(bar_file, dpi=150)
        plt.close()

        # Line Plot for this year
        plt.figure(figsize=(8, 4))
        plt.plot(yr_df['month_year_label'],
                 yr_df['vandalism_count'],
                 marker='o', linestyle='-', color='salmon')
        plt.xlabel(f'Month of {yr}')
        plt.ylabel('Vandalism Count')
        plt.title(f'Vandalism by Month (Line) - {yr}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        line_file = os.path.join(output_dir, f"vandalism_line_{yr}.png")
        plt.savefig(line_file, dpi=150)
        plt.close()


def plot_interactive_heatmap(df, output_dir):
    """
    Creates an interactive Plotly scattergeo with a time slider (manual only).
    - Smaller dot sizes
    - Purple color
    - Manual slider (no autoplay)
    - Smoother transitions

    Each frame = one 'month_year'.
    """
    # Filter valid lat/lon
    df = df.dropna(subset=['centroid_x', 'centroid_y'])

    # Prepare sorting
    df['year'] = df['month_year'].apply(lambda x: int(x.split('-')[0]))
    df['month'] = df['month_year'].apply(lambda x: int(x.split('-')[1]))
    df = df.sort_values(by=['year', 'month'])

    # Create figure using scatter_geo with animation_frame
    fig = px.scatter_geo(
        df,
        lon='centroid_x',
        lat='centroid_y',
        hover_name='changeset_id',
        color_discrete_sequence=['purple'],  # all points in purple
        animation_frame='month_year',
        projection='natural earth'
    )

    fig.update_traces(
        marker=dict(size=3)  # Smaller dot size
    )

    # Manual slider config:
    # - Remove autoplay buttons by clearing updatemenus
    # - Make transitions smoother
    fig.layout.updatemenus = []  # no play/pause buttons
    fig.update_layout(
        title='Vandalism Heatmap Over Time (Manual Slider)',
        geo=dict(
            showland=True,
            landcolor='rgb(217, 217, 217)',
            showcountries=True,
            countrycolor='rgb(204, 204, 204)'
        ),
        # Smoother transitions
        transition={'duration': 600, 'easing': 'cubic-in-out'},
        sliders=[{
            'currentvalue': {'prefix': 'Month-Year: '},
            'transition': {'duration': 600, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 10},
            'len': 0.9,
            'x': 0.1,
            'steps': []
        }]
    )

    # Manually rebuild slider steps from the frames
    # Plotly auto-creates frames from animation_frame; we can override the slider "steps"
    # for a smoother manual experience:
    steps = []
    all_frames = fig.frames
    for fr in all_frames:
        steps.append({
            "args": [[fr.name],
                     {"frame": {"duration": 600, "redraw": False},
                      "mode": "immediate",
                      "transition": {"duration": 600, "easing": "cubic-in-out"}}],
            "label": fr.name,
            "method": "animate"
        })

    # Update the single slider's steps
    if len(fig.layout.sliders) > 0:
        fig.layout.sliders[0].steps = steps

    # Save to HTML
    html_file = os.path.join(output_dir, "vandalism_heatmap_timeline.html")
    fig.write_html(html_file)
    print(f"Saved interactive heatmap with time slider to {html_file}")


def main():
    folder_path = r"D:\PycharmProjects\vandalism_detection_osm\data\contribution_data\output\predictions_output\pcuf_full_dataset_detailed_2022_to_2024_monthly"
    output_dir = r"D:\PycharmProjects\vandalism_detection_osm\data\contribution_data\output\plots"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = load_vandalism_files(folder_path)
    if df.empty:
        print("No data found or no CSV files ending with '_prediction_output.csv' in the folder.")
        return

    print(f"Loaded {len(df)} vandalism entries in total.")

    # 1) Plot overall bar & line
    plot_bar_line_overall(df, output_dir)

    # 2) Plot bar & line per year
    plot_bar_line_by_year(df, output_dir)

    # 3) Plot interactive heatmap with manual slider
    plot_interactive_heatmap(df, output_dir)


if __name__ == "__main__":
    main()
