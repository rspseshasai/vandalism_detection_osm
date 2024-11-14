import pandas as pd
import os
from datetime import datetime

# Define the path to the input Parquet file and the output directory
input_parquet_file = "../../data/changeset_data/output/merged_osm_contributions.parquet"
output_directory = "../../data/changeset_data/output/quarterly_data"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Load the Parquet file into a DataFrame
print("Loading the Parquet file...")
df = pd.read_parquet(input_parquet_file)

# Convert 'created_at' to datetime if it's not already
if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
    df['created_at'] = pd.to_datetime(df['created_at'])

# Extract year and quarter from 'created_at' column
df['year'] = df['created_at'].dt.year
df['quarter'] = df['created_at'].dt.quarter

# Group the DataFrame by year and quarter and save each group to a separate Parquet file
for (year, quarter), group_df in df.groupby(['year', 'quarter']):
    # Define the filename for the quarter
    output_file = os.path.join(output_directory, f"osm_contributions_{year}_Q{quarter}.parquet")

    # Save the group to a Parquet file
    print(f"Saving data for {year} Q{quarter} to {output_file}...")
    group_df.to_parquet(output_file, index=False)

print("Completed splitting the file by quarters.")
