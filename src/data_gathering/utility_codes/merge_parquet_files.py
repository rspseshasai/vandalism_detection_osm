import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Path to the directory containing the .parquet files
input_directory = '../../../data/contribution_data/raw/temp/'

# List all .parquet files in the directory
parquet_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.parquet')]

# Initialize an empty list to store DataFrames
dataframes = []

# Read and concatenate all parquet files into a single DataFrame
for parquet_file in parquet_files:
    df = pd.read_parquet(parquet_file)
    dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
merged_df = pd.concat(dataframes, ignore_index=True)

# Shuffle the merged DataFrame
shuffled_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)


# Function to convert a DataFrame to a PyArrow Table using the specified schema
def convert_df_to_arrow_table(df):
    # Uncomment the following line to include a schema if required
    # table = pa.Table.from_pandas(df, schema=get_osm_contribution_schema(), preserve_index=False)
    table = pa.Table.from_pandas(df, preserve_index=False)
    return table


# Convert the shuffled DataFrame to PyArrow Table
arrow_table = convert_df_to_arrow_table(shuffled_df)

# Save the Arrow Table as a single Parquet file
output_file = '../../../data/contribution_data/raw/training_contributions_with_complex_features_balanced.parquet'
pq.write_table(arrow_table, output_file)

print(f"Merged and shuffled all parquet files into {output_file}")
