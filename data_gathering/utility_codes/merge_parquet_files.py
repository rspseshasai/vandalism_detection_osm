import pandas as pd

# List of all the 6 .parquet files
parquet_files = [
    'output/filtered_contributions_part_0.parquet',
    'output/filtered_contributions_part_1.parquet',
    'output/filtered_contributions_part_2.parquet',
    'output/filtered_contributions_part_3.parquet',
    'output/filtered_contributions_part_4.parquet',
    'output/filtered_contributions_part_5.parquet'
]

# Initialize an empty list to store DataFrames
dataframes = []

# Iterate over the parquet files and read each one
for parquet_file in parquet_files:
    df = pd.read_parquet(parquet_file)
    dataframes.append(df)

# Concatenate all DataFrames into one
merged_dataframe = pd.concat(dataframes, ignore_index=True)

# Save the merged DataFrame into a new .parquet file
output_file = 'merged_contributions.parquet'
merged_dataframe.to_parquet(output_file, index=False)

print(f"All files merged and saved to {output_file}")
