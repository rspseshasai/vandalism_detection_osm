import os
import sys
import datetime
import pyarrow.dataset as ds
import pandas as pd

from config import OUTPUT_DIR, RAW_DATA_DIR

# Parameters
start_date = datetime.date(2022, 1, 1)
end_date = datetime.date(2024, 2, 1)
per_file = 5769       # rows to read (random) per file
total_needed = 150000 # total rows required
output_file = os.path.join(OUTPUT_DIR, "non_vandalism_sample.parquet")

# Function to get the next month
def next_month(d):
    # increment month by 1
    if d.month == 12:
        return datetime.date(d.year + 1, 1, 1)
    else:
        return datetime.date(d.year, d.month + 1, 1)

current_date = start_date
all_dfs = []
current_count = 0

while current_date <= end_date and current_count < total_needed:
    file_name = os.path.join(RAW_DATA_DIR, f"{current_date.isoformat()}.parquet")
    print(f"reading {file_name}")
    if not os.path.exists(file_name):
        # If file doesn't exist, skip to next month
        current_date = next_month(current_date)
        continue

    # Create a dataset from the single file
    dataset = ds.dataset(file_name, format="parquet")

    # Read entire file into a DataFrame
    try:
        df = dataset.to_table().to_pandas()
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        current_date = next_month(current_date)
        continue

    if df.empty:
        # If file is empty, move on
        current_date = next_month(current_date)
        continue

    # If file has fewer rows than per_file, take all; else sample
    if len(df) <= per_file:
        sampled_df = df.copy()
    else:
        # Randomly sample per_file rows
        # Set a random_state if you want reproducible results
        sampled_df = df.sample(n=per_file, random_state=42)

    # Mark these entries as non-vandalism
    sampled_df['vandalism'] = 0

    all_dfs.append(sampled_df)
    current_count += len(sampled_df)

    # Move to next month
    current_date = next_month(current_date)

# If we collected more than needed, trim the last DataFrame
if current_count > total_needed and all_dfs:
    excess = current_count - total_needed
    last_df = all_dfs[-1]
    # Keep only the rows needed
    all_dfs[-1] = last_df.iloc[:-excess]

# Concatenate all collected DataFrames
if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True)
else:
    final_df = pd.DataFrame(columns=['vandalism'])  # Empty DF if no data found

# Save the combined sample to parquet
final_df.to_parquet(output_file, index=False)

print(f"Saved {len(final_df)} rows of non-vandalism samples to {output_file}")
