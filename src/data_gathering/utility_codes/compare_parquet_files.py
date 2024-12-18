import pandas as pd

def compare_parquet_files(file1, file2, output_diff_csv):
    """
    Compare two Parquet files and save differences to a CSV file for detailed review.
    Parameters:
    - file1: Path to the first Parquet file.
    - file2: Path to the second Parquet file.
    - output_diff_csv: Path to save the differences as a CSV file.
    """
    # Load the Parquet files
    df1 = pd.read_parquet(file1)
    df2 = pd.read_parquet(file2)

    print("Comparing schemas...")
    schema1 = set(df1.columns)
    schema2 = set(df2.columns)

    # Compare column names
    missing_in_file1 = schema2 - schema1
    missing_in_file2 = schema1 - schema2

    if missing_in_file1:
        print(f"Columns in File 2 but missing in File 1: {missing_in_file1}")
    else:
        print("No columns are missing in File 1 compared to File 2.")

    if missing_in_file2:
        print(f"Columns in File 1 but missing in File 2: {missing_in_file2}")
    else:
        print("No columns are missing in File 2 compared to File 1.")

    print("\nComparing data...")
    # Align columns for data comparison
    common_columns = schema1 & schema2
    df1_common = df1[list(common_columns)].sort_index(axis=1).reset_index(drop=True)
    df2_common = df2[list(common_columns)].sort_index(axis=1).reset_index(drop=True)

    # Check for data type mismatches
    dtype_mismatches = {}
    for col in common_columns:
        if df1[col].dtype != df2[col].dtype:
            dtype_mismatches[col] = (df1[col].dtype, df2[col].dtype)

    if dtype_mismatches:
        print("\nData type mismatches found in the following columns:")
        for col, dtypes in dtype_mismatches.items():
            print(f"Column '{col}': File 1 dtype = {dtypes[0]}, File 2 dtype = {dtypes[1]}")
    else:
        print("\nNo data type mismatches found.")

    # Convert all data to string to avoid issues with dtype mismatches
    df1_common = df1_common.astype(str)
    df2_common = df2_common.astype(str)

    # Compare row counts
    if len(df1) != len(df2):
        print(f"Row counts differ: File 1 has {len(df1)} rows, File 2 has {len(df2)} rows.")
    else:
        print("Both files have the same number of rows.")

    # Compare common column data
    differences = []
    for col in common_columns:
        mismatch = df1_common[col] != df2_common[col]
        if mismatch.any():
            mismatch_indices = df1_common[mismatch].index
            for idx in mismatch_indices:
                differences.append({
                    'Index': idx,
                    'Column': col,
                    'File1_Value': df1_common.at[idx, col],
                    'File2_Value': df2_common.at[idx, col]
                })

    if differences:
        print(f"\nFound {len(differences)} differences. Saving to {output_diff_csv}...")
        differences_df = pd.DataFrame(differences)
        differences_df.to_csv(output_diff_csv, index=False)
        print(f"Differences saved to {output_diff_csv}.")
    else:
        print("\nNo differences found in data for common columns.")

    print("\nComparison completed.")


# Example usage
file1_path = "C:\\Users\\Pavan\\Downloads\\test_data\\itr1\\parallel\\_processed_features.parquet"
file2_path = "C:\\Users\\Pavan\\Downloads\\test_data\\itr1\\non-parallel\\_processed_features.parquet"
output_diff_csv = "C:\\Users\\Pavan\\Downloads\\test_data\\itr1\\differences_between_files.csv"

compare_parquet_files(file1_path, file2_path, output_diff_csv)
