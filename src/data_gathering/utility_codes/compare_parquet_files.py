import pandas as pd

def compare_parquet_files(file1, file2):
    """
    Compare two Parquet files and highlight differences in schema and data.
    Parameters:
    - file1: Path to the first Parquet file.
    - file2: Path to the second Parquet file.
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

    # Compare row counts
    if len(df1) != len(df2):
        print(f"Row counts differ: File 1 has {len(df1)} rows, File 2 has {len(df2)} rows.")
    else:
        print("Both files have the same number of rows.")

    # Compare common column data
    data_mismatch = df1_common.compare(df2_common, align_axis=1, keep_equal=False)

    if not data_mismatch.empty:
        print("\nDifferences found in data for the following columns:")
        print(data_mismatch)
    else:
        print("\nNo differences found in data for common columns.")

    print("\nComparison completed.")


# Example usage
file1_path = "path_to_non_parallel_features.parquet"
file2_path = "path_to_parallel_features.parquet"
compare_parquet_files(file1_path, file2_path)
