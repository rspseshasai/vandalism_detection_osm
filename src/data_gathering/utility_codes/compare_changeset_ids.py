import pandas as pd


def compare_and_save_common_changesets(prefix):
    """
    Compare changeset_id columns in two Parquet files and a TSV file, find common IDs, and save the first 1000 to a file.

    Parameters:
        parquet1_path (str): Path to the first Parquet file.
        parquet2_path (str): Path to the second Parquet file.
        tsv_file_path (str): Path to the TSV file.
        output_file_path (str): Path to save the output file with 1000 common changeset IDs.
    """

    # Define file paths
    # parquet1_path = f'../../../data/contribution_data/processed/test_processed_features.parquet'
    # parquet2_path = f'../../../data/changeset_data/processed/test_processed_features.parquet'
    # tsv_file_path = f'../../../data/changeset_data/raw/changeset_labels.tsv'

    parquet1_path = '../../../data/contribution_data/processed/_processed_features.parquet'
    parquet2_path = '../../../data/changeset_data/processed/_processed_features.parquet'
    tsv_file_path = '../../../data/changeset_data/raw/changeset_labels.tsv'

    # Read the first Parquet file
    df_parquet1 = pd.read_parquet(parquet1_path)
    changeset_ids_parquet1 = set(df_parquet1['changeset_id'].tolist())
    print(f"Loaded {len(changeset_ids_parquet1)} changeset IDs from Parquet file 1.")

    # Read the second Parquet file
    df_parquet2 = pd.read_parquet(parquet2_path)
    changeset_ids_parquet2 = set(df_parquet2['changeset_id'].tolist())
    print(f"Loaded {len(changeset_ids_parquet2)} changeset IDs from Parquet file 2.")

    # Read the TSV file
    df_tsv = pd.read_csv(tsv_file_path, sep='\t')
    changeset_ids_tsv = set(df_tsv['changeset'].tolist())
    print(f"Loaded {len(changeset_ids_tsv)} changeset IDs from TSV file.")

    # Find common changeset IDs across all three files
    common_changeset_ids = changeset_ids_parquet1.intersection(changeset_ids_parquet2, changeset_ids_tsv)
    print(f"Found {len(common_changeset_ids)} common changeset IDs across all files.")

    # Save the common changesets to a file
    if prefix == '':
        output_file_path = f'../../../data/changeset_data/raw/{prefix}_common_changeset_ids.csv'
        df_common_sample = pd.DataFrame({'changeset_id': list(common_changeset_ids)})
        df_common_sample.to_csv(output_file_path, index=False)
        print(f"Saved the common changeset IDs to {output_file_path}.")
    else:
        output_file_path = f'../../../data/changeset_data/raw/{prefix}_common_changesets_1000.csv'
        df_common_sample = pd.DataFrame({'changeset_id': list(common_changeset_ids)[:1000]})
        df_common_sample.to_csv(output_file_path, index=False)
        print(f"Saved the first 1000 common changeset IDs to {output_file_path}.")


# Run the function
compare_and_save_common_changesets('')
# compare_and_save_common_changesets('test')
