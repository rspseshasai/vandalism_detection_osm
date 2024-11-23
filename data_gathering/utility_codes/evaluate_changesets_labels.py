import pandas as pd
import pyarrow.parquet as pq

from logger_config import logger

# Path to the merged parquet file
merged_parquet_file = '../../data/contribution_data/osm_labelled_contributions_v2.parquet'
# Path to the TSV file containing the original changeset data
tsv_file_path = '../../data/ovid_data/ovid_labels.tsv'

logger.info(f"Reading merged parquet file: {merged_parquet_file}")

# Read the merged parquet file into a Pandas DataFrame
try:
    table = pq.read_table(merged_parquet_file)
    df = table.to_pandas()
    logger.info(f"Successfully loaded parquet file with {df.shape[0]} rows and {df.shape[1]} columns.")
except Exception as e:
    logger.error(f"Error while reading parquet file: {e}")
    raise


def group_changeset_with_vandalism(df):
    # Extract the changeset 'id' from the 'changeset' field
    logger.info("Extracting 'changeset_id' from 'changeset' field.")
    df['changeset_id'] = df['changeset'].apply(lambda x: x['id'] if pd.notnull(x) and 'id' in x else None)

    logger.info("Extracting 'vandalism' field.")
    df['vandalism'] = df['vandalism']

    # Group by 'changeset_id' and collect 'vandalism' values into a set
    logger.info("Grouping by 'changeset_id' and aggregating 'vandalism' into sets.")
    try:
        grouped_df = df.groupby('changeset_id')['vandalism'].apply(set).reset_index()
        logger.info(f"Successfully grouped data. Resulting DataFrame has {grouped_df.shape[0]} rows.")
    except Exception as e:
        logger.error(f"Error during grouping: {e}")
        raise
    return grouped_df


def check_single_vandalism_value_per_changeset(grouped_df):
    """Checks if each changeset has only one unique vandalism value (yes or no)."""

    # Find rows where the 'vandalism' set has more than 1 unique element
    logger.info("Filtering rows where the 'vandalism' set contains more than 1 unique element.")
    try:
        filtered_df = grouped_df[grouped_df['vandalism'].apply(lambda x: len(x) > 1)]
        logger.info(f"Filtered data contains {filtered_df.shape[0]} rows with multiple unique 'vandalism' values.")
    except Exception as e:
        logger.error(f"Error during filtering: {e}")
        raise


def update_vandalism_dtype(grouped_df):
    """Updates the vandalism values from set to bool (yes or no)."""

    logger.info("Updating vandalism values from set to bool (yes or no).")
    try:
        grouped_df['vandalism'] = grouped_df['vandalism'].apply(lambda x: 'yes' if True in x else 'no')
        logger.info("Successfully updated vandalism values.")
    except Exception as e:
        logger.error(f"Error during updating vandalism values: {e}")
        raise


def count_vandalism_values(grouped_df):
    """Counts the number of changesets with 'yes' and 'no' vandalism values."""
    yes_count = (grouped_df['vandalism'] == 'yes').sum()
    no_count = (grouped_df['vandalism'] == 'no').sum()

    logger.info(f"Number of changesets with vandalism 'yes': {yes_count}")
    logger.info(f"Number of changesets with vandalism 'no': {no_count}")

    print(f"Number of changesets with vandalism 'yes': {yes_count}")
    print(f"Number of changesets with vandalism 'no': {no_count}")


def read_tsv_file(tsv_file_path):
    """Reads the original changeset data from a TSV file."""
    logger.info(f"Reading TSV file: {tsv_file_path}")
    try:
        original_df = pd.read_csv(tsv_file_path, sep='\t')
        logger.info(f"Successfully loaded TSV file with {original_df.shape[0]} rows and {original_df.shape[1]} columns.")
    except Exception as e:
        logger.error(f"Error while reading TSV file: {e}")
        raise
    return original_df


def normalize_vandalism_value(value):
    """Normalizes the vandalism value to a string format."""
    if pd.isna(value):
        return None
    if value in ['yes', 'TRUE', True]:
        return 'yes'
    elif value in ['no', 'FALSE', False]:
        return 'no'
    return None


def compare_changeset_data(grouped_df, original_df):
    """Compares grouped changeset data with the original data and generates a report."""
    logger.info("Comparing changeset data with original data.")

    # Keep only entries from grouped_df
    comparison_df = pd.merge(grouped_df, original_df, on='changeset_id', how='inner')
    comparison_df.rename(columns={'vandalism': 'computed_vandalism', 'label': 'original_vandalism'}, inplace=True)

    # Normalize original vandalism values for comparison
    comparison_df['original_vandalism'] = comparison_df['original_vandalism'].apply(normalize_vandalism_value)

    # Identify discrepancies
    comparison_df['discrepancy'] = comparison_df.apply(
        lambda row: row['computed_vandalism'] != row['original_vandalism'], axis=1)

    logger.info("Generating detailed comparison report.")
    report_df = comparison_df[['changeset_id', 'computed_vandalism', 'original_vandalism', 'discrepancy']]

    logger.info(f"Comparison report generated with {report_df.shape[0]} rows.")
    return report_df


# Run the functions
grouped_df = group_changeset_with_vandalism(df)
check_single_vandalism_value_per_changeset(grouped_df)
update_vandalism_dtype(grouped_df)

# Count the number of changesets with 'yes' and 'no' vandalism values
count_vandalism_values(grouped_df)

# Read the original changeset data
original_df = read_tsv_file(tsv_file_path)

# Compare the changeset data
comparison_report = compare_changeset_data(grouped_df, original_df)

# Print the comparison report
print(comparison_report)

# Save the report to a CSV file
# comparison_report.to_csv('output/comparison_report.csv', index=False)
# logger.info("Comparison report saved as 'comparison_report.csv'.")
