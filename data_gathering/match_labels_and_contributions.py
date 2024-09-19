import pandas as pd

# Load the labels.tsv file
labels_file = '../data/ovid_labels.tsv'
labels_df = pd.read_csv(labels_file, sep='\t')

# Load the merged OSM contributions parquet file
merged_parquet_file = '../data/osm_labelled_contributions.parquet'
merged_df = pd.read_parquet(merged_parquet_file)

# Extract the changeset ids from both files
label_changeset_ids = set(labels_df['changeset'])
osm_changeset_ids = set(merged_df['changeset'].apply(lambda x: x['id']))

# Find how many changeset IDs have matching OSM contributions
matching_ids = label_changeset_ids.intersection(osm_changeset_ids)
missing_ids = label_changeset_ids.difference(osm_changeset_ids)

# Output the counts
print(f"Total changeset IDs in labels.tsv: {len(label_changeset_ids)}")
print(f"Changeset IDs with matching OSM contributions: {len(matching_ids)}")
print(f"Changeset IDs without matching OSM contributions: {len(missing_ids)}")

# Filter the labels_df to get missing IDs and their vandalism labels
missing_labels_df = labels_df[labels_df['changeset'].isin(missing_ids)]

# Count how many are labeled as vandalism (true) and how many are not (false)
vandalism_count = missing_labels_df[missing_labels_df['label'] == True].shape[0]
non_vandalism_count = missing_labels_df[missing_labels_df['label'] == False].shape[0]

# Output the counts
print(f"Missing Changeset IDs labeled as vandalism (true): {vandalism_count}")
print(f"Missing Changeset IDs labeled as non-vandalism (false): {non_vandalism_count}")

# Optionally, display the actual missing changeset IDs
print(f"Missing Changeset IDs: {missing_ids}")
