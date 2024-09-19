from feature_extraction import extract_features
from load_parquet_data import load

contributions_df = load('data/osm_labelled_contributions.parquet', False)

# Extract features for all contributions and store them in a DataFrame
features_df = extract_features(contributions_df)
