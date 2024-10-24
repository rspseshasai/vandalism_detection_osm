from data_gathering.load_parquet_data import load
from feature_extraction import get_or_generate_features

contributions_df = load('data/contribution_data/osm_labelled_contributions.parquet', False)

# Extract features for all contributions and store them in a DataFrame
features_df = get_or_generate_features(contributions_df, False)
