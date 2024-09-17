from feature_extraction import extract_features
from load_parquet_data import load

contributions_df = load('data/contri_test_0.parquet', False)

# Extract features for all contributions and store them in a DataFrame
features_df = extract_features(contributions_df)
