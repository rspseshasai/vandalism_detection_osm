from data_gathering.load_parquet_data import load
from feature_extraction import get_or_generate_features


def main():
    contributions_df = load('data/contribution_data/osm_labelled_contributions_v2.parquet', False)

    # Extract features for all contributions and store them in a DataFrame
    features_df = get_or_generate_features(contributions_df, True)


if __name__ == '__main__':
    main()
