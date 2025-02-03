import joblib
import pandas as pd

from config import FINAL_MODEL_PATH, FINAL_TRAINED_FEATURES_PATH

# Load the trained model and features
model = joblib.load(FINAL_MODEL_PATH)
feature_names = joblib.load(FINAL_TRAINED_FEATURES_PATH)

# Metrics to evaluate
metrics = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']

# Create a DataFrame to store feature importance for all metrics
importance_data = []

for metric in metrics:
    booster = model.get_booster()
    score_dict = booster.get_score(importance_type=metric)

    # Map internal feature names to actual feature names
    importance_df = pd.DataFrame({
        'feature': list(score_dict.keys()),
        metric: list(score_dict.values())
    })
    importance_df['feature'] = importance_df['feature'].apply(
        lambda x: feature_names[int(x[1:])] if x.startswith('f') else x
    )
    importance_data.append(importance_df)

# Merge all metrics into a single DataFrame
feature_importance_df = pd.concat(importance_data, axis=0).groupby('feature').sum().reset_index()

# Sort by 'gain' for better interpretation
feature_importance_df = feature_importance_df.sort_values(by='total_gain', ascending=False)

# Top 15 important features
top_15_features = feature_importance_df.head(30)

# Features with very low importance
gain_threshold = 0.01 * feature_importance_df['gain'].max()  # 1% of max gain
low_importance_features = feature_importance_df[
    (feature_importance_df['gain'] < gain_threshold) &
    (feature_importance_df['weight'] == 0)
    ]

# Print the results
print("Top 15 Important Features:")
print(top_15_features)

print("\nFeatures That Can Be Removed (Low Gain and Zero Weight):")
print(low_importance_features)
