import os
import json
from datetime import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from ml_training_and_eval_pipeline import pipeline

# Load specified combinations from a file
with open("full_specified_combinations.json", "r") as f:
    specified_combinations = json.load(f)

def run_pipeline(train_regions, val_regions, test_regions):
    """
    Run the ML pipeline by overriding the train, test, and val regions.
    """
    os.environ['TRAIN_REGIONS'] = ','.join(train_regions)
    os.environ['VAL_REGIONS'] = ','.join(val_regions)
    os.environ['TEST_REGIONS'] = ','.join(test_regions)

    return pipeline(train_regions, val_regions, test_regions)

def calculate_metrics(y_true, y_pred):
    """
    Calculate key metrics from true and predicted values.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return tn, fp, fn, tp, accuracy, precision, recall, f1

# Initialize results storage
results = []
output_path = rf"D:\PycharmProjects\vandalism_detection_osm\data\contribution_data\output\geo_bootstrap_results\bootstrap_geo_split_results_nuof_high_recall_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.parquet"

# Iterate over specified combinations
combination_count = 0
for train_group, val_group, test_group in specified_combinations:
    combination_count += 1
    print(f"\nProcessing combination {combination_count}: Train={train_group}, Val={val_group}, Test={test_group}")

    try:
        # Run the pipeline and get evaluation results
        eval_results = run_pipeline(train_group, val_group, test_group)

        # Extract true and predicted values
        y_true = eval_results['y_true']
        y_pred = eval_results['y_pred_main']

        # Calculate metrics
        tn, fp, fn, tp, accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)

        # Store results
        results.append({
            'train_regions': train_group,
            'val_regions': val_group,
            'test_regions': test_group,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

        print(f"\nMetrics for combination {combination_count}: TN={tn}, FP={fp}, FN={fn}, TP={tp}, Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1={f1}")

        # Save results every 10 combinations
        if combination_count % 10 == 0:
            results_df = pd.DataFrame(results)
            results_df.to_parquet(output_path, index=False)
            print(f"Results updated to {output_path} after {combination_count} combinations.")

    except Exception as e:
        print(f"Error running pipeline for combination Train: {train_group}, Val: {val_group}, Test: {test_group}. Error: {e}")

# Final save of results
results_df = pd.DataFrame(results)
results_df.to_parquet(output_path, index=False)
print(f"Final bootstrap results saved to {output_path}")
