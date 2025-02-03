# XGBoost-Based ML Framework for Vandalism Detection in OpenStreetMap

**Brief**  
This project focuses on **automatic vandalism detection** in OpenStreetMap (OSM) contributions and changesets. It provides data ingestion, preprocessing, feature engineering, model training (using XGBoost), evaluation, and inference pipelines—enabling both **monthly** and **daily** data processing.

## Project Overview
- **Goal**: Build and evaluate robust pipelines for classifying OSM edits (vandalism vs. legitimate).

- **Main Components**:
  - Parallelizable **data loading** and **feature engineering** for large datasets
  - Modular **XGBoost-based training pipeline** designed to seamlessly handle both **OSM changesets and contributions data**
  - Pipeline scripts for **monthly** (bulk) and **daily** (frequent) inference
  - Logging, evaluation artifacts (e.g., confusion matrix, ROC, PR/PR curves)
- **Extensibility**: Easily integrate new features, clustering, or alternative classifiers using a chunk-based approach and flexible config files.

## Installation & Setup
1. **Clone** the repository:
   ```bash
   git clone https://github.com/rspseshasai/vandalism_detection_osm.git
   ```
2. **Create & activate** a virtual environment (optional):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   ```
3. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Prepare Data**:  
   - Place OSM data (raw extracts) in the `data/` directory.  
   - Update any paths in scripts or configuration (`config.py`) if needed.


2. **Configure Settings**:  
   - Edit `config.py` to specify hyperparameters, chunk sizes for data loading, logging levels, etc.  
   - You may also define custom splits (random, geographic, or temporal) or advanced model configurations.


3. **Run the Pipeline**:  
   - Execute `ml_training_and_eval_pipeline.py` to orchestrate data loading, preprocessing, training, and evaluation in one go:
     ```bash
     python ml_training_and_eval_pipeline.py
     ```
   - Logs, models, and evaluation outputs appear under `logs/`, `models/`, and `data/output/`.


4. **Review Results**:  
   - Analyze results efficiently by reviewing logs, exploring the output directory, or examining notebook visualizations for detailed run summaries, including key performance metrics like confusion matrices, ROC/PR curves, and model evaluations.

### Additional Notes
- **Inference Pipelines**:  
  - `prediction_pipeline_for_monthly_files.py` and `prediction_pipeline_for_smaller_daily_files.py` enable inference on distinct data ingestion schedules (monthly vs. daily).  
- **Notebooks**:  
  - Use `notebooks/` for interactive data exploration, ablations, or advanced visualization.  
- **Model Checkpoints**:  
  - Stored in `models/` with subdirectories for hyper-classifier and meta-classifier variants.  

## Repository Structure
```
.
├── ml_training_and_eval_pipeline.py
├── prediction_pipeline_for_monthly_files.py
├── prediction_pipeline_for_smaller_daily_files.py
├── README.md
├── requirements.txt
├── data/
│   ├── changeset_data/
│   ├── contribution_data/
│   │   ├── output/
│   │   ├── processed/
│   │   └── raw/
│   │   └── Visualization/
│   ...
├── logs/
│   ├── changeset_pipeline.log
│   ├── contribution_pipeline.log
│   └── pipeline.log
├── models/
│   ├── changeset_model/
│   │   ├── random/
│   │   ├── geographic/
│   │   └── temporal/
│   │       └── (hyper_classifier, meta_classifier subdirs)
│   └── contribution_model/
├── notebooks/
│   ├── visualization_and_analysis_changeset_data.ipynb
│   ├── visualization_and_analysis_contribution_data.ipynb
│   └── ... (other exploratory notebooks)
├── src/
│   ├── data_loading.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── training.py           # XGBoost training logic
│   ├── evaluation.py         # Metrics & PR-Gain plots
│   ├── config.py
│   ├── hyper_classifier/
│   │   ├── hyper_classifier_main.py
│   │   ├── hyper_classifier_training.py
│   │   ├── hyper_classifier_data_splitting.py
├── ...
```

---

**Happy Mapping and Machine Learning!**
