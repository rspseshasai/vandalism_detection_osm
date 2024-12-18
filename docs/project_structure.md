project/
│
├── data/                           # Data files
│   ├── raw/                        # Original raw data
│   │   ├── changeset_data/
│   │   └── contribution_data/
│   ├── processed/                  # Processed data
│   │   ├── changeset_data/
│   │   └── contribution_data/
│   └── external/                   # External data (e.g., from APIs)
│
├── notebooks/                      # Jupyter notebooks for EDA and reporting
│   ├── EDA_changeset.ipynb
│   ├── EDA_contribution.ipynb
│   ├── Model_Performance.ipynb
│   └── Feature_Importance.ipynb
│
├── src/                            # Source code for the project
│   ├── __init__.py
│   ├── data_loading.py             # Data loading functions
│   ├── preprocessing.py            # Data cleaning and preprocessing
│   ├── feature_engineering.py      # Feature extraction and engineering
│   ├── data_splitting.py           # Data splitting strategies
│   ├── modeling.py                 # Model training and hyperparameter tuning
│   ├── evaluation.py               # Model evaluation and metrics
│   ├── geo_temporal_evaluation.py  # Geographic and temporal evaluation
│   ├── utilities.py                # Helper functions and classes
│   ├── logger_config.py            # Logging configuration
│   └── config.py                   # Configuration settings
│
├── tests/                          # Unit tests
│   ├── test_data_loading.py
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_modeling.py
│   ├── test_evaluation.py
│   └── test_data_splitting.py
│
├── models/                         # Saved models and parameters
│   ├── changeset_model/
│   │   ├── best_model.pkl
│   │   ├── best_hyperparameters.json
│   │   └── bootstrapping_results/
│   └── contribution_model/
│       ├── best_model.pkl
│       ├── best_hyperparameters.json
│       └── bootstrapping_results/
│
├── docs/                           # Documentation and notes
│   ├── README.md
│   └── *.md                        # Other documentation files
│
├── scripts/                        # Scripts to run parts of the pipeline
│   ├── run_data_preprocessing.py
│   ├── run_feature_engineering.py
│   ├── run_training.py
│   ├── run_evaluation.py
│   └── run_geo_temporal_evaluation.py
│
├── requirements.txt                # Project dependencies
├── setup.py                        # For packaging and distribution (if needed)
└── .gitignore                      # Git ignore file
