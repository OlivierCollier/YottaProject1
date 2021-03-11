"""
    This class is one of the two entry points with predict.py.
    It is used to train a ML model with some training data.
"""

import argparse
import os

import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import src.config.base as base
import src.config.column_names as col
from src.domain.build_features import feature_engineering_transformer
from src.domain.cleaning import correct_wrong_entries, impute_missing_eco_data, MissingValueTreatment
from src.infrastructure.build_dataset import DataBuilderFactory, DataMerger


# Parse arguments
parser = argparse.ArgumentParser(description='Files containing the training datasets', \
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--client-data',
                '-c', 
                help='Provide the client dataset file',
                default=os.path.join(base.RAW_DATA_DIR, base.CLIENT_DATA_FILE_NAME),
                dest='client_file')
parser.add_argument('--eco-data',
                '-e',
                help='Provide the economic information dataset file',
                default=os.path.join(base.RAW_DATA_DIR, base.ECO_DATA_FILE_NAME),
                dest='eco_file')
args = parser.parse_args()

# Build datasets
client_builder = DataBuilderFactory(args.client_file, base.config_client_data, 'production', \
                                base.CLIENT_COLUMNS_TO_DROP, base.ALL_CLIENT_DATA_TRANSLATION)
client_data = client_builder.preprocess_data().data

eco_builder = DataBuilderFactory(args.eco_file, base.config_eco_data, 'production', \
                            base.ECO_COLUMNS_TO_DROP)
eco_data = eco_builder.preprocess_data().data

# Impute NaN from the eco dataset
# This step is done outside the pipeline to avoid duplication of NaN after the merge
eco_data = impute_missing_eco_data(eco_data)
# Fix wrong entries in client dataset
client_data = correct_wrong_entries(client_data, base.config_client_data.get('wrong_entries'))

# Merger client and eco datasets
merged = DataMerger(client_data, eco_data, col.MERGER_FIELD)
merged.merge_datasets()
merged_data = merged.joined_datasets
merged_data_X = merged_data.drop(columns=col.TARGET)
merged_data_y = merged_data[col.TARGET]

# Load pipeline
class_weight = {0: 1, 1: 9}
pipeline = Pipeline([('imputer', MissingValueTreatment()),
                     ('feature_engineering', feature_engineering_transformer()),
                     ('rf_clf', RandomForestClassifier(class_weight=class_weight))
                     ])

# Train-test split
merged_data_y = merged_data_y.eq('Yes').astype(int)
X_train, X_test, y_train, y_test = train_test_split(merged_data_X, merged_data_y,
                                                    test_size=0.2,
                                                    random_state=base.SEED,
                                                    stratify=merged_data_y)

# Initialize Random search
clf = RandomizedSearchCV(estimator=pipeline,
                         param_distributions=base.RF_PARAM,
                         scoring='average_precision',
                         random_state=base.SEED,
                         cv=5)

# Fit the model
clf.fit(X_train, y_train)

# Save model
with open(base.SAVED_MODEL_PATH, 'wb') as file:
    pickle.dump(clf.best_estimator_, file)