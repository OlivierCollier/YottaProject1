"""
    This module is one of the two entrypoints with train.py
    This class is used to make some predictions based on a model and new data
"""

import argparse
import pickle
import os

import pandas as pd

import src.config.base as base
import src.config.column_names as col
from src.infrastructure.build_dataset import DataBuilderFactory, DataMerger


def load_pipeline():
    try:
        print('Loading the fitted pipeline..')
        with open(base.SAVED_MODEL_PATH, 'rb') as model_file:
            pipeline = pickle.load(model_file)
        print('Loading completed successfully')
    except FileNotFoundError:
        print('Model file has not been found')
        raise
    return pipeline


# Parse arguments
print('Parsing input arguments...')
parser = argparse.ArgumentParser(description='Files containing the test datasets', \
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--client-data',
                '-c', 
                help='Provide the client dataset file',
                dest='client_file')
parser.add_argument('--eco-data',
                '-e',
                help='Provide the economic information dataset file',
                dest='eco_file')

args = parser.parse_args()


# Build datasets
print('Building datasets...')
client_builder = DataBuilderFactory(args.client_file, base.config_client_data, 'production', \
                                base.CLIENT_COLUMNS_TO_DROP, base.ALL_CLIENT_DATA_TRANSLATION)
client_data = client_builder.preprocess_data().data

eco_builder = DataBuilderFactory(args.eco_file, base.config_eco_data, 'production', \
                            base.ECO_COLUMNS_TO_DROP)
eco_data = eco_builder.preprocess_data().data


# Merger client and eco datasets
print('Merging the client and economic datasets together...')
merged = DataMerger(client_data, eco_data, col.MERGER_FIELD)
merged.merge_datasets()
X_pred = merged.joined_datasets
if col.TARGET in X_pred.columns:
    X_pred.drop(col.TARGET, axis= 1, inplace=True)


# Load pipeline
pipeline = load_pipeline()


# Make predictions
y_pred = pipeline.predict(X_pred)


# Write predictions
y_pred = pd.Series(y_pred)
y_pred.to_csv(base.PREDICTIONS_FILE_PATH)