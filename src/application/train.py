"""
    This class is one of the two entrypoints with predict.py.
    This class is used to train a ML model with some training data.
"""
import argparse
import os
import pickle
from warnings import simplefilter

import ipdb
import pandas as pd
import src.config.base as base
import src.config.column_names as col
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import precision_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


from src.domain.build_features import feature_engineering_transformer
from src.domain.cleaning import correct_wrong_entries, impute_missing_eco_data, MissingValueTreatment
from src.infrastructure.build_dataset import DataBuilderFactory, DataMerger

# Ignore warnings
# simplefilter(action='ignore', category=SettingWithCopyWarning)
# simplefilter(action='ignore', category=FutureWarning)


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


# Impute NaN from the socio-eco dataset
# This step is done outside the pipeline to avoid duplication of Nan after the merge
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
pipeline = Pipeline([('imputer', MissingValueTreatment())
                        ,('feature_engineering', feature_engineering_transformer())
                        ,('log_reg_clf', LogisticRegression())
                        ])


# Train-test split
merged_data_y = merged_data_y.eq('Yes').astype(int)
X_train, X_test, y_train, y_test = train_test_split(merged_data_X, merged_data_y, 
                                        test_size=0.2, random_state=base.SEED)


# Initialize Random search
# clf = RandomizedSearchCV(estimator=processing_pipeline, param_distributions = base.LOGISTIC_REGRESSION_PARAM, 
                        # scoring='precision', random_state=base.SEED, cv=5)

# Fit the model
pipeline.fit(X_train, y_train)


# Make prediction on test set
print('Make a prediction on test set')
y_pred = pipeline.predict(X_test)
ipdb.set_trace()

# Save model 
# with open(base.SAVED_MODEL_PATH, 'wb') as file:
#     pickle.dump(processing_pipeline, file)
