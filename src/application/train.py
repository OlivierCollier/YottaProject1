"""
    This class is one of the two entry points with predict.py.
    It is used to train our model.
"""

import pickle
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import src.config.base as base
import src.config.column_names as col
from src.domain.build_features import feature_engineering_transformer
from src.domain.cleaning import correct_wrong_entries, impute_missing_eco_data, MissingValueTreatment
from src.infrastructure.build_dataset import DataBuilderFactory, DataMerger


def main():

    # Builds datasets.
    print('Building datasets...')
    client_builder = DataBuilderFactory(base.TRAIN_CLIENT_DATA_PATH,
                                        base.config_client_data,
                                        base.ALL_CLIENT_DATA_TRANSLATION)
    client_data = client_builder.preprocess_data().data

    eco_builder = DataBuilderFactory(base.TRAIN_ECO_DATA_PATH,
                                     base.config_eco_data)
    eco_data = eco_builder.preprocess_data().data

    print('Preprocessing...')
    # Imputes NaN from the eco dataset.
    # This step is done outside the pipeline to avoid duplication of NaN while merging.
    eco_data = impute_missing_eco_data(eco_data)
    # Fixes erroneous entries in client dataset.
    client_data = correct_wrong_entries(client_data, base.config_client_data.get('wrong_entries'))

    # Merges client and eco datasets.
    print('Merging the client and economic datasets together...')
    merged = DataMerger(client_data, eco_data, col.MERGER_FIELD)
    merged.merge_datasets()
    merged_data = merged.joined_datasets
    merged_data_X = merged_data.drop(columns=col.TARGET)
    merged_data_y = merged_data[col.TARGET]

    # Loads pipeline.
    class_weight = {0: 1, 1: 9}
    pipeline = Pipeline([('imputation', MissingValueTreatment()),
                         ('feature_engineering', feature_engineering_transformer()),
                         ('rf_clf', RandomForestClassifier(class_weight=class_weight))
                         ])

    # Splits train and test sets.
    print('Splitting train and test...')
    merged_data_y = merged_data_y.eq('Yes').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(merged_data_X, merged_data_y,
                                                        test_size=0.2,
                                                        random_state=base.SEED,
                                                        stratify=merged_data_y)

    pipeline.fit(X_train, y_train)

    # Initializes random search.
    print('Initializing random search...')
    clf = RandomizedSearchCV(estimator=pipeline,
                             param_distributions=base.RF_PARAM,
                             scoring='average_precision',
                             random_state=base.SEED,
                             cv=5)

    # Fits the model.
    print('Fitting model...')
    clf.fit(X_train, y_train)

    # Saves model.
    print('Saving model...')
    with open(base.SAVED_MODEL_PATH, 'wb') as file:
        pickle.dump(clf, file)

if __name__ == '__main__':
    main()
