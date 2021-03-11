import os
from dataclasses import dataclass

import pandas as pd

from src.domain.cleaning import impute_missing_eco_data, correct_wrong_entries
import src.config.base as base
import src.config.column_names as col


# Global variables
MERGING_METHODS = ['left', 'right', 'outer', 'inner', 'cross']


class DataBuilder:

    def __init__(self, path: str, config: dict, text_translation: dict = None):
        self.path = path
        self.config = config
        self.text_translation = text_translation
        self.data = self.read()


    def read(self):
        return NotImplementedError


    def _add_merger_field(self) -> None:
        """Reformat date field to have unique merge field for the join"""
        self.data[col.MERGER_FIELD] = self.data[self.config.get('date_column')].dt.strftime('%Y-%m')
        return self


    def _cast_types(self) -> None:
        """Cast columns to adequate types"""
        mapping_cols_types = self.config.get('cast_types')
        self.data = self.data.astype(mapping_cols_types)
        return self

    def _replace_translation(self) -> None:
        """Replace text translation from french into english"""
        if self.text_translation:
            self.data.replace(self.text_translation, inplace=True)
        return self
    
    def _drop_very_incomplete_rows(self) -> None:
        """Drop rows with many missing values (e.g. AGE and JOB_TYPE)"""
        missing_cols = self.config.get('filters').get('missing')
        if missing_cols:
            subset_df = self.data.loc[:, missing_cols]
            subset_bool = subset_df.isna().all(axis=1)
            missing_idx = subset_bool[subset_bool].index
            self.data = self.data.drop(index=missing_idx)
        return self

    def preprocess_data(self) -> pd.DataFrame:
        """Run preprocess tasks"""
        processed_data = self._cast_types()\
            ._replace_translation()\
            ._add_merger_field()\
            ._drop_very_incomplete_rows()

        return processed_data

    def transform(self, data_type: str) -> pd.DataFrame:
        """Run preprocess tasks, including imputation, with logs"""
        print(f'========== Processing {data_type} data ==========')
        print('- Casting types.')
        self._cast_types()
        print('- Translating French words to English.')
        self._replace_translation()
        self._add_merger_field()
        print('- Dropping rows with too many missing values.')
        self._drop_very_incomplete_rows()

        processed_data = self.data

        if data_type == 'eco':
            print('- Imputing missing data.')
            processed_data = impute_missing_eco_data(processed_data)
        elif data_type == 'client':
            print('- Correcting erroneous entries.')
            processed_data = correct_wrong_entries(processed_data,
                                                   base.config_client_data.get('wrong_entries'))

        return processed_data


class DataBuilderCSV(DataBuilder):
    """
    Data loader class for csv format
    """
    def read(self) -> pd.DataFrame:
        data = pd.read_csv(self.path, sep=self.config.get('sep'))
        return data


class DataBuilderFactory:
    """
    DataBuilder factory class
    Supported formats are:
        - csv
    To support another format, create another DataBuilder<format> class and add 
    and if statement in this class
    """

    def __new__(cls, path: str, config: dict, text_translation: dict = None):
        if path.endswith('.csv'):
            return DataBuilderCSV(path, config, text_translation)
        else:
            raise ValueError('Unsupported data format.')
    

class DataMerger:
    """
        Class to merge two datasets together
    """

    def __init__(self, left_dataset: pd.DataFrame, right_dataset: pd.DataFrame,
                 merge_field: str, how: str = 'left') -> None:
        self.left_dataset = left_dataset
        self.right_dataset = right_dataset
        self.merge_field = merge_field
        if how not in MERGING_METHODS:
            raise ValueError(f'How argument must be in {MERGING_METHODS}')
        self.how = how
        self.joined_datasets = None

    def merge_datasets(self):
        """Merge both datasets"""
        self._drop_duplicate_columns()
        self.joined_datasets = self.left_dataset.merge(self.right_dataset, how=self.how, on=self.merge_field)
        self.joined_datasets.drop([col.MERGER_FIELD], axis=1, inplace=True)

    def _drop_duplicate_columns(self):
        """Drop columns that are present in both datasets"""
        columns_in_left_dataset = set(self.left_dataset.columns)
        columns_in_right_dataset = set(self.right_dataset.columns)
        columns_to_keep_in_right_dataset = list(columns_in_right_dataset - columns_in_left_dataset)\
                                           + [col.MERGER_FIELD]
        self.right_dataset = self.right_dataset[columns_to_keep_in_right_dataset]

    def save(self, out_path: str):
        """Save the merged dataset"""
        self.joined_datasets.to_csv(out_path)

    def transform(self):
        """Merge datasets with logs"""
        print('========== Merging datasets ==========')
        self.merge_datasets()
        merged_data = self.joined_datasets
        print('========== Separating target from explanatory variables ==========')
        X = merged_data.drop(columns=col.TARGET)
        y = merged_data[col.TARGET]
        return X, y


if __name__ == '__main__':

    # read and clean both datasets
    client_builder = DataBuilderFactory(base.DATA_PATH,
                                        base.config_client_data,
                                        base.ALL_CLIENT_DATA_TRANSLATION)
    client_data = client_builder.preprocess_data().data

    economic_builder = DataBuilderFactory(base.ECO_DATA_PATH,
                                          base.config_eco_data)
    economic_data = economic_builder.preprocess_data().data

    # join datasets and merge
    merged = DataMerger(client_data, economic_data, merge_field=col.MERGER_FIELD)
    merged.merge_datasets()
<<<<<<< HEAD
<<<<<<< HEAD
    merged.save(base.PROCESSED_DATA_PATH)
=======
    merged.save(PROCESSED_DATA_PATH)
>>>>>>> :memo: Add README instructions for the train/predict
=======
    merged.save(base.PROCESSED_DATA_PATH)
>>>>>>> 6fa20d3799e8318676408cdca92913f0772c6ad2
