import os
from dataclasses import dataclass

import ipdb
import pandas as pd
from src.config.base import *
from src.config.column_names import *


BUILD_MODES = ['production', 'eda']


class DataBuilder:

    def __init__(self, path: str, config: dict, build_mode: str, \
                 cols_to_drop: list = None, text_translation: dict = None):
        self.path = path
        self.config = config
        if build_mode not in BUILD_MODES:
            raise ValueError(f'Build mode should be one of {BUILD_MODES}')
        self.build_mode = build_mode
        self.cols_to_drop = cols_to_drop
        self.text_translation = text_translation
        self.data = self.read()


    def read(self):
        return NotImplementedError


    def _add_merger_field(self) -> None:
        """ Reformat date field to have unique merge field for the join """
        self.data[MERGER_FIELD] = self.data[self.config.get('date_column')].dt.strftime('%Y-%m')
        return self


    def _cast_types(self) -> None:
        """ Cast columns to adequate types"""
        mapping_cols_types = self.config.get('cast_types')
        self.data = self.data.astype(mapping_cols_types)
        return self


    def _drop_columns(self) -> None:
        """ 
        Drop columns specified in column_names.py only in 'production' build mode
        because we cannot use them. 
        In 'eda' mode, we want to keep them for the plots and explanations.
        """
        if self.build_mode == 'production':
            self.data.drop(columns=self.cols_to_drop, inplace=True)
        return self


    def _replace_translation(self) -> None:
        """ Replace text translation from french into english """
        if self.text_translation:
            self.data.replace(self.text_translation, inplace=True)
        return self


    def preprocess_data(self) -> pd.DataFrame:
        """ Run preprocess tasks """
        processed_data = self._cast_types() \
                             ._drop_columns() \
                             ._replace_translation() \
                             ._add_merger_field()

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

    def __new__(cls, path: str, config: dict, build_mode: str, cols_to_drop: list, \
            text_translation: dict = None):
        if path.endswith('.csv'):
            return DataBuilderCSV(path, config, build_mode, cols_to_drop, text_translation)
        else:
            raise ValueError('Unsupported data format.')
    

@dataclass
class Join:
    client_data_left: pd.DataFrame
    economic_data_right: pd.DataFrame
    joined_data = None

    def left_join(self) -> pd.DataFrame:
        self.joined_data = self.client_data_left.merge(self.economic_data_right, how='left', on=MERGER_FIELD)
        self.joined_data.drop([MERGER_FIELD], axis=1, inplace=True)
        #return joined_data

    def save(self, out_path: str):
        self.joined_data.to_csv(out_path)


if __name__ == '__main__':

    # read and clean both datasets
    client_builder = DataBuilderFactory(DATA_PATH,
                       config_client_data,
                       'production',
                       CLIENT_COLUMNS_TO_DROP,
                       ALL_CLIENT_DATA_TRANSLATION
                       )
    client_data = client_builder.preprocess_data().data
    
    economic_builder = DataBuilderFactory(ECO_DATA_PATH,
                                      config_eco_data,
                                      'production',
                                      ECO_COLUMNS_TO_DROP)
    economic_data = economic_builder.preprocess_data().data

    # join datasets and merge 
    join = Join(client_data, economic_data)
    join.left_join()
    join.save(PROCESSED_DATA_PATH)

