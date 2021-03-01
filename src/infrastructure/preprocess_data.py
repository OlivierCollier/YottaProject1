import datetime as dt
import os
from dataclasses import dataclass

import ipdb
import pandas as pd
from src.config.base import *
from src.config.column_names import *
from src.config.config import read_yaml


BUILD_MODES = ['production', 'eda']


class DataBuilder:

    def __init__(self, path: str, date_column_name: str, date_format: str, sep: str,\
        config: dict, build_mode: str, cols_to_drop : list = None):
        self.path = path
        self.date_column_name = date_column_name
        self.date_format = date_format
        self.sep = sep
        self.config = config
        if build_mode not in BUILD_MODES:
            raise ValueError(f'Build mode should be one of {BUILD_MODES}')
        self.build_mode = build_mode
        self.cols_to_drop = cols_to_drop
        self.data = self.read()

    def read(self):
        return NotImplementedError

    def _add_merger_field(self, data: pd.DataFrame) -> pd.DataFrame:
        data[MERGER_FIELD] = data[self.date_column_name].dt.strftime('%Y-%m')
        # datetime = data['DATE'].apply(lambda x: dt.datetime.strptime(x, self.date_format))
        # year_month = datetime.apply(lambda x: f'{x.month}-{x.year}')
        # enriched_data = data.assign(YEAR_MONTH=year_month)
        return data

    def _cast_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Cast columns to adequate types"""
        mapping_cols_types = self.config.get('cast_types')
        casted_data = data.astype(mapping_cols_types)
        return casted_data

    def _drop_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Drop columns specified in column_names.py """
        if self.cols_to_drop == 'eda':
            data_cleaned = data.drop(columns=self.cols_to_drop)
            return data_cleaned
        else:
            return data

    @property
    def preprocess_data(self) -> pd.DataFrame:
        """ Run preprocess tasks """
        ipdb.set_trace()
        raw_data = self.read()
        types_casted_data = self._cast_types(raw_data)
        dropped_cols_data = self._drop_columns(types_casted_data)
        processed_data = self._add_merger_field(dropped_cols_data)
        return processed_data


class DataBuilderCSV(DataBuilder):
    """
    Data loader class for csv format
    """
    def read(self) -> pd.DataFrame:
        data = pd.read_csv(self.path, sep=self.sep)
        return data


class DataBuilderFactory:
    """
    DataBuilder factory class
    Supported formats are:
        - csv
    To support another format, create another DataBuilder<format> class and add 
    and if statement in this class
    """
    def __new__(cls, path: str, date_column_name: str, date_format: str, sep: str, \
        config: dict, build_mode: str, cols_to_drop: list):
        if path.endswith('.csv'):
            return DataBuilderCSV(path, date_column_name, date_format, sep, \
                config, build_mode, cols_to_drop)
        else:
            raise ValueError('Unsupported data format.')
    

@dataclass
class Join:
    data: pd.DataFrame
    external_month_info: pd.DataFrame

    @property
    def left_join(self) -> pd.DataFrame:
        joined_data = self.data.merge(self.external_month_info, how='left', on=MERGER_FIELD)
        joined_data.drop([MERGER_FIELD], axis=1, inplace=True)
        return joined_data

    def save(self, out_path: str):
        self.left_join.to_csv(out_path)


if __name__ == '__main__':

    # read and clean data
    client_data = DataBuilderFactory(DATA_PATH,
                       LAST_CONTACT_DATE,
                       DATA_DATE_FORMAT,
                       DATA_SEP,
                       config_client_data,
                       'production',
                       CLIENT_COLUMNS_TO_DROP).preprocess_data
    external_month_info = DataBuilderFactory(ECO_DATA_PATH,
                                      END_MONTH,
                                      ECO_DATA_DATE_FORMAT,
                                      ECO_DATA_SEP,
                                      config_eco_data,
                                      'production',
                                      ECO_COLUMNS_TO_DROP).preprocess_data

    # gather data and save
    join = Join(client_data, external_month_info)
    preprocessed_data = join.left_join
    join.save(PROCESSED_DATA_PATH)



