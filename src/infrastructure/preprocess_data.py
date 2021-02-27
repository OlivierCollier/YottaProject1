import os
import pandas as pd
import datetime as dt
from dataclasses import dataclass
from src.config.config import read_yaml
from src.config.base import *
from src.config.column_names import *


class Data:

    def __init__(self, path: str, date_column_name: str, date_format: str, sep: str):
        self.path = path
        self.date_column_name = date_column_name
        self.date_format = date_format
        self.sep = sep

    def read(self):
        return NotImplementedError

    def _rename_date_column(self, data: pd.DataFrame) -> pd.DataFrame:
        renamed_data = data.rename({self.date_column_name: 'DATE'})
        return renamed_data

    def _add_year_and_month_column(self, data: pd.DataFrame) -> pd.DataFrame:
        datetime = data['DATE'].apply(lambda x: dt.datetime.strptime(x, self.date_format))
        year_month = datetime.apply(lambda x: f'{x.month}-{x.year}')
        enriched_data = data.assign(YEAR_MONTH=year_month)
        return enriched_data

    @property
    def clean_data(self) -> pd.DataFrame:
        raw_data = self.read()
        renamed_data = self._rename_date_column(raw_data)
        cleaned_data = self._add_year_and_month_column(renamed_data)
        return cleaned_data


class DataCSV(Data):

    def read(self) -> pd.DataFrame:
        data = pd.read_csv(self.path, sep=self.sep)
        return data


class DataFactory:

    def __new__(cls, path: str, date_column_name: str, date_format: str, sep: str):
        if path.endswith('.csv'):
            return DataCSV(path, date_column_name, date_format, sep)
        else:
            raise ValueError('Unsupported data format.')


@dataclass
class Join:
    data: pd.DataFrame
    external_month_info: pd.DataFrame

    @property
    def left_join(self) -> pd.DataFrame:
        join = self.data.merge(self.external_month_info, how='left')
        join.drop(['YEAR_MONTH'], axis=1, inplace=True)
        return join

    def save(self, out_path: str):
        self.left_join.to_csv(out_path)


if __name__ == '__main__':

    # read and clean data
    data = DataFactory(DATA_PATH,
                       LAST_CONTACT_DATE,
                       DATA_DATE_FORMAT,
                       DATA_SEP).clean_data
    external_month_info = DataFactory(ECO_DATA_PATH,
                                      END_MONTH,
                                      ECO_DATA_DATE_FORMAT,
                                      ECO_DATA_SEP).clean_data

    # gather data and save
    join = Join(data, external_month_info)
    preprocessed_data = join.left_join
    join.save(PROCESSED_DATA_PATH)



