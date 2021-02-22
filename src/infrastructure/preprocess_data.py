
import pandas as pd
import datetime as dt
from dataclasses import dataclass
from config.config import read_yaml


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

    # read config file
    CONFIG = read_yaml('../config/config.yml')
    DATA_PATH = CONFIG['subscription']['path']
    DATA_DATE_COLUMN_NAME = CONFIG['subscription']['date_column_name']
    DATA_DATE_FORMAT = CONFIG['subscription']['date_format']
    DATA_SEP = CONFIG['subscription']['sep']
    ECO_DATA_PATH = CONFIG['economic_info']['path']
    ECO_DATA_DATE_COLUMN_NAME = CONFIG['economic_info']['date_column_name']
    ECO_DATA_DATE_FORMAT = CONFIG['economic_info']['date_format']
    ECO_DATA_SEP = CONFIG['economic_info']['sep']
    PREPROCESSING_DATA_PATH = CONFIG['out']['preprocessing']

    # read and clean data
    data = DataFactory(DATA_PATH,
                       DATA_DATE_COLUMN_NAME,
                       DATA_DATE_FORMAT,
                       DATA_SEP).clean_data
    external_month_info = DataFactory(ECO_DATA_PATH,
                                      ECO_DATA_DATE_COLUMN_NAME,
                                      ECO_DATA_DATE_FORMAT,
                                      ECO_DATA_SEP).clean_data

    # gather data and save
    join = Join(data, external_month_info)
    preprocessed_data = join.left_join
    join.save(PREPROCESSING_DATA_PATH)



