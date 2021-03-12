
import pandas as pd
from warnings import simplefilter
from pandas.core.common import SettingWithCopyWarning

from src.domain.cleaning import impute_missing_eco_data, correct_wrong_entries
import src.config.base as base
import src.config.column_names as col

# Ignorer les warnings pour améliorer la lisibilité
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action="ignore", category=SettingWithCopyWarning)

# Global variables
MERGING_METHODS = ['left', 'right', 'outer', 'inner', 'cross']


class DataBuilder:
    """Loads data.

    Attributes
    ----------
    path: string
        Path to raw data.
    config: dict
        Dictionary containing configuration options, probably coming from a yaml file.
    text_translation: dict
        Dictionary encoding expected translations.
    data: pd.DataFrame
        Loaded data.

    Methods
    -------
    preprocess_data()
        Applies basic preprocessing treatments.
    transform(data_type)
        Applies basic preprocessing treatments, plus imputation for economic data
        or correction of erroneous entries for client data.
    """

    def __init__(self, path: str, config: dict, text_translation: dict = None):
        self.path = path
        self.config = config
        self.text_translation = text_translation
        self.data = self.read()

    def read(self):
        return NotImplementedError

    def _add_merger_field(self):
        """Reformats date field to have unique merge field for the join."""
        self.data[col.MERGER_FIELD] = self.data[self.config.get('date_column')].dt.strftime('%Y-%m')
        return self

    def _cast_types(self):
        """Casts columns to adequate types."""
        mapping_cols_types = self.config.get('cast_types')
        self.data = self.data.astype(mapping_cols_types)
        return self

    def _replace_translation(self):
        """Translates French terms to English."""
        if self.text_translation:
            self.data.replace(self.text_translation, inplace=True)
        return self
    
    def _drop_very_incomplete_rows(self):
        """Drops rows with many missing values (e.g. AGE and JOB_TYPE)."""
        missing_cols = self.config.get('filters').get('missing')
        if missing_cols:
            subset_df = self.data.loc[:, missing_cols]
            subset_bool = subset_df.isna().all(axis=1)
            missing_idx = subset_bool[subset_bool].index
            self.data = self.data.drop(index=missing_idx)
        return self

    def preprocess_data(self):
        """Runs preprocess tasks."""
        processed_data = self._cast_types()\
            ._replace_translation()\
            ._add_merger_field()\
            ._drop_very_incomplete_rows()
        return processed_data

    def transform(self, data_type: str) -> pd.DataFrame:
        """Runs preprocess tasks, plus imputation, with logs."""
        print(f'========== Processing {data_type} data ==========')
        print('- Casting types.')
        self._cast_types()
        print('- Translating French terms to English.')
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
    """Loads data in csv format.

    Methods
    -------
    read()
        Loads csv data from path.
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
    if statement in constructor.
    """

    def __new__(cls, path: str, config: dict, text_translation: dict = None):
        if path.endswith('.csv'):
            return DataBuilderCSV(path, config, text_translation)
        else:
            raise ValueError('Unsupported data format.')
    

class DataMerger:
    """Merges two datasets.

    Attributes
    ----------
    left_dataset: pd.DataFrame
    right_dataset: pd.DataFrame
    merge_field: string
        Name of columns used to merge.
    how: string
        Name of merging method.
    joined_datasets: pd.DataFrame

    Methods
    -------
    merge_datasets()
    save(out_path)
        Saves merged dataset in csv format.
    transform()
        Merges datasets and separates target from explanatory variables.
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

    def merge_datasets(self) -> None:
        """Merges both datasets."""
        self._drop_duplicate_columns()
        self.joined_datasets = self.left_dataset.merge(self.right_dataset, how=self.how, on=self.merge_field)
        self.joined_datasets.drop([col.MERGER_FIELD], axis=1, inplace=True)

    def _drop_duplicate_columns(self) -> None:
        """Drops columns that are present in both datasets."""
        columns_in_left_dataset = list(self.left_dataset.columns)
        columns_in_right_dataset = list(self.right_dataset.columns)
        columns_to_keep_in_right_dataset = [column_name for column_name in columns_in_right_dataset if column_name not in columns_in_left_dataset]\
                                           + [col.MERGER_FIELD]
        self.right_dataset = self.right_dataset[columns_to_keep_in_right_dataset]

    def save(self, out_path: str) -> None:
        """Saves the merged dataset."""
        self.joined_datasets.to_csv(out_path)

    def transform(self) -> (pd.DataFrame, pd.DataFrame):
        """Merges datasets with logs."""
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
    merged.save(base.PROCESSED_DATA_PATH)