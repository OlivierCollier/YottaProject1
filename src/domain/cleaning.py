import numpy as np
import pandas as pd
from warnings import simplefilter
from pandas.core.common import SettingWithCopyWarning
from sklearn.base import BaseEstimator, TransformerMixin

import src.config.column_names as col
import src.config.base as base

# Ignorer les warnings pour améliorer la lisibilité
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action="ignore", category=SettingWithCopyWarning)

def impute_missing_eco_data(eco_data: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in economic data using interpolation.

	Parameters
	----------
	eco_data: pd.Dataframe

	Returns
	-------
	complete_data: pd.Dataframe
	"""
    complete_data = eco_data.interpolate(limit_direction='both')
    return complete_data


def correct_wrong_entries(data: pd.DataFrame, corrections: dict) -> pd.DataFrame:
    """Correct erroneous entries in data.

	Parameters
	----------
	data: pd.DataFrame
	corrections: dict
		A dictionary whose keys are the columns to modify,
		and whose values are dictionaries characterizing the changes to be made.

	Returns
	-------
	corrected_data: pd.DataFrame
	"""
    replacements = {column: {value: np.nan} for column, value in corrections.items()}
    corrected_data = data.replace(replacements)
    return corrected_data


class MissingValueTreatment(BaseEstimator, TransformerMixin):
    """Data transformations to deal with missing values.

	When JOB_TYPE is provided, other related variables can be filled
	by the most common value among observations with same JOB_TYPE value.
	In other cases, the observation is removed.

	Attributes
	----------
	data: pd.DataFrame
	categorical_variables: list
		List of categorical variables.
	continuous_variables: list
		List of continuous variables.

	Methods
	-------
	transform(data)
	"""

    def __init__(self):
        self.data = None
        self.categorical_variables = None
        self.continuous_variables = None

    def fit(self, X: pd.DataFrame, y):
        cat_variables = X.select_dtypes(include='object').columns.tolist()
        cat_variables.remove(col.JOB_TYPE)
        self.categorical_variables = cat_variables
        self.continuous_variables = X.select_dtypes(include='number').columns.tolist()
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """Imputes JOB_TYPE when missing, then imputes other missing values from JOB_TYPE."""
        self.data = X
        self._impute_job_type()
        self._impute_from_job_type()
        return self.data

    def _imputation_from_age(self, age):
        if age < 25:
            return base.JOB_TYPE_TRANSLATION.get('Etudiant')
        elif age > 60:
            return base.JOB_TYPE_TRANSLATION.get('Retraité')
        else:
            return self.data['JOB_TYPE'].mode()[0]

    def _impute_job_type(self):
        """Imputes missing values of JOB_TYPE from AGE."""
        self.data.loc[pd.isna(self.data[col.JOB_TYPE]), col.JOB_TYPE] = \
            self.data.loc[pd.isna(self.data[col.JOB_TYPE]), col.AGE].map(lambda x: self._imputation_from_age(x))
        return self

    def _impute_from_job_type(self):
        """Imputes missing values using JOB_TYPE."""
        for column_name in self.categorical_variables:
            method = (lambda x: x.mode()[0])
            self._impute_single_column_from_job_type(column_name, method)
        for column_name in self.continuous_variables:
            method = (lambda x: x.median())
            self._impute_single_column_from_job_type(column_name, method)
        return self

    def _impute_single_column_from_job_type(self, column_name: str, method):
        """Imputes missing values for a single variable using JOB_TYPE."""
        relevant_observations = (~self.data[col.JOB_TYPE].isnull()) & (self.data[column_name].isnull())
        corresponding_job_types = self.data.loc[relevant_observations, col.JOB_TYPE]
        replacements = self.data[[col.JOB_TYPE, column_name]] \
            .groupby(col.JOB_TYPE) \
            .agg(method) \
            .to_dict()
        self.data.loc[relevant_observations, column_name] = \
            corresponding_job_types.replace(replacements[column_name])
        return self
