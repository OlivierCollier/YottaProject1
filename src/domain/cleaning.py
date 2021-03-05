
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import src.config.column_names as col


def impute_missing_eco_data(eco_data: pd.DataFrame) -> pd.DataFrame:
	"""Impute missing values in economic data.

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
		A dictionary whose keys are the relevant columns to modify,
		and whose values are dictionaries characterizing the changes to be made.

	Returns
	-------
	corrected_data: pd.DataFrame
	"""
	corrected_data = data.replace(corrections)
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
		List of categorical variables having missing values.
	continuous_variables: list
		List of continuous variables having missing values.

	Methods
	-------
	fit(data)
	transform(data)

	"""

	def __init__(self, categorical_variables: list, continuous_variables: list):
		self.data = None
		self.categorical_variables = categorical_variables
		self.continuous_variables = continuous_variables

	def fit(self, data: pd.DataFrame):
		self.data = data
		return self

	def transform(self, data: pd.DataFrame) -> pd.DataFrame:
		"""Imputes missing values using JOB_TYPE then removes remaining ones."""
		self._impute_from_job_type()
		self._remove_remaining_rows_with_missing_values()
		return self.data

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

	def _remove_remaining_rows_with_missing_values(self):
		"""Removes remaining observations with missing values."""
		self.data.dropna(inplace=True)
		return self


class OutlierTreatment(BaseEstimator, TransformerMixin):
	"""Data transformations to deal with outliers.

	Attributes
	----------
	data: pd.DataFrame
	column_names: list

	Methods
	-------
	fit(data)
	transform(data)
	"""

	def __init__(self, column_names: list):
		self.data = None
		self.column_names = column_names

	def fit(self, data: pd.DataFrame):
		self.data = data
		return self

	def transform(self, data: pd.DataFrame, upper_clips: dict) -> pd.DataFrame:
		"""Deal with outliers for given columns."""
		self._clip(upper_clips)
		return self.data

	def _clip(self, upper_clips: dict):
		"""Clips outliers."""
		for column_name in self.column_names:
			self.data[column_name] = np.clip(self.data[column_name], a_min=0, a_max=upper_clips[column_name])
		return self
