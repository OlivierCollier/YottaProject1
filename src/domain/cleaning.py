
from inspect import indentsize
import ipdb
import numpy as np
import pandas as pd
import src.config.column_names as col
import src.config.base as base
from sklearn.base import BaseEstimator, TransformerMixin


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
	to_replace = {k: {v: np.nan} for k, v in corrections.items()}

	corrected_data = data.replace(to_replace)
	
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

	def __init__(self):
		self.data = None
		self.y = None
		self.categorical_variables = None
		self.continuous_variables = None


	def fit(self, data: pd.DataFrame, y):
		self.data = data
		# self.y = y
		cat_variables = data.select_dtypes(include='object').columns.tolist()
		cat_variables.remove(col.JOB_TYPE)
		self.categorical_variables = cat_variables
		
		self.continuous_variables = data.select_dtypes(include='number').columns.tolist()

		return self


	def transform(self, data: pd.DataFrame, y=None) -> pd.DataFrame:
		"""Imputes missing values using JOB_TYPE then removes remaining ones."""
		self.data = data
		self._impute_job_type()
		self._impute_from_job_type()
		return self.data


	def _impute_job_type(self):
		"""Imputes missing values of JOB_TYPE column given AGE value"""
		def _age_imputation(age):
			if age < 25:
				return base.JOB_TYPE_TRANSLATION.get('Etudiant')
			elif age > 60:
				return base.JOB_TYPE_TRANSLATION.get('Retraité')
			else:
				return self.data['JOB_TYPE'].mode()[0]
		
		self.data.loc[pd.isna(self.data[col.JOB_TYPE]),col.JOB_TYPE] = \
			self.data.loc[pd.isna(self.data[col.JOB_TYPE]),col.AGE].map(lambda x: _age_imputation(x))
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


