
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats.mstats import winsorize
import datetime as dt


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


	"""
	Assumes that data has column "date" in format datetime.

	"""

	def __init__(self, data, quarterly_features):
		self.data = data
		self.quarterly_features = quarterly_features

	@property
	def monthly_features(self):
		column_names = self.data.columns
		quarterly_features = self.data.quarterly_data
		monthly_features = column_names.difference(quarterly_features).to_list()
		return monthly_features

	def clean(self):
		self._check_if_consecutive_months()
		self._impute_monthly_data()
		for column_name in self.quarterly_features:
			self._check_if_quarterly_data(column_name)
			self.data[column_name] = self._impute_quarterly_data(column_name)

	def _check_if_consecutive_months(self):
		month_number = self.data['date'].apply(lambda x: 12 * x.year + x.month).to_numpy()
		differences = np.diff(month_number)
		test = all([element == 1 for element in differences])
		if not test:
			raise ValueError('Months are not consecutive.')

	def _impute_monthly_data(self):
		self.data[self.monthly_features].interpolate(limit_direction='both', inplace=True)

	def _check_if_quarterly_data(self, column_name):

	def _impute_quarterly_data(self, column_name):



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
		relevant_observations = (~self.data['JOB_TYPE'].isnull()) & (self.data[column_name].isnull())
		corresponding_job_types = self.data.loc[relevant_observations, 'JOB_TYPE']
		replacements = self.data[['JOB_TYPE', column_name]] \
			.groupby('JOB_TYPE') \
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
	def _detect_start_of_quarter(self, column_name):
		changes = self.data[column_name].diff().nonzero()


	def __init__(self, column_names: list):
		self.data = None
		self.column_names = column_names

	def fit(self, data: pd.DataFrame):
		self.data = data
		return self

	def transform(self, data: pd.DataFrame) -> pd.DataFrame:
		"""Deal with outliers for given columns."""
		self._winsorize()
		return self.data

	def _winsorize(self):
		"""Winsorizes outliers."""
		for column_name in self.column_names:
			self.data[column_name] = winsorize(self.data[column_name], limits=[0, .01])
