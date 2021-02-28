
import numpy as np
import pandas as pd
import datetime as dt


class ClientDataCleaner:

	def __init__(self, data):
		self.data = data

	def clean(self):


	def _remove_incoherent_values(self, column_name, value):
		self.data[column_name].replace(to_replace=value, value=np.nan, inplace=True)

	def _impute_from_job_type(self, column_name, method):
		select_relevant_values = (~self.data['job_type'].isnull()) & (self.data[column_name].isnull())
		to_replace = self.data.loc[select_relevant_values, column_name]
		corresponding_job_types = self.data.loc[select_relevant_values, 'job_type']
		replacements = self.data[['job_type', column_name]]\
			.groupby('job_type')\
			.agg(method)\
			.to_dict()
		to_replace = corresponding_job_types.replace(replacements[column_name])

	def _impute_categorical_from_job_type(self, column_name):
		method = (lambda x: x.mode()[0])
		self._impute_from_job_type(column_name, method)

	def _impute_continuous_from_job_type(self):
		method = (lambda x: x.median())
		self._impute_from_job_type(column_name, method)

	def _deal_with_outliers(self, column_name, method):
		median = self.data[column_name].median()
		variance = self.data[column_name].var()
		outliers = abs(self.data[column_name]) > median + 2 * variance
		if method == 'drop':
			self.data = self.data[~outliers]
		elif method == 'clip':
			self.data[column_name].clip(lower=mean-2*variance, upper=mean+2*variance, inplace=True)
		else:
			raise ValueError(f'Method {method} not implemented')

	def _remove_remaining_rows_with_missing_values(self):
		self.data.dropna(inplace=True)

class EcoDataCleaner:

	"""
	Assumes that data has column "date" in format datetime.

	"""

	def __init__(self, data):#, quarterly_features):
		self.data = data
		#self.quarterly_features = quarterly_features

	#@property
	#def monthly_features(self):
		#		column_names = self.data.columns
		#quarterly_features = self.data.quarterly_data
		#monthly_features = column_names.difference(quarterly_features).to_list()
		#return monthly_features

	def clean(self):
		self._check_if_consecutive_months()
		self._impute_monthly_data()
		#for column_name in self.quarterly_features:
		#	self._impute_quarterly_data(column_name)

	def _check_if_consecutive_months(self):
		month_number = self.data['date'].apply(lambda x: 12 * x.year + x.month)
		differences = month_number.diff()
		test = all(differences.eq(1))
		if not test:
			raise ValueError('Months are not consecutive.')

	def _impute_monthly_data(self):
		self.data.interpolate(limit_direction='both', inplace=True)

	#def _impute_quarterly_data(self, column_name):
	#	length_of_first_quarter = self._detect_length_of_first_quarter(column_name)




	#def _detect_length_of_first_quarter(self, column_name):
	#	feature = self.data[column_name].reset_index(drop=True)
	#	for length_first_quarter in range(3):
	#		quarters = pd.Series(feature.index).apply(lambda x: (x-length_first_quarter) // 3)
	#		numbers_of_values_per_quarters = feature.groupby(quarters).nunique()
		#		test = all(numbers_of_values_per_quarters.isin([0, 1]))
		#	if test:
		#		return length_first_quarter
		#raise ValueError(f'Feature {column_name} is not quarterly.')






