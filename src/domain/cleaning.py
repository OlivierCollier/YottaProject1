
import numpy as np
import pandas as pd
import datetime as dt


class ClientDataCleaner:

	def __init__(self, data):
		self.data = data

	def _remove_





class EcoDataCleaner:

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



	def _detect_start_of_quarter(self, column_name):
		changes = self.data[column_name].diff().nonzero()





