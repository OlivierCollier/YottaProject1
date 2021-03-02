
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalToBinaryTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, category):
        self.category = category

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        transformed = X.eq(self.category).astype(int)
        return transformed


class ToIndicatorTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, inequality_type, threshold):
        self.inequality_type = inequality_type
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.inequality_type == 'is_smaller':
            transformed = X.le(threshold).astype(int)
        elif self.inequality_type == 'is_larger':
            transformed = X.ge(threshold).astype(int)
        return transformed




binary_features = ['HAS_DEFAULT', 'HAS_HOUSING_LOAN', 'HAS_PERSO_LOAN']



processing_pipeline = ColumnTransformer(
    [

    ]

)