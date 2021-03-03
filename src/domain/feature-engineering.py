
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, FunctionTransformer, OneHotEncoder, Binarizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalToBinaryTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, category):
        self.category = category

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        transformed = X.eq(self.category).astype(int)
        return transformed

AgeTransformer = FeatureUnion([
    ()
])


pipeline = ColumnTransformer([
    ('binary_transformer', OneHotEncoder(drop='first'), ['HAS_DEFAULT', 'HAS_HOUSING_LOAN', 'HAS_PERSO_LOAN']),
    ('categorical_simplifier1', CategoricalToBinaryTransformer('Célibataire'), ['STATUS']),
    ('categorical_simplifier2', CategoricalToBinaryTransformer('Succes'), ['RESULT_LAST_CAMPAIGN']),
    ('indicator_transformer', Binarizer(), ['BALANCE', 'NB_DAY_LAST_CONTACT']),
    ('inverse_transformer', FunctionTransformer(lambda x: 1/x), ['NB_DAY_LAST_CONTACT'])
]
)