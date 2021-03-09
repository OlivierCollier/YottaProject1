
from os import sendfile
from collections import defaultdict
import ipdb
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, Binarizer, KBinsDiscretizer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from category_encoders.target_encoder import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression


import src.config.column_names as col
from src.domain.cleaning import MissingValueTreatment


class CategoriesBinarizer(BaseEstimator, TransformerMixin):
    """
        Splits and binarizes all categorical values except one provided value on one hand,
        and the provided value on the other hand.
        
        Attributes
        ----------
        single_col_value: str
            Category value that is separated from all other category values

        Methods
        -------
        fit(X, y)
        transform(X, y)
    """

    def __init__(self, single_category: str) -> None:
        self.single_category = single_category

    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # single_value_match = {self.single_category: 1}
        # replacement = defaultdict(lambda: 0, single_value_match)
        # ipdb.set_trace()
        # X_replaced = X.map(replacement)

        for column in X.columns:
            X[column] = np.where(X[col].isin([self.single_category]), 1, 0)

        return X


def feature_engineering_transformer():
    """Creates pipeline for feature engineering."""

    AgeTransformer = FeatureUnion([
        ('is_not_young', Binarizer(25)),
        ('is_old', Binarizer(60)),
        ('distance_from_40', FunctionTransformer(lambda x: abs(x - 40)))
    ])

    transformer = ColumnTransformer([
        ('binary_transformer', FunctionTransformer(lambda x: (x == 'Yes')+0), [col.HAS_DEFAULT, col.HAS_HOUSING_LOAN, col.HAS_PERSO_LOAN]),
        ('status_transformer', FunctionTransformer(lambda x: (x == 'Single')+0), [col.MARITAL_STATUS]),
        ('result_transformer', FunctionTransformer(lambda x: (x == 'Success')+0), [col.RESULT_LAST_CAMPAIGN]),
        # ('bool_binarizer', CategoriesBinarizer('Yes'), [col.HAS_DEFAULT, col.HAS_HOUSING_LOAN, col.HAS_PERSO_LOAN]),
        # ('marital_binarizer', CategoriesBinarizer('Single'), [col.MARITAL_STATUS]),
        # ('result_binarizer', CategoriesBinarizer('Success'), [col.RESULT_LAST_CAMPAIGN]),
        ('indicator_transformer', Binarizer(0), [col.ACCOUNT_BALANCE, col.NB_DAY_LAST_CONTACT]),
        ('inverse_transformer', FunctionTransformer(lambda x: 1/x), [col.NB_DAY_LAST_CONTACT, col.NB_CONTACT_CURRENT_CAMPAIGN]),
        ('age_transformer', AgeTransformer, [col.AGE]),
        ('target_encoder', TargetEncoder(), [col.JOB_TYPE, col.EDUCATION]),
        ('identity', 'passthrough', [col.NB_CONTACT_CURRENT_CAMPAIGN])],
        remainder='drop')

    return transformer

