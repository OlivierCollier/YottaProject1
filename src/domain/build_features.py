
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, Binarizer, KBinsDiscretizer
from sklearn.pipeline import Pipeline, FeatureUnion
from category_encoders.target_encoder import TargetEncoder

import src.config.column_names as col


def build_feature_engineering_pipeline():
    """Creates pipeline for feature engineering."""

    AgeTransformer = FeatureUnion([
        ('is_not_young', Binarizer(25)),
        ('is_old', Binarizer(60)),
        ('distance_from_40', FunctionTransformer(lambda x: abs(x - 40)))
    ])

    pipeline = ColumnTransformer([
        ('binary_transformer', FunctionTransformer(lambda x: x == 'Yes'), [col.HAS_DEFAULT, col.HAS_HOUSING_LOAN, col.HAS_PERSO_LOAN]),
        ('status_transformer', FunctionTransformer(lambda x: x == 'Single'), [col.MARITAL_STATUS]),
        ('result_transformer', FunctionTransformer(lambda x: x == 'Success'), [col.RESULT_LAST_CAMPAIGN]),
        ('indicator_transformer', Binarizer(0), [col.ACCOUNT_BALANCE, col.NB_DAYS_LAST_CONTACT]),
        ('inverse_transformer', FunctionTransformer(lambda x: 1/x), [col.NB_DAYS_LAST_CONTACT, col.NB_CONTACTS]),
        ('age_transformer', AgeTransformer, [col.AGE]),
        ('target_encoder', TargetEncoder(), [col.JOB_TYPE, col.EDUCATION]),
        ('identity', 'passthrough', [col.NB_CONTACTS])],
        remainder='drop')

    return pipeline
