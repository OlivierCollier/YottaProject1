
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipdb
from scipy.sparse.dia import isspmatrix_dia

from sklearn.compose import ColumnTransformer
from sklearn.metrics import plot_precision_recall_curve, auc, precision_recall_curve
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, Binarizer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
from category_encoders.target_encoder import TargetEncoder

import src.config.column_names as col


class ClipTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, a_min, a_max):
        self.a_min = a_min
        self.a_max = a_max

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_clipped = np.clip(X, self.a_min, self.a_max)
        X_clipped_scaled = X_clipped / (self.a_max - self.a_min)
        return X_clipped_scaled


class ExtractCategoryTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, category):
        self.category = category

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_is_category = X.eq(self.category).astype(int)
        return X_is_category


def AgeTransformer():

    age_transformer = FeatureUnion([
        ('is-not-young-indicator', Binarizer(25)),
        ('is-old-indicator', Binarizer(60)),
        ('scaled', StandardScaler())
    ])

    return age_transformer


class LogicalUnionTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.eq('Yes').astype(int)
        sum = X.sum(axis=1)
        prod = X.prod(axis=1)
        return pd.DataFrame(sum - prod)


class DateTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.y = None

    def fit(self, X, y=None):
        self.encoder = TargetEncoder().fit(X, y)
        return self

    def transform(self, X):
        month = X.apply(lambda x: x.apply(lambda y: str(y.month)))
        target_encoded_month = self.encoder.transform(month)
        return target_encoded_month


class NbDaysLastContactTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, value_to_replace, n_bins):
        self.value_to_replace = value_to_replace
        self.n_bins = n_bins
        self.X = None

    def fit(self, X, y=None):
        X = X.replace({self.value_to_replace: 800})
        data_to_bin = X.to_numpy().reshape(1, -1).tolist()[0]
        groups = pd.qcut(data_to_bin, self.n_bins)
        self.bins = groups
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed = X_transformed.replace({self.value_to_replace: 800})
        # data_to_bin = X[X.ne(self.value_to_replace).to_numpy()]
        ipdb.set_trace()
        binned_X = X_transformed.groupby(self.bins).transform('mean')
        # X_transformed[X.eq(self.value_to_replace)] = binned_X.max()[0]
        # tmp = X_transformed[X.ne(self.value_to_replace)].dropna()
        # X_transformed[X.ne(self.value_to_replace)].dropna() = binned_X
        ipdb.set_trace()
        return binned_X


def feature_engineering_transformer():
    """Creates pipeline for feature engineering."""

    one_hot_encoded_features = [col.MARITAL_STATUS,
                                col.EDUCATION,
                                col.HAS_HOUSING_LOAN,
                                col.HAS_PERSO_LOAN,
                                col.HAS_DEFAULT]
                                #col.RESULT_LAST_CAMPAIGN]
    eco_features = [col.EMPLOYMENT_VARIATION_RATE, col.IDX_CONSUMER_PRICE, col.IDX_CONSUMER_CONFIDENCE]

    feature_eng_transformer = ColumnTransformer([
        ('balance-clipper', ClipTransformer(a_min=-4000, a_max=4000), [col.ACCOUNT_BALANCE]),
        ('nb-clipper', ClipTransformer(a_min=0, a_max=15), [col.NB_CONTACTS_CURRENT_CAMPAIGN, col.NB_CONTACTS_BEFORE_CAMPAIGN]),
        ('one-hot-encoder', OneHotEncoder(drop='first'), one_hot_encoded_features),
        ('category-retired-extracter', ExtractCategoryTransformer('Retired'), [col.JOB_TYPE]),
        ('category-success-extracter', ExtractCategoryTransformer('Success'), [col.RESULT_LAST_CAMPAIGN]),
        ('category-single-extracter', ExtractCategoryTransformer('Single'), [col.MARITAL_STATUS]),
        ('age-transformer', AgeTransformer(), [col.AGE]),
        ('date-transformer', DateTransformer(), [col.LAST_CONTACT_DATE]),
        ('sum-transformer', LogicalUnionTransformer(), [col.HAS_PERSO_LOAN, col.HAS_HOUSING_LOAN]),
        # ('nb-days-last-contact-transformer', NbDaysLastContactTransformer(-1, 4), [col.NB_DAYS_LAST_CONTACT])
        ('identity', StandardScaler(), eco_features)
    ])

    return feature_eng_transformer
