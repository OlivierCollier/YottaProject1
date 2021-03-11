
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, Binarizer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
from category_encoders.target_encoder import TargetEncoder

import src.config.column_names as col


class ClipTransformer(BaseEstimator, TransformerMixin):
    """Transforms pandas dataframe by clipping values then scaling columns.

    Attributes
    ----------
    a_min: float
        Lower clip.
    a_max: float
        Upper clip.
    """

    def __init__(self, a_min: float, a_max: float):
        self.a_min = a_min
        self.a_max = a_max

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data_clipped = np.clip(X, self.a_min, self.a_max)
        data_clipped_scaled = data_clipped / (self.a_max - self.a_min)
        return data_clipped_scaled


class ExtractCategoryTransformer(BaseEstimator, TransformerMixin):
    """Transforms into indicator of a given value.

    Attributes
    ----------
    value: Any
    """

    def __init__(self, value):
        self.value = value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        indicator_is_value = X.eq(self.value).astype(int)
        return indicator_is_value


def age_transformer():
    """Returns transformer to apply to AGE.

    Returns
    -------
    transformer: sklearn.pipeline.FeatureUnion
        Transforms AGE using:
            - indicator that AGE is larger than 25,
            - indicator that AGE is larger than 60,
            - scaling with sklearn StandardScaler.
    """

    transformer = FeatureUnion([
        ('is-not-young-indicator', Binarizer(25)),
        ('is-old-indicator', Binarizer(60)),
        ('scaled', StandardScaler())
    ])
    return transformer


class LogicalOrTransformer(BaseEstimator, TransformerMixin):
    """Implements logical or for two columns."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        boolean = X.eq('Yes').astype(int)
        sum = boolean.sum(axis=1)
        prod = boolean.prod(axis=1)
        disjunction = pd.DataFrame(sum - prod)
        return disjunction


class DateTransformer(BaseEstimator, TransformerMixin):
    """Transforms DATE using target encoding of MONTH."""

    def __init__(self):
        self.encoder = None

    def fit(self, X, y=None):
        self.encoder = TargetEncoder().fit(X, y)
        return self

    ############
    def transform(self, X, y=None):
        month = X.apply(lambda x: x.apply(lambda y: str(y.month)))
        target_encoded_month = self.encoder.transform(month)
        return target_encoded_month
    ############

class NbDaysLastContactTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, value_to_replace, n_bins: int):
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
    eco_features = [col.EMPLOYMENT_VARIATION_RATE, col.IDX_CONSUMER_PRICE, col.IDX_CONSUMER_CONFIDENCE]

    feature_eng_transformer = ColumnTransformer([
        ('balance-clipper', ClipTransformer(a_min=-4000, a_max=4000), [col.ACCOUNT_BALANCE]),
        ('nb-clipper', ClipTransformer(a_min=0, a_max=15), [col.NB_CONTACTS_CURRENT_CAMPAIGN, col.NB_CONTACTS_BEFORE_CAMPAIGN]),
        ('one-hot-encoder', OneHotEncoder(drop='first'), one_hot_encoded_features),
        ('category-retired-extractor', ExtractCategoryTransformer('Retired'), [col.JOB_TYPE]),
        ('category-success-extractor', ExtractCategoryTransformer('Success'), [col.RESULT_LAST_CAMPAIGN]),
        ('category-single-extractor', ExtractCategoryTransformer('Single'), [col.MARITAL_STATUS]),
        ('age-transformer', age_transformer(), [col.AGE]),
        ('date-transformer', DateTransformer(), [col.LAST_CONTACT_DATE]),
        ('disjunction-transformer', LogicalOrTransformer(), [col.HAS_PERSO_LOAN, col.HAS_HOUSING_LOAN]),
        # ('nb-days-last-contact-transformer', NbDaysLastContactTransformer(-1, 4), [col.NB_DAYS_LAST_CONTACT])
        ('scaler', StandardScaler(), eco_features)
    ])

    return feature_eng_transformer
