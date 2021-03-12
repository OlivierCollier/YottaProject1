
import numpy as np
import pandas as pd
from warnings import simplefilter
from pandas.core.common import SettingWithCopyWarning
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, Binarizer, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
from category_encoders.target_encoder import TargetEncoder

import src.config.column_names as col

# Ignorer les warnings pour améliorer la lisibilité
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action="ignore", category=SettingWithCopyWarning)


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
        self.month = None

    def fit(self, X, y=None):
        self.month = X.apply(lambda x: x.apply(lambda y: str(y.month)))
        self.encoder = TargetEncoder().fit(self.month, y)
        return self

    def transform(self, X, y=None):
        months = X.apply(lambda x: x.apply(lambda y: str(y.month)))
        target_encoded_month = self.encoder.transform(months)
        return target_encoded_month


class NbDaysLastContactTransformer(BaseEstimator, TransformerMixin):
    """Transforms NB_DAYS_LAST_CONTACT using binning."""

    def __init__(self, value_to_replace, n_bins: int):
        self.value_to_replace = value_to_replace
        self.n_bins = n_bins
        self.bins = None
        self.middles = None

    def fit(self, X, y=None):
        positive_values = X[X.ne(self.value_to_replace)]
        positive_values = pd.DataFrame(positive_values)
        positive_values.dropna(axis=0, inplace=True)
        positive_values = positive_values.to_numpy().reshape(1, -1).tolist()[0]
        interval_groups = pd.qcut(positive_values, self.n_bins)
        boundaries = set((interval_groups.map(lambda x: x.right).unique().tolist() + interval_groups.map(lambda x: x.left).unique().tolist()))
        boundaries = list(boundaries)
        boundaries.sort()
        boundaries[-1] += 1000
        self.bins = boundaries
        middles = (np.array(boundaries[:-1]) + np.array(boundaries[1:])) / 2
        self.middles = middles
        return self

    def transform(self, X):
        transformed = X.copy()
        transformed = transformed.replace({self.value_to_replace: max(self.bins)})
        transformed = transformed.to_numpy().reshape(1, -1).tolist()[0]
        binned = pd.cut(transformed, bins=self.bins, labels=self.middles)
        return pd.DataFrame(binned)


def feature_engineering_transformer():
    """Creates pipeline for feature engineering."""

    one_hot_encoded_features = [col.EDUCATION,
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
        ('nb-days-last-contact-transformer', NbDaysLastContactTransformer(value_to_replace=-1, n_bins=4), [col.NB_DAYS_LAST_CONTACT]),
        ('scaler', StandardScaler(), eco_features)
    ])

    return feature_eng_transformer
