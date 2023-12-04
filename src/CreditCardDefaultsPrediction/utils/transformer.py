# transformer.py

import numpy as np
import pandas as pd
from scipy import stats
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin


class UpperBoundCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Calculate the upper bounds during fitting
        self.upper_bounds_ = X.quantile(0.99)
        return self

    def transform(self, X):
        # This transformer doesn't modify the data, just returns it
        return X


class ClipTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        # The clip transformer doesn't require any training, so fit is just a pass-through
        return self

    def transform(self, X):
        # Clip the values in the DataFrame or Series using the upper bounds
        return X.clip(lower=None, upper=self.upper_bounds_, axis=1)


class PositiveTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No training needed, just return self
        return self

    def transform(self, X):
        # Ensure data is strictly positive
        return X + np.abs(X.min()) + 1  # Add 1 to avoid zero values
