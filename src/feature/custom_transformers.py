import sys
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import pandas as pd
from pandas import Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

# Import local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import describe_as_df


class ElementwiseSummaryStats(BaseEstimator, TransformerMixin):
    """Constructs a transformer which computes the summary stats of each element of the input
    and concatenates the results into a single DataFrame.
    """

    def __init__(self, desc_kw_args=None):
        self.desc_kw_args = desc_kw_args

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data_dict = {}

        for name, col in X.items():
            stats = col.apply(lambda x: describe_as_df(x))
            data_dict[name] = (
                pd.concat(stats.to_dict(), axis=0).droplevel(1, axis=1).droplevel(1)
            )
            data_dict[name].drop(columns=["nobs"], axis=1, inplace=True)

        return pd.concat(data_dict, axis=1)

    def set_output(
        self, *, transform: None | Literal["default"] | Literal["pandas"] = None
    ) -> BaseEstimator:
        return self


class ElementwiseTransformer(FunctionTransformer):
    """Constructs a transformer which applies a given function to each element of the input.
    Expects the data to be an iterable of elements, where each element is a single sample.
    Expects the output be of a consistent shape for each element.
    """

    def _transform(self, X, func=None, kw_args=None):
        # Construct a vectorized version of the function
        vfunc = np.vectorize(lambda x, **kwargs: func(x, **kwargs), otypes=[np.ndarray])

        # Apply the vectorized function to the input
        res = super()._transform(X, func=vfunc, kw_args=kw_args)

        if isinstance(X, Series):
            res = Series(res)
            res.index = X.index

        return res


class LibrosaTransformer(BaseEstimator, TransformerMixin):
    """Constructs a transformer which applies a librosa function to each element of the input."""

    def __init__(self, feature: str = "chroma_stft", **kwargs):
        self.feature = feature  # TODO: Consider passing function instead of string
        self.kwargs = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        func = self._get_librosa_func(self.feature)

        if isinstance(X, pd.DataFrame):
            if X.shape[1] == 1:
                return self.transform(X.iloc[:, 0])

            x_dict = {k: self.transform(v) for k, v in X.items()}
            X = pd.concat(x_dict, axis=1, keys=x_dict.keys())
        elif isinstance(X, pd.Series):
            X = X.apply(lambda x: func(y=x, **self.kwargs))
            X = X.apply(lambda x: pd.Series(x.tolist())).rename(columns=lambda x: x + 1)
        else:
            X = np.array([func(y=x, **self.kwargs) for x in X])

        return X

    def _get_librosa_func(self, feature):
        try:
            func = getattr(librosa.feature, feature)
            return func
        except AttributeError:
            raise ValueError(
                f"The feature '{feature}' was not found in librosa.feature."
                # TODO: Try search in librosa module
            )

    def get_params(self, deep=True):
        params = {"feature": self.feature}
        params.update(self.kwargs)
        return params

    def set_params(self, **parameters):

        self.kwargs = {}
        for parameter, value in parameters.items():
            if parameter == "feature":
                self.feature = value
            else:
                self.kwargs[parameter] = value

        return self

    def set_output(
        self, *, transform: None | Literal["default"] | Literal["pandas"] = None
    ) -> BaseEstimator:
        return self


class WindowSelector(BaseEstimator, TransformerMixin):
    """Constructs a transformer which selects a window of a given size from the input."""

    def __init__(self, win_size: int = 2048):
        self.win_size = win_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            if self.win_size == "all":
                return X
            else:
                return X.loc[:, self.win_size]
        else:
            raise ValueError("Input must be a DataFrame.")

        return X
