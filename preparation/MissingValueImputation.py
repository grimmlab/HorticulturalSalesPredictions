import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
import sklearn.impute
import sklearn.compose


def get_simple_imputer(df: pd.DataFrame, strategy: str) -> sklearn.impute.SimpleImputer:
    """
    Get simple imputer for each column according to specified strategy
    :param df: DataFrame to impute
    :param strategy: strategy to use, e.g. 'mean' or 'median'
    :return: imputer
    """
    simple_imputer = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy=strategy).fit(X=df)
    return simple_imputer


def get_iter_imputer(df: pd.DataFrame, sample_posterior: bool = True, max_iter: int = 100,
                     min_value: int = 0, max_value: int = None) -> sklearn.impute.IterativeImputer:
    """
    Multivariate, iterative imputer fitted to df with specified parameters
    :param df: DataFrame to fit for imputation
    :param sample_posterior: sample from predictive posterior of fitted estimator (standard: BayesianRidge())
    :param max_iter: maximum number of iterations for imputation
    :param min_value: min value for imputation
    :param max_value: max value for imputation
    :return: imputer
    """
    iterative_imputer = sklearn.impute.IterativeImputer(sample_posterior=sample_posterior, max_iter=max_iter,
                                                        min_value=min_value, max_value=max_value,
                                                        random_state=0)
    iterative_imputer.fit(X=df)
    return iterative_imputer


def get_knn_imputer(df: pd.DataFrame, n_neighbors: int = 10) -> sklearn.impute.KNNImputer:
    """
    Imputer of missing values according to k-nearest neighbors in feature space
    :param df: DataFrame to use for imputation
    :param n_neighbors: number of neighbors to use for imputation
    :return: imputer
    """
    knn_imputer = sklearn.impute.KNNImputer(n_neighbors=n_neighbors)
    knn_imputer.fit(X=df)
    return knn_imputer
