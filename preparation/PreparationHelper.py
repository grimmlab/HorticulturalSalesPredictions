import pandas as pd
import numpy as np
import datetime


def drop_columns(df: pd.DataFrame, columns: list):
    """
    Function dropping all columns specified
    :param df: dataset used for dropping
    :param columns: columns which should be dropped
    """
    df.drop(columns=columns, inplace=True)


def drop_rows_by_dates(df: pd.DataFrame, start: datetime.date, end: datetime.date):
    """
    Function dropping rows within specified dates
    :param df: dataset used for dropping
    :param start: start date for dropped period
    :param end: end date for dropped period
    """
    df.drop(pd.date_range(start=start, end=end).tolist(), inplace=True)


def get_one_hot_encoded_df(df: pd.DataFrame, columns_to_encode: list) -> pd.DataFrame:
    """
    Function delivering dataframe with specified columns one hot encoded
    :param df: dataset to use for encoding
    :param columns_to_encode: columns to encode
    :return: dataset with encoded columns
    """
    return pd.get_dummies(df, columns=columns_to_encode)


def custom_resampler(arraylike: pd.Series, summation_cols: list):
    """
    Custom resampling function when resampling frequency of dataset
    :param arraylike: Series to use for calculation
    :param summation_cols: Columns which should be applied summation
    :return: sum or mean of arraylike
    """
    if arraylike.name in summation_cols:
        return np.sum(arraylike)
    else:
        return np.mean(arraylike)
