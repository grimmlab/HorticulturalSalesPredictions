import numpy as np
import pandas as pd


def value_nan_zero(value) -> int:
    """
    Return value or 0 if value is nan
    :param value: value to check
    :return: value or 0
    """
    return 0 if np.isnan(value) else value


def set_dtypes(df: pd.DataFrame, cols_to_str: list = None):
    """
    Function setting dtypes of dataset. cols_to_str are converted to string, rest except date to float.
    Needed due to structure of raw file
    :param df: DataFrame whose columns data types should be set
    :param cols_to_str: Columns which should be converted to a string
    """
    if cols_to_str is None:
        cols_to_str = []
    for col in df.columns:
        if col in cols_to_str:
            df[col] = df[col].astype(dtype='string')
        elif col != 'Date':
            df[col] = df[col].astype(dtype='float')


def get_product_len_dict(dictionary: dict) -> int:
    """
    Calc product of lengths of values in dictionary
    :param dictionary: dictionary to use
    :return: product of lengths of all values
    """
    product_len = 1
    for value in dictionary.values():
        product_len *= len(value)
    return product_len


def get_duplicated_value_columns(df: pd.DataFrame) -> list:
    """
    Get column names if values are duplicated
    :param df: DataFrame to analyse
    :return: list of columns which are a duplicate
    """
    duplicate_column_names = set()
    for x in range(df.shape[1]):
        # iterate over all columns
        col = df.iloc[:, x]
        # compare with all index+1 columns and add if duplicate
        for y in range(x + 1, df.shape[1]):
            compare_col = df.iloc[:, y]
            if col.equals(compare_col):
                duplicate_column_names.add(df.columns.values[y])
    return list(duplicate_column_names)
