import numpy as np
import pandas as pd


def nan_checker(series: pd.Series):
    """Function checking if any value in series is nan"""
    return series.isna().values.any()


def drop_dummy_values(actual: pd.Series, prediction: pd.Series) -> tuple:
    """Function dropping indices with dummy values in prediction at both series"""
    indices = prediction.index[prediction == -9999]
    return actual.drop(indices), prediction.drop(indices)


def rmse(actual: pd.Series, prediction: pd.Series) -> float:
    """
    Function delivering Root Mean Squared Error between prediction and actual values
    :param actual: actual values
    :param prediction: prediction values
    :return: RMSE between prediciton and actual values
    """
    if nan_checker(actual) or nan_checker(prediction):
        raise NameError('Found NaNs - stopped calculation of evaluation metric')
    return np.mean((prediction - actual) ** 2) ** 0.5


def smape(actual: pd.Series, prediction: pd.Series) -> float:
    """
    Function delivering Symmetric Mean Absolute Percentage Error between prediction and actual values
    :param actual: actual values
    :param prediction: prediction values
    :return: sMAPE between prediction and actual values
    """
    if nan_checker(actual) or nan_checker(prediction):
        raise NameError('Found NaNs - stopped calculation of evaluation metric')
    return 100 / len(actual) * np.sum(np.abs(prediction - actual) / ((np.abs(actual) + np.abs(prediction)) / 2))


def mape(actual: pd.Series, prediction: pd.Series) -> float:
    """
    Function delivering Mean Absolute Percentage Error between prediction and actual values
    :param actual: actual values
    :param prediction: prediction values
    :return: MAPE between prediction and actual values
    """
    if nan_checker(actual) or nan_checker(prediction):
        raise NameError('Found NaNs - stopped calculation of evaluation metric')
    return np.mean(np.abs((actual - prediction) / (actual + 0.1))) * 100  # +0.1 to avoid div by zero


def get_all_eval_vals(actual: pd.Series, prediction: pd.Series) -> tuple:
    """
    Get all implemented eval vals (currently RMSE, MAPE, sMAPE) for handed over actual and prediction Series
    :param actual: actual values
    :param prediction: prediction values
    :return: RMSE, MAPE and sMAPE between prediction and actual values
    """
    actual, prediction = drop_dummy_values(actual=actual, prediction=prediction)
    return (rmse(actual=actual, prediction=prediction), mape(actual=actual, prediction=prediction),
            smape(actual=actual, prediction=prediction))
