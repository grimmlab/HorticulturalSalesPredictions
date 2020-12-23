import pandas as pd
import numpy as np


class SimpleBaseline:
    """
    Base Class for SimpleBaselines with init and methods to implement
    """
    def __init__(self, one_step_ahead: bool = False):
        """
        :param one_step_ahead: Determine if prediction should be done in one step ahead fashion
        """
        self.one_step_ahead = one_step_ahead

    def get_insample_prediction(self, train: pd.DataFrame, test: pd.DataFrame, target_column: str, **kwargs) -> tuple:
        """
        Function for retrieving insample and prediction DataFrames
        :param train: training set to use
        :param test: test set to use
        :param target_column: target_column to predict
        """
        raise NotImplementedError


class RandomWalk(SimpleBaseline):
    """
    Random Walk / Seasonal Random Walk(predict last known value) as SimpleBaseline
    """

    def get_insample_prediction(self, train: pd.DataFrame, test: pd.DataFrame, target_column: str, **kwargs) -> tuple:
        """
        Create insample and prediction DataFrames
        :param train: training set to use
        :param test: test set to use
        :param target_column: target_column to predict
        :param kwargs: seasonal_period if performing seasonal random walk
        :return: return insample and prediction DataFrame
        """
        if 'seasonal_periods' in kwargs:
            shift_param = kwargs.get('seasonal_periods')
        else:
            shift_param = 1
        # shift and fillna (values before shift_param is reached) with dummy value
        ds_shifted = train.append(test)[target_column].shift(shift_param).fillna(-9999)
        insample = pd.DataFrame(ds_shifted[train.index].values, index=train.index, columns=['Insample'])
        if self.one_step_ahead:
            prediction = pd.DataFrame(ds_shifted[test.index].values, index=test.index, columns=['Prediction'])
        else:
            if shift_param == 1:
                # naive forecast: take last train value as forecast
                prediction = pd.DataFrame(data=train[target_column].tail(shift_param).values[0],
                                          index=test.index, columns=['Prediction'])
            else:
                # seasonal naive forecast: take last season train values as forecasts
                last_season = train[target_column].tail(shift_param).values
                # concat last season to match length of test set
                while len(last_season) < test.shape[0]:
                    last_season = np.append(last_season, last_season)
                prediction = pd.DataFrame(data=last_season[0:test.shape[0]], index=test.index, columns=['Prediction'])
        return insample, prediction


class HistoricalAverage(SimpleBaseline):
    """
    Historical Average (Average of all known values) as SimpleBaseline
    """
    def get_insample_prediction(self, train: pd.DataFrame, test: pd.DataFrame, target_column: str, **kwargs) -> tuple:
        """
        Get insample and prediction DataFrames
        :param train: training set to use
        :param test: test set to use
        :param target_column: target_column to predict
        :return: return insample and prediction DataFrame
        """
        # use average of train set for insample prediction (insample -> knowledge of whole train set)
        insample = pd.DataFrame(train[target_column].mean(), index=train.index, columns=['Insample'])
        if self.one_step_ahead:
            preds = []
            vals = train[target_column].tolist()
            for ind in test[target_column].index:
                preds.append(sum(vals) / len(vals))
                vals.append(test[target_column][ind])
            prediction = pd.DataFrame(preds, index=test.index, columns=['Prediction'])
        else:
            prediction = pd.DataFrame(train[target_column].mean(), index=test.index, columns=['Prediction'])
        return insample, prediction
