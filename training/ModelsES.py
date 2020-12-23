import pandas as pd
import numpy as np
import statsmodels.tsa.api
import sklearn
import copy

from training import ModelsBaseClass, TrainHelper


class ExponentialSmoothing(ModelsBaseClass.BaseModel):
    """
    Class containing Exponential Smoothing model and methods
    """

    def __init__(self, target_column: str, trend: str = None, damped: bool = False, seasonal: str = None,
                 seasonal_periods: int = None, remove_bias: bool = False, use_brute: bool = False,
                 one_step_ahead: bool = False, power_transf: bool = False, log: bool = False):
        """
        :param target_column: target column for prediction
        :param trend: trend component 'add' | 'mul' | None
        :param damped: damping of trend
        :param seasonal: seasonal component 'add' | 'mul' | None
        :param seasonal_periods: number of periods in a seasonal cycle
        :param remove_bias: remove bias from forecast and fitted values forcing average residual equal 0
        :param use_brute: use brute force optimizer for starting values
        :param one_step_ahead: perform one step ahead prediction
        :param power_transf: use power transform
        :param log: use log transform
        """
        super().__init__(target_column=target_column, seasonal_periods=seasonal_periods, name='ExponentialSmoothing',
                         one_step_ahead=one_step_ahead)
        self.model = None
        self.model_results = None
        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.remove_bias = remove_bias
        self.use_brute = use_brute
        self.log = log
        self.power_transformer = sklearn.preprocessing.PowerTransformer() if power_transf else None
        self.contains_zeros = False

    def train(self, train: pd.DataFrame, cross_val_call: bool = False) -> dict:
        """
        Train Exponential Smoothing model
        :param train: train set
        :param cross_val_call: called to perform cross validation
        """
        cross_val_score_dict = {}
        if cross_val_call:
            cross_val_score_dict = self.get_cross_val_score(train=train)
        if (self.power_transformer is not None) or self.log:
            train = TrainHelper.get_transformed_set(dataset=train, target_column=self.target_column,
                                                    power_transformer=self.power_transformer, log=self.log)
        if (0 in train[self.target_column].values) and (self.trend == 'mul' or self.seasonal == 'mul'):
            # multiplicative trend or seasonal only working with strictly-positive data
            # only done if no transform was performed, otherwise values would need to be corrected a lot
            train = train.copy()
            train[self.target_column] += 0.01
        self.model = statsmodels.tsa.api.ExponentialSmoothing(endog=train[self.target_column], trend=self.trend,
                                                              damped=self.damped, seasonal=self.seasonal,
                                                              seasonal_periods=self.seasonal_periods)
        self.model_results = self.model.fit(remove_bias=self.remove_bias, use_brute=self.use_brute)
        return cross_val_score_dict

    def update(self, train: pd.DataFrame):
        """
        Update Exponential Smoothing model, e.g. for one step ahead, if applicable with already transformed data
        :param train: train set
        """
        if (0 in train[self.target_column].values) and (self.trend == 'mul' or self.seasonal == 'mul'):
            # multiplicative trend or seasonal only working with strictly-positive data
            # only done if no transform was performed, otherwise values would need to be corrected a lot
            train = train.copy()
            train[self.target_column] += 0.01
        model = statsmodels.tsa.api.ExponentialSmoothing(endog=train[self.target_column], trend=self.trend,
                                                         damped=self.damped, seasonal=self.seasonal,
                                                         seasonal_periods=self.seasonal_periods)
        model_results = model.fit(remove_bias=self.remove_bias, use_brute=self.use_brute)
        return model_results

    def insample(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Deliver (back-transformed) insample predictions
        :param train: train set
        :return: DataFrame with insample predictions
        """
        insample = pd.DataFrame(data=self.model_results.predict(start=train.index[0], end=train.index[-1]),
                                index=train.index, columns=['Insample'])
        if self.power_transformer is not None:
            insample = pd.DataFrame(data=self.power_transformer.inverse_transform(insample['Insample']
                                                                                  .values.reshape(-1, 1)),
                                    index=insample.index, columns=['Insample'])
        if self.log:
            if 0 in train[self.target_column].values:
                self.contains_zeros = True
                insample = np.exp(insample) - 1
            else:
                insample = np.exp(insample)
        return insample

    def predict(self, test: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
        """
        Deliver (back-transformed), if specified one step ahead, out-of-sample predictions
        :param test: test set
        :param train: train set
        :return: DataFrame with predictions
        """
        if (self.power_transformer is not None) or self.log:
            test = TrainHelper.get_transformed_set(dataset=test, target_column=self.target_column,
                                                   power_transformer=self.power_transformer, log=self.log,
                                                   only_transform=True)
            train = TrainHelper.get_transformed_set(dataset=train, target_column=self.target_column,
                                                    power_transformer=self.power_transformer, log=self.log)
        if self.one_step_ahead:
            train_manip = train.copy()[self.target_column]
            predict = []
            # deep copy model as predict function should not change class model
            model_results = copy.deepcopy(self.model_results)
            for ind in test.index:
                fc = model_results.forecast()
                predict.append(fc[ind])
                train_manip = train_manip.append(pd.Series(data=test[self.target_column], index=[ind]))
                model_results = self.update(train=pd.DataFrame(data=train_manip, columns=[self.target_column]))
        else:
            predict = self.model_results.predict(start=test.index[0], end=test.index[-1])
        predictions = pd.DataFrame({'Prediction': predict}, index=test.index)

        if self.power_transformer is not None:
            predictions = pd.DataFrame({'Prediction': self.power_transformer.inverse_transform(
                                            predictions['Prediction'].values.reshape(-1, 1)).flatten()},
                                       index=predictions.index)
        if self.log:
            if self.contains_zeros:
                predictions = predictions.apply(np.exp) + 1
            else:
                predictions = predictions.apply(np.exp)

        return predictions
