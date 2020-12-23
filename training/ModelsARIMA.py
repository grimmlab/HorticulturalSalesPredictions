import pmdarima
import pandas as pd
import numpy as np
import sklearn
import copy

from training import TrainHelper, ModelsBaseClass
from preparation import PreparationHelper


class ARIMA(ModelsBaseClass.BaseModel):
    """
    Class containing (S)ARIMA(X) model and methods
    """

    def __init__(self, target_column: str, order: tuple, seasonal_order: tuple, method: str = 'lbfgs',
                 use_exog: bool = False, with_intercept: bool = True, trend: str = None, log: bool = False,
                 power_transf: bool = False, one_step_ahead: bool = False):
        """
        :param target_column: target_column for prediction
        :param order: (p, d, q) of (S)ARIMA(X) model
        :param seasonal_order: (P, D, Q, m) of (S)ARIMA(X) model
        :param method: method to use for optimization
        :param use_exog: use exogenous input
        :param with_intercept: use intercept
        :param trend: trend component
        :param log: use log transform
        :param power_transf: use power transform
        :param one_step_ahead: perform one step ahead prediction
        """
        super().__init__(target_column=target_column, seasonal_periods=seasonal_order[3], name='(S)ARIMA(X)',
                         one_step_ahead=one_step_ahead)
        self.model = pmdarima.ARIMA(order=order, seasonal_order=seasonal_order, maxiter=50, disp=1, method=method,
                                    with_intercept=with_intercept, enforce_stationarity=False,
                                    suppress_warnings=True)
        self.use_exog = use_exog
        self.exog_cols_dropped = None
        self.trend = trend
        self.log = log
        self.power_transformer = sklearn.preprocessing.PowerTransformer() if power_transf else None
        self.contains_zeros = False

    def train(self, train: pd.DataFrame, cross_val_call: bool = False) -> dict:
        """
        Train (S)ARIMA(X) model
        :param train: train set
        :param cross_val_call: called to perform cross validation
        :return dictionary with cross validated scores (if specified)
        """
        cross_val_score_dict = {}
        if cross_val_call:
            cross_val_score_dict = self.get_cross_val_score(train=train)
        train_exog = None
        if (self.power_transformer is not None) or self.log:
            train = TrainHelper.get_transformed_set(dataset=train, target_column=self.target_column,
                                                    power_transformer=self.power_transformer, log=self.log)
        if self.use_exog:
            train_exog = train.drop(labels=[self.target_column], axis=1)
            self.exog_cols_dropped = train_exog.columns[train_exog.isna().any()].tolist()
            PreparationHelper.drop_columns(train_exog, self.exog_cols_dropped)
            train_exog = train_exog.to_numpy(dtype=float)
        self.model.fit(y=train[self.target_column], exogenous=train_exog, trend=self.trend)
        return cross_val_score_dict

    def insample(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Deliver (back-transformed) insample predictions
        :param train: train set
        :return: DataFrame with insample predictions
        """
        train_exog = None
        if self.use_exog:
            train_exog = train.drop(labels=[self.target_column], axis=1)
            PreparationHelper.drop_columns(train_exog, self.exog_cols_dropped)
            train_exog = train_exog.to_numpy(dtype=float)
        insample = pd.DataFrame(data=self.model.predict_in_sample(exogenous=train_exog), index=train.index,
                                columns=['Insample'])
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
        :return: DataFrame with predictions, upper and lower confidence level
        """
        test_exog = None
        if (self.power_transformer is not None) or self.log:
            test = TrainHelper.get_transformed_set(dataset=test, target_column=self.target_column,
                                                   power_transformer=self.power_transformer, log=self.log,
                                                   only_transform=True)
        if self.use_exog:
            test_exog = test.drop(labels=[self.target_column], axis=1)
            PreparationHelper.drop_columns(test_exog, self.exog_cols_dropped)
            test_exog = test_exog.to_numpy(dtype=float)
        if self.one_step_ahead:
            predict = []
            conf_low = []
            conf_up = []
            # deep copy model as predict function should not change class model
            model = copy.deepcopy(self.model)
            for i in range(0, test.shape[0]):
                if self.use_exog:
                    fc, conf = model.predict(n_periods=1, exogenous=pd.DataFrame(test_exog[i].reshape(1, -1)),
                                             return_conf_int=True, alpha=0.05)
                    model.update(test[self.target_column][i],
                                 exogenous=pd.DataFrame(test_exog[i].reshape(1, -1)))
                else:
                    fc, conf = model.predict(n_periods=1, return_conf_int=True, alpha=0.05)
                    model.update(test[self.target_column][i])
                predict.append(fc[0])
                conf_low.append(conf[0][0])
                conf_up.append(conf[0][1])
        else:
            predict, conf = self.model.predict(n_periods=test.shape[0], exogenous=test_exog,
                                               return_conf_int=True, alpha=0.05)
            conf_low = conf[:, 0]
            conf_up = conf[:, 1]
        predictions = pd.DataFrame({'Prediction': predict, 'LowerConf': conf_low, 'UpperConf': conf_up},
                                   index=test.index)

        if self.power_transformer is not None:
            predictions = pd.DataFrame({'Prediction': self.power_transformer.inverse_transform(
                predictions['Prediction'].values.reshape(-1, 1)).flatten(),
                                        'LowerConf': self.power_transformer.inverse_transform(
                                            predictions['LowerConf'].values.reshape(-1, 1)).flatten(),
                                        'UpperConf': self.power_transformer.inverse_transform(
                                            predictions['UpperConf'].values.reshape(-1, 1)).flatten()},
                                       index=predictions.index)
        if self.log:
            predict_backtr = np.exp(predictions['Prediction'])
            if self.contains_zeros:
                predict_backtr += 1
            lower_dist = ((predictions['Prediction'] - predictions['LowerConf'])
                          / predictions['Prediction']) * predict_backtr
            upper_dist = ((predictions['UpperConf'] - predictions['Prediction'])
                          / predictions['Prediction']) * predict_backtr
            predictions = pd.DataFrame({'Prediction': predict_backtr,
                                        'LowerConf': predict_backtr - lower_dist,
                                        'UpperConf': predict_backtr + upper_dist},
                                       index=predictions.index)
        return predictions
