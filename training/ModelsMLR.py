import sklearn
import pandas as pd
import numpy as np
import copy

from training import ModelsBaseClass


class MultipleLinearRegression(ModelsBaseClass.BaseModel):
    """Class containing Lasso, Ridge, ElasticNet, BayesianRidge and ARD Regression Model"""
    def __init__(self, model_to_use: str, target_column: str, seasonal_periods: int, one_step_ahead: bool = False,
                 normalize: bool = False,
                 alpha: float = 1.0, l1_ratio: float = 0.5,
                 alpha_1: float = 1e-6, alpha_2: float = 1e-6, lambda_1: float = 1e-6, lambda_2: float = 1e-6,
                 threshold_lambda: float = 10000):
        """
        ### Parameters relevant for all algorithms ###
        :param model_to_use: switch model (regularization) based on it ('lasso'|'ridge'|'elasticnet'|'bayesridge'|'ard')
        :param target_column: target_column for prediction
        :param seasonal_periods: period of seasonality
        :param one_step_ahead: perform one step ahead prediction
        :param normalize: normalize regressors X or not
        ### Parameters for Lasso, Ridge and Elastic Net Regression ###
        :param alpha: constant multiplying regularization term
        :param l1_ratio: ratio of l1-regularization term in case of elastic net regression
        ### Parameters for BayesianRidge and ARD Regression ###
        :param alpha_1: shape parameter Gamma distribution prior over alpha
        :param alpha_2: inverse scale parameter Gamma distribution prior over alpha
        :param lambda_1: shape parameter Gamma distribution prior over lambda
        :param lambda_2: inverse scale parameter Gamma distribution prior over lambda
        :param threshold_lambda: threshold for removing (pruning) weights with high precision from the computation
        """
        super().__init__(target_column=target_column, seasonal_periods=seasonal_periods,
                         name=model_to_use, one_step_ahead=one_step_ahead)
        if model_to_use == 'lasso':
            self.model = sklearn.linear_model.Lasso(
                alpha=alpha, normalize=normalize, fit_intercept=True, precompute=False, copy_X=True, max_iter=10000,
                tol=1e-4, warm_start=False, positive=False, random_state=42, selection='cyclic')
            self.probabilistic = False
        elif model_to_use == 'ridge':
            self.model = sklearn.linear_model.Ridge(
                alpha=alpha, normalize=normalize, fit_intercept=True, copy_X=True, max_iter=None, tol=1e-3,
                random_state=42, solver='auto')
            self.probabilistic = False
        elif model_to_use == 'elasticnet':
            self.model = sklearn.linear_model.ElasticNet(
                alpha=alpha, normalize=normalize, l1_ratio=l1_ratio, fit_intercept=True, precompute=False, copy_X=True,
                max_iter=10000, tol=1e-4, warm_start=False, positive=False, random_state=42, selection='cyclic')
            self.probabilistic = False
        elif model_to_use == 'bayesridge':
            self.model = sklearn.linear_model.BayesianRidge(
                n_iter=10000, tol=1e-3, alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2,
                alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=True, normalize=normalize,
                copy_X=True, verbose=False)
            self.probabilistic = True
        elif model_to_use == 'ard':
            self.model = sklearn.linear_model.ARDRegression(
                n_iter=1000, tol=1e-3, alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2,
                threshold_lambda=threshold_lambda, compute_score=False, fit_intercept=True, normalize=normalize,
                copy_X=True, verbose=False)
            # currently a bug in predict method of ARDRegression when using normalizing
            self.probabilistic = False if normalize is True else True

    def train(self, train: pd.DataFrame, cross_val_call: bool = False) -> dict:
        """
        Train model
        :param train: train set
        :param cross_val_call: called to perform cross validation
        :return dictionary with cross validated scores (if specified)
        """
        cross_val_score_dict = {}
        if cross_val_call:
            cross_val_score_dict = self.get_cross_val_score(train=train)
        self.model.fit(X=train.drop([self.target_column], axis=1), y=train[self.target_column])
        return cross_val_score_dict

    def update(self, train: pd.DataFrame, model):
        """
        Update existing model due to new samples
        :param train: train set with new samples
        :param model: model to update
        :return: updated model
        """
        return model.fit(X=train.drop([self.target_column], axis=1), y=train[self.target_column])

    def insample(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Deliver  insample predictions
        :param train: train set
        :return: DataFrame with insample predictions
        """
        insample = pd.DataFrame(data=self.model.predict(X=train.drop([self.target_column], axis=1)),
                                index=train.index, columns=['Insample'])
        return insample

    def predict(self, test: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
        """
        Deliver, if specified one step ahead, out-of-sample predictions
        :param test: test set
        :param train: train set
        :return: DataFrame with predictions, upper and lower confidence level
        """
        if self.one_step_ahead:
            train_manip = train.copy()
            predict_lst = []
            if self.probabilistic:
                sig_lst = []
            # deep copy model as predict function should not change class model
            model = copy.deepcopy(self.model)
            for i in range(0, test.shape[0]):
                if self.probabilistic:
                    fc, sigma = model.predict(
                        X=test.drop([self.target_column], axis=1).iloc[[i]].values.astype(np.float64), return_std=True)
                    sig_lst.append(sigma)
                else:
                    fc = model.predict(X=test.drop([self.target_column], axis=1).iloc[[i]].values.astype(np.float64))
                train_manip = train_manip.append(test.iloc[[i]])
                model = self.update(train=train_manip, model=model)
                predict_lst.append(fc)
            predict = np.array(predict_lst).flatten()
            if self.probabilistic:
                sig = np.array(sig_lst).flatten()
        else:
            if self.probabilistic:
                predict, sig = self.model.predict(X=test.drop([self.target_column], axis=1).values.astype(np.float64),
                                                  return_std=True)
            else:
                predict = self.model.predict(X=test.drop([self.target_column], axis=1).values.astype(np.float64))
        if self.probabilistic:
            predictions = pd.DataFrame({'Prediction': predict, 'LowerConf': predict - sig, 'UpperConf': predict + sig},
                                       index=test.index)
        else:
            predictions = pd.DataFrame({'Prediction': predict}, index=test.index)
        return predictions
