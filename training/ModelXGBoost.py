import xgboost as xgb
import pandas as pd
import numpy as np
import copy

from training import ModelsBaseClass


class XGBoostRegression(ModelsBaseClass.BaseModel):
    """Class containing XGBoost Regression Model"""
    def __init__(self, target_column: str, seasonal_periods: int, tree_meth: str = 'auto', learning_rate: float = 0.3,
                 max_depth: int = 6, subsample: float = 1, colsample_by_tree: float = 1, n_estimators: int = 100,
                 gamma: float = 0, alpha: int = 0, reg_lambda: int = 1, one_step_ahead: bool = False):
        """
        :param target_column: target_column for prediction
        :param seasonal_periods: seasonal periodicity
        :param tree_meth: tree_method to use
        :param learning_rate: boosting learning rate
        :param max_depth: maximum depth for base learners
        :param subsample: subsample ration of training instance
        :param colsample_by_tree: subsample ratio of columns for constructing each tree
        :param n_estimators: number of trees
        :param gamma: minimum loss reduction required to make a further partition on leaf node
        :param alpha: l1 regularization term
        :param reg_lambda: l2 regularization term
        :param one_step_ahead: perform one step ahead prediction
        """
        super().__init__(target_column=target_column, seasonal_periods=seasonal_periods, name='XGBoostRegression',
                         one_step_ahead=one_step_ahead)
        self.model = xgb.XGBRegressor(tree_method=tree_meth, objective='reg:squarederror', learning_rate=learning_rate,
                                      max_depth=max_depth, subsample=subsample, colsample_by_tree=colsample_by_tree,
                                      random_state=42, n_estimators=n_estimators, gamma=gamma, alpha=alpha,
                                      reg_lambda=reg_lambda, verbosity=0)

    def train(self, train: pd.DataFrame, cross_val_call: bool = False) -> dict:
        """
        Train XGB model
        :param train: train set
        :param cross_val_call: called to perform cross validation
        :return dictionary with cross validated scores (if specified)
        """
        cross_val_score_dict = {}
        if cross_val_call:
            cross_val_score_dict = self.get_cross_val_score(train=train)
        self.model.fit(X=train.drop([self.target_column], axis=1), y=train[self.target_column])
        return cross_val_score_dict

    def update(self, train: pd.DataFrame, model: xgb.XGBRegressor) -> xgb.XGBRegressor:
        """
        Update existing model due to new samples
        :param train: train set with new samples
        :param model: model to update
        :return: updated model
        """
        return model.fit(X=train.drop([self.target_column], axis=1), y=train[self.target_column])

    def insample(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Deliver (back-transformed) insample predictions
        :param train: train set
        :return: DataFrame with insample predictions
        """
        insample = pd.DataFrame(data=self.model.predict(data=train.drop([self.target_column], axis=1)),
                                index=train.index, columns=['Insample'])
        return insample

    def predict(self, test: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
        """
        Deliver (back-transformed), if specified one step ahead, out-of-sample predictions
        :param test: test set
        :param train: train set
        :return: DataFrame with predictions, upper and lower confidence level
        """
        if self.one_step_ahead:
            train_manip = train.copy()
            predict_lst = []
            # deep copy model as predict function should not change class model
            model = copy.deepcopy(self.model)
            for i in range(0, test.shape[0]):
                fc = model.predict(data=test.drop([self.target_column], axis=1).iloc[[i]])
                train_manip = train_manip.append(test.iloc[[i]])
                model = self.update(train=train_manip, model=model)
                predict_lst.append(fc)
            predict = np.array(predict_lst).flatten()
        else:
            predict = self.model.predict(data=test.drop([self.target_column], axis=1))
        predictions = pd.DataFrame({'Prediction': predict}, index=test.index)

        return predictions

    def plot_feature_importance(self, importance_type: str = 'weight'):
        """
        Plot feature importance for XGB Regressor
        :param importance_type: importance type to use
            ‘weight’: the number of times a feature is used to split the data across all trees.
            ‘gain’: the average gain across all splits the feature is used in.
            ‘cover’: the average coverage across all splits the feature is used in.
            ‘total_gain’: the total gain across all splits the feature is used in.
            ‘total_cover’: the total coverage across all splits the feature is used in.
        """
        feature_important = self.model.get_booster().get_score(importance_type=importance_type)
        keys = list(feature_important.keys())
        values = list(feature_important.values())

        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
        data.plot(kind='barh')
