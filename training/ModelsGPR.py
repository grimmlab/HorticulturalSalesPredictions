import copy
import gpflow
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn

from training import ModelsBaseClass


class GaussianProcessRegressionGPFlow(ModelsBaseClass.BaseModel):
    """Class containing Gaussian Process Regression Model based on gpflow lib"""
    def __init__(self, target_column: str, seasonal_periods: int, kernel: gpflow.kernels.Kernel = None,
                 mean_function: gpflow.mean_functions.MeanFunction = None, noise_variance: float = 1.0,
                 optimizer=gpflow.optimizers.Scipy(), standardize_x: bool = False, standardize_y: bool = False,
                 one_step_ahead: bool = False):
        """
        :param target_column: target_column for prediction
        :param kernel: kernel to use for GPR
        :param mean_function: mean function to use for GPR
        :param noise_variance: variance parameter for gaussian likelihood
        :param optimizer: optimizer to use for model fitting
        :param standardize_x: standardize X according to X_train
        :param standardize_y: standardize Y according to Y_train
        :param one_step_ahead: perform one step ahead prediction
        """
        super().__init__(target_column=target_column, seasonal_periods=seasonal_periods,
                         name='GaussianProcessRegression_gpflow',  one_step_ahead=one_step_ahead)
        self.model = gpflow.models.GPR(data=(np.zeros((5, 1)), np.zeros((5, 1))), kernel=kernel,
                                       mean_function=mean_function, noise_variance=noise_variance)
        self.optimizer = optimizer
        self.standardize_x = standardize_x
        self.standardize_y = standardize_y
        self.x_scaler = None
        self.y_scaler = None

    def train(self, train: pd.DataFrame, cross_val_call: bool = False) -> dict:
        """
        Train GPR model
        :param train: train set
        :param cross_val_call: called to perform cross validation
        :return dictionary with cross validated scores (if specified)
        """
        cross_val_score_dict = {}
        if cross_val_call:
            cross_val_score_dict_ts, self.model = self.get_cross_val_score(train=train)
            cross_val_score_dict_shuf, self.model = self.get_cross_val_score(train=train, normal_cv=True)
            cross_val_score_dict = {**cross_val_score_dict_ts, **cross_val_score_dict_shuf}
        x_train = train.drop(self.target_column, axis=1).values.reshape(-1, train.shape[1]-1)
        y_train = train[self.target_column].values.reshape(-1, 1)
        if self.standardize_x:
            self.x_scaler = sklearn.preprocessing.StandardScaler()
            x_train = self.x_scaler.fit_transform(x_train)
        if self.standardize_y:
            self.y_scaler = sklearn.preprocessing.StandardScaler()
            y_train = self.y_scaler.fit_transform(y_train)

        self.model.data = (tf.convert_to_tensor(value=x_train.astype(float), dtype=tf.float64),
                           tf.convert_to_tensor(value=y_train.astype(float), dtype=tf.float64))
        self.optimizer.minimize(self.model.training_loss, self.model.trainable_variables)
        return cross_val_score_dict

    def insample(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Deliver (back-transformed) insample predictions
        :param train: train set
        :return: DataFrame with insample predictions and variance
        """
        x_train = train.drop(self.target_column, axis=1).values.reshape(-1, train.shape[1] - 1)
        if self.standardize_x:
            x_train = self.x_scaler.transform(x_train)
        insample_mean, insample_var = self.model.predict_y(Xnew=tf.convert_to_tensor(value=x_train.astype(float),
                                                                                     dtype=tf.float64))
        insample_mean, insample_var = insample_mean.numpy(), insample_var.numpy()
        if self.standardize_y:
            insample_mean = self.y_scaler.inverse_transform(insample_mean)
            insample_var = self.y_scaler.inverse_transform(insample_var)
        insample = pd.DataFrame({'Insample': insample_mean.flatten(), 'Variance': insample_var.flatten()},
                                index=train.index)
        return insample

    def predict(self, test: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
        """
        Deliver (back-transformed), if specified one step ahead, out-of-sample predictions
        :param test: test set
        :param train: train set
        :return: DataFrame with predictions, variance, upper and lower confidence level
        """
        x_test = test.drop(self.target_column, axis=1).values.reshape(-1, test.shape[1]-1)
        if self.one_step_ahead:
            x_train_osa = train.drop(self.target_column, axis=1).values.reshape(-1, train.shape[1]-1)
            y_train_osa = train[self.target_column].values.reshape(-1, 1)
            y_test = test[self.target_column].values.reshape(-1, 1)
            predict_lst = []
            var_lst = []
            # deep copy scalers and model as predict function should not change class model and scalers
            x_scaler_osa = copy.deepcopy(self.x_scaler)
            y_scaler_osa = copy.deepcopy(self.y_scaler)
            model = gpflow.utilities.deepcopy(self.model)
            for i in range(0, x_test.shape[0]):
                sample = x_test[i].reshape(1, -1)
                if self.standardize_x:
                    sample = x_scaler_osa.transform(sample)
                mean, var = model.predict_y(Xnew=tf.convert_to_tensor(value=sample.astype(float), dtype=tf.float64))
                mean, var = mean.numpy(), var.numpy()
                x_train_osa = np.vstack((x_train_osa, sample))
                y_train_osa = np.vstack((y_train_osa, y_test[i]))
                if self.standardize_x:
                    x_train_osa = x_scaler_osa.fit_transform(x_train_osa)
                if self.standardize_y:
                    mean = y_scaler_osa.inverse_transform(mean)
                    var = y_scaler_osa.inverse_transform(var)
                    y_train_osa = y_scaler_osa.fit_transform(y_train_osa)
                predict_lst.append(mean[0][0])
                var_lst.append(var[0][0])
                model.data = (tf.convert_to_tensor(value=x_train_osa.astype(float), dtype=tf.float64),
                              tf.convert_to_tensor(value=y_train_osa.astype(float), dtype=tf.float64))
                self.optimizer.minimize(model.training_loss, model.trainable_variables)
            predict_mean = np.array(predict_lst)
            predict_var = np.array(var_lst)
        else:
            if self.standardize_x:
                x_test = self.x_scaler.transform(x_test)
            predict_mean, predict_var = self.model.predict_y(Xnew=tf.convert_to_tensor(value=x_test.astype(float),
                                                                                       dtype=tf.float64))
            predict_mean, predict_var = predict_mean.numpy(), predict_var.numpy()
            if self.standardize_y:
                predict_mean = self.y_scaler.inverse_transform(predict_mean)
                predict_var = self.y_scaler.inverse_transform(predict_var)
        predictions = pd.DataFrame({'Prediction': predict_mean.flatten(), 'Variance': predict_var.flatten(),
                                    'LowerConf': (predict_mean - 2 * np.sqrt(predict_var)).flatten(),
                                    'UpperConf': (predict_mean + 2 * np.sqrt(predict_var)).flatten()},
                                   index=test.index)
        return predictions
