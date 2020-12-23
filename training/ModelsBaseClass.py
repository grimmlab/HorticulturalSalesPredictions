import os
import pandas as pd
import numpy as np
import sklearn
import torch
import datetime

from evaluation import EvaluationHelper, SimpleBaselines
from training import TrainHelper


class BaseModel:
    """
    Class containing Base model and methods
    """

    def __init__(self, target_column: str, seasonal_periods: int, name: str, one_step_ahead: bool = False):
        self.target_column = target_column
        self.one_step_ahead = one_step_ahead
        self.seasonal_periods = seasonal_periods
        self.name = name

    def train(self, train: pd.DataFrame, cross_val_call: bool = False) -> dict:
        """
        Train model
        :param train: train set
        :param cross_val_call: called to perform cross validation
        :return dictionary with cross validated scores (if specified)
        """
        raise NotImplementedError

    def get_cross_val_score(self, train: pd.DataFrame) -> dict:
        """
        Deliver cross validated evaluation scores
        :param train: train set
        :return: dictionary with mean and std of cross validated evaluation scores
        """
        if train.shape[0] < 80:
            print('Train set too small for Cross Validation')
            return {}
        train = train.copy()
        rmse_lst, mape_lst, smape_lst = [], [], []
        tscv = sklearn.model_selection.TimeSeriesSplit(n_splits=3)
        for train_index, test_index in tscv.split(train):
            cv_train, cv_test = train.loc[train.index[train_index]], train.loc[train.index[test_index]]
            # ES Model with seasonality is only working if n_samples is bigger than seasonality
            if self.name == 'ExponentialSmoothing' and self.seasonal is not None:
                if cv_train.shape[0] <= self.seasonal_periods:
                    print('CV train set too small for seasonality')
                    return {}
            self.train(train=cv_train)
            predictions = self.predict(test=cv_test, train=cv_train)
            rmse_test, mape_test, smape_test = EvaluationHelper.get_all_eval_vals(
                actual=cv_test[self.target_column], prediction=predictions['Prediction'])
            rmse_lst.append(rmse_test)
            mape_lst.append(mape_test)
            smape_lst.append(smape_test)
        rmse_mean, mape_mean, smape_mean = \
            np.mean(np.asarray(rmse_lst)), np.mean(np.asarray(mape_lst)), np.mean(np.asarray(smape_lst))
        rmse_std, mape_std, smape_std = \
            np.std(np.asarray(rmse_lst)), np.std(np.asarray(mape_lst)), np.std(np.asarray(smape_lst))
        return {'cv_rmse_mean': rmse_mean, 'cv_rmse_std': rmse_std,
                'cv_mape_mean': mape_mean, 'cv_mape_std': mape_std,
                'cv_smape_mean': smape_mean, 'cv_smape_std': smape_std}

    def insample(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Deliver (back-transformed) insample predictions
        :param train: train set
        :return: DataFrame with insample predictions
        """
        raise NotImplementedError

    def predict(self, test: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
        """
        Deliver (back-transformed), if specified one step ahead, out-of-sample predictions
        :param test: test set
        :param train: train set
        :return: DataFrame with predictions, upper and lower confidence level
        """
        raise NotImplementedError

    def evaluate(self, train: pd.DataFrame, test: pd.DataFrame) -> dict:
        """
        Evaluate model against all implemented evaluation metrics and baseline methods.
        Deliver dictionary with evaluation metrics.
        :param train: train set
        :param test: test set
        :return: dictionary with evaluation metrics of model and all baseline methods
        """
        TrainHelper.init_pytorch_seeds()
        insample_rw, prediction_rw = SimpleBaselines.RandomWalk(one_step_ahead=self.one_step_ahead)\
            .get_insample_prediction(train=train, test=test, target_column=self.target_column)
        insample_seasrw, prediction_seasrw = SimpleBaselines.RandomWalk(one_step_ahead=self.one_step_ahead)\
            .get_insample_prediction(train=train, test=test, target_column=self.target_column,
                                     seasonal_periods=self.seasonal_periods)
        insample_ha, prediction_ha = SimpleBaselines.HistoricalAverage(one_step_ahead=self.one_step_ahead)\
            .get_insample_prediction(train=train, test=test, target_column=self.target_column)
        insample_model = self.insample(train=train)
        prediction_model = self.predict(test=test, train=train)
        rmse_train_rw, mape_train_rw, smape_train_rw = EvaluationHelper.get_all_eval_vals(
            actual=train[self.target_column], prediction=insample_rw['Insample'])
        rmse_test_rw, mape_test_rw, smape_test_rw = EvaluationHelper.get_all_eval_vals(
            actual=test[self.target_column], prediction=prediction_rw['Prediction'])
        rmse_train_seasrw, mape_train_seasrw, smape_train_seasrw = EvaluationHelper.get_all_eval_vals(
            actual=train[self.target_column], prediction=insample_seasrw['Insample'])
        rmse_test_seasrw, mape_test_seasrw, smape_test_seasrw = EvaluationHelper.get_all_eval_vals(
            actual=test[self.target_column], prediction=prediction_seasrw['Prediction'])
        rmse_train_ha, mape_train_ha, smape_train_ha = EvaluationHelper.get_all_eval_vals(
            actual=train[self.target_column], prediction=insample_ha['Insample'])
        rmse_test_ha, mape_test_ha, smape_test_ha = EvaluationHelper.get_all_eval_vals(
            actual=test[self.target_column], prediction=prediction_ha['Prediction'])
        rmse_train_model, mape_train_model, smape_train_model = EvaluationHelper.get_all_eval_vals(
            actual=train[self.target_column], prediction=insample_model['Insample'])
        rmse_test_model, mape_test_model, smape_test_model = EvaluationHelper.get_all_eval_vals(
            actual=test[self.target_column], prediction=prediction_model['Prediction'])
        return {'RMSE_Train_RW': rmse_train_rw, 'MAPE_Train_RW': mape_train_rw, 'sMAPE_Train_RW': smape_train_rw,
                'RMSE_Test_RW': rmse_test_rw, 'MAPE_Test_RW': mape_test_rw, 'sMAPE_Test_RW': smape_test_rw,
                'RMSE_Train_seasRW': rmse_train_seasrw, 'MAPE_Train_seasRW': mape_train_seasrw,
                'sMAPE_Train_seasRW': smape_train_seasrw,
                'RMSE_Test_seasRW': rmse_test_seasrw, 'MAPE_Test_seasRW': mape_test_seasrw,
                'sMAPE_Test_seasRW': smape_test_seasrw,
                'RMSE_Train_HA': rmse_train_ha, 'MAPE_Train_HA': mape_train_ha, 'sMAPE_Train_HA': smape_train_ha,
                'RMSE_Test_HA': rmse_test_ha, 'MAPE_Test_HA': mape_test_ha, 'sMAPE_Test_HA': smape_test_ha,
                'RMSE_Train': rmse_train_model, 'MAPE_Train': mape_train_model, 'sMAPE_Train': smape_train_model,
                'RMSE_Test': rmse_test_model, 'MAPE_Test': mape_test_model, 'sMAPE_Test': smape_test_model
                }

    def run_pytorch_optim_loop(self, train_loader, x_valid, y_valid, model, checkpoint_name: str = 'train'):
        """
        Optimization of hyperparameters
        :param train_loader: DataLoader with sequenced train batches
        :param x_valid: sequenced validation data
        :param y_valid: validation labels
        :param model: model to optimize
        :param checkpoint_name: save name for best checkpoints
        :return:
        """
        TrainHelper.init_pytorch_seeds()
        # name for checkpoint for temporary storing during optimization with early stopping
        # detailed timestamp to prevent interference with parallel running jobs using same directory
        checkpoint_name += '_' + datetime.datetime.now().strftime("%d-%b-%Y_%H-%M-%S-%f")
        min_valid_loss = 99999999
        epochs_wo_improvement_threshold = 0
        epochs_wo_improvement_total = 0
        # instantiate new optimizer to ensure independence of previous runs
        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.learning_rate)
        # get device and shift model and data to it
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        x_valid, y_valid = x_valid.to(device), y_valid.to(device)
        for e in range(self.epochs):
            model.train()
            for (batch_x, batch_y) in train_loader:
                TrainHelper.init_pytorch_seeds()
                # copy data to device
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                # gradients are summed up so they need to be zeroed for new run
                optimizer.zero_grad()
                y_pred = model(batch_x)
                loss_train = self.loss(y_pred, batch_y)
                loss_train.backward()
                optimizer.step()
            model.eval()
            y_pred_valid = model(x_valid)
            loss_valid = self.loss(y_pred_valid, y_valid).item()
            if loss_valid < min_valid_loss:
                min_valid_loss = loss_valid
                epochs_wo_improvement_threshold = 0
                epochs_wo_improvement_total = 0
                torch.save(model.state_dict(), 'Checkpoints/checkpoint_' + checkpoint_name + '.pt')
            elif (loss_valid - min_valid_loss) > self.min_val_loss_improvement:
                # Early Stopping with thresholds for counter incrementing and max_epochs
                epochs_wo_improvement_threshold += 1
                if epochs_wo_improvement_threshold > self.max_epochs_wo_improvement:
                    print('Early Stopping after epoch ' + str(e))
                    break
            elif loss_valid >= min_valid_loss:
                # Early stopping if there is no improvement with a higher threshold
                epochs_wo_improvement_total += 1
                if epochs_wo_improvement_total > 2 * self.max_epochs_wo_improvement:
                    print('Early Stopping after epoch ' + str(e))
                    break
            if e % 100 == 0:
                print('Epoch ' + str(e) + ': valid loss = ' + str(loss_valid)
                      + ', min_valid_loss = ' + str(min_valid_loss))
        model.load_state_dict(state_dict=torch.load('Checkpoints/checkpoint_' + checkpoint_name + '.pt'))
        os.remove('Checkpoints/checkpoint_' + checkpoint_name + '.pt')
