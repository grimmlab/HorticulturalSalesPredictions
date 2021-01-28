import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.preprocessing
import pandas as pd
import numpy as np
import copy

from training import ModelsBaseClass, TrainHelper


class AnnRegression(ModelsBaseClass.BaseModel):
    """Class containing ANN Regression Model and code for hyperparameter search"""
    def __init__(self, target_column: str, seasonal_periods: int, one_step_ahead: bool, n_feature: int,
                 n_hidden: int = 10, num_hidden_layer: int = 3, n_output: int = 1, dropout_rate: float = 0.0,
                 epochs: int = 5000, batch_size: int = 16, learning_rate: float = 1e-3, loss=nn.MSELoss(),
                 min_val_loss_improvement: float = 1000, max_epochs_wo_improvement: int = 100):
        """
        :param target_column: target_column for prediction
        :param seasonal_periods: period of seasonality
        :param one_step_ahead: perform one step ahead prediction
        :param n_feature: number of features for ANN input
        :param n_hidden: number of hidden neurons (first hidden layer)
        :param num_hidden_layer: number of hidden layers
        :param n_output: number of outputs
        :param dropout_rate: probability of element being zeroed in dropout layer
        :param epochs: number of epochs
        :param batch_size: size of a batch
        :param learning_rate: learning rate for optimizer
        :param loss: loss function to use
        :param min_val_loss_improvement: deviation validation loss to min_val_loss for being counted for early stopping
        :param max_epochs_wo_improvement: maximum number of epochs without improvement before early stopping
        """
        TrainHelper.init_pytorch_seeds()
        super().__init__(target_column=target_column, seasonal_periods=seasonal_periods,
                         name='ANN', one_step_ahead=one_step_ahead)
        self.model = ANN(n_feature=n_feature, n_hidden=n_hidden, num_hidden_layer=num_hidden_layer,
                         n_output=n_output, dropout_rate=dropout_rate)
        self.optimizer = 'adam'
        self.learning_rate = learning_rate
        self.loss = loss
        self.x_scaler = sklearn.preprocessing.StandardScaler()
        self.batch_size = batch_size
        self.epochs = epochs
        self.min_val_loss_improvement = min_val_loss_improvement
        self.max_epochs_wo_improvement = max_epochs_wo_improvement

    def train(self, train: pd.DataFrame, cross_val_call: bool = False) -> dict:
        """
        Train model
        :param train: train set
        :param cross_val_call: called to perform cross validation
        :return dictionary with cross validated scores (if specified)
        """
        TrainHelper.init_pytorch_seeds()
        cross_val_score_dict = {}
        if cross_val_call:
            cross_val_score_dict_ts, self.model = self.get_cross_val_score(train=train)
            cross_val_score_dict_shuf, self.model = self.get_cross_val_score(train=train, normal_cv=True)
            cross_val_score_dict = {**cross_val_score_dict_ts, **cross_val_score_dict_shuf}
        # create train and validation set
        train_loader, x_valid, y_valid = self.create_train_valid_sets(train=train)
        # run optim loop
        self.run_pytorch_optim_loop(train_loader=train_loader, x_valid=x_valid, y_valid=y_valid, model=self.model,
                                    checkpoint_name='ann_train')
        return cross_val_score_dict

    def update(self, train: pd.DataFrame, model):
        """
        Update existing model due to new samples
        :param train: train set with new samples
        :param model: model to update
        """
        TrainHelper.init_pytorch_seeds()
        train_loader, x_valid, y_valid = self.create_train_valid_sets(train=train)
        self.run_pytorch_optim_loop(train_loader=train_loader, x_valid=x_valid, y_valid=y_valid, model=model,
                                    checkpoint_name='ann_update')

    def insample(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Deliver  insample predictions
        :param train: train set
        :return: DataFrame with insample predictions
        """
        TrainHelper.init_pytorch_seeds()
        self.model.eval()
        # predict on cpu
        self.model.to(torch.device("cpu"))
        x_train = torch.tensor(data=self.x_scaler.transform(train.drop(self.target_column, axis=1)).astype(np.float32))
        insample = pd.DataFrame(data=self.model(x=x_train).data.numpy(),
                                index=train.index, columns=['Insample'])
        return insample

    def predict(self, test: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
        """
        Deliver, if specified one step ahead, out-of-sample predictions
        :param test: test set
        :param train: train set
        :return: DataFrame with predictions, upper and lower confidence level
        """
        TrainHelper.init_pytorch_seeds()
        x_test = torch.tensor(data=self.x_scaler.transform(test.drop(self.target_column, axis=1)).astype(np.float32))
        if self.one_step_ahead:
            train_manip = train.copy()
            predict_lst = []
            # deep copy model as predict function should not change class model
            model = copy.deepcopy(self.model)
            for i in range(0, test.shape[0]):
                model.eval()
                # predict on cpu
                model.to(torch.device("cpu"))
                fc = model(x=x_test[i].view(1, -1)).item()
                train_manip = train_manip.append(test.iloc[[i]])
                self.update(train=train_manip, model=model)
                predict_lst.append(fc)
            predict = np.array(predict_lst).flatten()
        else:
            self.model.eval()
            # predict on cpu
            self.model.to(torch.device("cpu"))
            predict = self.model(x=x_test).data.numpy().flatten()
        predictions = pd.DataFrame({'Prediction': predict}, index=test.index)
        return predictions

    def create_train_valid_sets(self, train: pd.DataFrame) -> tuple:
        """
        Create train and validation set respective train loader with batches
        :param train: train dataset
        :return: DataLoader with batches of train data as well as validation data
        """
        TrainHelper.init_pytorch_seeds()
        # create train and validation set
        valid_size = 0.2
        split_ind = int(train.shape[0] * (1 - valid_size))
        train_data = train.iloc[:split_ind]
        valid_data = train.iloc[split_ind:]
        # scale input data
        x_train = self.x_scaler.fit_transform(train_data.drop(self.target_column, axis=1))
        x_valid = self.x_scaler.transform(valid_data.drop(self.target_column, axis=1))
        # create train ready data
        x_train = torch.tensor(x_train.astype(np.float32))
        x_valid = torch.tensor(x_valid.astype(np.float32))
        y_train = torch.tensor(data=train_data[self.target_column].values.reshape(-1, 1).astype(np.float32))
        y_valid = torch.tensor(data=valid_data[self.target_column].values.reshape(-1, 1).astype(np.float32))
        train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(x_train, y_train),
                                                   batch_size=self.batch_size, shuffle=False, drop_last=False,
                                                   worker_init_fn=np.random.seed(0))
        return train_loader, x_valid, y_valid


class ANN(torch.nn.Module):
    """Class containing ANN Regression Model"""

    def __init__(self, n_feature: int, n_hidden: int, num_hidden_layer: int, n_output: int = 1,
                 dropout_rate: float = 0.0):
        """
        :param n_feature: number of features for ANN input
        :param n_hidden: number of hidden neurons (first hidden layer)
        :param num_hidden_layer: number of hidden layers
        :param n_output: number of outputs
        :param dropout_rate: probability of element being zeroed in dropout layer
        """
        super(ANN, self).__init__()
        TrainHelper.init_pytorch_seeds()
        self.hidden_layer = nn.ModuleList()
        hidden_in = n_feature
        hidden_out = n_hidden
        for layer_num in range(num_hidden_layer):
            self.hidden_layer.append(nn.Linear(in_features=hidden_in, out_features=hidden_out))
            hidden_in = hidden_out
            hidden_out = int(hidden_in / 2)
        self.output_layer = nn.Linear(in_features=hidden_in, out_features=n_output)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """
        Feedforward path
        :param x: data to process
        :return: prediction value
        """
        TrainHelper.init_pytorch_seeds()
        for layer in self.hidden_layer:
            x = F.relu(layer(x))
            x = self.dropout(x)
        out = self.output_layer(x)
        return out
