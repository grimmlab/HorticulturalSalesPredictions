import torch
import torch.nn as nn
import sklearn.preprocessing
import pandas as pd
import numpy as np
import copy

from training import ModelsBaseClass, TrainHelper


class LstmRegression(ModelsBaseClass.BaseModel):
    """Class containing LSTM Regression Model and code for hyperparameter search"""
    def __init__(self, target_column: str, seasonal_periods: int, one_step_ahead: bool,
                 n_feature: int, lstm_hidden_dim: int = 10, lstm_num_layers: int = 1, seq_length: int = 7,
                 n_output: int = 1, dropout_rate: float = 0.0, epochs: int = 5000, batch_size: int = 16,
                 learning_rate: float = 1e-3, loss=nn.MSELoss(), min_val_loss_improvement: float = 1000,
                 max_epochs_wo_improvement: int = 100):
        """
        :param target_column: target_column for prediction
        :param seasonal_periods: period of seasonality
        :param one_step_ahead: perform one step ahead prediction
        :param n_feature: number of features for ANN input
        :param lstm_hidden_dim: dimensionality of hidden layer
        :param lstm_num_layers: depth of lstm network
        :param seq_length: sequence length for input of lstm network
        :param n_output: number of outputs
        :param dropout_rate: probability of element being zeroed in dropout layer
        :param epochs: number of epochs
        :param batch_size: size of a batch
        :param learning_rate: learning rate for optimizer
        :param loss: loss function to use
        :param min_val_loss_improvement: deviation validation loss to min_val_loss for being counted for early stopping
        :param max_epochs_wo_improvement: maximum number of epochs without improvement before early stopping
        """
        super().__init__(target_column=target_column, seasonal_periods=seasonal_periods,
                         name='LSTM', one_step_ahead=one_step_ahead)
        TrainHelper.init_pytorch_seeds()
        self.model = LSTM(n_feature=n_feature, lstm_hidden_dim=lstm_hidden_dim, lstm_num_layers=lstm_num_layers,
                          n_output=n_output, dropout_rate=dropout_rate)
        self.seq_length = seq_length
        self.optimizer = 'adam'
        self.learning_rate = learning_rate
        self.loss = loss
        self.x_scaler = sklearn.preprocessing.StandardScaler()
        self.y_scaler = sklearn.preprocessing.StandardScaler()
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
        train_loader, x_valid, y_valid = self.create_train_valid_sequence_sets(train=train)
        # run optim loop
        self.run_pytorch_optim_loop(train_loader=train_loader, x_valid=x_valid, y_valid=y_valid, model=self.model,
                                    checkpoint_name='lstm_train')
        return cross_val_score_dict

    def update(self, train: pd.DataFrame, model):
        """
        Update existing model due to new samples
        :param train: train set with new samples
        :param model: model to update
        """
        TrainHelper.init_pytorch_seeds()
        train_loader, x_valid, y_valid = self.create_train_valid_sequence_sets(train=train)
        self.run_pytorch_optim_loop(train_loader=train_loader, x_valid=x_valid, y_valid=y_valid, model=model,
                                    checkpoint_name='lstm_update')

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
        # scale
        x_train_scaled = self.x_scaler.transform(train.drop(self.target_column, axis=1))
        y_train_scaled = self.y_scaler.transform(train[self.target_column].values.reshape(-1, 1))
        # create sequences
        x_seq_train, _ = self.create_sequences(data=np.hstack((x_train_scaled, y_train_scaled)))
        x_train = torch.tensor(x_seq_train.astype(np.float32))
        # predict and transform back
        y_insample = self.y_scaler.inverse_transform(self.model(x_train).data.numpy())
        # insert dummy values for train samples before first full sequence
        y_insample = np.insert(y_insample, 0, self.seq_length * [-9999])
        insample = pd.DataFrame(data=y_insample, index=train.index, columns=['Insample'])
        return insample

    def predict(self, test: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
        """
        Deliver, if specified one step ahead, out-of-sample predictions
        :param test: test set
        :param train: train set
        :return: DataFrame with predictions, upper and lower confidence level
        """
        TrainHelper.init_pytorch_seeds()
        x_train_scaled = self.x_scaler.transform(train.drop(self.target_column, axis=1))
        y_train_scaled = self.y_scaler.transform(train[self.target_column].values.reshape(-1, 1))
        x_test_scaled = self.x_scaler.transform(test.drop(self.target_column, axis=1))
        y_test_scaled = self.y_scaler.transform(test[self.target_column].values.reshape((-1, 1)))
        # add last elements of train to complete first test sequence
        x_test_full = np.vstack((x_train_scaled[-self.seq_length:], x_test_scaled))
        y_test_full = np.vstack((y_train_scaled[-self.seq_length:], y_test_scaled))
        # create test sequences
        x_seq_test, _ = self.create_sequences(data=np.hstack((x_test_full, y_test_full)))
        if self.one_step_ahead:
            predict_lst = []
            train_manip = train.copy()
            # deep copy model as predict function should not change class model
            model = copy.deepcopy(self.model)
            for i in range(0, test.shape[0]):
                test_seq = x_seq_test[i].reshape(1, self.seq_length, -1)
                model.eval()
                # predict on cpu
                model.to(torch.device("cpu"))
                fc = self.y_scaler.inverse_transform(model(x=torch.tensor(test_seq.astype(np.float32))).data.numpy())
                train_manip = train_manip.append(test.iloc[[i]])
                self.update(train=train, model=model)
                predict_lst.append(fc)
            predict = np.array(predict_lst).flatten()
        else:
            # predict on cpu
            self.model.to(torch.device("cpu"))
            self.model.eval()
            predict = self.y_scaler.inverse_transform(
                self.model(x=torch.tensor(x_seq_test.astype(np.float32))).data.numpy()).flatten()
        predictions = pd.DataFrame({'Prediction': predict}, index=test.index)
        return predictions

    def create_train_valid_sequence_sets(self, train: pd.DataFrame) -> tuple:
        """
        Create train and validation sequenced set respective train loader with sequenced batches
        :param train: train data to use
        :return: DataLoader with batches of sequenced train data as well as sequenced validation data
        """
        TrainHelper.init_pytorch_seeds()
        # scale input data
        x_train_scaled = self.x_scaler.fit_transform(train.drop(self.target_column, axis=1))
        y_train_scaled = self.y_scaler.fit_transform(train[self.target_column].values.reshape(-1, 1))
        # create sequences
        x_seq_train, y_seq_train = self.create_sequences(data=np.hstack((x_train_scaled, y_train_scaled)))
        # split into train and validation set
        valid_size = 0.2
        split_ind = int(x_seq_train.shape[0] * (1 - valid_size))
        x_train = torch.tensor(x_seq_train[:split_ind, :, :].astype(np.float32))
        x_valid = torch.tensor(x_seq_train[split_ind:, :, :].astype(np.float32))
        y_train = torch.tensor(y_seq_train[:split_ind].reshape(-1, 1).astype(np.float32))
        y_valid = torch.tensor(y_seq_train[split_ind:].reshape(-1, 1).astype(np.float32))
        train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(x_train, y_train),
                                                   batch_size=self.batch_size, shuffle=False, drop_last=False)
        return train_loader, x_valid, y_valid

    def create_sequences(self, data: np.array) -> tuple:
        """
        Create sequenced data according to self.seq_length
        :param data: data to sequence
        :return: sequenced data and labels
        """
        xs = []
        ys = []
        for i in range(data.shape[0] - self.seq_length):
            xs.append(data[i:(i + self.seq_length), :])
            ys.append(data[i + self.seq_length, -1])
        return np.array(xs), np.array(ys)


class LSTM(torch.nn.Module):
    """Class containing LSTM Regression Model"""

    def __init__(self, n_feature: int, lstm_hidden_dim: int, lstm_num_layers: int = 1, n_output: int = 1,
                 dropout_rate: float = 0.0):
        """
        :param n_feature: number of features for ANN input
        :param lstm_hidden_dim: dimensionality of hidden layer
        :param lstm_num_layers: depth of lstm network
        :param n_output: number of outputs
        :param dropout_rate: probability of element being zeroed in dropout layer
        """
        super(LSTM, self).__init__()
        TrainHelper.init_pytorch_seeds()
        self.lstm = nn.LSTM(input_size=n_feature, hidden_size=lstm_hidden_dim, num_layers=lstm_num_layers,
                            batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(in_features=lstm_hidden_dim, out_features=n_output)

    def forward(self, x):
        """
        Feedforward path
        :param x: data to process
        :return: prediction value
        """
        TrainHelper.init_pytorch_seeds()
        # input (batch x seq_length x input_size) (batch_first is set True)
        lstm_out, (hn, cn) = self.lstm(x.view(x.shape[0], x.shape[1], -1))
        # only take last output of sequence
        out = self.dropout(lstm_out[:, -1, :])
        out = self.output_layer(out)
        return out

