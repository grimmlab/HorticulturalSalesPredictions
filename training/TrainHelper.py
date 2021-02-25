import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import PowerTransformer
import datetime
import sklearn
import configparser
import itertools
import torch
import random

from preparation import PreparationHelper, MissingValueImputation
from feateng import FeatureAdder, StatisticalFeatures
from utils import MixedHelper


def create_train_test_set(df: pd.DataFrame, split_perc: float) -> list:
    """
    Deliver train and test set according to split_perc
    :param df: DataFrame to use for train and test set
    :param split_perc: percentage of samples to use for train set
    :return: train and test set in a list for compatibility reasons
    """
    split_ind = int(len(df) * split_perc)
    train = df.iloc[0:split_ind]
    test = df.iloc[split_ind:]
    train_test_sets = list()
    train_test_sets.append((train, test))
    return train_test_sets


def create_tssplit_train_test_set(df: pd.DataFrame, init_train_len: int, test_len: int) -> list:
    """
    deliver train and test set according to time series split with init_train_len and test_len
    :param df:  DataFrame to use for train and test sets
    :param init_train_len: length of first train set
    :param test_len: usual length of test set (could be shorter for last test set)
    :return: list of train and test sets
    """
    train_test_sets = list()
    split_ind = init_train_len
    for i in range(0, math.ceil(df.shape[0] / (init_train_len + test_len))):
        train = df.iloc[0:split_ind]
        if (split_ind + test_len) > df.shape[0]:
            test = df.iloc[split_ind:]
        else:
            test = df.iloc[split_ind:split_ind+test_len]
        train_test_sets.append((train, test))
        split_ind += test_len
    return train_test_sets


def get_transformed_set(dataset: pd.DataFrame, target_column: str,
                        power_transformer: PowerTransformer = None,
                        log: bool = False, only_transform: bool = False) -> pd.DataFrame:
    """
    Function returning dataset with (log or power) transformed column
    :param dataset: dataset to transform
    :param target_column: column to transform
    :param power_transformer: power_transformer instance
    :param log: use log or not
    :param only_transform: only transform, no new fit of parameters
    :return: dataset with transformed column
    """
    dataset_manip = dataset.copy()
    if power_transformer is not None:
        if only_transform:
            dataset_manip[target_column] = power_transformer \
                .transform(dataset_manip[target_column].values.reshape(-1, 1))
        else:
            dataset_manip[target_column] = power_transformer \
                .fit_transform(dataset_manip[target_column].values.reshape(-1, 1))
    if log:
        if any(dataset_manip[target_column] < 0):
            raise NameError('Negative values for log-transform')
        if 0 in dataset_manip[target_column].values:
            dataset_manip[target_column] = np.log(dataset_manip[target_column] + 1)
        else:
            dataset_manip[target_column] = np.log(dataset_manip[target_column])
    return dataset_manip


def load_datasets(config: configparser.ConfigParser, target_column: str) -> list:
    """
    Load datasets according to info specified in config file
    :param config: config with dataset specific info
    :param target_column: target_column for prediction
    :return: list of datasets to use for optimization
    """
    datasets_lst = list()
    # load and name raw dataset
    dataset_raw = pd.read_csv(config['General']['base_dir'] + 'Data/' + config[target_column]['dataset_raw'] + '.csv',
                              sep=';', decimal=',', index_col=0)
    try:
        dataset_raw.index = pd.to_datetime(dataset_raw.index, format='%d.%m.%Y')
    except:
        dataset_raw.index = pd.to_datetime(dataset_raw.index, format='%Y-%m-%d')
    dataset_raw = dataset_raw.asfreq('D')
    dataset_raw.name = config[target_column]['dataset_raw']
    datasets_lst.append(dataset_raw)

    # split dataset at before_break_date
    if 'before_break_date' in config[target_column]:
        dataset_before_break = dataset_raw.copy()
        dataset_before_break.name = dataset_raw.name + '_before_break'
        before_break_date = datetime.datetime.strptime(config[target_column]['before_break_date'], '%Y-%m-%d').date()
        PreparationHelper.drop_rows_by_dates(df=dataset_before_break, start=before_break_date,
                                             end=dataset_before_break.index[-1])
        datasets_lst.append(dataset_before_break)

    return datasets_lst


def get_train_test_lst(dataset: pd.DataFrame, init_train_len: int, test_len: int, split_perc: float) -> list:
    """
    Get list of train and test sets
    :param dataset: dataset to use
    :param init_train_len: length of first train set
    :param test_len: usual length of test set (could be shorter for last test set)
    :param split_perc: percentage of samples to use for train set
    :return: list of tuples with train and test set
    """
    if split_perc is None:
        train_test_list = create_tssplit_train_test_set(df=dataset, init_train_len=init_train_len,
                                                        test_len=test_len)
    else:
        train_test_list = create_train_test_set(df=dataset, split_perc=split_perc)
    return train_test_list


def get_imputed_train_test_lst(dataset: pd.DataFrame, init_train_len: int, test_len: int, split_perc: float,
                               imputation: str) -> list:
    """
    Deliver imputed train_test_lst
    :param dataset: dataset to use
    :param init_train_len: length of first train set
    :param test_len: usual length of test set (could be shorter for last test set)
    :param split_perc: percentage of samples to use for train set
    :param imputation: imputation method to use
    :return: list with imputed train test tuples
    """
    train_test_list = list()
    train_test_list_mv = get_train_test_lst(dataset=dataset, init_train_len=init_train_len,
                                            test_len=test_len, split_perc=split_perc)
    for train_mv, test_mv in train_test_list_mv:
        if imputation is not None:
            _, train, test = impute_dataset_train_test(imputation=imputation, dataset=dataset,
                                                       train=train_mv, test=test_mv)
        else:
            train = train_mv
            test = test_mv
        train_test_list.append((train, test))
    return train_test_list


def print_best_vals(evaluation_dict: dict, best_rmse: float, best_mape: float, best_smape: float,
                    run_number: int) -> tuple:
    """
    print and return new best eval vals
    :param evaluation_dict: evaluation dictionary
    :param best_rmse: current best rmse
    :param best_mape: current best mape
    :param best_smape: current best smape
    :param run_number: current optimization loop iteration
    :return: new best eval values
    """
    if evaluation_dict['RMSE_Test'] < best_rmse:
        best_rmse = evaluation_dict['RMSE_Test']
        print('Best RMSE Run: ', run_number, ', RMSE Value = ', best_rmse)
    if evaluation_dict['MAPE_Test'] < best_mape:
        best_mape = evaluation_dict['MAPE_Test']
        print('Best MAPE Run: ', run_number, ', MAPE Value = ', best_mape)
    if evaluation_dict['sMAPE_Test'] < best_smape:
        best_smape = evaluation_dict['sMAPE_Test']
        print('Best sMAPE Run: ', run_number, ', sMAPE Value = ', best_smape)
    return best_rmse, best_mape, best_smape


def get_failure_eval_dict():
    """
    get eval dictionary in case of a optim failure
    :return: failure eval dict
    """
    return {'RMSE_Train_RW': np.nan, 'MAPE_Train_RW': np.nan, 'sMAPE_Train_RW': np.nan,
            'RMSE_Test_RW': np.nan, 'MAPE_Test_RW': np.nan, 'sMAPE_Test_RW': np.nan,
            'RMSE_Train_seasRW': np.nan, 'MAPE_Train_seasRW': np.nan, 'sMAPE_Train_seasRW': np.nan,
            'RMSE_Test_seasRW': np.nan, 'MAPE_Test_seasRW': np.nan, 'sMAPE_Test_seasRW': np.nan,
            'RMSE_Train_HA': np.nan, 'MAPE_Train_HA': np.nan, 'sMAPE_Train_HA': np.nan,
            'RMSE_Test_HA': np.nan, 'MAPE_Test_HA': np.nan, 'sMAPE_Test_HA': np.nan,
            'RMSE_Train': np.nan, 'MAPE_Train': np.nan, 'sMAPE_Train': np.nan,
            'RMSE_Test': np.nan, 'MAPE_Test': np.nan, 'sMAPE_Test': np.nan
            }


def save_csv_results(doc_results: pd.DataFrame, save_dir: str, company_model_desc: str, target_column: str,
                     datasets: list, imputations: list, split_perc: float, seasonal_periods: int,
                     featuresets: list = None):
    """
    save optim results to disc as csv
    :param doc_results: DataFrame to save
    :param save_dir: directory for saving file
    :param company_model_desc: description of company and used algorithm
    :param target_column: target_column for prediction
    :param datasets: of used datasets
    :param imputations: used imputations
    :param featuresets: used featuresets
    :param split_perc: split_percentage used
    :param seasonal_periods: seasonal_periods used
    """
    ds_names = ''
    for ds in datasets:
        ds_names = ds_names + '-' + ds.name
    imp_names = 'imp'
    for imp in imputations:
        imp_names = imp_names + '_' + str('None' if imp is None else imp)
    if featuresets is not None:
        feat_names = 'feat'
        for feat in featuresets:
            feat_names = feat_names + '_' + feat
    else:
        feat_names = ''
    doc_results.to_csv(save_dir + company_model_desc + '-' + target_column + ds_names + '-' + feat_names + '-'
                       + imp_names + '-splitperc_' + str(split_perc).replace('.', '')
                       + '-SeasPer_' + str(seasonal_periods)
                       + '-' + datetime.datetime.now().strftime("%d-%b-%Y_%H-%M") + '.csv',
                       sep=';', decimal=',', float_format='%.10f')


def get_pw_l_for_transf(transf: str) -> tuple:
    """
    get parameters pw (power transform) and l (log transform) according to transform parameter
    :param transf: transform parameter
    :return: pw and l parameters
    """
    if transf == 'log':
        pw = False
        l = True
    elif transf == 'pw':
        pw = True
        l = False
    else:
        pw = False
        l = False
    return pw, l


def impute_dataset_train_test(imputation: str, train: pd.DataFrame, test: pd.DataFrame = None,
                              dataset: pd.DataFrame = None) -> tuple:
    """
    Get imputed dataset as well as train and test set (fitted to train set)
    :param imputation: imputation method to use
    :param train: train set to impute (and to use for fitting imputer)
    :param test: test set
    :param dataset: dataset to impute
    :return: imputed dataset, train and test set
    """
    cols_to_impute = train.loc[:, train.isna().any()].select_dtypes(exclude=['string', 'object']).columns.tolist()
    if len(cols_to_impute) == 0:
        return dataset, train, test
    cols_to_add = [col for col in train.columns.tolist() if col not in cols_to_impute]
    if imputation == 'mean' or imputation == 'median':
        imputer = MissingValueImputation.get_simple_imputer(df=train.filter(cols_to_impute), strategy=imputation)
    elif imputation == 'iterative':
        imputer = MissingValueImputation.get_iter_imputer(df=train.filter(cols_to_impute))
    elif imputation == 'knn':
        imputer = MissingValueImputation.get_knn_imputer(df=train.filter(cols_to_impute))

    train_imp = pd.concat([pd.DataFrame(data=imputer.transform(X=train.filter(cols_to_impute)),
                                        columns=cols_to_impute, index=train.index), train[cols_to_add]],
                          axis=1, sort=False)
    if test is None:
        test_imp = None
    else:
        test_imp = pd.concat([pd.DataFrame(data=imputer.transform(X=test.filter(cols_to_impute)),
                                           columns=cols_to_impute, index=test.index), test[cols_to_add]],
                             axis=1, sort=False)
    if dataset is None:
        dataset_imp = None
    else:
        dataset_imp = pd.concat([pd.DataFrame(data=imputer.transform(X=dataset.filter(cols_to_impute)),
                                              columns=cols_to_impute, index=dataset.index), dataset[cols_to_add]],
                                axis=1, sort=False)
    return dataset_imp, train_imp, test_imp


def pca_transform_train_test(train: pd. DataFrame, test: pd.DataFrame, target_column: str) -> tuple:
    """
    Deliver PCA transformed train and test set
    :param train: train set to transform (and used to fit pca)
    :param test: test set to transform
    :param target_column: target_column to add in the end
    :return: tuple of transformed train and test dataset
    """
    scaler = sklearn.preprocessing.StandardScaler()
    train_stand = scaler.fit_transform(train.drop(target_column, axis=1))
    pca = sklearn.decomposition.PCA(0.95)
    train_transf = pca.fit_transform(train_stand)
    test_stand = scaler.transform(test.drop(target_column, axis=1))
    test_transf = pca.transform(test_stand)
    train_data = pd.DataFrame(data=train_transf, columns=['PC' + str(i) for i in range(train_transf.shape[1])],
                              index=train.index)
    train_data[target_column] = train[target_column]
    test_data = pd.DataFrame(data=test_transf, columns=['PC' + str(i) for i in range(test_transf.shape[1])],
                             index=test.index)
    test_data[target_column] = test[target_column]
    return train_data, test_data


def get_ready_train_test_lst(dataset: pd.DataFrame, config: configparser.ConfigParser,
                             init_train_len: int, test_len: int, split_perc: float, imputation: str, target_column: str,
                             reset_index: bool = False, dimensionality_reduction: str = None,
                             featureset: str = 'full') -> list:
    """
    Function preparing train and test sets for training based on raw dataset:
    - Missing Value imputation
    - Feature Extraction
    (- Resampling if specified)
    - Deletion of non-target sales columns
    - Split into train and test set(s)
    :param dataset: dataset with raw samples
    :param config: config with dataset specific info
    :param init_train_len: length of first train set
    :param test_len: usual length of test set (could be shorter for last test set)
    :param split_perc: percentage of samples to use for train set
    :param imputation: imputation method to use
    :param target_column: target_column used for predictions
    :param reset_index: reset_index of dataset (relevant for Exponential Smoothing)
    :param dimensionality_reduction: perform dimensionality reduction
    :param featureset: featureset to use ('full', 'cal', 'stat', 'none')
    :return: list with train and test set(s)
    """
    print('##### Preparing Train and Test Sets #####')
    # get dataset specific parameters
    seasonal_periods = config[target_column].getint('seasonal_periods')
    features_for_stats = config[target_column]['features_for_stats'].replace(" ", "").split(',')
    resample_weekly = config[target_column].getboolean('resample_weekly')
    possible_target_cols = config[target_column]['possible_target_cols'].replace(" ", "").split(',')
    cols_to_condense, condensed_col_name = None, None
    # use stat and cal features according to specified featureset
    stat_features = True
    cal_features = True
    if featureset == 'none':
        stat_features = False
        cal_features = False
    elif featureset == 'cal':
        stat_features = False
    elif featureset == 'stat':
        cal_features = False

    if 'cols_to_condense' in config[target_column]:
        cols_to_condense = config[target_column]['cols_to_condense'].replace(" ", "").split(',')
        condensed_col_name = config[target_column]['condensed_col_name']

    # load train and test set with missing values
    train_test_list_mv = get_train_test_lst(dataset=dataset, init_train_len=init_train_len, test_len=test_len,
                                            split_perc=split_perc)
    train_test_list = list()
    counter_list_tuple = 0
    for train_mv, test_mv in train_test_list_mv:
        # impute dataset according to fitting on train set with missing values
        if imputation is not None:
            dataset_imputed, _, _ = impute_dataset_train_test(imputation=imputation, dataset=dataset,
                                                              train=train_mv, test=test_mv)
        else:
            dataset_imputed = dataset.copy()
        MixedHelper.set_dtypes(df=dataset_imputed, cols_to_str=['public_holiday', 'school_holiday'])

        # feature extraction on imputed dataset
        if resample_weekly:
            # stats features after resampling, if resampling is done, to avoid information leak due to resampling
            FeatureAdder.add_features(dataset=dataset_imputed, cols_to_condense=cols_to_condense,
                                      condensed_col_name=condensed_col_name, use_stat_features=False,
                                      use_calendar_features=cal_features)
        else:
            FeatureAdder.add_features(dataset=dataset_imputed, cols_to_condense=cols_to_condense,
                                      condensed_col_name=condensed_col_name, seasonal_periods=seasonal_periods,
                                      features_for_stats=features_for_stats, use_stat_features=stat_features,
                                      use_calendar_features=cal_features)

        dataset_feat = PreparationHelper.get_one_hot_encoded_df(
            df=dataset_imputed,
            columns_to_encode=list(set(dataset_imputed.columns).intersection(['public_holiday', 'school_holiday'])))
        dataset_feat.dropna(subset=[target_column], inplace=True)

        # resample if specified
        if resample_weekly:
            dataset_feat = dataset_feat.resample('W').apply(
                lambda x: PreparationHelper.custom_resampler(arraylike=x, summation_cols=possible_target_cols)
            )
            if 'cal_date_weekday' in dataset_feat.columns:
                PreparationHelper.drop_columns(df=dataset_feat, columns=['cal_date_weekday'])
            # drop rows added due to resampling of quarter dataset
            dataset_feat.dropna(inplace=True)
            init_train_len = int(train_test_list_mv[0][0].shape[0] / 7)
            test_len = int(train_test_list_mv[0][1].shape[0] / 7)
            seasonal_periods = int(seasonal_periods / 7)
            if stat_features:
                FeatureAdder.add_features(dataset=dataset_feat, seasonal_periods=seasonal_periods,
                                          features_for_stats=features_for_stats,
                                          use_calendar_features=False, with_weekday_stats=False,
                                          lags=[1, 4], windowsize_rolling=4, windowsize_rolling_seas=4)
                StatisticalFeatures.add_rolling_statistics_features(dataset=dataset_feat, windowsize=2,
                                                                    features=features_for_stats)
                StatisticalFeatures.add_rolling_seasonal_statistics_features(dataset=dataset_feat,
                                                                             windowsize=2,
                                                                             features=features_for_stats,
                                                                             seasonal_periods=seasonal_periods)

        # drop non-target columns
        cols_to_drop = possible_target_cols.copy()
        cols_to_drop.remove(target_column)
        PreparationHelper.drop_columns(df=dataset_feat, columns=cols_to_drop)

        # split into train and test set(s)
        if reset_index:
            dataset_feat.reset_index(drop=True, inplace=True)
        train_test_list_feat = get_train_test_lst(dataset=dataset_feat, init_train_len=init_train_len,
                                                  test_len=test_len, split_perc=split_perc)
        # impute missing values after adding statistical features (e.g. due to lagged features)
        if imputation is not None:
            _, train_feat_imp, test_feat_imp = impute_dataset_train_test(
                imputation=imputation, train=train_test_list_feat[counter_list_tuple][0],
                test=train_test_list_feat[counter_list_tuple][1])
        else:
            train_feat_imp = train_test_list_feat[counter_list_tuple][0]
            test_feat_imp = train_test_list_feat[counter_list_tuple][1]

        # perform dimensionality reduction if specified
        if not train_feat_imp.isna().any().any() and dimensionality_reduction == 'pca':
            train_feat_imp, test_feat_imp = pca_transform_train_test(train=train_feat_imp, test=test_feat_imp,
                                                                     target_column=target_column)
        train_test_list.append((train_feat_imp, test_feat_imp))
        if len(train_test_list_feat) > 1:
            # special treatment for time series split with multiple train test pairs in train_test_list
            # first iteration of for loop: imputation based on train_1, second iteration: imputation based on train_2
            # in both cases creation of multiple (train, test) pairs based on imputed dataset
            # only the one related to the set used for imputation shall be kept
            counter_list_tuple += 1
        print('##### Prepared Train and Test Sets #####')
    return train_test_list


def get_optimization_run_parameters(config: configparser.ConfigParser, target_column: str, split_perc: float) -> tuple:
    """
    Function loading and preparing parameters needed for optimization run
    :param config: config with dataset specific info
    :param target_column: target_column for prediction
    :param split_perc: percentage of samples to use for train set
    :return: parameters needed for optimization run
    """
    # general parameters
    base_dir = config['General']['base_dir']
    # dataset specific parameters
    seasonal_periods = config[target_column].getint('seasonal_periods')
    split_perc = None if split_perc == 0 else split_perc
    init_train_len = None if split_perc is not None else 2 * seasonal_periods
    test_len = None if split_perc is not None else seasonal_periods
    resample_weekly = config[target_column].getboolean('resample_weekly')
    if resample_weekly:
        seasonal_periods = int(seasonal_periods / 7)

    return base_dir, seasonal_periods, split_perc, init_train_len, test_len, resample_weekly


def extend_kernel_combinations(kernels: list, base_kernels: list):
    """
    Function extending kernels list with combinations based on base_kernels
    :param kernels: list of kernels for optimization
    :param base_kernels: list of base_kernels
    """
    kernels.extend(base_kernels)
    for el in list(itertools.combinations(*[base_kernels], r=2)):
        kernels.append(el[0] + el[1])
        kernels.append(el[0] * el[1])
    for el in list(itertools.combinations(*[base_kernels], r=3)):
        kernels.append(el[0] + el[1] + el[2])
        kernels.append(el[0] * el[1] * el[2])
        kernels.append(el[0] * el[1] + el[2])
        kernels.append(el[0] + el[1] * el[2])
        kernels.append(el[0] * el[2] + el[1])


def random_sample_parameter_grid(param_grid: dict, sample_share: float) -> list:
    """
    Function random sampling from parameter grid and returning sorted list
    :param param_grid: parameter grid to sample from
    :param sample_share: percentage value of combinations to use (related to a full combination of all variants)
    :return: sorted list with parameters for optimization
    """
    params_lst = sorted(list(sklearn.model_selection.ParameterSampler(
        param_distributions=param_grid,
        n_iter=int(sample_share * MixedHelper.get_product_len_dict(dictionary=param_grid)),
        random_state=np.random.RandomState(42))),
        key=lambda d: (d['dataset'].name, d['imputation'], d['featureset'], d['dim_reduction']))
    return params_lst


def get_imputation_featureset_combis(imputations: list, featuresets: list, target_column: str) -> list:
    """
    Function delivering imputation and featureset combinations which make sense
    :param imputations: imputations to use
    :param featuresets: featuresets to use
    :param target_column: target_column for retrieving info from config file
    :return: list of combinations which make sense
    """
    config = configparser.ConfigParser()
    config.read('Configs/dataset_specific_config.ini')
    multiple_nans_raw_set = config[target_column].getboolean('multiple_nans_raw_set')
    imp_feat_combis = list(itertools.product(*[imputations, featuresets]))
    if not multiple_nans_raw_set:
        for (imp, feat) in imp_feat_combis.copy():
            if (feat == 'cal' or feat == 'none') and (imp == 'iterative' or imp == 'knn'):
                imp_feat_combis.remove((imp, feat))
    return imp_feat_combis


def init_pytorch_seeds():
    """
    Initialize all seeds for pytorch optimization
    """
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
