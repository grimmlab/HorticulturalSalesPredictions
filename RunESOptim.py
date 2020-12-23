import os
import sys
import warnings
import configparser

import numpy as np
import pandas as pd
import sklearn
from tqdm import tqdm

from training import TrainHelper, ModelsES
from utils import MixedHelper


def run_es_optim(target_column: str, split_perc: float, imputation: str):
    """
    Run whole ES optimization loop
    :param target_column: target variable for predictions
    :param split_perc: percentage of samples to use for train set
    :param imputation: imputation method for missing values
    """
    config = configparser.ConfigParser()
    config.read('Configs/dataset_specific_config.ini')
    # get optim parameters
    base_dir, seasonal_periods, split_perc, init_train_len, test_len, resample_weekly = \
        TrainHelper.get_optimization_run_parameters(config=config, target_column=target_column, split_perc=split_perc)
    # load datasets
    datasets = TrainHelper.load_datasets(config=config, target_column=target_column)
    # prepare parameter grid
    param_grid = {'dataset': datasets,
                  'imputation': [imputation],
                  'trend': ['add', None],
                  'damp': [False, True],
                  'seasonality': ['add', 'mul', None],
                  'remove_bias': [False, True],
                  'brute': [False, True],
                  'osa': [False, True],
                  'transf': [False, 'log', 'pw']
                  }
    # random sample from parameter grid
    params_lst = sorted(list(sklearn.model_selection.ParameterSampler(
        param_distributions=param_grid, n_iter=int(1 * MixedHelper.get_product_len_dict(dictionary=param_grid)),
        random_state=np.random.RandomState(42))),
                        key=lambda d: (d['dataset'].name, d['imputation']))

    doc_results = None
    best_rmse = 5000000.0
    best_mape = 5000000.0
    best_smape = 5000000.0
    dataset_last_name = 'Dummy'
    imputation_last = 'Dummy'

    for i in tqdm(range(len(params_lst))):
        warnings.simplefilter('ignore')
        dataset = params_lst[i]['dataset']
        imputation = params_lst[i]['imputation']
        tr = params_lst[i]['trend']
        damp = params_lst[i]['damp']
        season = params_lst[i]['seasonality']
        remo_bias = params_lst[i]['remove_bias']
        brute = params_lst[i]['brute']
        one_step_ahead = params_lst[i]['osa']
        transf = params_lst[i]['transf']
        power, log = TrainHelper.get_pw_l_for_transf(transf=transf)

        if not((dataset.name == dataset_last_name) and (imputation == imputation_last)):
            if resample_weekly and 'weekly' not in dataset.name:
                dataset.name = dataset.name + '_weekly'
            print(dataset.name + ' ' + str('None' if imputation is None else imputation) + ' ' + target_column)
            train_test_list = TrainHelper.get_ready_train_test_lst(dataset=dataset, config=config,
                                                                   init_train_len=init_train_len,
                                                                   test_len=test_len, split_perc=split_perc,
                                                                   imputation=imputation,
                                                                   target_column=target_column,
                                                                   reset_index=True)
            if dataset.name != dataset_last_name:
                best_rmse = 5000000.0
                best_mape = 5000000.0
                best_smape = 5000000.0
            dataset_last_name = dataset.name
            imputation_last = imputation

        sum_dict = None
        try:
            for train, test in train_test_list:
                model = ModelsES.ExponentialSmoothing(target_column=target_column, trend=tr, damped=damp,
                                                      seasonal=season, seasonal_periods=seasonal_periods,
                                                      remove_bias=remo_bias, use_brute=brute,
                                                      one_step_ahead=one_step_ahead, power_transf=power, log=log)
                cross_val_dict = model.train(train=train, cross_val_call=False)
                eval_dict = model.evaluate(train=train, test=test)
                eval_dict.update(cross_val_dict)
                if sum_dict is None:
                    sum_dict = eval_dict
                else:
                    for k, v in eval_dict.items():
                        sum_dict[k] += v
            evaluation_dict = {k: v / len(train_test_list) for k, v in sum_dict.items()}
            params_dict = {'dataset': dataset.name, 'imputation': str('None' if imputation is None else imputation),
                           'init_train_len': init_train_len, 'test_len': test_len, 'split_perc': split_perc,
                           'trend': tr, 'damped': damp, 'seasonal': season, 'seasonal_periods': seasonal_periods,
                           'remove_bias': remo_bias, 'use_brute': brute, 'one_step_ahead': one_step_ahead,
                           'power_transform': power, 'log': log}
            save_dict = params_dict.copy()
            save_dict.update(evaluation_dict)
            if doc_results is None:
                doc_results = pd.DataFrame(columns=save_dict.keys())
            doc_results = doc_results.append(save_dict, ignore_index=True)
            best_rmse, best_mape, best_smape = TrainHelper.print_best_vals(evaluation_dict=evaluation_dict,
                                                                           best_rmse=best_rmse, best_mape=best_mape,
                                                                           best_smape=best_smape, run_number=i)
        except KeyboardInterrupt:
            print('Got interrupted')
            break
        except Exception as exc:
            print(exc)
            params_dict = {'dataset': 'Failure', 'imputation': str('None' if imputation is None else imputation),
                           'init_train_len': init_train_len, 'test_len': test_len, 'split_perc': split_perc,
                           'trend': tr, 'damped': damp, 'seasonal': season, 'seasonal_periods': seasonal_periods,
                           'remove_bias': remo_bias, 'use_brute': brute, 'one_step_ahead': one_step_ahead,
                           'power_transform': power, 'log': log}
            save_dict = params_dict.copy()
            save_dict.update(TrainHelper.get_failure_eval_dict())
            if doc_results is None:
                doc_results = pd.DataFrame(columns=save_dict.keys())
            doc_results = doc_results.append(save_dict, ignore_index=True)
    TrainHelper.save_csv_results(doc_results=doc_results, save_dir=base_dir+'OptimResults/',
                                 company_model_desc='es', target_column=target_column,
                                 seasonal_periods=seasonal_periods, datasets=datasets,
                                 imputations=param_grid['imputation'],
                                 split_perc=split_perc)
    print('Optimization Done. Saved Results.')


if __name__ == '__main__':
    target_column = str(sys.argv[1])
    split_perc = float(sys.argv[2])
    # univariate method -> imputation after statistical features not needed for raw dataset without missing values
    config = configparser.ConfigParser()
    config.read('Configs/dataset_specific_config.ini')
    imputations = [None]
    if config[target_column]['univariate_imputation_needed']:
        imputations = ['mean', 'iterative', 'knn']
    for imputation in imputations:
        new_pid = os.fork()
        if new_pid == 0:
            run_es_optim(target_column=target_column, split_perc=split_perc, imputation=imputation)
            sys.exit()
        else:
            os.waitpid(new_pid, 0)
            print('finished run with ' + str('None' if imputation is None else imputation))
