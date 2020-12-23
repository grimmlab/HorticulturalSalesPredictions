import configparser
import os
import sys
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm

from training import TrainHelper, ModelsMLR


def run_regressions_optim(target_column: str, split_perc: float, algo: str):
    """
    Run whole multiple linear regression optimization loops
    :param target_column: target variable for predictions
    :param split_perc: percentage of samples to use for train set
    :param algo: algo to use for optimization (['lasso', 'ridge', 'elasticnet', 'bayesridge', 'ard'])
    """
    config = configparser.ConfigParser()
    config.read('Configs/dataset_specific_config.ini')
    # get optim parameters
    base_dir, seasonal_periods, split_perc, init_train_len, test_len, resample_weekly = \
        TrainHelper.get_optimization_run_parameters(config=config, target_column=target_column, split_perc=split_perc)
    multiple_nans_raw_set = config[target_column].getboolean('multiple_nans_raw_set')
    # load datasets
    datasets = TrainHelper.load_datasets(config=config, target_column=target_column)
    # prepare parameter grid
    # parameters relevant for all algos
    param_grid = {'dataset': datasets,
                  'imputation': ['mean', 'iterative', 'knn'],
                  'featureset': ['full', 'cal', 'stat', 'none'],
                  'dim_reduction': ['None', 'pca'],
                  'normalize': [False, True],
                  'osa': [False, True]
                  }
    # parameters relevant for lasso, ridge and elasticnet
    if algo in ['lasso', 'ridge', 'elasticnet']:
        param_grid['alpha'] = [10**x for x in range(-5, 5)]
        if algo == 'elasticnet':
            param_grid['l1_ratio'] = np.arange(0.1, 1, 0.1)
        # random sample from parameter grid: all combis for lasso, ridge, elasticnet
        params_lst = TrainHelper.random_sample_parameter_grid(param_grid=param_grid, sample_share=1)
    # parameters relevant for bayesian ridge and ard regression
    else:
        param_grid['alpha_1'] = [10**x for x in range(-6, 1)]
        param_grid['alpha_2'] = [10**x for x in range(-6, -4)]
        param_grid['lambda_1'] = [10**x for x in range(-6, 1)]
        param_grid['lambda_2'] = [10**x for x in range(-6, 1)]
        # random sample from parameter grid: 0.25 share for bayesridge
        params_lst = TrainHelper.random_sample_parameter_grid(param_grid=param_grid, sample_share=0.25)
        if algo == 'ard':
            param_grid['threshold_lambda'] = [10**x for x in range(2, 6)]
            # random sample from parameter grid: 0.2 share for ard
            params_lst = TrainHelper.random_sample_parameter_grid(param_grid=param_grid, sample_share=0.2)
    # remove non-relevant featureset imputation combis
    if not multiple_nans_raw_set:
        params_lst_small = params_lst.copy()
        for param_set in params_lst:
            feat = param_set['featureset']
            imp = param_set['imputation']
            if (feat == 'cal' or feat == 'none') and (imp == 'iterative' or imp == 'knn'):
                params_lst_small.remove(param_set)
        params_lst = params_lst_small

    doc_results = None
    best_rmse = 5000000.0
    best_mape = 5000000.0
    best_smape = 5000000.0
    dataset_last_name = 'Dummy'
    imputation_last = 'Dummy'
    dim_reduction_last = 'Dummy'
    featureset_last = 'Dummy'

    for i in tqdm(range(len(params_lst))):
        warnings.simplefilter('ignore')
        dataset = params_lst[i]['dataset']
        imputation = params_lst[i]['imputation']
        featureset = params_lst[i]['featureset']
        dim_reduction = None if params_lst[i]['dim_reduction'] == 'None' else params_lst[i]['dim_reduction']
        normalize = params_lst[i]['normalize']
        one_step_ahead = params_lst[i]['osa']
        l1_ratio = params_lst[i]['l1_ratio'] if 'l1_ratio' in params_lst[i] else None
        alpha = params_lst[i]['alpha'] if 'alpha' in params_lst[i] else None
        alpha_1 = params_lst[i]['alpha_1'] if 'alpha_1' in params_lst[i] else None
        alpha_2 = params_lst[i]['alpha_2'] if 'alpha_2' in params_lst[i] else None
        lambda_1 = params_lst[i]['lambda_1'] if 'lambda_1' in params_lst[i] else None
        lambda_2 = params_lst[i]['lambda_2'] if 'lambda_2' in params_lst[i] else None
        threshold_lambda = params_lst[i]['threshold_lambda'] if 'threshold_lambda' in params_lst[i] else None

        # dim_reduction does not make sense for few features
        if featureset == 'none' and dim_reduction is not None:
            continue

        if not((dataset.name == dataset_last_name) and (imputation == imputation_last) and
               (dim_reduction == dim_reduction_last) and (featureset == featureset_last)):
            if resample_weekly and 'weekly' not in dataset.name:
                dataset.name = dataset.name + '_weekly'
            print(dataset.name + ' ' + str('None' if imputation is None else imputation) + ' '
                  + str('None' if dim_reduction is None else dim_reduction) + ' '
                  + featureset + ' ' + target_column)
            train_test_list = TrainHelper.get_ready_train_test_lst(dataset=dataset, config=config,
                                                                   init_train_len=init_train_len,
                                                                   test_len=test_len, split_perc=split_perc,
                                                                   imputation=imputation,
                                                                   target_column=target_column,
                                                                   dimensionality_reduction=dim_reduction,
                                                                   featureset=featureset)
            if dataset.name != dataset_last_name:
                best_rmse = 5000000.0
                best_mape = 5000000.0
                best_smape = 5000000.0
            dataset_last_name = dataset.name
            imputation_last = imputation
            dim_reduction_last = dim_reduction
            featureset_last = featureset

        sum_dict = None
        try:
            for train, test in train_test_list:
                model = ModelsMLR.MultipleLinearRegression(
                    model_to_use=algo, target_column=target_column, seasonal_periods=seasonal_periods,
                    one_step_ahead=one_step_ahead, normalize=normalize, l1_ratio=l1_ratio, alpha=alpha,
                    alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2,
                    threshold_lambda=threshold_lambda
                )
                cross_val_dict = model.train(train=train, cross_val_call=False)
                eval_dict = model.evaluate(train=train, test=test)
                eval_dict.update(cross_val_dict)
                if sum_dict is None:
                    sum_dict = eval_dict
                else:
                    for k, v in eval_dict.items():
                        sum_dict[k] += v
            evaluation_dict = {k: v / len(train_test_list) for k, v in sum_dict.items()}
            params_dict = {'dataset': dataset.name, 'featureset': featureset,
                           'imputation': str('None' if imputation is None else imputation),
                           'dim_reduction': str('None' if dim_reduction is None else dim_reduction),
                           'init_train_len': init_train_len, 'test_len': test_len, 'split_perc': split_perc,
                           'algo': model.name, 'normalize': normalize, 'alpha': alpha, 'l1_ratio': l1_ratio,
                           'alpha_1': alpha_1, 'alpha_2': alpha_2, 'lambda_1': lambda_1, 'lambda_2': lambda_2,
                           'threshold_lambda': threshold_lambda, 'one_step_ahead': one_step_ahead,
                           'fitted_coef': model.model.coef_, 'fitted_intercept': model.model.intercept_}
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
            params_dict = {'dataset': 'Failure', 'featureset': featureset,
                           'imputation': str('None' if imputation is None else imputation),
                           'dim_reduction': str('None' if dim_reduction is None else dim_reduction),
                           'init_train_len': init_train_len, 'test_len': test_len, 'split_perc': split_perc,
                           'algo': model.name, 'normalize': normalize, 'alpha': alpha, 'l1_ratio': l1_ratio,
                           'alpha_1': alpha_1, 'alpha_2': alpha_2, 'lambda_1': lambda_1, 'lambda_2': lambda_2,
                           'threshold_lambda': threshold_lambda, 'one_step_ahead': one_step_ahead,
                           'fitted_coef': 'failed', 'fitted_intercept': 'failed'}
            save_dict = params_dict.copy()
            save_dict.update(TrainHelper.get_failure_eval_dict())
            if doc_results is None:
                doc_results = pd.DataFrame(columns=save_dict.keys())
            doc_results = doc_results.append(save_dict, ignore_index=True)
    TrainHelper.save_csv_results(doc_results=doc_results, save_dir=base_dir+'OptimResults/',
                                 company_model_desc=algo, target_column=target_column,
                                 seasonal_periods=seasonal_periods, datasets=datasets,
                                 featuresets=param_grid['featureset'], imputations=param_grid['imputation'],
                                 split_perc=split_perc)
    print('Optimization Done. Saved Results.')


if __name__ == '__main__':
    target_column = str(sys.argv[1])
    split_perc = float(sys.argv[2])
    algos = ['lasso', 'ridge', 'elasticnet', 'bayesridge', 'ard']
    for algo in algos:
        new_pid = os.fork()
        if new_pid == 0:
            run_regressions_optim(target_column=target_column, split_perc=split_perc, algo=algo)
            sys.exit()
        else:
            os.waitpid(new_pid, 0)
            print('finished run with ' + algo)
