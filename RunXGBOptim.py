import configparser
import os
import sys
import warnings

import pandas as pd
from tqdm import tqdm

from training import TrainHelper, ModelXGBoost


def run_xgb_optim(target_column: str, split_perc: float, imputation: str, featureset: str):
    """
    Run whole XGB optimization loop
    :param target_column: target variable for predictions
    :param split_perc: percentage of samples to use for train set
    :param imputation: imputation method for missing values
    :param featureset: featureset to use
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
                  'featureset': [featureset],
                  'dim_reduction': ['None', 'pca'],
                  'learning_rate': [0.05, 0.1, 0.3],
                  'max_depth': [3, 5, 10],
                  'subsample': [0.3, 0.7, 1],
                  'n_estimators': [10, 100, 1000],
                  'gamma': [0, 1, 10],
                  'alpha': [0, 0.1, 1, 10],
                  'reg_lambda': [0, 0.1, 1, 10],
                  'osa': [True]
                  }

    # random sample from parameter grid
    params_lst = TrainHelper.random_sample_parameter_grid(param_grid=param_grid, sample_share=0.25)

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
        learning_rate = params_lst[i]['learning_rate']
        max_depth = params_lst[i]['max_depth']
        subsample = params_lst[i]['subsample']
        n_estimators = params_lst[i]['n_estimators']
        gamma = params_lst[i]['gamma']
        alpha = params_lst[i]['alpha']
        reg_lambda = params_lst[i]['reg_lambda']
        one_step_ahead = params_lst[i]['osa']

        # dim_reduction only done without NaNs
        if imputation is None and dim_reduction is not None:
            continue
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
                model = ModelXGBoost.XGBoostRegression(target_column=target_column,
                                                       seasonal_periods=seasonal_periods,
                                                       learning_rate=learning_rate,
                                                       max_depth=max_depth, subsample=subsample,
                                                       n_estimators=n_estimators, gamma=gamma, alpha=alpha,
                                                       reg_lambda=reg_lambda, one_step_ahead=one_step_ahead)
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
                           'learning_rate': learning_rate, 'max_depth': max_depth, 'subsample': subsample,
                           'n_estimators': n_estimators, 'gamma': gamma, 'alpha': alpha, 'lambda': reg_lambda,
                           'one_step_ahead': one_step_ahead}
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
                           'learning_rate': learning_rate, 'max_depth': max_depth, 'subsample': subsample,
                           'n_estimators': n_estimators, 'gamma': gamma, 'alpha': alpha, 'lambda': reg_lambda,
                           'one_step_ahead': one_step_ahead}
            save_dict = params_dict.copy()
            save_dict.update(TrainHelper.get_failure_eval_dict())
            if doc_results is None:
                doc_results = pd.DataFrame(columns=save_dict.keys())
            doc_results = doc_results.append(save_dict, ignore_index=True)
    TrainHelper.save_csv_results(doc_results=doc_results, save_dir=base_dir+'OptimResults/',
                                 company_model_desc='xgb', target_column=target_column,
                                 seasonal_periods=seasonal_periods, datasets=datasets,
                                 featuresets=param_grid['featureset'], imputations=param_grid['imputation'],
                                 split_perc=split_perc)
    print('Optimization Done. Saved Results.')


if __name__ == '__main__':
    target_column = str(sys.argv[1])
    split_perc = float(sys.argv[2])
    imputations = [None, 'mean', 'iterative', 'knn']
    featuresets = ['full', 'cal', 'stat', 'none']
    imp_feat_combis = TrainHelper.get_imputation_featureset_combis(imputations=imputations, featuresets=featuresets,
                                                                   target_column=target_column)
    for (imputation, featureset) in imp_feat_combis:
        new_pid = os.fork()
        if new_pid == 0:
            run_xgb_optim(target_column=target_column, split_perc=split_perc, imputation=imputation,
                          featureset=featureset)
            sys.exit()
        else:
            os.waitpid(new_pid, 0)
            print('finished run with ' + featureset + ' ' + str('None' if imputation is None else imputation))
