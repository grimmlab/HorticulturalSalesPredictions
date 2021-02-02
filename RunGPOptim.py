import configparser
import os
import sys
import warnings

import gpflow
import pandas as pd
import tensorflow as tf
from gpflow.kernels import Matern52, White, RationalQuadratic, Periodic, \
    SquaredExponential, Polynomial, IsotropicStationary, Sum, Product
from tqdm import tqdm

from training import TrainHelper, ModelsGPR

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel('ERROR')


def get_docresults_strings(kernel: gpflow.kernels.Kernel, mean_function: gpflow.mean_functions.MeanFunction,
                           optimizer) -> tuple:
    """
    get string with infos from kernel, meanfct and optimizer for documentation in optim results file
    :param kernel: kernel with info to be converted
    :param mean_function: mean_function with info to be converted
    :param optimizer: optimizer with info to be converted
    :return: tupel with strings containing documentation info
    """
    kernel_string = kernel.name
    if isinstance(kernel, Sum) or isinstance(kernel, Product):
        for base_k in kernel.kernels:
            if isinstance(base_k, Sum) or isinstance(base_k, Product):
                kernel_string += base_k.name + '---'
                for k in base_k.kernels:
                    kernel_string += '-' + k.name + '_' + str(k.parameters) + '---'
                    if isinstance(k, Periodic):
                        kernel_string += 'with_' + k.base_kernel.name + '_' + str(k.base_kernel.parameters) \
                                         + '---'
            else:
                kernel_string += '-' + base_k.name + '_' + str(base_k.parameters) + '-'
                if isinstance(base_k, Periodic):
                    kernel_string += 'with_' + base_k.base_kernel.name + '_' + str(base_k.base_kernel.parameters) \
                                     + '---'
    elif isinstance(kernel, Periodic):
        kernel_string += str(kernel.parameters) + 'with' + kernel.base_kernel.name + '_' + \
                         str(kernel.base_kernel.parameters)
    else:
        kernel_string += str(kernel.parameters)

    if mean_function is not None:
        mean_fct_string = mean_function.name + '-' + str(mean_function.parameters)
    else:
        mean_fct_string = 'None'

    optimizer_string = 'gpflow_scipy'
    if isinstance(optimizer, tf.optimizers.Adam):
        optimizer_string = 'tfAdam_lr' + str(optimizer.learning_rate)

    return kernel_string, mean_fct_string, optimizer_string


def run_gp_optim(target_column: str, split_perc: float, imputation: str, featureset: str):
    """
    Run whole GPR optimization loop
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
    kernels = []
    base_kernels = [SquaredExponential(), Matern52(), White(), RationalQuadratic(), Polynomial()]
    for kern in base_kernels:
        if isinstance(kern, IsotropicStationary):
            base_kernels.append(Periodic(kern, period=seasonal_periods))
    TrainHelper.extend_kernel_combinations(kernels=kernels, base_kernels=base_kernels)
    param_grid = {'dataset': datasets,
                  'imputation': [imputation],
                  'featureset': [featureset],
                  'dim_reduction': ['None', 'pca'],
                  'kernel': kernels,
                  'mean_function': [None, gpflow.mean_functions.Constant()],
                  'noise_variance': [0.01, 1, 10, 100],
                  'optimizer': [gpflow.optimizers.Scipy()],
                  'standardize_x': [False, True],
                  'standardize_y': [False, True],
                  'osa': [True]
                  }
    # random sample from parameter grid
    params_lst = TrainHelper.random_sample_parameter_grid(param_grid=param_grid, sample_share=0.2)

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
        # deepcopy to prevent impact of previous optimizations
        kernel = gpflow.utilities.deepcopy(params_lst[i]['kernel'])
        mean_fct = gpflow.utilities.deepcopy(params_lst[i]['mean_function'])
        noise_var = params_lst[i]['noise_variance']
        optimizer = gpflow.utilities.deepcopy(params_lst[i]['optimizer'])
        stand_x = params_lst[i]['standardize_x']
        stand_y = params_lst[i]['standardize_y']
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

        kernel_string, mean_fct_string, optimizer_string = get_docresults_strings(kernel=kernel,
                                                                                  mean_function=mean_fct,
                                                                                  optimizer=optimizer)
        sum_dict = None
        try:
            for train, test in train_test_list:
                model = ModelsGPR.GaussianProcessRegressionGPFlow(
                    target_column=target_column, seasonal_periods=seasonal_periods, kernel=kernel,
                    mean_function=mean_fct, noise_variance=noise_var, optimizer=optimizer,
                    standardize_x=stand_x, standardize_y=stand_y, one_step_ahead=one_step_ahead
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
                           'kernel': kernel_string, 'mean_function': mean_fct_string, 'noise_variance': noise_var,
                           'optimizer': optimizer_string, 'standardize_x': stand_x, 'standardize_y': stand_y,
                           'one_step_ahead': one_step_ahead, 'optim_mod_params': model.model.parameters}
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
            # print(exc)
            params_dict = {'dataset': 'Failure', 'featureset': featureset,
                           'imputation': str('None' if imputation is None else imputation),
                           'dim_reduction': str('None' if dim_reduction is None else dim_reduction),
                           'init_train_len': init_train_len, 'test_len': test_len, 'split_perc': split_perc,
                           'kernel': kernel_string, 'mean_function': mean_fct_string, 'noise_variance': noise_var,
                           'optimizer': optimizer_string, 'standardize_x': stand_x, 'standardize_y': stand_y,
                           'one_step_ahead': one_step_ahead, 'optim_mod_params': 'failed'}
            save_dict = params_dict.copy()
            save_dict.update(TrainHelper.get_failure_eval_dict())
            if doc_results is None:
                doc_results = pd.DataFrame(columns=save_dict.keys())
            doc_results = doc_results.append(save_dict, ignore_index=True)
    TrainHelper.save_csv_results(doc_results=doc_results, save_dir=base_dir+'OptimResults/',
                                 company_model_desc='gpr', target_column=target_column,
                                 seasonal_periods=seasonal_periods, datasets=datasets,
                                 featuresets=param_grid['featureset'], imputations=param_grid['imputation'],
                                 split_perc=split_perc)
    print('Optimization Done. Saved Results.')


if __name__ == '__main__':
    target_column = str(sys.argv[1])
    split_perc = float(sys.argv[2])
    imputations = ['mean', 'iterative', 'knn']
    featuresets = ['full', 'cal', 'stat', 'none']
    imp_feat_combis = TrainHelper.get_imputation_featureset_combis(imputations=imputations, featuresets=featuresets,
                                                                   target_column=target_column)
    for (imputation, featureset) in imp_feat_combis:
        new_pid = os.fork()
        if new_pid == 0:
            run_gp_optim(target_column=target_column, split_perc=split_perc, imputation=imputation,
                         featureset=featureset)
            sys.exit()
        else:
            os.waitpid(new_pid, 0)
            print('finished run with ' + featureset + ' ' + str('None' if imputation is None else imputation))
