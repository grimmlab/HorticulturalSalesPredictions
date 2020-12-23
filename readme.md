# TimeSeriesPredictionFramework
This framework provides a full pipeline for Time Series Predictions using several different classical forecsting as well as Machine Learning algorithms.
We initially designed it for Horticultural Sales Predictions, see our publication below. Nevertheless, it can be adapted for other domains and datasets as described in the Remarks.
Providing a dataset with daily observations, you can use several preprocessing methods, e.g. for missing value imputation or feature engineering, and optimize as well as evaluate different models.
For more information on the preprocessing, feature engineering and training implementations, we recommend reading our publication.

Currently, the following algorithms are included:
- Exponential Smoothing (ES)
- (Seasonal) auto-regressive integrated moving average (with external factors) ((S)ARIMA(X))
- Lasso Regression
- Ridge Regression
- ElasticNet Regression
- BayesianRidge Regression
- Automatic Relevance Determination Regression (ARD)
- Artificial Neural Networks (ANN)
- Long short-term memory networks (LSTM)
- XGBoost Regression
- Gaussian Process Regression (GPR)

Furthermore, the Simple Baselines Random Walk, Seasonal Random Walk and Historical Average as well as the Evaluation Metrics Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE) and symmetric MAPE (sMAPE) are implemented.

## Provided Datasets
The pipeline was initially designed for Horticultural Sales Predictions. Therefore, we provide two datasets the publication mentioned below is based on, see `Data`.
*OwnDoc* consists of daily observations of a manual documentation of tulips production, wholesale trade and end customer sales ranging from February to May 2020.
*CashierData* instead was derived from the store’s electronic cash register and shows daily turnovers for several product groups from December 2016 to August 2020.
More information on both datasets can be found in our publication. 

## Requirements
We recommend a workflow using [Docker](https://www.docker.com/) to ensure a stable working environment.
Subsequently, we describe the setup and operation according to it. 
If you want to follow our recommendation, **Docker** needs to be installed and running on your machine. We provide a Dockerfile as described below.

You can optionally use GPU support for ANN and LSTM optimization. 
The Docker image is based on **CUDA 11.1**, so if you have another version installed, please update it on your machine or adjust the Dockerfile.
Otherwise, ANN and LSTM optimization will be performed on CPU.

As an alternative, you can run all programs directly on your machine. 
The pipeline was developed and tested with Python 3.8 and Ubuntu 20.04.
All used Python packages and their versions are specified in `Configs/packages.txt`.

## Setup and Operation
1. Open a Terminal and navigate to the directory where you want to set up the project
2. Clone this repository
    ```bash
    git clone https://github.com/grimmlab/TimeSeriesPredictionFramework
    ```
3. Navigate to `Configs` after cloning the repository
4. Build a Docker image using the provided Dockerfile
    ```bash
    docker build -t IMAGENAME .
    ```
5. Run an interactive Docker container based on the created image. Mount the directory where the repository is placed on your machine. If you want to use GPU support, specify the GPUs to mount.
    ```bash
    docker run -it -v PATH/TO/REPO/FOLDER:/REPO/DIRECTORY/IN/CONTAINER --gpus=all --name CONTAINERNAME IMAGENAME
    ```
6. In the Docker container, navigate to the to top level of the repository in the mounted directory
7. Start one of the training scripts with specifying the target variable and train percentage
    ```bash
    python3 SCRIPTNAME TARGETVARIABLE TRAIN_PERCENTAGE
   (e.g.: python3 RunXGBOptim.py SoldTulips 0.8)
    ```
   The datasets we provide are based on the target variables *SoldTulips* for *OwnDoc* as well as *PotTotal* and *CutFlowers* for *CashierData*.
   
   A good choice for an initial test is `python3 RunESOptim.py SoldTulips 0.8` due to its short runtime.
    
   You should see outputs regarding the current progress and best values on your Terminal. 
8. Once all optimization runs are finished, you can find a file with the results, their configurations and a comparison with simple baselines in `OptimResults`

You can also quit a running program with `Ctrl+C`. Already finished optimizations will be saved.

## Remarks
- The optimization file names provide important information like the algorithm name, target variable, used datasets, featuresets, imputation methods, split percentage and seasonal period.
- The framework is parametrized via the config file `Configs/dataset_specific_config.ini`. 
There is a `General` section where we specify the *base_dir*, which is currently the same as the repository itself. 
If you want to move the storage of the datasets and optimization results, which need to be in the same base directory, you need to adjust the path there. 
Be careful with mounting the volume in your Docker container then, as the repository itself as well as the directory containing `Data` and `OptimResults` need to be accessible.  
Beyond that, there is a section for each target variable. Feel free to add a new one, if you want to use the framework for another dataset or change the settings for the existing ones.
- Furthermore, when using the framework for another dataset, you may want to adapt the function `get_ready_train_test_lst()` in `training/TrainHelper.py`.
- Hyperparameter ranges and random sample shares have to be adjusted in the Run script of the specific algorithm.
- For the target variables *PotTotal* and *SoldTulips*, we use a second split besides the full dataset. This is automatically included when starting a training script for one of these. See our publication linked below for further details.
- In the Checkpoints directory, we temporary store checkpoints during ANN and LSTM training. The files should be deleted automatically, if the program is not interrupted.


## Contributors
This pipeline is developed and maintened by members of the [Bioinformatics](https://bit.cs.tum.de) lab of [Prof. Dr. Dominik Grimm](https://bit.cs.tum.de/team/dominik-grimm/):
- [Florian Haselbeck, M.Sc.](https://bit.cs.tum.de/team/florian-haselbeck/)

## Citation
We are currently working on a publication regarding Horticultural Sales Predictions, for which we developed this framework.

When using this workflow, please cite our publication:

tbd.