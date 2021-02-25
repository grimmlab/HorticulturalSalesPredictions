import pandas as pd


def add_lags_seaslags_features(dataset: pd.DataFrame, lags: list, features: list, seasonal_periods: int,
                               non_seasonal_features: list = None):
    """
    Function adding lagged and seasonal-lagged features to dataset
    :param dataset: dataset for adding features
    :param lags: lags / seasonal lags to add of the features specified
    :param features: features for which lagged values should be added
    :param seasonal_periods: seasonal_period used for seasonal-lagged features
    :param non_seasonal_features: features for which no seasonal lags shall be added
    """
    if non_seasonal_features is None:
        non_seasonal_features = ['mean_temp', 'total_prec_height_mm', 'total_sun_dur_h']
    for feat in features:
        for i in lags:
            dataset['stat_' + feat + '_lag' + str(i)] = dataset[feat].shift(i)
            if feat not in non_seasonal_features:
                dataset['stat_' + feat + '_seaslag' + str(i)] = dataset[feat].shift(seasonal_periods + i - 1)


def add_rolling_statistics_features(dataset: pd.DataFrame, windowsize: int, features: list):
    """
    Function adding rolling statistics
    :param dataset: dataset for adding features
    :param windowsize: windowsize used for rolling statistics
    :param features: features for which the statistics are added
    """
    for feat in features:
        # shift by 1 so rolling statistics value is calculated without current value
        dataset['stat_' + feat + '_rolling_mean' + str(windowsize)] = dataset[feat].shift(1).rolling(windowsize).mean()
        dataset['stat_' + feat + '_rolling_median' + str(windowsize)] = \
            dataset[feat].shift(1).rolling(windowsize).median()
        dataset['stat_' + feat + '_rolling_max' + str(windowsize)] = dataset[feat].shift(1).rolling(windowsize).max()


def add_rolling_seasonal_statistics_features(dataset: pd.DataFrame, windowsize: int, features: list,
                                             seasonal_periods: int, non_seasonal_features: list = None):
    """
    Function adding rolling seasonal statistics
    :param dataset: dataset for adding features
    :param windowsize: windowsize used for rolling statistics
    :param features: features for which statistics are added
    :param seasonal_periods: seasonal_period used for seasonal rolling statistics
    :param non_seasonal_features: features for which no seasonal features should be added
    """
    # statistics does not make sense if seasonal periods is too small compared to windowsize
    if non_seasonal_features is None:
        non_seasonal_features = ['mean_temp', 'total_prec_height_mm', 'total_sun_dur_h']
    if seasonal_periods <= windowsize:
        return
    # separate function as different window sizes might be interesting compared to non-seasonal statistics
    for feat in set(features) - set(non_seasonal_features):
        # shift by 1 + seasonal_period so rolling statistics value is calculated without current value
        dataset['stat_' + feat + '_rolling_seasonal_mean' + str(windowsize)] = \
            dataset[feat].shift(seasonal_periods + 1).rolling(windowsize).mean()
        dataset['stat_' + feat + '_rolling_seasonal_median' + str(windowsize)] = \
            dataset[feat].shift(seasonal_periods + 1).rolling(windowsize).median()
        dataset['stat_' + feat + '_rolling_seasonal_max' + str(windowsize)] = \
            dataset[feat].shift(seasonal_periods + 1).rolling(windowsize).max()


def add_rolling_weekday_statistics_features(dataset: pd.DataFrame, windowsize: int, features: list,
                                            non_seasonal_features: list = None):
    """
    Function adding rolling statistics for each weekday
    :param dataset: dataset for adding features
    :param windowsize: windowsize used for rolling statistics of each weekday
    :param features: features for which statistics are added
    :param non_seasonal_features: features for which no seasonal features should be added
    """
    if non_seasonal_features is None:
        non_seasonal_features = ['mean_temp', 'total_prec_height_mm', 'total_sun_dur_h']
    weekday_indices = list()
    for day in range(0, 7):
        weekday_indices.append([index for index in dataset.index.date if index.weekday() == day])
    for indices in weekday_indices:
        for feat in set(features) - set(non_seasonal_features):
            # shift by 1 so rolling statistics value is calculated without current value
            dataset.at[indices, 'stat_' + feat + '_weekday_rolling_mean' + str(windowsize)] = \
                dataset.loc[indices, feat].shift(1).rolling(windowsize).mean()
            dataset.at[indices, 'stat_' + feat + '_weekday_rolling_median' + str(windowsize)] = \
                dataset.loc[indices, feat].shift(1).rolling(windowsize).median()
            dataset.at[indices, 'stat_' + feat + '_weekday_rolling_max' + str(windowsize)] = \
                dataset.loc[indices, feat].shift(1).rolling(windowsize).max()
