import pandas as pd

from feateng import DateCalendarFeatures, StatisticalFeatures
from preparation import PreparationHelper


def add_features(dataset: pd.DataFrame, cols_to_condense: list = None, condensed_col_name: str = None,
                 seasonal_periods: int = 0, features_for_stats: list = None, use_calendar_features: bool = True,
                 use_stat_features: bool = True, event_lags: list = None, special_days: list = None, lags: list = None,
                 windowsize_rolling: int = 7, windowsize_rolling_seas: int = 7, windowsize_rolling_weekday: int = 4,
                 with_weekday_stats: bool = True):
    """
    Function adding all specified features to dataset
    :param dataset: dataset used for adding features
    :param cols_to_condense: cols which should be condensed to one column
    :param condensed_col_name: name of condensed column
    :param seasonal_periods: seasonality used for seasonal-based features
    :param features_for_stats: features used for calculating statistical features
    :param use_calendar_features: specify if calendar features should be added
    :param use_stat_features: specify if statistical features should be added
    :param event_lags: lags for event counter features
    :param special_days: days with their own event counter
    :param lags: lags to use for lagged sales numbers
    :param windowsize_rolling: windowsize used for rolling statistics
    :param windowsize_rolling_seas: windowsize used for rolling seasonal statistics
    :param windowsize_rolling_weekday: windowsize used for rolling statistics for each weekday
    :param with_weekday_stats: specify if weekday specific stats should be added
    """
    if event_lags is None:
        event_lags = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3]
    if special_days is None:
        special_days = ['Valentine', 'MothersDay', 'Karfreitag']
    if lags is None:
        lags = [1, 2, 3, 4, 5, 6, 7]

    print('---Starting to add features---')
    if cols_to_condense is not None and condensed_col_name is not None:
        dataset[condensed_col_name] = 0
        for col in cols_to_condense:
            dataset[condensed_col_name] += dataset[col]
        PreparationHelper.drop_columns(df=dataset, columns=cols_to_condense)
    if use_calendar_features:
        print('---Adding calendar features---')
        add_calendar_features(dataset=dataset, event_lags=event_lags, special_days=special_days)
    if use_stat_features:
        print('---Adding statistical features---')
        add_statistical_features(dataset=dataset, seasonal_periods=seasonal_periods,
                                 features_for_stats=features_for_stats, lags=lags,
                                 windowsize_rolling=windowsize_rolling,
                                 windowsize_rolling_seas=windowsize_rolling_seas,
                                 windowsize_rolling_weekday=windowsize_rolling_weekday,
                                 with_weekday_stats=with_weekday_stats)
    print('---Features added---')


def add_calendar_features(dataset: pd.DataFrame, event_lags: list, special_days: list):
    """Function adding all calendar-based features"""
    DateCalendarFeatures.add_date_based_features(dataset=dataset)
    DateCalendarFeatures.add_valentine_mothersday(dataset=dataset)
    DateCalendarFeatures.add_public_holiday_counters(dataset=dataset, event_lags=event_lags, special_days=special_days)


def add_statistical_features(dataset: pd.DataFrame, seasonal_periods: int, features_for_stats: list, lags: list,
                             windowsize_rolling: int, windowsize_rolling_seas: int, windowsize_rolling_weekday: int,
                             with_weekday_stats: bool = True):
    """Function adding all statistical features"""
    StatisticalFeatures.add_lags_seaslags_features(dataset=dataset, lags=lags, seasonal_periods=seasonal_periods,
                                                   features=features_for_stats)
    StatisticalFeatures.add_rolling_statistics_features(dataset=dataset, windowsize=windowsize_rolling,
                                                        features=features_for_stats)
    StatisticalFeatures.add_rolling_seasonal_statistics_features(dataset=dataset,
                                                                 windowsize=windowsize_rolling_seas,
                                                                 features=features_for_stats,
                                                                 seasonal_periods=seasonal_periods)
    if with_weekday_stats:
        StatisticalFeatures.add_rolling_weekday_statistics_features(dataset=dataset,
                                                                    windowsize=windowsize_rolling_weekday,
                                                                    features=features_for_stats)
