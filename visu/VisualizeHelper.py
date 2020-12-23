import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math


def plt_line_mult_columns(df: pd.DataFrame, columns: list = None, subplots: bool = False) -> plt.Figure:
    """
    Function plotting selected columns of specified DataFrame in one line plot or subplots each with one line plot.
    Plotting all plotable columns if no columns specified.
    Plotting all specified in one subplot if subplots not set to True.
    :param df: DataFrame for selecting columns
    :param columns: columns of DataFrame to select
    :param subplots: optional parameter specifying if line plots should be separated in subplots
    :return: figure containing subplot[s]
    """
    plt_data, plt_labels = get_plot_list(df=df, columns=columns)
    number_of_cols = 2
    if subplots:
        fig, axs = plt.subplots(nrows=math.ceil(len(plt_data) / number_of_cols), ncols=number_of_cols, sharex='all')
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    for (col, el, i) in zip(plt_labels, plt_data, range(0, len(plt_data))):
        if subplots:
            ax = axs[int(i / number_of_cols)][i % number_of_cols]
        if 'Date' in df.columns:
            ax.plot(df.dropna(subset=[col])['Date'], el, label=col)
        else:
            ax.plot(df.dropna(subset=[col]).index, el, label=col)
        ax.legend()
    plt.xticks(rotation=90)
    return fig


def plt_autocorrplot_mult_columns(df: pd.DataFrame, columns: list = None, subplots: bool = False) -> plt.Figure:
    """
    Function plotting selected columns of specified DataFrame in one autocorrelation plot or subplots each with one.
    Plotting all plotable columns if no columns specified.
    Plotting all specified in one subplot if subplots not set to True.
    :param df: DataFrame for selecting columns
    :param columns: columns of DataFrame to select
    :param subplots: optional parameter specifying if line plots should be separated in subplots
    :return: figure containing subplot[s]
    """
    plt_data, plt_labels = get_plot_list(df=df, columns=columns)
    if subplots:
        fig, axs = plt.subplots(nrows=3, ncols=math.ceil(len(plt_data) / 3), sharex='all')
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    for (col, el, i) in zip(plt_labels, plt_data, range(0, len(plt_data))):
        if subplots:
            ax = axs[i]
        pd.plotting.autocorrelation_plot(el, label=col, ax=ax)
        ax.legend()
    plt.xticks(rotation=90)
    return fig


def plt_hist_mult_columns(df: pd.DataFrame, columns: list = None,
                          bins: list = None, subplots: bool = False) -> plt.Figure:
    """
    Function plotting selected columns of specified DataFrame in one histogram or subplots each with one.
    Plotting all plotable columns if no columns specified.
    Plotting all specified in one subplot if subplots not set to True.
    :param df: DataFrame for selecting columns
    :param columns: columns of DataFrame to select
    :param bins: optional parameter specifying boundaries of bins for histogram
    :param subplots: optional parameter specifying if line plots should be separated in subplots
    :return: figure containing plotted histogram
    """
    plt_data, plt_labels = get_plot_list(df=df, columns=columns)
    if subplots:
        fig, axs = plt.subplots(nrows=3, ncols=math.ceil(len(plt_data) / 3), sharex='all')
        for (col, el, i) in zip(plt_labels, plt_data, range(0, len(plt_data))):
            axs[i].hist(el, label=col, bins=bins)
            axs[i].legend()
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.hist(plt_data, label=plt_labels, bins=bins)
        ax.legend()
    return fig


def plt_boxplot_mult_columns(df: pd.DataFrame, columns: list = None) -> plt.Figure:
    """
    Function plotting selected columns of specified DataFrame in one boxplot with labels.
    Plotting all plotable columns if no columns specified.
    :param df: DataFrame for selecting columns
    :param columns: columns of DataFrame to select
    :return: figure containing plotted boxplot
    """
    plt_data, plt_labels = get_plot_list(df=df, columns=columns)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.boxplot(plt_data, labels=plt_labels)
    return fig


def plt_scattermat(df: pd.DataFrame, columns: list = None) -> sns.PairGrid:
    """
    Function plotting selected columns of specified DataFrame in a scatter matrix.
    Plotting all plotable columns if no columns specified.
    :param df: DataFrame for selecting columns
    :param columns: columns of DataFrame to select
    :return: figure containing plotted scatter matrix
    """
    plt_data, _ = get_plot_list(df=df, columns=columns)
    plt_data = pd.DataFrame(plt_data).transpose()
    return sns.pairplot(plt_data)


def plt_corr_heatmap(df: pd.DataFrame, columns: list = None) -> plt.Axes:
    """
    Function plotting correlation heatmap of selected columns of specified DataFrame.
    Plotting correlation between all possible columns if no columns specified.
    :param df: DataFrame for selecting columns
    :param columns: columns of DataFrame to select
    :return: figure containing plotted correlation heatmap
    """
    if columns is not None:
        corr = df[columns].corr()
    else:
        corr = df.corr()
    ax = sns.heatmap(corr, cmap=sns.color_palette('coolwarm'))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    return ax


def get_plot_list(df: pd.DataFrame, columns: list = None) -> tuple:
    """
    Function returning selected, plotable columns of DataFrame in an array and their names in a list
    :param df: DataFrame for selecting plotable columns
    :param columns: columns of DataFrame to select
    :return: list containing columns to plot
    """
    plt_data = []
    df_plotable = df.select_dtypes(include=['float', 'int'])
    if columns is None:
        columns = list(df_plotable.columns)
    for el in columns:
        if el in df_plotable.columns:
            plt_data.append(pd.to_numeric(df[el].dropna()))
        else:
            columns.remove(el)
            print(el, " is not plotable. skipped it.")
    return plt_data, columns
