import pandas as pd
from datetime import datetime
import numpy as np

def convert_transaction_to_user(df, recency_end_date=None):
    id_col = df.columns[0]
    date_col = df.columns[1]
    transaction_col = df.columns[2]
    df[date_col] = pd.to_datetime(df[date_col])

    if recency_end_date is not None:
        end_date = recency_end_date
    elif recency_end_date is None:
        end_date = datetime.today()

    df_recency = df.groupby([id_col])[date_col].max().apply(lambda date: (end_date - date).days).astype(float)
    df_frequency = df.groupby([id_col])[date_col].count().astype(float)
    df_monetary = df.groupby([id_col])[transaction_col].sum().astype(float)
    full_df = pd.concat([df_recency, df_frequency, df_monetary], axis=1)
    full_df.columns = ['recency', 'frequency', 'monetary']
    return full_df


def get_func(func):
    func = func.lower()
    if func == 'quintile':
        func = 'quantile'
    func = getattr(pd.Series, func)
    return func


def build_cutoffs(dataset, variable, func):
    __dict = {}
    dataset_copy = dataset
    func = get_func(func)
    if (func == pd.Series.quantile) and ('recency' != variable.lower()):
        qnt = func(dataset_copy[variable], [0.2, 0.4, 0.6, 0.8])
        qnt.index = np.arange(len(qnt))
        mn = dataset[variable].min()
        mx = dataset[variable].max()
        cutoffs = [(mn, qnt[0]),
                   (qnt[0],qnt[1]),
                   (qnt[1],qnt[2]),
                   (qnt[2],qnt[3]),
                   (qnt[3], mx)]
    elif (func == pd.Series.quantile) and ('recency' == variable.lower()):
        qnt = func(dataset_copy[variable], [0.2, 0.4, 0.6, 0.8])
        qnt.index = np.arange(len(qnt))
        mn = dataset[variable].min()
        mx = dataset[variable].max()
        cutoffs = [(mx, qnt[3]),
                   (qnt[3], qnt[2]),
                   (qnt[2], qnt[1]),
                   (qnt[1], qnt[0]),
                   (qnt[0], mn)]
    #### Other case where the function of choice is not quintile.
    else:
        if 'recency' != variable.lower():
            mean_list = list()
            for i in range(0, 4):
                divider = func(dataset_copy[variable])
                mean_list.append(divider)
                dataset_copy = dataset_copy[dataset_copy[variable] >= divider]
            mn = dataset_copy[variable].min()
            mx = dataset_copy[variable].max()
            cutoffs = [(mn, mean_list[0]),
                       (mean_list[0], mean_list[1]),
                       (mean_list[1], mean_list[2]),
                       (mean_list[2], mean_list[3]),
                       (mean_list[3], mx)]
        else:
            mean_list = list()
            for i in range(0,4):
                divider = func(dataset_copy[variable])
                mean_list.append(divider)
                dataset_copy = dataset_copy[dataset_copy[variable] <= divider]
            mn = dataset_copy[variable].min()
            mx = dataset_copy[variable].max()
            cutoffs = [(mx, mean_list[3]),
                       (mean_list[3], mean_list[2]),
                       (mean_list[2], mean_list[1]),
                       (mean_list[1], mean_list[0]),
                       (mean_list[0], mn)]
    return cutoffs

def mf_mapper(cutoffs, val):
    for i,bounds in enumerate(cutoffs):
        if val == bounds[0]:
            return i+1
        elif val == bounds[1]:
            return i+1
        elif (val >= bounds[0]) and (val < bounds[1]):
            return i+1

def r_mapper(cutoffs, val):
    for i, bounds in enumerate(cutoffs):
        if val == bounds[0]:
            return i + 1
        elif val == bounds[1]:
            return i + 1
        elif (val < bounds[0]) and (val >= bounds[1]):
            return i + 1


def input_scores(dataset, cutoffs, variable):
    if variable.lower() == 'recency':
        dataset[variable+'_scores'] = dataset[variable].apply(lambda val: r_mapper(cutoffs, val))
    else:
        dataset[variable + '_scores'] = dataset[variable].apply(lambda val: mf_mapper(cutoffs, val))
    return dataset


def apply_weights(dataset, variables, weights):
    for variable, weight in zip(variables, weights):
        dataset[variable+'_weighted'] = dataset[variable+'_scores'] * weight
    return dataset