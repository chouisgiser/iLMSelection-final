# -*- coding: utf-8 -*-
"""
# @time    : 01.09.20 12:04
# @author  : zhouzy
# @file    : sensitivity_analysis.py
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file
from utils.letor_metrics import ndcg_score

import matplotlib.pyplot as plt
import seaborn as sns

def estimator_sensitivity(split_time, familiarity):
    hr_list = list()
    mae_list = list()
    ndcg_score_list = list()
    n_estimators_list = list()

    for n_estimators in range(1, 51, 1):
        data_dir = '../../data/processed/cv_dataset/ranking/familiarity/fold' + str(split_time) + '/'
        # results_dir = '../../results/fit_results/'
        hyperparams_dir = '../../results/cv_results/familiarity/hyperparams/'

        infile_train = data_dir + familiarity + '_rank.train'
        infile_query_train = data_dir + familiarity + '_rank.train.query'

        infile_valid = data_dir + familiarity + '_rank.test'
        infile_query_valid = data_dir + familiarity + '_rank.test.query'

        print('Loading data...')
        # load or create your dataset
        X_train, y_train = load_svmlight_file(infile_train)
        X_train = X_train[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]]
        X_test, y_test = load_svmlight_file(infile_valid)
        X_test = X_test[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]]

        query_train = np.loadtxt(infile_query_train)
        query_test = np.loadtxt(infile_query_valid)

        df_hyperparams = pd.read_csv(hyperparams_dir + 'lgbrank_' + familiarity + '_best_params.csv')
        hyperparams = df_hyperparams[(df_hyperparams['fold'] == split_time)].iloc[-1, :]

        estimator_params = {'boosting_type': 'gbdt',
                            'objective': 'lambdarank',
                            'min_child_samples': 5,
                            # 'importance_type': 'split',
                            'importance_type': 'gain',
                            # 'reg_lambda': 0.1,
                            # 'n_estimators': 30,  # , 10, 20,  40
                            # 'num_leaves': 30,  # 10, 20, 30
                            # 'max_depth': 10,  # 5,
                            # 'learning_rate': 0.05,  # 0.05,
                            'n_estimators': n_estimators,
                            'num_leaves': int(hyperparams['num_leaves']),
                            'max_depth': int(hyperparams['max_depth']),
                            'learning_rate': hyperparams['learning_rate'],
                            }

        print('Defining Ranker...')
        ranker = lgb.LGBMRanker(**estimator_params)

        print('Start Training...')
        params_fit = {
            'eval_set': [(X_test, y_test)],
            'eval_group': [query_test],
            'eval_metric': 'ndcg',
            'early_stopping_rounds': 100,
            'eval_at': [1, 2],
            'feature_name': ['color', 'intensity', 'shape_size', 'integration', 'choice',
                             'visibility', 'proximity2dp', 'proximity2fe', 'proximity2be',
                             'uniqueness', 'wellknownness', 'relevance']
        }
        ranker.fit(X_train, y_train, group=query_train, **params_fit)

        y_pred = ranker.predict(X_test)

        amin, amax = min(y_pred), max(y_pred)
        for index, value in enumerate(y_pred):
            y_pred[index] = (value - amin) / (amax - amin) * 4 + 1

        # amin, amax = min(y_pred), max(y_pred)
        # for index, value in enumerate(y_pred):
        #     y_pred[index] = (value - amin) / (amax - amin)

        # amin, amax = min(y_pred), max(y_ground)
        # for index, value in enumerate(y_ground):
        #     y_ground[index] = (value - amin) / (amax - amin)

        y_ground_splits = list()
        y_pred_splits = list()
        for i in range(0, len(y_pred), 2):
            y_ground_splits.append(y_test[i: i + 2])
            y_pred_splits.append(y_pred[i: i + 2])

        pred_score = 0
        true_num = 0
        absolute_error = 0
        for i in range(0, len(y_ground_splits)):
            test_index = y_ground_splits[i].tolist().index(max(y_ground_splits[i].tolist()))
            pred_index = y_pred_splits[i].tolist().index(max(y_pred_splits[i].tolist()))
            if test_index == pred_index:
                true_num += 1

            absolute_error += (abs(y_ground_splits[i][0] - y_pred_splits[i][0]) + abs(
                y_ground_splits[i][1] - y_pred_splits[i][1])) / 2

            pred_score += ndcg_score(y_ground_splits[i], y_pred_splits[i], 1)

        n_estimators_list.append(n_estimators)
        ndcg_score_list.append( pred_score / len(y_ground_splits) )
        mae_list.append( absolute_error / len(y_ground_splits) )
        hr_list.append(true_num / len(y_ground_splits))

    data = {'n_estimators': n_estimators_list, 'HR': hr_list, 'MAE': mae_list, 'nDCG': ndcg_score_list}
    dataframe = pd.DataFrame(data)
    return dataframe

def learning_rate_sensitivity(split_time, familiarity):
    hr_list = list()
    mae_list = list()
    ndcg_score_list = list()
    learning_rate_list = list()

    for learning_rate in np.arange(0.001, 0.2, 0.001):
        data_dir = '../../data/processed/cv_dataset/ranking/familiarity/fold' + str(split_time) + '/'
        results_dir = '../../results/fit_results/'
        hyperparams_dir = '../../results/cv_results/familiarity/hyperparams/'

        infile_train = data_dir + familiarity + '_rank.train'
        infile_query_train = data_dir + familiarity + '_rank.train.query'

        infile_valid = data_dir + familiarity + '_rank.test'
        infile_query_valid = data_dir + familiarity + '_rank.test.query'

        print('Loading data...')
        # load or create your dataset
        X_train, y_train = load_svmlight_file(infile_train)
        X_train = X_train[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]]
        X_test, y_test = load_svmlight_file(infile_valid)
        X_test = X_test[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]]

        query_train = np.loadtxt(infile_query_train)
        query_test = np.loadtxt(infile_query_valid)

        df_hyperparams = pd.read_csv(hyperparams_dir + 'lgbrank_' + familiarity + '_best_params.csv')
        hyperparams = df_hyperparams[(df_hyperparams['fold'] == split_time)].iloc[-1, :]

        if familiarity == 'familiar':
            estimator_params = {'boosting_type': 'gbdt',
                                'objective': 'lambdarank',
                                'min_child_samples': 5,
                                # 'importance_type': 'split',
                                'importance_type': 'gain',
                                # 'reg_lambda': 0.1,
                                # 'n_estimators': 30,  # , 10, 20,  40
                                # 'num_leaves': 30,  # 10, 20, 30
                                # 'max_depth': 10,  # 5,
                                # 'learning_rate': 0.05,  # 0.05,
                                'n_estimators': 28,
                                'num_leaves': 14,
                                'max_depth': 6,
                                'learning_rate': learning_rate,
                                }
        else:
            estimator_params = {'boosting_type': 'gbdt',
                                'objective': 'lambdarank',
                                'min_child_samples': 5,
                                # 'importance_type': 'split',
                                'importance_type': 'gain',
                                # 'reg_lambda': 0.1,
                                # 'n_estimators': 30,  # , 10, 20,  40
                                # 'num_leaves': 30,  # 10, 20, 30
                                # 'max_depth': 10,  # 5,
                                # 'learning_rate': 0.05,  # 0.05,
                                'n_estimators': 36,
                                'num_leaves': 23,
                                'max_depth': 4,
                                'learning_rate': learning_rate,
                                }

        # estimator_params = {'boosting_type': 'gbdt',
        #                     'objective': 'lambdarank',
        #                     'min_child_samples': 5,
        #                     # 'importance_type': 'split',
        #                     'importance_type': 'gain',
        #                     # 'reg_lambda': 0.1,
        #                     # 'n_estimators': 30,  # , 10, 20,  40
        #                     # 'num_leaves': 30,  # 10, 20, 30
        #                     # 'max_depth': 10,  # 5,
        #                     # 'learning_rate': 0.05,  # 0.05,
        #                     'n_estimators': int(hyperparams['n_estimators']),
        #                     'num_leaves': int(hyperparams['num_leaves']),
        #                     'max_depth': int(hyperparams['max_depth']),
        #                     'learning_rate': learning_rate,
        #                     }

        print('Defining Ranker...')
        ranker = lgb.LGBMRanker(**estimator_params)

        print('Start Training...')
        params_fit = {
            'eval_set': [(X_test, y_test)],
            'eval_group': [query_test],
            'eval_metric': 'ndcg',
            'early_stopping_rounds': 100,
            'eval_at': [1, 2],
            'feature_name': ['color', 'intensity', 'shape_size', 'integration', 'choice',
                             'visibility', 'proximity2dp', 'proximity2fe', 'proximity2be',
                             'uniqueness', 'wellknownness', 'relevance']
        }
        ranker.fit(X_train, y_train, group=query_train, **params_fit)

        y_pred = ranker.predict(X_test)

        amin, amax = min(y_pred), max(y_pred)
        for index, value in enumerate(y_pred):
            y_pred[index] = (value - amin) / (amax - amin)

        amin, amax = min(y_pred), max(y_test)
        for index, value in enumerate(y_test):
            y_test[index] = (value - amin) / (amax - amin)

        y_ground_splits = list()
        y_pred_splits = list()
        for i in range(0, len(y_pred), 2):
            y_ground_splits.append(y_test[i: i + 2])
            y_pred_splits.append(y_pred[i: i + 2])

        pred_score = 0
        true_num = 0
        absolute_error = 0
        for i in range(0, len(y_ground_splits)):
            test_index = y_ground_splits[i].tolist().index(max(y_ground_splits[i].tolist()))
            pred_index = y_pred_splits[i].tolist().index(max(y_pred_splits[i].tolist()))
            if test_index == pred_index:
                true_num += 1

            absolute_error += abs(y_ground_splits[i][0] - y_pred_splits[i][0]) + abs(
                y_ground_splits[i][1] - y_pred_splits[i][1])

            pred_score += ndcg_score(y_ground_splits[i], y_pred_splits[i], 1)

        learning_rate_list.append(learning_rate)
        ndcg_score_list.append( pred_score / len(y_ground_splits) )
        mae_list.append( absolute_error / len(y_ground_splits) )
        hr_list.append(true_num / len(y_ground_splits))

    data = {'learning_rate': learning_rate_list, 'HR': hr_list, 'MAE': mae_list, 'nDCG': ndcg_score_list}
    dataframe = pd.DataFrame(data)
    return dataframe
    # return learning_rate_list, hr_list, mae_list, ndcg_score_list

def max_depth_sensitivity(split_time, familiarity):
    hr_list = list()
    mae_list = list()
    ndcg_score_list = list()
    max_depth_list = list()

    for max_depth in range(1, 51, 1):
        data_dir = '../../data/processed/cv_dataset/ranking/familiarity/fold' + str(split_time) + '/'
        results_dir = '../../results/fit_results/'
        hyperparams_dir = '../../results/cv_results/familiarity/hyperparams/'

        infile_train = data_dir + familiarity + '_rank.train'
        infile_query_train = data_dir + familiarity + '_rank.train.query'

        infile_valid = data_dir + familiarity + '_rank.test'
        infile_query_valid = data_dir + familiarity + '_rank.test.query'

        print('Loading data...')
        # load or create your dataset
        X_train, y_train = load_svmlight_file(infile_train)
        X_train = X_train[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]]
        X_test, y_test = load_svmlight_file(infile_valid)
        X_test = X_test[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]]

        query_train = np.loadtxt(infile_query_train)
        query_test = np.loadtxt(infile_query_valid)

        df_hyperparams = pd.read_csv(hyperparams_dir + 'lgbrank_' + familiarity + '_best_params.csv')
        hyperparams = df_hyperparams[(df_hyperparams['fold'] == split_time)].iloc[-1, :]

        estimator_params = {'boosting_type': 'gbdt',
                            'objective': 'lambdarank',
                            'min_child_samples': 5,
                            # 'importance_type': 'split',
                            'importance_type': 'gain',
                            # 'reg_lambda': 0.1,
                            # 'n_estimators': 30,  # , 10, 20,  40
                            # 'num_leaves': 30,  # 10, 20, 30
                            # 'max_depth': 10,  # 5,
                            # 'learning_rate': 0.05,  # 0.05,
                            'n_estimators': int(hyperparams['n_estimators']),
                            'num_leaves': int(hyperparams['num_leaves']),
                            'max_depth': max_depth,
                            'learning_rate': hyperparams['learning_rate'],
                            }

        print('Defining Ranker...')
        ranker = lgb.LGBMRanker(**estimator_params)

        print('Start Training...')
        params_fit = {
            'eval_set': [(X_test, y_test)],
            'eval_group': [query_test],
            'eval_metric': 'ndcg',
            'early_stopping_rounds': 100,
            'eval_at': [1, 2],
            'feature_name': ['color', 'intensity', 'shape_size', 'integration', 'choice',
                             'visibility', 'proximity2dp', 'proximity2fe', 'proximity2be',
                             'uniqueness', 'wellknownness', 'relevance']
        }
        ranker.fit(X_train, y_train, group=query_train, **params_fit)

        y_pred = ranker.predict(X_test)

        amin, amax = min(y_pred), max(y_pred)
        for index, value in enumerate(y_pred):
            y_pred[index] = (value - amin) / (amax - amin)

        amin, amax = min(y_pred), max(y_test)
        for index, value in enumerate(y_test):
            y_test[index] = (value - amin) / (amax - amin)

        y_ground_splits = list()
        y_pred_splits = list()
        for i in range(0, len(y_pred), 2):
            y_ground_splits.append(y_test[i: i + 2])
            y_pred_splits.append(y_pred[i: i + 2])

        pred_score = 0
        true_num = 0
        absolute_error = 0
        for i in range(0, len(y_ground_splits)):
            test_index = y_ground_splits[i].tolist().index(max(y_ground_splits[i].tolist()))
            pred_index = y_pred_splits[i].tolist().index(max(y_pred_splits[i].tolist()))
            if test_index == pred_index:
                true_num += 1

            absolute_error += abs(y_ground_splits[i][0] - y_pred_splits[i][0]) + abs(
                y_ground_splits[i][1] - y_pred_splits[i][1])

            pred_score += ndcg_score(y_ground_splits[i], y_pred_splits[i], 1)

        max_depth_list.append(max_depth)
        ndcg_score_list.append( pred_score / len(y_ground_splits) )
        mae_list.append( absolute_error / len(y_ground_splits) )
        hr_list.append(true_num / len(y_ground_splits))

    data = {'max_depth': max_depth_list, 'HR': hr_list, 'MAE': mae_list, 'nDCG': ndcg_score_list}
    dataframe = pd.DataFrame(data)
    return dataframe
    # return max_depth_list, hr_list, mae_list, ndcg_score_list

def num_leaves_sensitivity(split_time, familiarity):
    hr_list = list()
    mae_list = list()
    ndcg_score_list = list()
    num_leaves_list = list()

    for num_leaves in range(2, 51, 1):
        data_dir = '../../data/processed/cv_dataset/ranking/familiarity/fold' + str(split_time) + '/'
        results_dir = '../../results/fit_results/'
        hyperparams_dir = '../../results/cv_results/familiarity/hyperparams/'

        infile_train = data_dir + familiarity + '_rank.train'
        infile_query_train = data_dir + familiarity + '_rank.train.query'

        infile_valid = data_dir + familiarity + '_rank.test'
        infile_query_valid = data_dir + familiarity + '_rank.test.query'

        print('Loading data...')
        # load or create your dataset
        X_train, y_train = load_svmlight_file(infile_train)
        X_train = X_train[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]]
        X_test, y_test = load_svmlight_file(infile_valid)
        X_test = X_test[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]]

        query_train = np.loadtxt(infile_query_train)
        query_test = np.loadtxt(infile_query_valid)

        df_hyperparams = pd.read_csv(hyperparams_dir + 'lgbrank_' + familiarity + '_best_params.csv')
        hyperparams = df_hyperparams[(df_hyperparams['fold'] == split_time)].iloc[-1, :]

        estimator_params = {'boosting_type': 'gbdt',
                            'objective': 'lambdarank',
                            'min_child_samples': 5,
                            # 'importance_type': 'split',
                            'importance_type': 'gain',
                            # 'reg_lambda': 0.1,
                            # 'n_estimators': 30,  # , 10, 20,  40
                            # 'num_leaves': 30,  # 10, 20, 30
                            # 'max_depth': 10,  # 5,
                            # 'learning_rate': 0.05,  # 0.05,
                            'n_estimators': int(hyperparams['n_estimators']),
                            'num_leaves': num_leaves,
                            'max_depth': int(hyperparams['max_depth']),
                            'learning_rate': hyperparams['learning_rate'],
                            }

        print('Defining Ranker...')
        ranker = lgb.LGBMRanker(**estimator_params)

        print('Start Training...')
        params_fit = {
            'eval_set': [(X_test, y_test)],
            'eval_group': [query_test],
            'eval_metric': 'ndcg',
            'early_stopping_rounds': 100,
            'eval_at': [1, 2],
            'feature_name': ['color', 'intensity', 'shape_size', 'integration', 'choice',
                             'visibility', 'proximity2dp', 'proximity2fe', 'proximity2be',
                             'uniqueness', 'wellknownness', 'relevance']
        }
        ranker.fit(X_train, y_train, group=query_train, **params_fit)

        y_pred = ranker.predict(X_test)

        amin, amax = min(y_pred), max(y_pred)
        for index, value in enumerate(y_pred):
            y_pred[index] = (value - amin) / (amax - amin)

        amin, amax = min(y_pred), max(y_test)
        for index, value in enumerate(y_test):
            y_test[index] = (value - amin) / (amax - amin)

        y_ground_splits = list()
        y_pred_splits = list()
        for i in range(0, len(y_pred), 2):
            y_ground_splits.append(y_test[i: i + 2])
            y_pred_splits.append(y_pred[i: i + 2])

        pred_score = 0
        true_num = 0
        absolute_error = 0
        for i in range(0, len(y_ground_splits)):
            test_index = y_ground_splits[i].tolist().index(max(y_ground_splits[i].tolist()))
            pred_index = y_pred_splits[i].tolist().index(max(y_pred_splits[i].tolist()))
            if test_index == pred_index:
                true_num += 1

            absolute_error += abs(y_ground_splits[i][0] - y_pred_splits[i][0]) + abs(
                y_ground_splits[i][1] - y_pred_splits[i][1])

            pred_score += ndcg_score(y_ground_splits[i], y_pred_splits[i], 1)

        num_leaves_list.append(num_leaves)
        ndcg_score_list.append( pred_score / len(y_ground_splits) )
        mae_list.append( absolute_error / len(y_ground_splits) )
        hr_list.append(true_num / len(y_ground_splits))

    data = {'num_leaves': num_leaves_list, 'HR': hr_list, 'MAE': mae_list, 'nDCG': ndcg_score_list}
    dataframe = pd.DataFrame(data)
    return dataframe


def sensitivity_plot(fig_dir, params_type, params_list, metric_type, metric_value_list, ):
    data = {params_type: params_list, metric_type: metric_value_list}
    df = pd.DataFrame(data)
    # plt.figure(figsize=(8, 6))
    ax = sns.lineplot(x = params_type, y=metric_type, data=df)

    ax.set_xlabel(params_type, fontsize= 12, fontfamily='Times New Roman')
    ax.set_ylabel(metric_type, fontsize= 12, fontfamily='Times New Roman')

    font_tick = {'family': 'Times New Roman',
                   'size': 10,
                }
    # xticks = ax.get_xticks()
    # ax.set_xticklabels(ax.get_xticks(), font_tick)
    # ax.set_yticklabels(ax.get_yticks(), font_tick)

    ax.tick_params(axis='y', labelsize= 10)
    ax.tick_params(axis='x', labelsize= 10)
    plt.xticks(fontname = "Times New Roman")
    plt.yticks(fontname = "Times New Roman")

    plt.savefig(fig_dir + params_type + ".pdf", bbox_inches='tight')

    plt.show()

fig_dir = '../../results/reports/figures/sensitivity/'
table_dir = '../../results/reports/tables/sensitivity/'

# df_total = pd.DataFrame(columns=['n_estimators','HR','MAE','nDCG'])
# for split_time in range(1, 18):
#     df_split = estimator_sensitivity(split_time, 'unfamiliar')
#     df_total = pd.concat([df_total, df_split])
# df_mean = df_total.groupby('n_estimators').mean()
# df_mean.to_csv(path_or_buf=table_dir + 'n_estimators_unfamiliar.csv', index=True)

# df_mean = pd.read_csv(table_dir + 'n_estimators_unfamiliar.csv')
# params_list = df_mean['n_estimators'].tolist()
# hr_list = df_mean['HR']
# sensitivity_plot(fig_dir, 'n_estimators', params_list, 'HR', hr_list)


for familiarity in ['familiar', 'unfamiliar']:
    # df_total = pd.DataFrame(columns=['learning_rate', 'HR', 'MAE', 'nDCG'])
    # for split_time in range(1, 18):
    #     df_split = learning_rate_sensitivity(split_time, familiarity )
    #     df_total = pd.concat([df_total, df_split])
    # df_mean = df_total.groupby('learning_rate').mean()
    # df_mean.to_csv(path_or_buf=table_dir + 'learning_rate_' + familiarity + '.csv', index=True)

    df_mean = pd.read_csv(table_dir + 'learning_rate_' + familiarity + '.csv')
    params_list = df_mean['learning_rate'].tolist()
    # ndcg_score_list = df_mean['nDCG']
    hr_list = df_mean['HR']
    sensitivity_plot(fig_dir, 'learning_rate', params_list, 'HR', hr_list)


# df_total = pd.DataFrame(columns=['num_leaves','HR','MAE','nDCG'])
# for split_time in range(1, 18):
#     df_split = num_leaves_sensitivity(split_time, 'unfamiliar')
#     df_total = pd.concat([df_total, df_split])
# df_mean = df_total.groupby('num_leaves').mean()
# df_mean.to_csv(path_or_buf=table_dir + 'num_leaves_unfamiliar.csv', index=True)


# df_mean = pd.read_csv(table_dir + 'num_leaves_unfamiliar.csv')
# params_list = df_mean['num_leaves'].tolist()
# hr_list = df_mean['HR']
# sensitivity_plot(fig_dir, 'num_leaves', params_list, 'HR', hr_list)



# df_total = pd.DataFrame(columns=['max_depth','HR','MAE','nDCG'])
# for split_time in range(1, 18):
#     df_split = max_depth_sensitivity(split_time, 'unfamiliar')
#     df_total = pd.concat([df_total, df_split])
# df_mean = df_total.groupby('max_depth').mean()
# df_mean.to_csv(path_or_buf=table_dir + 'max_depth_unfamiliar.csv', index=True)

#
# df_mean = pd.read_csv(table_dir + 'max_depth_unfamiliar.csv')
# params_list = df_mean['max_depth'].tolist()
# hr_list = df_mean['HR']
# sensitivity_plot(fig_dir, 'max_depth', params_list, 'HR', hr_list)

