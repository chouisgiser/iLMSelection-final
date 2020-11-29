# -*- coding: utf-8 -*-
"""
# @time    : 27.08.20 10:37
# @author  : zhouzy
# @file    : hit_rate_decrease.py
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file
from utils.letor_metrics import ndcg_score
import csv
import os

def ranking_iteration(split_time, feature_index_list, feature_name_list):
    hr_list = list()
    mae_list = list()

    for familiarity in ['familiar', 'unfamiliar']:
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
        X_train = X_train[:, feature_index_list]
        X_test, y_test = load_svmlight_file(infile_valid)
        X_test = X_test[:, feature_index_list]

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
            'feature_name': feature_name_list
        }
        ranker.fit(X_train, y_train, group=query_train, **params_fit)

        ground_number_list = list()

        y_ground = y_test
        y_pred = ranker.predict(X_test)

        amin, amax = min(y_pred), max(y_pred)
        # print(amin)
        # print(amax)
        for index, value in enumerate(y_pred):
            if amax == amin:
                y_pred[index] = 0
            else:
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
            y_ground_splits.append(y_ground[i: i + 2])
            y_pred_splits.append(y_pred[i: i + 2])

        pred_score = 0
        true_num = 0
        absolute_error = 0

        for i in range(0, len(y_ground_splits)):
            test_index = y_ground_splits[i].tolist().index(max(y_ground_splits[i].tolist()))
            pred_index = y_pred_splits[i].tolist().index(max(y_pred_splits[i].tolist()))
            if test_index == pred_index:
                true_num += 1
            # absolute_error += abs(y_ground_splits[i][0] - y_pred_splits[i][0]) + abs(
            #     y_ground_splits[i][1] - y_pred_splits[i][1])

            absolute_error += (abs(y_ground_splits[i][0] - y_pred_splits[i][0]) + abs(
                y_ground_splits[i][1] - y_pred_splits[i][1])) / 2

            pred_score += ndcg_score(y_ground_splits[i], y_pred_splits[i], 1)

        ground_number_list.append(len(y_ground_splits))
        hr_list.append(true_num / len(y_ground_splits))
        mae_list.append(absolute_error / len(y_ground_splits))
        # ground_number_list.append(run_count)
        # hr_list.append(true_num / run_count)
        # mae_list.append(absolute_error / run_count)


    return hr_list, mae_list

# feature_index_list = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
# feature_name_list = ['color', 'intensity', 'shape_size', 'integration', 'choice',
#                          'visibility', 'proximity2dp', 'proximity2fe', 'proximity2be',
#                          'uniqueness', 'wellknownness', 'relevance', ]

# familiar
# feature_name_list = ['relevance', 'visibility', 'intensity', 'color', 'proximity2fe',
#                      'shape_size', 'proximity2dp', 'proximity2be', 'wellknownness',  'integration',
#                      'uniqueness', 'choice']
# feature_index_list = [12, 6, 1, 0, 8, 2, 7, 9, 11, 3, 10, 4,]

# unfamiliar
feature_name_list = ['intensity', 'visibility', 'color', 'relevance', 'proximity2be', 'proximity2fe',
                     'proximity2dp', 'shape_size', 'uniqueness', 'wellknownness',
                       'choice',  'integration',]
feature_index_list = [1, 6, 0, 12, 9, 8, 7, 2, 10, 11, 4, 3,]


for index in range(0, 11):
    print(index)
    # reducing from the least to the most important
    feature_index = feature_index_list[11 - index]
    feature_name = feature_name_list[11 - index]

    # reducing from the most to the least important
    # feature_index = feature_index_list[index]
    # feature_name = feature_name_list[index]

    # feature_index_list.pop(index)
    # feature_name_list.pop(index)

    for split_time in range(1, 18):

        # reducing from the least to the most important
        hr_list, mae_list = ranking_iteration(split_time, feature_index_list[0:11-index], feature_name_list[0:11-index])

        # reducing from the most to the least important
        # hr_list, mae_list = ranking_iteration(split_time, feature_index_list[index+1:12],
        #                                       feature_name_list[index+1:12])

        # reducing the single feature
        # feature_index_list_tmp = feature_index_list[0:index] + feature_index_list[index+1:12]
        # feature_name_list_tmp = feature_name_list[0:index] + feature_name_list[index + 1:12]
        # hr_list, mae_list = ranking_iteration(split_time, feature_index_list_tmp,
        #                                       feature_name_list_tmp)

        headers = ['removed_feature', 'HR', 'MAE']

        # familiar_model_hr_decrease_file = '../../results/reports/tables/feature_remove/familiar_model_features.csv'
        # if not os.path.exists(familiar_model_hr_decrease_file):
        #     with open(familiar_model_hr_decrease_file, 'w', newline='') as file:
        #         writer = csv.DictWriter(file, headers)
        #         writer.writeheader()
        #
        # with open(familiar_model_hr_decrease_file, 'a', newline='') as file:
        #     writer = csv.DictWriter(file, headers)
        #     row = {'removed_feature': feature_name, 'HR': hr_list[0], 'MAE': mae_list[0]}
        #     writer.writerow(row)

        unfamiliar_model_hr_decrease_file = '../../results/reports/tables/feature_remove/unfamiliar_model_features.csv'
        if not os.path.exists(unfamiliar_model_hr_decrease_file):
            with open(unfamiliar_model_hr_decrease_file, 'w', newline='') as file:
                writer = csv.DictWriter(file, headers)
                writer.writeheader()

        with open(unfamiliar_model_hr_decrease_file, 'a', newline='') as file:
            writer = csv.DictWriter(file, headers)
            row = {'removed_feature':  feature_name, 'HR': hr_list[1], 'MAE': mae_list[1]}
            writer.writerow(row)

    # feature_index_list.insert(index, feature_index)
    # feature_name_list.insert(index, feature_name)
    # feature_index_list = [10, 11, 12]
    # feature_name_list = ['uniqueness', 'wellknownness', 'relevance',  ]



    # with open(unfamiliar_model_hr_decrease_file, 'a', newline='') as file:
    #     writer = csv.DictWriter(file, headers)
    #     row = {'removed_feature': 'full', 'hit_rate': hr_list[1]}
    #     writer.writerow(row)

    # hr_list = ranking_iteration(split_time, feature_index_list, feature_name_list)
    # with open(familiar_model_hr_decrease_file, 'a', newline='') as file:
    #     writer = csv.DictWriter(file, headers)
    #     row = {'removed_feature':  'uniqueness',   'hit_rate': hr_list[0]}
    #     writer.writerow(row)


