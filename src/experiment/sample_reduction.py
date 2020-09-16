# -*- coding: utf-8 -*-
"""
# @time    : 02.09.20 11:56
# @author  : zhouzy
# @file    : sample_reduction.py
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file
from utils.letor_metrics import ndcg_score
import csv
import os
import math

def ranking_iteration(split_time, sample_percentage):

    hr_list = list()

    mae_list = list()

    feature_index_list = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
    feature_name_list = ['color', 'intensity', 'shape_size', 'integration', 'choice',
                         'visibility', 'proximity2dp', 'proximity2fe', 'proximity2be',
                         'uniqueness', 'wellknownness', 'relevance', ]

    for familiarity in ['familiar', 'unfamiliar']:
        data_dir = '../../data/processed/fit_dataset/iter' + str(split_time) + '/ranking/familiarity/'
        # results_dir = '../../results/fit_results/'
        hyperparams_dir = '../../results/cv_results/familiarity/hyperparams/'

        infile_train = data_dir + familiarity + '_rank.train'
        infile_query_train = data_dir + familiarity + '_rank.train.query'

        infile_valid = data_dir + familiarity + '_rank.test'
        infile_query_valid = data_dir + familiarity + '_rank.test.query'

        print('Loading data...')
        # load or create your dataset

        query_train = np.loadtxt(infile_query_train)
        query_test = np.loadtxt(infile_query_valid)

        sample_size = math.floor(len(query_train) * sample_percentage)

        X_train, y_train = load_svmlight_file(infile_train)
        X_train = X_train[0: 2 * sample_size, feature_index_list]
        y_train = y_train[0: 2 * sample_size, ]
        query_train = query_train[0:sample_size, ]

        X_test, y_test = load_svmlight_file(infile_valid)
        X_test = X_test[:, feature_index_list]



        df_hyperparams = pd.read_csv(hyperparams_dir + 'lgbrank_' + familiarity + '_best_params.csv')
        hyperparams = df_hyperparams[(df_hyperparams['iter'] == split_time)].iloc[-1, :]

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
        for index, value in enumerate(y_pred):
            y_pred[index] = (value - amin) / (amax - amin)

        amin, amax = min(y_pred), max(y_ground)
        for index, value in enumerate(y_ground):
            y_ground[index] = (value - amin) / (amax - amin)

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

            absolute_error += abs(y_ground_splits[i][0] - y_pred_splits[i][0]) + abs(
                y_ground_splits[i][1] - y_pred_splits[i][1])

            pred_score += ndcg_score(y_ground_splits[i], y_pred_splits[i], 1)

        ground_number_list.append(len(y_ground_splits))
        # hr = true_num/len(y_ground_splits)
        hr = pred_score / len(y_ground_splits)
        hr_list.append(hr)
        mae = absolute_error / len(y_ground_splits)
        mae_list.append(mae)

    return hr_list, mae_list

for split_time in range(1, 21):

    for sample_percentage in np.arange(1, 0, -0.1):

        hr_list, mae_list = ranking_iteration(split_time, sample_percentage)

        headers = ['sample_percentage', 'hit_rate', 'MAE']
        familiar_model_hr_decrease_file = '../../results/reports/tables/sample_size_influence/familiar_model_sampling.csv'
        if not os.path.exists(familiar_model_hr_decrease_file):
            with open(familiar_model_hr_decrease_file, 'w', newline='') as file:
                writer = csv.DictWriter(file, headers)
                writer.writeheader()

        with open(familiar_model_hr_decrease_file, 'a', newline='') as file:
            writer = csv.DictWriter(file, headers)
            row = {'sample_percentage': sample_percentage, 'hit_rate': hr_list[0], 'MAE': mae_list[0]}
            writer.writerow(row)

        unfamiliar_model_hr_decrease_file = '../../results/reports/tables/sample_size_influence/unfamiliar_model_sampling.csv'
        if not os.path.exists(unfamiliar_model_hr_decrease_file):
            with open(unfamiliar_model_hr_decrease_file, 'w', newline='') as file:
                writer = csv.DictWriter(file, headers)
                writer.writeheader()

        with open(unfamiliar_model_hr_decrease_file, 'a', newline='') as file:
            writer = csv.DictWriter(file, headers)
            row = {'sample_percentage': sample_percentage, 'hit_rate': hr_list[1], 'MAE': mae_list[1]}
            writer.writerow(row)