# -*- coding: utf-8 -*-
"""
# @time    : 28.07.20 10:43
# @author  : zhouzy
# @file    : rel_ranking_role_cv.py
"""

import numpy as np
import lightgbm as lgb
import csv
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

from sklearn.metrics import make_scorer
from utils.letor_metrics import ndcg_score

def ranking_iteration(split_time):
    data_dir = '../../../data/processed/dataset_split_iteration/iter' + str(
        split_time) + '/ranking/5folds_role/fold'
    results_dir = '../../../results/dataset_split_iteration/role/iter' + str(split_time) + '/'
    # hyperparams_dir = '../../results/dataset_split_iteration/split' + str(split_time) + '/experiments/hyperparams/'
    #
    #
    # data_dir = '../../../data/processed/dataset/ranking/1-5-scale/5folds_familiarity/fold'
    #
    # results_dir = '../../../results/'

    for fold_index in range(1, 6):
        for model_role in ['staff', 'student']:
            train_file = data_dir + str(fold_index) + '/' + model_role + '/rank.train'
            train_query_file = data_dir + str(fold_index) + '/' + model_role + '/rank.train.query'

            eval_file = data_dir + str(fold_index) + '/rank.test'
            eval_query_file = data_dir + str(fold_index) + '/rank.test.query'

            print('Loading data...')
            # load or create your dataset
            X_train, y_train = load_svmlight_file(train_file)
            X_train = X_train[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13]]
            query_train = np.loadtxt(train_query_file)

            X_eval, y_eval = load_svmlight_file(eval_file)
            X_eval = X_eval[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13]]
            query_eval = np.loadtxt(eval_query_file)

            estimator_params = {'boosting_type': 'gbdt',
                                'objective': 'lambdarank',
                                'min_child_samples': 5,
                                'feature_fraction': 0.8,
                                'bagging_fraction': 0.9,
                                'bagging_freq': 5,
                                'importance_type': 'gain',
                                }

            print('Defining Ranker...')
            ranker = lgb.LGBMRanker(**estimator_params)

            print('Defining Cross Validation Dataset...')
            cv_group_info = query_train.astype(int)
            flatted_group = np.repeat(range(len(cv_group_info)), repeats=cv_group_info)

            logo = LeaveOneGroupOut()
            cv = logo.split(X_train, y_train, groups=flatted_group)
            cv_group = logo.split(X_train, groups=flatted_group)

            def group_gen(flatted_group, cv):
                for train, test in cv:
                    yield np.unique(flatted_group[train], return_counts=True)[1]

            params_grid = {'n_estimators': [10, 20, ],  # , 10, 20,  40
                           'num_leaves': [10, 20, 30],  # 10, 20, 30
                           'max_depth': [10],  # 5,
                           'learning_rate': [0.05, 0.1],  # 0.05,
                           }

            grid = GridSearchCV(ranker, params_grid, cv=cv, verbose=2,
                                scoring=make_scorer(ndcg_score, greater_is_better=True), refit=False)

            gen = group_gen(flatted_group, cv_group)

            print('Start Training...')
            params_fit = {
                'eval_set': [(X_eval, y_eval)],
                'eval_group': [query_eval],
                'eval_metric': 'ndcg',
                'early_stopping_rounds': 100,
                'eval_at': [1, 2],
                'feature_name': ['color', 'intensity', 'shape_size', 'integration', 'choice',
                                 'visibility', 'proximity2dp', 'proximity2fe', 'proximity2be',
                                 'uniqueness', 'wellknownness', 'relevance', 'frequency']
            }

            grid.fit(X_train, y_train, group=next(gen), **params_fit)

            print('Writing hyper parameters ...')
            headers = ['fold', 'learning_rate', 'max_depth', 'n_estimators', 'num_leaves']
            hp_params_dic = grid.best_params_
            hp_params_dic['fold'] = fold_index
            hp_params_file = results_dir + 'experiments/hyperparams/lgbrank_' + model_role + '_params.csv'

            with open(hp_params_file, 'a', newline='') as file:
                file_csv = csv.DictWriter(file, headers)
                with open(hp_params_file) as readfile:
                    params_dic = csv.DictReader(readfile)
                    if params_dic.fieldnames == None:
                        file_csv.writeheader()
                    readfile.close()
                file_csv.writerow(hp_params_dic)


for split_time in range(1, 11):
    ranking_iteration(split_time)