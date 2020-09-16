# -*- coding: utf-8 -*-
"""
# @time    : 29.07.20 11:15
# @author  : zhouzy
# @file    : rel_rannking_fam_role.py
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file
from utils.letor_metrics import ndcg_score

def ranking_iteration(split_time):
    for fold_index in range(1, 6):
        for familiarity in ['familiar', 'unfamiliar']:
            for role in ['staff', 'student']:
                data_dir = '../../data/processed/dataset_split_iteration/iter' + str(
                    split_time) + '/ranking/5folds_role/fold'
                results_dir = '../../results/dataset_split_iteration/fam_role/iter' + str(split_time) + '/'
                hyperparams_dir = '../../results/dataset_split_iteration/fam_role/iter' + str(
                    split_time) + '/experiments/hyperparams'

                infile_train = data_dir + str(fold_index) + '/' + role + '/rank.train'
                infile_query_train = data_dir + str(fold_index) + '/' + role + '/rank.train.query'

                infile_valid = data_dir + str(fold_index) + '/rank.test'
                infile_query_valid = data_dir + str(fold_index) + '/rank.test.query'

                print('Loading data...')
                # load or create your dataset
                X_train, y_train = load_svmlight_file(infile_train)
                X_train = X_train[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]]
                X_test, y_test = load_svmlight_file(infile_valid)
                X_test = X_test[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]]

                query_train = np.loadtxt(infile_query_train)
                query_test = np.loadtxt(infile_query_valid)

                df_hyperparams = pd.read_csv(hyperparams_dir + '/lgbrank_' + familiarity + '_' + role + '_params.csv')
                hyperparams = df_hyperparams[(df_hyperparams['fold'] == fold_index)].iloc[-1, :]

                estimator_params = {'boosting_type': 'gbdt',
                                    'objective': 'lambdarank',
                                    'min_child_samples': 5,
                                    # 'importance_type': 'split',
                                    'importance_type': 'gain',
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
                    'feature_name': ['color', 'intensity', 'shape_size', 'integration', 'choice',
                                     'visibility', 'proximity2dp', 'proximity2fe', 'proximity2be',
                                     'uniqueness', 'wellknownness', 'relevance']
                    # 'feature_name': ['color', 'intensity', 'shape_size', 'integration', 'choice',
                    #                   'control', 'visibility', 'proximity2dp', 'proximity2fe', 'proximity2be',
                    #                   'uniqueness', 'wellknownness','relevance', 'frequency']
                }
                ranker.fit(X_train, y_train, group=query_train, **params_fit)

                print('Print importances:', list(ranker.feature_importances_))
                file_name = results_dir + 'experiments/feature_importance/ranking_' + role + '_' + familiarity + '/fold' + str(
                    fold_index) + '_importance.csv'
                column_name = ['features', 'importance_gain']
                column_data = {'features': ranker.feature_name_, 'importance_gain': list(ranker.feature_importances_)}
                pd_eval_his = pd.DataFrame(columns=column_name, data=column_data)
                pd_eval_his.to_csv(file_name, encoding='utf-8')

                print('Saving model...')
                ranker.booster_.save_model(
                    results_dir + 'models/ranking_' + role + '_' + familiarity + '/fold' + str(fold_index) + '_model.txt')


for split_time in range(1, 11):
    ranking_iteration(split_time)