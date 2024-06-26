import numpy as np
import pandas as pd
import lightgbm as lgb
import csv
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, GroupKFold, RandomizedSearchCV

from sklearn.metrics import make_scorer
from utils.letor_metrics import ndcg_score

def ranking_iteration(split_time):
    data_dir = '../../../data/processed/cv_dataset/ranking/familiarity/fold' + str(split_time)
    results_dir = '../../../results/cv_results/familiarity/'

    # , 'unfamiliar'
    for model_familiarity in ['familiar']:
        best_params = {'n_estimators': 0, 'num_leaves': 0, 'max_depth': 0, 'learning_rate': 0}
        best_score = float('-inf')
        eval_his = list()
        # [10, 20, 30, 40, 50]
        for n_estimators in range(5, 55, 5):
            # [5, 10, 15, 20]
            for num_leaves in range(5, 25, 5) :
                # [5, 10]
                for max_depth in range(5, 25, 5):
                    # np.arange(0.02, 0.22, 0.02)
                    for learning_rate in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]:
                        tmp_score = 0
                        train_file = data_dir + '/' + model_familiarity + '_rank.train'
                        train_query_file = data_dir + '/' + model_familiarity + '_rank.train.query'

                        eval_file = data_dir + '/' + model_familiarity + '_rank.test'
                        eval_query_file = data_dir + '/' + model_familiarity + '_rank.test.query'

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
                                            'importance_type': 'gain',
                                            'n_estimators': n_estimators,
                                            'num_leaves': num_leaves,
                                            'max_depth': max_depth,
                                            'learning_rate': learning_rate,
                                            }

                        ranker = lgb.LGBMRanker(**estimator_params)

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
                        ranker.fit(X_train, y_train, group=query_train, **params_fit)
                        tmp_score += ranker.best_score_['valid_0']['ndcg@1']


                        eval_dic = {'fold': split_time, 'n_estimators': n_estimators, 'num_leaves': num_leaves, 'max_depth': max_depth, 'learning_rate': learning_rate,
                                    'ndcg@1_score': tmp_score}
                        eval_his.append(eval_dic)

                        if best_score < tmp_score:
                            best_score = tmp_score
                            best_params['n_estimators'] = n_estimators
                            best_params['num_leaves'] = num_leaves
                            best_params['max_depth'] = max_depth
                            best_params['learning_rate'] = learning_rate

        print('Writing parameter setting history ...')
        headers = ['fold', 'n_estimators', 'num_leaves', 'max_depth', 'learning_rate', 'ndcg@1_score']
        params_eval_his_file = results_dir + 'hyperparams/lgbrank_' + model_familiarity + '_params_setting_history.csv'
        with open(params_eval_his_file, 'a', newline='') as file:
            file_csv = csv.DictWriter(file, headers)
            with open(params_eval_his_file) as readfile:
                params_dic = csv.DictReader(readfile)
                if params_dic.fieldnames == None:
                    file_csv.writeheader()
                readfile.close()
            file_csv.writerows(eval_his)

        print('Writing hyper parameters ...')
        headers = ['fold', 'learning_rate', 'max_depth', 'n_estimators', 'num_leaves']
        hp_params_dic = best_params
        hp_params_dic['fold'] = split_time
        hp_params_file = results_dir + 'hyperparams/lgbrank_' + model_familiarity + '_best_params.csv'

        with open(hp_params_file, 'a', newline='') as file:
            file_csv = csv.DictWriter(file, headers)
            with open(hp_params_file) as readfile:
                params_dic = csv.DictReader(readfile)
                if params_dic.fieldnames == None:
                    file_csv.writeheader()
                readfile.close()
            file_csv.writerow(hp_params_dic)


# ranking()
for split_time in range(4, 18):
    ranking_iteration(split_time)
