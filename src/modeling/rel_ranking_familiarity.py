import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file
from utils.letor_metrics import ndcg_score

def ranking_label_influence():
    label_scale = '1-7-scale'
    for familiarity in ['familiar', 'unfamiliar']:
        total_num = 0
        total_true_num = 0

        for fold_index in range(1, 6):
            data_dir = '../../data/processed/dataset/ranking/' + label_scale + '/5folds_familiarity/fold'

            infile_train = data_dir + str(fold_index) + '/' + familiarity + '/rank.train'
            infile_query_train = data_dir + str(fold_index) + '/' + familiarity + '/rank.train.query'

            infile_valid = data_dir + str(fold_index) + '/rank.test'
            infile_query_valid = data_dir + str(fold_index) + '/rank.test.query'

            print('Loading data...')
            # load or create your dataset
            X_train, y_train = load_svmlight_file(infile_train)
            X_train = X_train[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13]]
            X_test, y_test = load_svmlight_file(infile_valid)
            X_test = X_test[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13]]

            query_train = np.loadtxt(infile_query_train)
            query_test = np.loadtxt(infile_query_valid)

            estimator_params = {'boosting_type': 'gbdt',
                                'objective': 'lambdarank',
                                'min_child_samples': 5,
                                # 'importance_type': 'split',
                                'importance_type': 'gain',
                                # 'reg_lambda': 0.1,
                                'n_estimators': 30,  # , 10, 20,  40
                                'num_leaves': 30,  # 10, 20, 30
                                'max_depth': 10,  # 5,
                                'learning_rate': 0.05,  # 0.05,
                                # 'n_estimators': int(hyperparams['n_estimators']),
                                # 'num_leaves': int(hyperparams['num_leaves']),
                                # 'max_depth': int(hyperparams['max_depth']),
                                # 'learning_rate': hyperparams['learning_rate'],
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
                                 'uniqueness', 'wellknownness', 'relevance', 'frequency']

            }
            ranker.fit(X_train, y_train, group=query_train, **params_fit)

            pred_fold_list = list()
            pred_fam_list = list()
            pred_role_list = list()
            ground_number_list = list()
            true_number_list = list()

            for pred_role in ['staff', 'student']:
                for pred_familiarity in ['familiar', 'unfamiliar']:
                    pred_fold_list.append(fold_index)
                    pred_role_list.append(pred_role)
                    pred_fam_list.append(pred_familiarity)

                    pred_data_dir = '../../data/processed/dataset/ranking/' + label_scale + '/5folds_fam_role/fold'
                    pred_results_dir = '../../results/experiments/label_influence/' + label_scale + '/ranking_' + familiarity + '/'

                    pred_file = pred_data_dir + str(
                        fold_index) + '/' + pred_familiarity + '_' + pred_role + '/rank.test'

                    X_pred, y_ground = load_svmlight_file(pred_file)
                    X_pred = X_pred[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13]]
                    y_pred = ranker.predict(X_pred)

                    y_ground_splits = list()
                    y_pred_splits = list()
                    for i in range(0, len(y_pred), 2):
                        y_ground_splits.append(y_ground[i: i + 2])
                        y_pred_splits.append(y_pred[i: i + 2])

                    pred_score = 0
                    true_num = 0
                    for i in range(0, len(y_ground_splits)):
                        test_index = y_ground_splits[i].tolist().index(max(y_ground_splits[i].tolist()))
                        pred_index = y_pred_splits[i].tolist().index(max(y_pred_splits[i].tolist()))
                        if test_index == pred_index:
                            true_num += 1
                        pred_score += ndcg_score(y_ground_splits[i], y_pred_splits[i], 1)

                    ground_number_list.append(len(y_ground_splits))
                    total_num += len(y_ground_splits)
                    true_number_list.append(true_num)
                    total_true_num += true_num

            pred_summary_file = pred_results_dir + 'fold{}'.format(fold_index) + '_model_evaluation.csv'
            df_pred_summary_file = pd.DataFrame()
            df_pred_summary_file['fold'] = pred_fold_list
            df_pred_summary_file['role'] = pred_role_list
            df_pred_summary_file['familiarity'] = pred_fam_list
            df_pred_summary_file['test set size'] = ground_number_list
            df_pred_summary_file['hit number'] = true_number_list
            df_pred_summary_file.to_csv(pred_summary_file)

        print(total_true_num)

# def ranking_iteration(split_time):
#     for fold_index in range(1, 6):
#         for familiarity in [ 'familiar', 'unfamiliar']:
#             data_dir = '../../data/processed/dataset_split_iteration/iter' + str(split_time) + '/ranking/5folds_familiarity/fold'
#             results_dir = '../../results/dataset_split_iteration/familiarity/iter' + str(split_time) + '/'
#             hyperparams_dir = '../../results/dataset_split_iteration/familiarity/iter' + str(split_time) + '/experiments/hyperparams/'
#
#             infile_train = data_dir + str(fold_index) + '/' + familiarity + '/rank.train'
#             infile_query_train = data_dir + str(fold_index) + '/' + familiarity + '/rank.train.query'
#
#             infile_valid = data_dir + str(fold_index) + '/rank.test'
#             infile_query_valid = data_dir + str(fold_index) + '/rank.test.query'
#
#             print('Loading data...')
#             # load or create your dataset
#             X_train, y_train = load_svmlight_file(infile_train)
#             X_train = X_train[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]]
#             X_test, y_test = load_svmlight_file(infile_valid)
#             X_test = X_test[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]]
#
#             query_train = np.loadtxt(infile_query_train)
#             query_test = np.loadtxt(infile_query_valid)
#
#             df_hyperparams = pd.read_csv(hyperparams_dir + '/lgbrank_' + familiarity + '_params.csv')
#             hyperparams = df_hyperparams[(df_hyperparams['fold'] == fold_index)].iloc[-1, :]
#
#             estimator_params = {'boosting_type': 'gbdt',
#                                 'objective': 'lambdarank',
#                                 'min_child_samples': 5,
#                                 # 'importance_type': 'split',
#                                 'importance_type': 'gain',
#                                 # 'reg_lambda': 0.1,
#                                 # 'n_estimators': 30,  # , 10, 20,  40
#                                 # 'num_leaves': 30,  # 10, 20, 30
#                                 # 'max_depth': 10,  # 5,
#                                 # 'learning_rate': 0.05,  # 0.05,
#                                 'n_estimators': int(hyperparams['n_estimators']),
#                                 'num_leaves': int(hyperparams['num_leaves']),
#                                 'max_depth': int(hyperparams['max_depth']),
#                                 'learning_rate': hyperparams['learning_rate'],
#                                 }
#
#             print('Defining Ranker...')
#             ranker = lgb.LGBMRanker(**estimator_params)
#
#             print('Start Training...')
#             params_fit = {
#                 'eval_set': [(X_test, y_test)],
#                 'eval_group': [query_test],
#                 'eval_metric': 'ndcg',
#                 'early_stopping_rounds': 100,
#                 'eval_at': [1, 2],
#                 'feature_name': ['color', 'intensity', 'shape_size', 'integration', 'choice',
#                                  'visibility', 'proximity2dp', 'proximity2fe', 'proximity2be',
#                                  'uniqueness', 'wellknownness', 'relevance']
#             }
#             ranker.fit(X_train, y_train, group=query_train, **params_fit)
#
#             print('Print importances:', list(ranker.feature_importances_))
#             file_name = results_dir + 'experiments/feature_importance/ranking_' + familiarity + '/fold' + str(
#                 fold_index) + '_importance.csv'
#             column_name = ['features', 'importance_gain']
#             column_data = {'features': ranker.feature_name_, 'importance_gain': list(ranker.feature_importances_)}
#             pd_eval_his = pd.DataFrame(columns=column_name, data=column_data)
#             pd_eval_his.to_csv(file_name, encoding='utf-8')
#
#             print('Saving model...')
#             ranker.booster_.save_model(
#                 results_dir + 'models/ranking_' + familiarity + '/fold' + str(fold_index) + '_model.txt')

def ranking_iteration(split_time):
    for familiarity in ['familiar', 'unfamiliar']:
        data_dir = '../../data/processed/cv_dataset/ranking/familiarity/fold' + str(split_time) + '/'
        results_dir = '../../results/cv_results/familiarity/'
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
                                # 'num_leaves': 14,
                                'num_leaves': 28,
                                # 'max_depth': 6,
                                'max_depth': 9,
                                'learning_rate': 0.113,
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
                                # 'max_depth': 4,
                                'max_depth': 4,
                                'learning_rate': 0.049,
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
        #                     'learning_rate': hyperparams['learning_rate'],
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

        print('Print importances:', list(ranker.feature_importances_))
        file_name = results_dir + 'feature_importance/ranking_' + familiarity + '/fold' + \
                    str(split_time) + '_importance.csv'
        column_name = ['features', 'importance_gain']
        column_data = {'features': ranker.feature_name_, 'importance_gain': list(ranker.feature_importances_)}
        pd_eval_his = pd.DataFrame(columns=column_name, data=column_data)
        pd_eval_his.to_csv(file_name, encoding='utf-8')

        print('Saving model...')
        ranker.booster_.save_model(
            results_dir + 'models/ranking_' + familiarity + '/fold' + str(split_time) + '_model.txt')

# ranking()

for split_time in range(1, 18):
    ranking_iteration(split_time)

# ranking_label_influence()