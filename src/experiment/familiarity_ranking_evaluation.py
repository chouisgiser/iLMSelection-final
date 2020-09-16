import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, GroupKFold, RandomizedSearchCV

from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler

from utils.letor_metrics import ndcg_score


def evaluation_iteration(split_time):
    for model_familiarity in ['familiar', 'unfamiliar']:
        print(model_familiarity)
        total_num = 0
        total_true_num = 0
        model_dir = '../../results/cv_results/familiarity/models/ranking_' + model_familiarity + '/'
        pred_data_dir = '../../data/processed/cv_dataset/ranking/fam_role/fold' + str(split_time)
        pred_results_dir = '../../results/cv_results/familiarity/evaluation/ranking_' + model_familiarity + '/'

        pred_iter_list = list()
        pred_fam_list = list()
        pred_role_list = list()
        ground_number_list = list()
        true_number_list = list()
        hr_list = list()
        mae_list = list()

        ranker = lgb.Booster(model_file=model_dir + 'fold' + str(split_time) + '_model.txt')
        # pred_summary = open(pred_results_dir + 'overall_model_{}'.format(fold_index) + '_evaluation.txt', 'a')
        # pred_summary.write('Test set of fold {}'.format(fold_index) + '\n')
        for pred_role in ['staff', 'student']:
            for pred_familiarity in ['familiar', 'unfamiliar']:
                for root, dirs, files in os.walk(pred_data_dir):
                    if pred_familiarity + '_' + pred_role + '_rank.test' in files:
                        pred_iter_list.append(split_time)
                        pred_role_list.append(pred_role)
                        pred_fam_list.append(pred_familiarity)
                        pred_file = pred_data_dir + '/' + pred_familiarity + '_' + pred_role + '_rank.test'
                        X_pred, y_ground = load_svmlight_file(pred_file)
                        X_pred = X_pred[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]]
                        y_pred = ranker.predict(X_pred)

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
                            # ground_pre_lable = y_ground_splits[i][0]
                            # ground_next_label = y_ground_splits[i][1]
                            # pred_pre_lable = y_pred_splits[i][0]
                            # pred_next_lable = y_pred_splits[i][1]

                            absolute_error += (abs(y_ground_splits[i][0] - y_pred_splits[i][0]) + abs(
                                y_ground_splits[i][1] - y_pred_splits[i][1]))/2

                            pred_score += ndcg_score(y_ground_splits[i], y_pred_splits[i], 1)

                        ground_number_list.append(len(y_ground_splits))
                        total_num += len(y_ground_splits)
                        true_number_list.append(true_num)
                        total_true_num += true_num
                        hr_list.append(true_num / len(y_ground_splits))
                        mae_list.append(absolute_error / len(y_ground_splits))

        pred_summary_file = pred_results_dir + 'fold{}'.format(split_time) + '_model_evaluation.csv'
        df_pred_summary_file = pd.DataFrame()
        df_pred_summary_file['fold'] = pred_iter_list
        df_pred_summary_file['role'] = pred_role_list
        df_pred_summary_file['familiarity'] = pred_fam_list
        df_pred_summary_file['test set size'] = ground_number_list
        df_pred_summary_file['hit number'] = true_number_list
        df_pred_summary_file['HR'] = hr_list
        df_pred_summary_file['MAE'] = mae_list
        df_pred_summary_file.to_csv(pred_summary_file)

        print(total_true_num)

# evaluation()
for split_time in range(1, 18):
    evaluation_iteration(split_time)