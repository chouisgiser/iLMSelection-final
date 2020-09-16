import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, GroupKFold, RandomizedSearchCV

from sklearn.metrics import make_scorer
from utils.letor_metrics import ndcg_score


def evaluation_iteration(split_time):
    total_num = 0
    total_true_num = 0
    model_dir = '../../results/dataset_split_iteration/overall/iter' + str(
        split_time) + '/models/'
    pred_data_dir = '../../data/processed/dataset_split_iteration/iter' + str(
        split_time) + '/ranking/5folds_fam_role/fold'
    pred_results_dir = '../../results/dataset_split_iteration/overall/iter' + str(
        split_time) + '/experiments/evaluation/ranking_overall/'
    for fold_index in range(1, 6):
        pred_fold_list = list()
        pred_fam_list = list()
        pred_role_list = list()
        ground_number_list = list()
        true_number_list = list()

        ranker = lgb.Booster(model_file=model_dir + 'fold' + str(fold_index) + '_model.txt')
        # pred_summary = open(pred_results_dir + 'overall_model_{}'.format(fold_index) + '_evaluation.txt', 'a')
        # pred_summary.write('Test set of fold {}'.format(fold_index) + '\n')
        for pred_role in ['staff', 'student']:
            for pred_familiarity in ['familiar', 'unfamiliar']:
                pred_fold_list.append(fold_index)
                pred_role_list.append(pred_role)
                pred_fam_list.append(pred_familiarity)

                pred_file = pred_data_dir + str(
                    fold_index) + '/' + pred_familiarity + '_' + pred_role + '/rank.test'

                X_pred, y_ground = load_svmlight_file(pred_file)
                X_pred = X_pred[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]]
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
                    # else:
                    #     print(i)

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

# evaluation()
for split_time in range(1,11):
    evaluation_iteration(split_time)