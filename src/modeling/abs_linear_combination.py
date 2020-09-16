import pandas as pd
import os
from sklearn.datasets import load_svmlight_file

def linear_combination():
    pred_data_dir = '../../data/processed/fit_dataset/regression/5folds_fam_role/fold'
    pred_results_dir = '../../results/experiments/evaluation/linear_combination/'

    total_num = 0
    total_true_num = 0

    for fold_index in range(1, 6):
        pred_fold_list = list()
        pred_fam_role_list = list()
        ground_number_list = list()
        true_number_list = list()

        for pred_role in ['staff', 'student']:
            for pred_familiarity in ['familiar', 'unfamiliar']:
                pred_fold_list.append(fold_index)
                pred_fam_role_list.append(pred_role + ' + ' + pred_familiarity)

                # pred_file = '../../data/processed/label/'  + pred_role + '_feature_norm.csv'

                ground_truth_file = pred_data_dir + str(
                    fold_index) + '/' + pred_familiarity + '_' + pred_role + '/abs_reg_test.csv'

                df_pred = pd.read_csv(ground_truth_file, header=None, sep=' ')
                y_ground = df_pred[0]
                X_pred = df_pred.drop(0, axis=1)
                y_pred = X_pred.sum(axis=1)

                y_ground_splits = list()
                y_pred_splits = list()
                for i in range(0, len(y_pred), 2):
                    y_ground_splits.append(y_ground[i: i + 2])
                    y_pred_splits.append(y_pred[i: i + 2])

                # pred_score = 0
                true_num = 0
                for i in range(0, len(y_ground_splits)):
                    test_index = y_ground_splits[i].tolist().index(max(y_ground_splits[i].tolist()))
                    pred_index = y_pred_splits[i].tolist().index(max(y_pred_splits[i].tolist()))
                    if test_index == pred_index:
                        true_num += 1

                ground_number_list.append(len(y_ground_splits))
                total_num += len(y_ground_splits)
                true_number_list.append(true_num)
                total_true_num += true_num

        pred_summary_file = pred_results_dir + 'linear_combination_{}'.format(fold_index) + '_evaluation.csv'
        df_pred_summary_file = pd.DataFrame()
        df_pred_summary_file['Fold'] = pred_fold_list
        df_pred_summary_file['Familiarity + Role'] = pred_fam_role_list
        df_pred_summary_file['Test set size'] = ground_number_list
        df_pred_summary_file['Hit number'] = true_number_list
        df_pred_summary_file.to_csv(pred_summary_file)

    print(total_true_num)

# def linear_combination_iteration(split_time):
#     pred_data_dir = '../../data/processed/dataset_split_iteration/iter' + str(split_time) + '/regression/5folds_fam_role/fold'
#     pred_results_dir = '../../results/dataset_split_iteration/iter' + str(split_time) + '/experiments/evaluation/linear_combination/'
#
#     total_num = 0
#     total_true_num = 0
#
#     for fold_index in range(1, 6):
#         pred_fold_list = list()
#         pred_fam_list = list()
#         pred_role_list = list()
#         ground_number_list = list()
#         true_number_list = list()
#
#         for pred_role in ['staff', 'student']:
#             for pred_familiarity in ['familiar', 'unfamiliar']:
#                 pred_fold_list.append(fold_index)
#                 pred_role_list.append(pred_role)
#                 pred_fam_list.append(pred_familiarity)
#
#                 ground_truth_file = pred_data_dir + str(
#                     fold_index) + '/' + pred_familiarity + '_' + pred_role + '/abs_reg_test.csv'
#
#                 df_pred = pd.read_csv(ground_truth_file, header=None, sep=' ')
#                 y_ground = df_pred[0]
#                 X_pred = df_pred.drop(0, axis=1)
#                 # y_pred = X_pred.sum(axis=1)
#                 y_pred = X_pred.loc[:, [1, 2, 3]].sum(axis=1) * 1/3 + X_pred.loc[:, [4, 5, 7, 8, 9, 10]].sum(axis=1) * 1/6 \
#                          + X_pred.loc[:, [11, 12, 13]].sum(axis=1) * 1/3
#
#                 y_ground_splits = list()
#                 y_pred_splits = list()
#                 for i in range(0, len(y_pred), 2):
#                     y_ground_splits.append(y_ground[i: i + 2])
#                     y_pred_splits.append(y_pred[i: i + 2])
#
#                 true_num = 0
#                 for i in range(0, len(y_ground_splits)):
#                     test_index = y_ground_splits[i].tolist().index(max(y_ground_splits[i].tolist()))
#                     pred_index = y_pred_splits[i].tolist().index(max(y_pred_splits[i].tolist()))
#                     if test_index == pred_index:
#                         true_num += 1
#
#                 ground_number_list.append(len(y_ground_splits))
#                 total_num += len(y_ground_splits)
#                 true_number_list.append(true_num)
#                 total_true_num += true_num
#
#         pred_summary_file = pred_results_dir + 'linear_combination_{}'.format(fold_index) + '_evaluation.csv'
#         df_pred_summary_file = pd.DataFrame()
#         df_pred_summary_file['fold'] = pred_fold_list
#         df_pred_summary_file['role'] = pred_role_list
#         df_pred_summary_file['familiarity'] = pred_fam_list
#         df_pred_summary_file['test set size'] = ground_number_list
#         df_pred_summary_file['hit number'] = true_number_list
#         df_pred_summary_file.to_csv(pred_summary_file)
#     print(total_true_num)

def linear_combination_iteration(split_time):
    total_num = 0
    total_true_num = 0

    pred_data_dir = '../../data/processed/cv_dataset/regression/fam_role/fold' + str(split_time)
    pred_results_dir = '../../results/cv_results/familiarity/evaluation/linear_combination/'

    pred_fold_list = list()
    pred_fam_list = list()
    pred_role_list = list()
    ground_number_list = list()
    true_number_list = list()
    hr_list = list()
    mae_list = list()

    for pred_role in ['staff', 'student']:
        for pred_familiarity in ['familiar', 'unfamiliar']:
            ground_truth_file = pred_data_dir + '/' + pred_familiarity + '_' + pred_role + '_abs_reg_test.csv'
            if os.path.getsize(ground_truth_file):
                pred_fold_list.append(split_time)
                pred_role_list.append(pred_role)
                pred_fam_list.append(pred_familiarity)

                df_pred = pd.read_csv(ground_truth_file, header=None, sep=' ')
                y_ground = df_pred[0]
                X_pred = df_pred.drop(0, axis=1)
                # y_pred = X_pred.sum(axis=1)
                y_pred = X_pred.loc[:, [1, 2, 3]].sum(axis=1) * 1 / 3 + X_pred.loc[:, [4, 5, 7, 8, 9, 10]].sum(
                    axis=1) * 1 / 6 + X_pred.loc[:, [11, 12, 13]].sum(axis=1) * 1 / 3

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

                true_num = 0
                absolute_error = 0
                for i in range(0, len(y_ground_splits)):
                    test_index = y_ground_splits[i].tolist().index(max(y_ground_splits[i].tolist()))
                    pred_index = y_pred_splits[i].tolist().index(max(y_pred_splits[i].tolist()))
                    if test_index == pred_index:
                        true_num += 1

                    print(y_ground_splits[i].iloc[0])
                    print(y_pred_splits[i].iloc[0])
                    print(y_ground_splits[i].iloc[1])
                    print(y_pred_splits[i].iloc[1])

                    absolute_error += abs(y_ground_splits[i].iloc[0] - y_pred_splits[i].iloc[0]) + \
                                      abs(y_ground_splits[i].iloc[1] - y_pred_splits[i].iloc[1])

                ground_number_list.append(len(y_ground_splits))
                total_num += len(y_ground_splits)
                true_number_list.append(true_num)
                total_true_num += true_num
                hr_list.append(true_num / len(y_ground_splits))
                mae_list.append(absolute_error / len(y_ground_splits))

    pred_summary_file = pred_results_dir + 'fold{}'.format(split_time) + '_model_evaluation.csv'
    df_pred_summary_file = pd.DataFrame()
    df_pred_summary_file['fold'] = pred_fold_list
    df_pred_summary_file['role'] = pred_role_list
    df_pred_summary_file['familiarity'] = pred_fam_list
    df_pred_summary_file['test set size'] = ground_number_list
    df_pred_summary_file['hit number'] = true_number_list
    df_pred_summary_file['HR'] = hr_list
    df_pred_summary_file['MAE'] = mae_list
    df_pred_summary_file.to_csv(pred_summary_file)

    print(total_true_num)

# linear_combination()
for split_time in range(1, 18):
    linear_combination_iteration(split_time)