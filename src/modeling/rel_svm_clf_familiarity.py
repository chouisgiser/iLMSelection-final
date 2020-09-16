import pandas as pd
import numpy as np
from sklearn import svm
import os

def svm_clf():
    for model_familiarity in ['familiar', 'unfamiliar']:
        total_num = 0
        total_true_num = 0
        data_dir = '../../data/processed/fit_dataset/classification/5folds_familiarity/fold'
        results_dir = '../../results/'
        # pred_results_dir = '../../results/experiments/evaluation/gp_abs_reg_' + model_familiarity + '/'
        test_dir = '../../data/processed/fit_dataset/classification/5folds_fam_role/fold'

        for fold_index in range(1, 6):
            train_file = data_dir + str(fold_index) + '/' + model_familiarity + '_multiclass_train.csv'
            df_train = pd.read_csv(train_file, header=None, sep=' ')

            y_train = df_train[0]
            X_train = df_train.drop(0, axis=1)
            X_train = X_train.loc[:, [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14]]
            model = svm.LinearSVC(C=8, loss="hinge")

            multiclf = model.fit(X_train, y_train)

            # rfc = grid.best_estimator_

            pred_fold_list = list()
            pred_fam_role_list = list()
            ground_number_list = list()
            true_number_list = list()
            for pred_role in ['staff', 'student']:
                for pred_familiarity in ['familiar', 'unfamiliar']:
                    pred_fold_list.append(fold_index)
                    pred_fam_role_list.append(pred_role + ' + ' + pred_familiarity)
                    test_file = test_dir + str(
                        fold_index) + '/' + pred_familiarity + '_' + pred_role + '/rel_clf_test.csv'
                    # pred_file = test_dir + str(
                    #     fold_index) + '/' + pred_familiarity + '_' + pred_role + '/rank.test'

                    df_test = pd.read_csv(test_file, header=None, sep=' ')
                    y_test = df_test[0]
                    X_test = df_test.drop(0, axis=1)
                    X_test = X_test.loc[:, [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14]]
                    y_pred = multiclf.predict(X_test)

                    true_num = 0
                    for i in range(0, len(y_pred)):
                        if y_pred[i] == y_test.iloc[i]:
                            true_num += 1

                    ground_number_list.append(len(y_test))
                    total_num += len(y_test)
                    true_number_list.append(true_num)
                    total_true_num += true_num

            pred_summary_file = results_dir + 'experiments/evaluation/svm_rel_clf_' + model_familiarity + '/svm_{}'.format(
                fold_index) + '_evaluation.csv'
            df_pred_summary_file = pd.DataFrame()
            df_pred_summary_file['Fold'] = pred_fold_list
            df_pred_summary_file['Familiarity + Role'] = pred_fam_role_list
            df_pred_summary_file['Test set size'] = ground_number_list
            df_pred_summary_file['Hit number'] = true_number_list
            df_pred_summary_file.to_csv(pred_summary_file)

        print(total_true_num)

def svm_clf_iteration(split_time):
    for model_familiarity in ['familiar', 'unfamiliar']:
        total_num = 0
        total_true_num = 0
        train_data_dir = '../../data/processed/cv_dataset/classification/familiarity/fold' + str(split_time) + '/'
        results_dir = '../../results/cv_results/familiarity/evaluation/svm_rel_clf_' + model_familiarity + '/'
        # pred_results_dir = '../../results/experiments/evaluation/gp_abs_reg_' + model_familiarity + '/'
        test_data_dir = '../../data/processed/cv_dataset/classification/fam_role/fold' + str(split_time) + '/'

        # for fold_index in range(1, 6):
        train_file = train_data_dir + model_familiarity + '_rel_clf_train.csv'
        df_train = pd.read_csv(train_file, header=None, sep=' ')

        y_train = df_train[0]
        X_train = df_train.drop(0, axis=1)
        X_train = X_train.loc[:, [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]]

        best_params = {'control': 0}
        best_score = float('-inf')

        for control in range(1, 11, 1):
            model = svm.LinearSVC(C=control, loss="hinge")
            model.fit(X_train, y_train)
            tmp_score = model.score(X_train, y_train)
            if best_score < tmp_score:
                best_score = tmp_score
                best_params['control'] = control

        model = svm.LinearSVC(C=best_params['control'], loss="hinge")
        multiclf = model.fit(X_train, y_train)

        # rfc = grid.best_estimator_

        pred_fold_list = list()
        pred_fam_list = list()
        pred_role_list = list()
        ground_number_list = list()
        true_number_list = list()
        hr_list = list()

        for pred_role in ['staff', 'student']:
            for pred_familiarity in ['familiar', 'unfamiliar']:
                test_file = test_data_dir + pred_familiarity + '_' + pred_role + '_rel_clf_test.csv'
                if os.path.getsize(test_file):
                    pred_fold_list.append(split_time)
                    pred_role_list.append(pred_role)
                    pred_fam_list.append(pred_familiarity)

                    df_test = pd.read_csv(test_file, header=None, sep=' ')
                    y_test = df_test[0]
                    X_test = df_test.drop(0, axis=1)
                    X_test = X_test.loc[:, [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]]
                    y_pred = multiclf.predict(X_test)

                    true_num = 0
                    for i in range(0, len(y_pred)):
                        if y_pred[i] == y_test.iloc[i]:
                            true_num += 1

                    ground_number_list.append(len(y_test))
                    total_num += len(y_test)
                    true_number_list.append(true_num)
                    total_true_num += true_num
                    hr_list.append(true_num / len(y_test))

        pred_summary_file = results_dir + 'fold{}'.format(split_time) + '_model_evaluation.csv'
        df_pred_summary_file = pd.DataFrame()
        df_pred_summary_file['fold'] = pred_fold_list
        df_pred_summary_file['role'] = pred_role_list
        df_pred_summary_file['familiarity'] = pred_fam_list
        df_pred_summary_file['test set size'] = ground_number_list
        df_pred_summary_file['hit number'] = true_number_list
        df_pred_summary_file['HR'] = hr_list
        df_pred_summary_file.to_csv(pred_summary_file)

        print(total_true_num)

# svm_clf()
for split_time in range(1, 18):
    svm_clf_iteration(split_time)