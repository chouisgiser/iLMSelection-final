import os

import pandas as pd
from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import root_mean_square_error

def gp_regression():
    for model_familiarity in ['familiar', 'unfamiliar']:
        total_num = 0
        total_true_num = 0
        data_dir = '../../data/processed/fit_dataset/regression/5folds_familiarity/fold'
        pred_results_dir = '../../results/experiments/evaluation/gp_abs_reg_' + model_familiarity + '/'
        test_dir = '../../data/processed/fit_dataset/regression/5folds_fam_role/fold'

        for fold_index in range(1, 6):
            train_file = data_dir + str(fold_index) + '/' + model_familiarity + '_abs_reg_train.csv'
            df_train = pd.read_csv(train_file, header=None, sep=' ')
            y_train = df_train[0]
            X_train = df_train.drop(0, axis=1)
            X_train = X_train.loc[:, [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14]]

            best_params = {'population_size': 0, 'generation': 0}
            best_score = float('inf')
            for size in range(1, 30, 1):
                for genegaration in range(1, 10, 1):
                    model = SymbolicRegressor(population_size=size,
                                              generations=genegaration, stopping_criteria=0.01,
                                              p_crossover=0.85, p_subtree_mutation=0.05,
                                              p_hoist_mutation=0.05, p_point_mutation=0.05,
                                              max_samples=0.9, verbose=1,
                                              parsimony_coefficient=0.01, random_state=0, metric=root_mean_square_error)
                    model.fit(X_train, y_train)
                    tmp_score = abs(model.score(X_train, y_train) - 1)
                    if tmp_score < best_score:
                        best_score = tmp_score
                        best_params['population_size'] = size
                        best_params['generation'] = genegaration

            model = SymbolicRegressor(population_size=best_params['population_size'],
                                      generations=best_params['generation'], stopping_criteria=0.01,
                                      p_crossover=0.85, p_subtree_mutation=0.05,
                                      p_hoist_mutation=0.05, p_point_mutation=0.05,
                                      max_samples=0.9, verbose=1,
                                      parsimony_coefficient=0.01, random_state=0, metric=root_mean_square_error)

            regressor = model.fit(X_train, y_train)

            pred_fold_list = list()
            pred_fam_role_list = list()
            ground_number_list = list()
            true_number_list = list()
            for pred_role in ['staff', 'student']:
                for pred_familiarity in ['familiar', 'unfamiliar']:
                    pred_fold_list.append(fold_index)
                    pred_fam_role_list.append(pred_role + ' + ' + pred_familiarity)
                    test_file = test_dir + str(
                        fold_index) + '/' + pred_familiarity + '_' + pred_role + '/abs_reg_test.csv'

                    df_test = pd.read_csv(test_file, header=None, sep=' ')
                    y_test = df_test[0]
                    X_test = df_test.drop(0, axis=1)
                    X_test = X_test.loc[:, [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14]]
                    y_pred = regressor.predict(X_test)

                    # print(model.score(X_test, y_test))

                    y_test_splits = list()
                    y_pred_splits = list()
                    for i in range(0, len(y_test), 2):
                        y_test_splits.append(y_test[i: i + 2])
                        y_pred_splits.append(y_pred[i: i + 2])

                    true_num = 0
                    for i in range(0, len(y_test_splits)):
                        test_index = y_test_splits[i].tolist().index(max(y_test_splits[i].tolist()))
                        pred_index = y_pred_splits[i].tolist().index(max(y_pred_splits[i].tolist()))
                        if test_index == pred_index:
                            true_num += 1

                    ground_number_list.append(len(y_test) / 2)
                    total_num += len(y_test) / 2
                    true_number_list.append(true_num)
                    total_true_num += true_num

            pred_summary_file = pred_results_dir + 'gp_{}'.format(fold_index) + '_evaluation.csv'
            df_pred_summary_file = pd.DataFrame()
            df_pred_summary_file['Fold'] = pred_fold_list
            df_pred_summary_file['Familiarity + Role'] = pred_fam_role_list
            df_pred_summary_file['Test set size'] = ground_number_list
            df_pred_summary_file['Hit number'] = true_number_list
            df_pred_summary_file.to_csv(pred_summary_file)

        print(total_true_num)

def gp_regression_iteration(split_time):
    for model_familiarity in ['familiar', 'unfamiliar']:
        total_num = 0
        total_true_num = 0
        train_data_dir = '../../data/processed/cv_dataset/regression/familiarity/fold' + str(split_time) + '/'
        pred_results_dir = '../../results/cv_results/familiarity/evaluation/gp_abs_reg_' + model_familiarity + '/'
        test_data_dir = '../../data/processed/cv_dataset//regression/fam_role/fold' + str(split_time)

        # for fold_index in range(1, 6):
        train_file = train_data_dir + model_familiarity + '_abs_reg_train.csv'
        df_train = pd.read_csv(train_file, header=None, sep=' ')
        y_train = df_train[0]
        X_train = df_train.drop(0, axis=1)
        X_train = X_train.loc[:, [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]]

        best_params = {'population_size': 0, 'generation': 0}
        best_score = float('inf')
        for size in range(1, 30, 2):
            for genegaration in range(1, 40, 1):
                model = SymbolicRegressor(population_size=size,
                                          generations=genegaration, stopping_criteria=0.01,
                                          p_crossover=0.85, p_subtree_mutation=0.05,
                                          p_hoist_mutation=0.05, p_point_mutation=0.05,
                                          max_samples=0.9, verbose=1,
                                          parsimony_coefficient=0.01, random_state=0, metric=root_mean_square_error)
                model.fit(X_train, y_train)
                tmp_score = abs(model.score(X_train, y_train) - 1)
                if tmp_score < best_score:
                    best_score = tmp_score
                    best_params['population_size'] = size
                    best_params['generation'] = genegaration

        model = SymbolicRegressor(population_size=best_params['population_size'],
                                  generations=best_params['generation'], stopping_criteria=0.01,
                                  p_crossover=0.85, p_subtree_mutation=0.05,
                                  p_hoist_mutation=0.05, p_point_mutation=0.05,
                                  max_samples=0.9, verbose=1,
                                  parsimony_coefficient=0.01, random_state=0, metric=root_mean_square_error)

        regressor = model.fit(X_train, y_train)

        pred_fold_list = list()
        pred_fam_list = list()
        pred_role_list = list()
        ground_number_list = list()
        true_number_list = list()
        hr_list = list()
        mae_list = list()
        for pred_role in ['staff', 'student']:
            for pred_familiarity in ['familiar', 'unfamiliar']:
                test_file = test_data_dir + '/' + pred_familiarity + '_' + pred_role + '_abs_reg_test.csv'
                if os.path.getsize(test_file):
                    pred_fold_list.append(split_time)
                    pred_role_list.append(pred_role)
                    pred_fam_list.append(pred_familiarity)

                    df_test = pd.read_csv(test_file, header=None, sep=' ')
                    y_test = df_test[0]
                    X_test = df_test.drop(0, axis=1)
                    X_test = X_test.loc[:, [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]]
                    y_pred = regressor.predict(X_test)

                    amin, amax = min(y_pred), max(y_pred)
                    for index, value in enumerate(y_pred):
                        y_pred[index] = (value - amin) / (amax - amin)

                    amin, amax = min(y_pred), max(y_test)
                    for index, value in enumerate(y_test):
                        y_test[index] = (value - amin) / (amax - amin)

                    # print(model.score(X_test, y_test))

                    y_test_splits = list()
                    y_pred_splits = list()
                    for i in range(0, len(y_test), 2):
                        y_test_splits.append(y_test[i: i + 2])
                        y_pred_splits.append(y_pred[i: i + 2])

                    true_num = 0
                    absolute_error = 0
                    for i in range(0, len(y_test_splits)):
                        print(i)
                        test_index = y_test_splits[i].tolist().index(max(y_test_splits[i].tolist()))
                        pred_index = y_pred_splits[i].tolist().index(max(y_pred_splits[i].tolist()))
                        if test_index == pred_index:
                            true_num += 1

                        absolute_error += abs(y_test_splits[i].iloc[0] - y_pred_splits[i][0]) + \
                                          abs(y_test_splits[i].iloc[1] - y_pred_splits[i][1])

                    ground_number_list.append(len(y_test) / 2)
                    total_num += len(y_test) / 2
                    true_number_list.append(true_num)
                    total_true_num += true_num
                    hr_list.append(true_num / len(y_test_splits))
                    mae_list.append(absolute_error / len(y_test_splits))

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

# gp_regression()

for split_time in range(16, 18):
    gp_regression_iteration(split_time)