import pandas as pd
import os


def combine_by_familiarity_iteration(split_time):
    fold_path = '../../../data/processed/cv_dataset/classification/fam_role/fold' + str(split_time)
    dirs = os.listdir(fold_path)

    familiarity_path = '../../../data/processed/cv_dataset/classification/familiarity/fold' + str(split_time)

    df_train_familiar = pd.DataFrame()
    df_train_unfamiliar = pd.DataFrame()
    df_test_familiar = pd.DataFrame()
    df_test_unfamiliar = pd.DataFrame()

    for dirpath, dirnames, filenames in os.walk(fold_path):
        for filename in filenames:
            if filename == 'familiar_staff_rel_clf_train.csv' or filename == 'familiar_student_rel_clf_train.csv':
                df_train = pd.read_csv(fold_path + '/' + filename, header=None)
                df_train_familiar = pd.concat([df_train_familiar, df_train])
            if filename == 'unfamiliar_staff_rel_clf_train.csv' or filename == 'unfamiliar_student_rel_clf_train.csv':
                df_train = pd.read_csv(fold_path + '/' + filename, header=None)
                df_train_unfamiliar = pd.concat([df_train_unfamiliar, df_train])
            if filename == 'familiar_staff_rel_clf_test.csv' or filename == 'familiar_student_rel_clf_test.csv':
                if os.path.getsize(fold_path + '/' + filename):
                    df_test = pd.read_csv(fold_path + '/' + filename, header=None)
                    df_test_familiar = pd.concat([df_test_familiar, df_test])
            if filename == 'unfamiliar_staff_rel_clf_test.csv' or filename == 'unfamiliar_student_rel_clf_test.csv':
                if os.path.getsize(fold_path + '/' + filename):
                    df_test = pd.read_csv(fold_path + '/' + filename, header=None)
                    df_test_unfamiliar = pd.concat([df_test_unfamiliar, df_test])

    df_train_file_path = familiarity_path + '/familiar_rel_clf_train.csv'
    df_train_familiar.to_csv(df_train_file_path, header=False, index=False)

    df_train_file_path = familiarity_path + '/unfamiliar_rel_clf_train.csv'
    df_train_unfamiliar.to_csv(df_train_file_path, header=False, index=False)

    df_train_file_path = familiarity_path + '/familiar_rel_clf_test.csv'
    df_test_familiar.to_csv(df_train_file_path, header=False, index=False)

    df_train_file_path = familiarity_path + '/unfamiliar_rel_clf_test.csv'
    df_test_unfamiliar.to_csv(df_train_file_path, header=False, index=False)

    # fold_path = '../../../data/processed/dataset_split_iteration/iter' + str(split_time) + '/classification/5folds_fam_role'
    # dirs = os.listdir(fold_path)
    #
    # familiarity_path = '../../../data/processed/dataset_split_iteration/iter' + str(split_time) + '/classification/5folds_familiarity'
    #
    # for dir_test in dirs:
    #     df_test = pd.DataFrame()
    #
    #     df_train_familiar = pd.DataFrame()
    #
    #     df_train_unfamiliar = pd.DataFrame()
    #
    #     if "fold" in dir_test:
    #         for dirpath, dirnames, filenames in os.walk(fold_path + '/' + dir_test):
    #             for dirname in dirnames:
    #                 for sub_dirpath, sub_dirnames, sub_filenames in os.walk(dirpath + '/' + dirname):
    #                     for sub_file in sub_filenames:
    #                         if sub_file == 'rel_clf_test.csv':
    #                             test_file = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
    #                             df_test = pd.concat([df_test, test_file])
    #
    #                         if 'familiar_staff' == dirname or 'familiar_student' == dirname:
    #                             if sub_file == 'rel_clf_train.csv':
    #                                 df_train_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
    #                                 df_train_familiar = pd.concat([df_train_familiar, df_train_fold])
    #
    #                         if 'unfamiliar_staff' == dirname or 'unfamiliar_student' == dirname:
    #                             if sub_file == 'rel_clf_train.csv':
    #                                 df_train_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
    #                                 df_train_unfamiliar = pd.concat([df_train_unfamiliar, df_train_fold])
    #
    #         df_train_file_path = familiarity_path + '/' + dir_test + '/familiar_multiclass_train.csv'
    #         df_train_familiar.to_csv(df_train_file_path, header=False, index=False)
    #
    #         df_train_file_path = familiarity_path + '/' + dir_test + '/unfamiliar_multiclass_train.csv'
    #         df_train_unfamiliar.to_csv(df_train_file_path, header=False, index=False)


# combine_by_familiarity()
for split_time in range(1, 18):
    combine_by_familiarity_iteration(split_time)