import pandas as pd
import os

def combine_overall():

    fold_path = '1-10-scale/5folds_fam_role'
    dirs = os.listdir(fold_path)

    overall_path = '1-10-scale/5folds_overall'

    for dir_test in dirs:
        df_test = pd.DataFrame()
        df_train = pd.DataFrame()
        for dirpath, dirnames, filenames in os.walk(fold_path + '/' + dir_test):
            for dirname in dirnames:
                # familiarity = dirname.split('_')[0]
                # role = dirname.split('_')[1]
                for sub_dirpath, sub_dirnames, sub_filenames in os.walk(dirpath + '/' + dirname):
                    for sub_file in sub_filenames:
                        if sub_file == 'lr_test_set.csv':
                            test_file = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                            df_test = pd.concat([df_test, test_file])
                            # df_test['familiarity'] = [familiarity] * df_test.shape[0]
                            # df_test['role'] = [familiarity] * df_test.shape[0]

                        if sub_file == 'lr_train_set.csv':
                            df_train_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                            df_train = pd.concat([df_train, df_train_fold])

        test_file_path = overall_path + '/' + dir_test + '/lr_test_set.csv'
        df_test.to_csv(test_file_path, header=False, index=False)

        df_train_file_path = overall_path + '/' + dir_test + '/lr_train_set.csv'
        df_train.to_csv(df_train_file_path, header=False, index=False)

def combine_by_familiarity_iteration(split_time):
    fold_path = '../../../data/processed/cv_dataset//regression/fam_role/fold' + str(split_time)
    dirs = os.listdir(fold_path)

    familiarity_path = '../../../data/processed/cv_dataset/regression/familiarity/fold' + str(split_time)

    df_train_familiar = pd.DataFrame()
    df_train_unfamiliar = pd.DataFrame()
    df_test_familiar = pd.DataFrame()
    df_test_unfamiliar = pd.DataFrame()

    for dirpath, dirnames, filenames in os.walk(fold_path):
        for filename in filenames:
            if filename == 'familiar_staff_abs_reg_train.csv' or filename == 'familiar_student_abs_reg_train.csv':
                df_train = pd.read_csv(fold_path + '/' + filename, header=None)
                df_train_familiar = pd.concat([df_train_familiar, df_train])

            if filename == 'unfamiliar_staff_abs_reg_train.csv' or filename == 'unfamiliar_student_abs_reg_train.csv':
                df_train = pd.read_csv(fold_path + '/' + filename, header=None)
                df_train_unfamiliar = pd.concat([df_train_unfamiliar, df_train])

            if filename == 'familiar_staff_abs_reg_test.csv' or filename == 'familiar_student_abs_reg_test.csv':
                if os.path.getsize(fold_path + '/' + filename):
                    df_test = pd.read_csv(fold_path + '/' + filename, header=None)
                    df_test_familiar = pd.concat([df_test_familiar, df_test])

            if filename == 'unfamiliar_staff_abs_reg_test.csv' or filename == 'unfamiliar_student_abs_reg_test.csv':
                if os.path.getsize(fold_path + '/' + filename):
                    df_test = pd.read_csv(fold_path + '/' + filename, header=None)
                    df_test_unfamiliar = pd.concat([df_test_unfamiliar, df_test])


    df_train_file_path = familiarity_path + '/familiar_abs_reg_train.csv'
    df_train_familiar.to_csv(df_train_file_path, header=False, index=False)

    df_train_file_path = familiarity_path + '/unfamiliar_abs_reg_train.csv'
    df_train_unfamiliar.to_csv(df_train_file_path, header=False, index=False)

    df_train_file_path = familiarity_path + '/familiar_abs_reg_test.csv'
    df_test_familiar.to_csv(df_train_file_path, header=False, index=False)

    df_train_file_path = familiarity_path + '/unfamiliar_abs_reg_test.csv'
    df_test_unfamiliar.to_csv(df_train_file_path, header=False, index=False)

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
    #                         if sub_file == 'abs_reg_set.csv':
    #                             test_file = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
    #                             df_test = pd.concat([df_test, test_file])
    #
    #                         if 'familiar_staff' == dirname or 'familiar_student' == dirname:
    #                             if sub_file == 'abs_reg_train.csv':
    #                                 df_train_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
    #                                 df_train_familiar = pd.concat([df_train_familiar, df_train_fold])
    #
    #                         if 'unfamiliar_staff' == dirname or 'unfamiliar_student' == dirname:
    #                             if sub_file == 'abs_reg_train.csv':
    #                                 df_train_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
    #                                 df_train_unfamiliar = pd.concat([df_train_unfamiliar, df_train_fold])
    #
    #         df_train_file_path = familiarity_path + '/' + dir_test + '/familiar_abs_reg_train.csv'
    #         df_train_familiar.to_csv(df_train_file_path, header=False, index=False)
    #
    #         df_train_file_path = familiarity_path + '/' + dir_test + '/unfamiliar_abs_reg_train.csv'
    #         df_train_unfamiliar.to_csv(df_train_file_path, header=False, index=False)

def combine_by_familiarity():
    fold_path = '../../../data/processed/fit_dataset/regression/5folds_fam_role'
    dirs = os.listdir(fold_path)

    familiarity_path = '../../../data/processed/fit_dataset/regression/5folds_familiarity'

    for dir_test in dirs:
        df_test = pd.DataFrame()

        df_train_familiar = pd.DataFrame()

        df_train_unfamiliar = pd.DataFrame()

        if "fold" in dir_test:
            for dirpath, dirnames, filenames in os.walk(fold_path + '/' + dir_test):
                for dirname in dirnames:
                    for sub_dirpath, sub_dirnames, sub_filenames in os.walk(dirpath + '/' + dirname):
                        for sub_file in sub_filenames:
                            if sub_file == 'abs_reg_set.csv':
                                test_file = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                df_test = pd.concat([df_test, test_file])

                            if 'familiar_staff' == dirname or 'familiar_student' == dirname:
                                if sub_file == 'abs_reg_train.csv':
                                    df_train_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                    df_train_familiar = pd.concat([df_train_familiar, df_train_fold])

                            if 'unfamiliar_staff' == dirname or 'unfamiliar_student' == dirname:
                                if sub_file == 'abs_reg_train.csv':
                                    df_train_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                    df_train_unfamiliar = pd.concat([df_train_unfamiliar, df_train_fold])


            df_train_file_path = familiarity_path + '/' + dir_test + '/familiar_abs_reg_train.csv'
            df_train_familiar.to_csv(df_train_file_path, header=False, index=False)

            df_train_file_path = familiarity_path + '/' + dir_test + '/unfamiliar_abs_reg_train.csv'
            df_train_unfamiliar.to_csv(df_train_file_path, header=False, index=False)


def combine_by_role():
    fold_path = '1-10-scale/5folds_fam_role'
    dirs = os.listdir(fold_path)

    role_path = '5folds_role'

    for dir_test in dirs:
        df_test = pd.DataFrame()

        df_train_staff = pd.DataFrame()

        df_train_student = pd.DataFrame()

        for dirpath, dirnames, filenames in os.walk(fold_path + '/' + dir_test):
            for dirname in dirnames:
                for sub_dirpath, sub_dirnames, sub_filenames in os.walk(dirpath + '/' + dirname):
                    for sub_file in sub_filenames:
                        if os.path.splitext(sub_file) == 'lr_test_set.csv':
                            test_file = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                            df_test = pd.concat([df_test, test_file])

                        if 'familiar_staff' == dirname or 'unfamiliar_staff' == dirname:
                            if os.path.splitext(sub_file) == 'lr_train_set.csv':
                                df_train_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                df_train_staff = pd.concat([df_train_staff, df_train_fold])

                        if 'familiar_student' == dirname or 'unfamiliar_student' == dirname:
                            if os.path.splitext(sub_file) == 'lr_train_set.csv':
                                df_train_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                df_train_student = pd.concat([df_train_student, df_train_fold])

        test_file_path = role_path + '/' + dir_test + '/lr_test_set.csv'
        df_test.to_csv(test_file_path, header=False, index=False)

        df_train_file_path = role_path + '/' + dir_test + '/staff/lr_train_set.csv'
        df_train_staff.to_csv(df_train_file_path, header=False, index=False)

        df_train_file_path = role_path + '/' + dir_test + '/student/lr_train_set.csv'
        df_train_student.to_csv(df_train_file_path, header=False, index=False)

# combine_by_familiarity()
for split_time in range(1, 18):
    combine_by_familiarity_iteration(split_time)