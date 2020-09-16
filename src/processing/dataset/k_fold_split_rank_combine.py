import pandas as pd
import os


def combine_overall_iteration(split_time):
    overall_path = '../../../data/processed/dataset_split_iteration/iter' + str(split_time) + '/ranking/5folds_overall'

    fold_path = '../../../data/processed/dataset_split_iteration/iter' + str(split_time) + '/ranking/5folds_fam_role'
    dirs = os.listdir(fold_path)

    for dir_test in dirs:
        df_test = pd.DataFrame()
        df_test_query = pd.DataFrame()
        df_train = pd.DataFrame()
        df_train_query = pd.DataFrame()
        if "fold" in dir_test:
            for dirpath, dirnames, filenames in os.walk(fold_path + '/' + dir_test):
                for dirname in dirnames:
                    for sub_dirpath, sub_dirnames, sub_filenames in os.walk(dirpath + '/' + dirname):
                        for sub_file in sub_filenames:
                            if os.path.splitext(sub_file)[1] == '.test':
                                test_file = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                df_test = pd.concat([df_test, test_file])
                            if os.path.splitext(sub_file)[0] == 'rank.test' and os.path.splitext(sub_file)[
                                1] == '.query':
                                test_query_file = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                df_test_query = pd.concat([df_test_query, test_query_file])
                            if os.path.splitext(sub_file)[1] == '.train':
                                df_train_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                df_train = pd.concat([df_train, df_train_fold])
                            if os.path.splitext(sub_file)[0] == 'rank.train' and os.path.splitext(sub_file)[
                                1] == '.query':
                                df_train_query_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                df_train_query = pd.concat([df_train_query, df_train_query_fold])

            test_file_path = overall_path + '/' + dir_test + '/rank.test'
            df_test.to_csv(test_file_path, header=False, index=False)
            test_query_file_path = overall_path + '/' + dir_test + '/rank.test.query'
            df_test_query.to_csv(test_query_file_path, header=False, index=False)

            df_train_file_path = overall_path + '/' + dir_test + '/rank.train'
            df_train.to_csv(df_train_file_path, header=False, index=False)
            df_train_query_file_path = overall_path + '/' + dir_test + '/rank.train.query'
            df_train_query.to_csv(df_train_query_file_path, header=False, index=False)


def combine_by_role_iteration(split_time):
    role_path = '../../../data/processed/dataset_split_iteration/iter' + str(split_time) + '/ranking/5folds_role'

    fold_path = '../../../data/processed/dataset_split_iteration/iter' + str(split_time) + '/ranking/5folds_fam_role'
    dirs = os.listdir(fold_path)

    for dir_test in dirs:
        df_test = pd.DataFrame()
        df_test_query = pd.DataFrame()
        df_train_staff = pd.DataFrame()
        df_train_query_staff = pd.DataFrame()
        df_train_student = pd.DataFrame()
        df_train_query_student = pd.DataFrame()
        if "fold" in dir_test:
            for dirpath, dirnames, filenames in os.walk(fold_path + '/' + dir_test):
                for dirname in dirnames:
                    for sub_dirpath, sub_dirnames, sub_filenames in os.walk(dirpath + '/' + dirname):
                        for sub_file in sub_filenames:
                            if os.path.splitext(sub_file)[1] == '.test':
                                test_file = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                df_test = pd.concat([df_test, test_file])
                            if os.path.splitext(sub_file)[0] == 'rank.test' and os.path.splitext(sub_file)[
                                1] == '.query':
                                test_query_file = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                df_test_query = pd.concat([df_test_query, test_query_file])
                            if 'familiar_staff' == dirname or 'unfamiliar_staff' == dirname:
                                if os.path.splitext(sub_file)[1] == '.train':
                                    df_train_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                    df_train_staff = pd.concat([df_train_staff, df_train_fold])
                                if os.path.splitext(sub_file)[0] == 'rank.train' and os.path.splitext(sub_file)[
                                    1] == '.query':
                                    df_train_query_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                    df_train_query_staff = pd.concat([df_train_query_staff, df_train_query_fold])
                            if 'familiar_student' == dirname or 'unfamiliar_student' == dirname:
                                if os.path.splitext(sub_file)[1] == '.train':
                                    df_train_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                    df_train_student = pd.concat([df_train_student, df_train_fold])
                                if os.path.splitext(sub_file)[0] == 'rank.train' and os.path.splitext(sub_file)[
                                    1] == '.query':
                                    df_train_query_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                    df_train_query_student = pd.concat(
                                        [df_train_query_student, df_train_query_fold])

            test_file_path = role_path + '/' + dir_test + '/rank.test'
            df_test.to_csv(test_file_path, header=False, index=False)
            test_query_file_path = role_path + '/' + dir_test + '/rank.test.query'
            df_test_query.to_csv(test_query_file_path, header=False, index=False)

            df_train_file_path = role_path + '/' + dir_test + '/staff/rank.train'
            df_train_staff.to_csv(df_train_file_path, header=False, index=False)
            df_train_query_file_path = role_path + '/' + dir_test + '/staff/rank.train.query'
            df_train_query_staff.to_csv(df_train_query_file_path, header=False, index=False)

            df_train_file_path = role_path + '/' + dir_test + '/student/rank.train'
            df_train_student.to_csv(df_train_file_path, header=False, index=False)
            df_train_query_file_path = role_path + '/' + dir_test + '/student/rank.train.query'
            df_train_query_student.to_csv(df_train_query_file_path, header=False, index=False)


def combine_by_familiarity_cv(split_time):
    familiarity_path = '../../../data/processed/cv_dataset/ranking/familiarity/fold' + str(split_time)

    fold_path = '../../../data/processed/cv_dataset/ranking/fam_role/fold' + str(split_time)
    dirs = os.listdir(fold_path)

    for dir_test in dirs:
        df_valid_familiar = pd.DataFrame()
        df_valid_query_familiar = pd.DataFrame()
        df_valid_unfamiliar = pd.DataFrame()
        df_valid_query_unfamiliar = pd.DataFrame()
        df_train_familiar = pd.DataFrame()
        df_train_query_familiar = pd.DataFrame()
        df_train_unfamiliar = pd.DataFrame()
        df_train_query_unfamiliar = pd.DataFrame()
        if "fold" in dir_test:
            for dirpath, dirnames, filenames in os.walk(fold_path + '/' + dir_test):
                for filename in filenames:
                    if filename == 'familiar_staff_rank.train' or filename == 'familiar_student_rank.train':
                        df_train = pd.read_csv(dirpath + '/' + filename, header=None)
                        df_train_familiar = pd.concat([df_train_familiar, df_train])
                    if filename == 'unfamiliar_staff_rank.train' or filename == 'unfamiliar_student_rank.train':
                        df_train = pd.read_csv(dirpath + '/' + filename, header=None)
                        df_train_unfamiliar = pd.concat([df_train_unfamiliar, df_train])
                    if filename == 'familiar_staff_rank.valid' or filename == 'familiar_student_rank.valid':
                        df_valid = pd.read_csv(dirpath + '/' + filename, header=None)
                        df_valid_familiar = pd.concat([df_valid_familiar, df_valid])
                    if filename == 'unfamiliar_staff_rank.valid' or filename == 'unfamiliar_student_rank.valid':
                        df_valid = pd.read_csv(dirpath + '/' + filename, header=None)
                        df_valid_unfamiliar = pd.concat([df_valid_unfamiliar, df_valid])

                    if filename == 'familiar_staff_rank.train.query' or filename == 'familiar_student_rank.train.query':
                        df_train_query = pd.read_csv(dirpath + '/' + filename, header=None)
                        df_train_query_familiar = pd.concat([df_train_query_familiar, df_train_query])
                    if filename == 'unfamiliar_staff_rank.train.query' or filename == 'unfamiliar_student_rank.train.query':
                        df_train_query = pd.read_csv(dirpath + '/' + filename, header=None)
                        df_train_query_unfamiliar = pd.concat([df_train_query_unfamiliar, df_train_query])
                    if filename == 'familiar_staff_rank.valid.query' or filename == 'familiar_student_rank.valid.query':
                        df_valid_query = pd.read_csv(dirpath + '/' + filename, header=None)
                        df_valid_query_familiar = pd.concat([df_valid_query_familiar, df_valid_query])
                    if filename == 'unfamiliar_staff_rank.valid.query' or filename == 'unfamiliar_student_rank.valid.query':
                        df_valid_query = pd.read_csv(dirpath + '/' + filename, header=None)
                        df_valid_query_unfamiliar = pd.concat([df_valid_query_unfamiliar, df_valid_query])

            df_train_file_path = familiarity_path + '/' + dir_test + '/familiar_rank.train'
            df_train_familiar.to_csv(df_train_file_path, header=False, index=False)
            df_train_query_file_path = familiarity_path + '/' + dir_test + '/familiar_rank.train.query'
            df_train_query_familiar.to_csv(df_train_query_file_path, header=False, index=False)

            df_train_file_path = familiarity_path + '/' + dir_test + '/unfamiliar_rank.train'
            df_train_unfamiliar.to_csv(df_train_file_path, header=False, index=False)
            df_train_query_file_path = familiarity_path + '/' + dir_test + '/unfamiliar_rank.train.query'
            df_train_query_unfamiliar.to_csv(df_train_query_file_path, header=False, index=False)

            df_valid_file_path = familiarity_path + '/' + dir_test + '/familiar_rank.valid'
            df_valid_familiar.to_csv(df_valid_file_path, header=False, index=False)
            df_valid_query_file_path = familiarity_path + '/' + dir_test + '/familiar_rank.valid.query'
            df_valid_query_familiar.to_csv(df_valid_query_file_path, header=False, index=False)

            df_valid_file_path = familiarity_path + '/' + dir_test + '/unfamiliar_rank.valid'
            df_valid_unfamiliar.to_csv(df_valid_file_path, header=False, index=False)
            df_valid_query_file_path = familiarity_path + '/' + dir_test + '/unfamiliar_rank.valid.query'
            df_valid_query_unfamiliar.to_csv(df_valid_query_file_path, header=False, index=False)



def combine_by_familiarity_fit(split_time):
    familiarity_path = '../../../data/processed/cv_dataset/ranking/familiarity/fold' + str(split_time)

    fold_path = '../../../data/processed/cv_dataset/ranking/fam_role/fold' + str(split_time)

    df_test_familiar = pd.DataFrame()
    df_test_query_familiar = pd.DataFrame()
    df_test_unfamiliar = pd.DataFrame()
    df_test_query_unfamiliar = pd.DataFrame()
    df_train_familiar = pd.DataFrame()
    df_train_query_familiar = pd.DataFrame()
    df_train_unfamiliar = pd.DataFrame()
    df_train_query_unfamiliar = pd.DataFrame()

    for dirpath, dirnames, filenames in os.walk(fold_path):
        for filename in filenames:
            if filename == 'familiar_staff_rank.train' or filename == 'familiar_student_rank.train':
                df_train = pd.read_csv(dirpath + '/' + filename, header=None)
                df_train_familiar = pd.concat([df_train_familiar, df_train])
            if filename == 'unfamiliar_staff_rank.train' or filename == 'unfamiliar_student_rank.train':
                df_train = pd.read_csv(dirpath + '/' + filename, header=None)
                df_train_unfamiliar = pd.concat([df_train_unfamiliar, df_train])
            if filename == 'familiar_staff_rank.test' or filename == 'familiar_student_rank.test':
                df_test = pd.read_csv(dirpath + '/' + filename, header=None)
                df_test_familiar = pd.concat([df_test_familiar, df_test])
            if filename == 'unfamiliar_staff_rank.test' or filename == 'unfamiliar_student_rank.test':
                df_test = pd.read_csv(dirpath + '/' + filename, header=None)
                df_test_unfamiliar = pd.concat([df_test_unfamiliar, df_test])

            if filename == 'familiar_staff_rank.train.query' or filename == 'familiar_student_rank.train.query':
                df_train_query = pd.read_csv(dirpath + '/' + filename, header=None)
                df_train_query_familiar = pd.concat([df_train_query_familiar, df_train_query])
            if filename == 'unfamiliar_staff_rank.train.query' or filename == 'unfamiliar_student_rank.train.query':
                df_train_query = pd.read_csv(dirpath + '/' + filename, header=None)
                df_train_query_unfamiliar = pd.concat([df_train_query_unfamiliar, df_train_query])
            if filename == 'familiar_staff_rank.test.query' or filename == 'familiar_student_rank.test.query':
                df_test_query = pd.read_csv(dirpath + '/' + filename, header=None)
                df_test_query_familiar = pd.concat([df_test_query_familiar, df_test_query])
            if filename == 'unfamiliar_staff_rank.test.query' or filename == 'unfamiliar_student_rank.test.query':
                df_test_query = pd.read_csv(dirpath + '/' + filename, header=None)
                df_test_query_unfamiliar = pd.concat([df_test_query_unfamiliar, df_test_query])

    df_train_file_path = familiarity_path + '/familiar_rank.train'
    df_train_familiar.to_csv(df_train_file_path, header=False, index=False)
    df_train_query_file_path = familiarity_path + '/familiar_rank.train.query'
    df_train_query_familiar.to_csv(df_train_query_file_path, header=False, index=False)

    df_train_file_path = familiarity_path  + '/unfamiliar_rank.train'
    df_train_unfamiliar.to_csv(df_train_file_path, header=False, index=False)
    df_train_query_file_path = familiarity_path + '/unfamiliar_rank.train.query'
    df_train_query_unfamiliar.to_csv(df_train_query_file_path, header=False, index=False)

    df_test_file_path = familiarity_path + '/familiar_rank.test'
    df_test_familiar.to_csv(df_test_file_path, header=False, index=False)
    df_test_query_file_path = familiarity_path + '/familiar_rank.test.query'
    df_test_query_familiar.to_csv(df_test_query_file_path, header=False, index=False)

    df_test_file_path = familiarity_path + '/unfamiliar_rank.test'
    df_test_unfamiliar.to_csv(df_test_file_path, header=False, index=False)
    df_test_query_file_path = familiarity_path + '/unfamiliar_rank.test.query'
    df_test_query_unfamiliar.to_csv(df_test_query_file_path, header=False, index=False)


def combine_overall():
    fold_path = '../../../data/processed/fit_dataset/ranking/1-5-scale/5folds_fam_role'

    dirs = os.listdir(fold_path)

    overall_path = '../../../data/processed/fit_dataset/ranking/1-5-scale/5folds_overall'

    for dir_test in dirs:
        df_test = pd.DataFrame()
        df_test_query = pd.DataFrame()
        df_train = pd.DataFrame()
        df_train_query = pd.DataFrame()
        for dirpath, dirnames, filenames in os.walk(fold_path + '/' + dir_test):
            for dirname in dirnames:
                for sub_dirpath, sub_dirnames, sub_filenames in os.walk(dirpath + '/' + dirname):
                    for sub_file in sub_filenames:
                        if os.path.splitext(sub_file)[1] == '.test':
                            test_file = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                            df_test = pd.concat([df_test, test_file])
                        if os.path.splitext(sub_file)[0] == 'rank.test' and os.path.splitext(sub_file)[1] == '.query':
                            test_query_file = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                            df_test_query = pd.concat([df_test_query, test_query_file])
                        if os.path.splitext(sub_file)[1] == '.train':
                            df_train_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                            df_train = pd.concat([df_train, df_train_fold])
                        if os.path.splitext(sub_file)[0] == 'rank.train' and os.path.splitext(sub_file)[1] == '.query':
                            df_train_query_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                            df_train_query = pd.concat([df_train_query, df_train_query_fold])

        test_file_path = overall_path + '/' + dir_test + '/rank.test'
        df_test.to_csv(test_file_path, header=False, index=False)
        test_query_file_path = overall_path + '/' + dir_test + '/rank.test.query'
        df_test_query.to_csv(test_query_file_path, header=False, index=False)

        df_train_file_path = overall_path + '/' + dir_test + '/rank.train'
        df_train.to_csv(df_train_file_path, header=False, index=False)
        df_train_query_file_path = overall_path + '/' + dir_test + '/rank.train.query'
        df_train_query.to_csv(df_train_query_file_path, header=False, index=False)


def combine_by_familiarity():

    familiarity_path = '../../../data/processed/fit_dataset/ranking/1-5-scale/5folds_familiarity'

    fold_path = '../../../data/processed/fit_dataset/ranking/1-5-scale/5folds_fam_role'
    dirs = os.listdir(fold_path)

    for dir_test in dirs:
        df_test = pd.DataFrame()
        df_test_query = pd.DataFrame()
        df_train_familiar = pd.DataFrame()
        df_train_query_familiar = pd.DataFrame()
        df_train_unfamiliar = pd.DataFrame()
        df_train_query_unfamiliar = pd.DataFrame()
        if "fold" in dir_test:
            for dirpath, dirnames, filenames in os.walk(fold_path + '/' + dir_test):
                for dirname in dirnames:
                    for sub_dirpath, sub_dirnames, sub_filenames in os.walk(dirpath + '/' + dirname):
                        for sub_file in sub_filenames:
                            if os.path.splitext(sub_file)[1] == '.test':
                                test_file = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                df_test = pd.concat([df_test, test_file])
                            if os.path.splitext(sub_file)[0] == 'rank.test' and os.path.splitext(sub_file)[
                                1] == '.query':
                                test_query_file = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                df_test_query = pd.concat([df_test_query, test_query_file])
                            if 'familiar_staff' == dirname or 'familiar_student' == dirname:
                                if os.path.splitext(sub_file)[1] == '.train':
                                    df_train_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                    df_train_familiar = pd.concat([df_train_familiar, df_train_fold])
                                if os.path.splitext(sub_file)[0] == 'rank.train' and os.path.splitext(sub_file)[
                                    1] == '.query':
                                    df_train_query_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                    df_train_query_familiar = pd.concat([df_train_query_familiar, df_train_query_fold])
                            if 'unfamiliar_staff' == dirname or 'unfamiliar_student' == dirname:
                                if os.path.splitext(sub_file)[1] == '.train':
                                    df_train_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                    df_train_unfamiliar = pd.concat([df_train_unfamiliar, df_train_fold])
                                if os.path.splitext(sub_file)[0] == 'rank.train' and os.path.splitext(sub_file)[
                                    1] == '.query':
                                    df_train_query_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                    df_train_query_unfamiliar = pd.concat(
                                        [df_train_query_unfamiliar, df_train_query_fold])

            test_file_path = familiarity_path + '/' + dir_test + '/rank.test'
            df_test.to_csv(test_file_path, header=False, index=False)
            test_query_file_path = familiarity_path + '/' + dir_test + '/rank.test.query'
            df_test_query.to_csv(test_query_file_path, header=False, index=False)

            df_train_file_path = familiarity_path + '/' + dir_test + '/familiar/rank.train'
            df_train_familiar.to_csv(df_train_file_path, header=False, index=False)
            df_train_query_file_path = familiarity_path + '/' + dir_test + '/familiar/rank.train.query'
            df_train_query_familiar.to_csv(df_train_query_file_path, header=False, index=False)

            df_train_file_path = familiarity_path + '/' + dir_test + '/unfamiliar/rank.train'
            df_train_unfamiliar.to_csv(df_train_file_path, header=False, index=False)
            df_train_query_file_path = familiarity_path + '/' + dir_test + '/unfamiliar/rank.train.query'
            df_train_query_unfamiliar.to_csv(df_train_query_file_path, header=False, index=False)


def combine_by_role():
    role_path = '../../../data/processed/fit_dataset/ranking/1-5-scale/5folds_familiarity'

    fold_path = '../../../data/processed/fit_dataset/ranking/1-5-scale/5folds_fam_role'
    dirs = os.listdir(fold_path)

    for dir_test in dirs:
        df_test = pd.DataFrame()
        df_test_query = pd.DataFrame()
        df_train_staff = pd.DataFrame()
        df_train_query_staff = pd.DataFrame()
        df_train_student = pd.DataFrame()
        df_train_query_student = pd.DataFrame()
        for dirpath, dirnames, filenames in os.walk(fold_path + '/' + dir_test):
            for dirname in dirnames:
                for sub_dirpath, sub_dirnames, sub_filenames in os.walk(dirpath + '/' + dirname):
                    for sub_file in sub_filenames:
                        if os.path.splitext(sub_file)[1] == '.test':
                            test_file = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                            df_test = pd.concat([df_test, test_file])
                        if os.path.splitext(sub_file)[0] == 'rank.test' and os.path.splitext(sub_file)[1] == '.query':
                            test_query_file = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                            df_test_query = pd.concat([df_test_query, test_query_file])
                        if 'familiar_staff' == dirname or 'unfamiliar_staff' == dirname:
                            if os.path.splitext(sub_file)[1] == '.train':
                                df_train_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                df_train_staff = pd.concat([df_train_staff, df_train_fold])
                            if os.path.splitext(sub_file)[0] == 'rank.train' and os.path.splitext(sub_file)[
                                1] == '.query':
                                df_train_query_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                df_train_query_staff = pd.concat([df_train_query_staff, df_train_query_fold])
                        if 'familiar_student' == dirname or 'unfamiliar_student' == dirname:
                            if os.path.splitext(sub_file)[1] == '.train':
                                df_train_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                df_train_student = pd.concat([df_train_student, df_train_fold])
                            if os.path.splitext(sub_file)[0] == 'rank.train' and os.path.splitext(sub_file)[1] == '.query':
                                df_train_query_fold = pd.read_csv(sub_dirpath + '/' + sub_file, header=None)
                                df_train_query_student = pd.concat([df_train_query_student, df_train_query_fold])

        test_file_path = role_path + '/' + dir_test + '/rank.test'
        df_test.to_csv(test_file_path, header=False, index=False)
        test_query_file_path = role_path + '/' + dir_test + '/rank.test.query'
        df_test_query.to_csv(test_query_file_path, header=False, index=False)

        df_train_file_path = role_path + '/' + dir_test + '/staff/rank.train'
        df_train_staff.to_csv(df_train_file_path, header=False, index=False)
        df_train_query_file_path = role_path + '/' + dir_test + '/staff/rank.train.query'
        df_train_query_staff.to_csv(df_train_query_file_path, header=False, index=False)

        df_train_file_path = role_path + '/' + dir_test + '/student/rank.train'
        df_train_student.to_csv(df_train_file_path, header=False, index=False)
        df_train_query_file_path = role_path + '/' + dir_test + '/student/rank.train.query'
        df_train_query_student.to_csv(df_train_query_file_path, header=False, index=False)

# combine_by_familiarity()
for split_time in range(1, 18):
    # combine_by_role_iteration(split_time)
    # combine_by_familiarity_cv(split_time)
    combine_by_familiarity_fit(split_time)
    # combine_overall_iteration(split_time)