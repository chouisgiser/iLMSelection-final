import pandas as pd
from utils import file_io
from src.processing.label import dataset_label
from sklearn.preprocessing import MinMaxScaler



def leave_out_lm_split(lm_name, X, y):
    """
    # lm_name   : the name of left out landmark
    # X         : the dataframe of read pairs of labelled data
    # y         : the dataframe of query numbers
    """

    X_train = pd.DataFrame(columns=X.columns.tolist())
    X_test = pd.DataFrame(columns=X.columns.tolist())
    y_train_list = list()
    y_test_list = list()

    for index, row in X.iterrows():
        if row['place1'] != lm_name and row['place2'] != lm_name:
            X_train = X_train.append(row, ignore_index=True)
            y_train_list.append(y[index])
        else:
            X_test = X_test.append(row, ignore_index=True)
            y_test_list.append(y[index])

    y_train = pd.Series(data=y_train_list, name=y.name)
    y_test = pd.Series(data=y_test_list, name=y.name)

    return X_train, X_test, y_train, y_test

def dataset_split(role, familiarity):
    label_file = '../../../data/processed/label/' + role + "4" + familiarity + ".csv"
    feature_file = '../../../data/processed/features/salience_' + role + ".csv"

    scale_features = ['color', 'intensity', 'shape_size', 'integration', 'choice', 'control', 'visibility',
                      'proximity2dp',
                      'proximity2fe', 'proximity2be', 'uniqueness', 'wellknownness', 'relevance', 'frequency']

    pd_feature_data = pd.read_csv(feature_file)
    pd_feature_data = pd_feature_data.drop(['role'], axis=1)

    ss = MinMaxScaler()
    pd_feature_data[scale_features] = ss.fit_transform(pd_feature_data[scale_features])
    print(pd_feature_data)


    cv_rank_dir = '../../../data/processed/cv_dataset/ranking/fam_role/'
    cv_clf_dir = '../../../data/processed/cv_dataset/classification/fam_role/'
    cv_reg_dir = '../../../data/processed/cv_dataset/regression/fam_role/'


    pd_label_data = pd.read_csv(label_file)

    # delete the equally suitable pairs
    for index, row in pd_label_data.iterrows():
        if row['rel1'] == row['rel2']:
            pd_label_data.drop(index, inplace=True)

    y = pd_label_data['query_num']
    X = pd_label_data.drop(['query_num'], axis=1)

    lm_name_list = ['Elevator', 'H38', 'H79', 'H92', 'J10', 'J39', 'J41', 'K13', 'K32', 'L11', 'L12', 'L26', 'L40', 'WC-H',  'WC-J',  'WC-K',  'WC-L']
    # lm_name_list = ['Elevator']

    fold_index = 1
    for lm_name in lm_name_list:
        X_train, X_test, y_train, y_test = leave_out_lm_split(lm_name, X, y)
        train_labels = pd.concat([X_train, y_train], axis=1)
        test_labels = pd.concat([X_test, y_test], axis=1)

        rank_train_rels, rank_train_features, rank_train_query = dataset_label.rel_salience_rank_label(train_labels, pd_feature_data, 5)
        rank_train_file = cv_rank_dir + 'fold{}'.format(fold_index) + '/' + familiarity + '_' + role + '_rank.train'
        file_io.write_rank_file(rank_train_file, rank_train_rels, rank_train_features)
        rank_train_query_file = cv_rank_dir + 'fold{}'.format(fold_index) + '/' + familiarity + '_' + role + '_rank.train.query'
        file_io.write_query_file(rank_train_query_file, rank_train_query)

        rank_valid_rels, rank_valid_features, rank_valid_query = dataset_label.rel_salience_rank_label(test_labels, pd_feature_data, 5)
        rank_valid_file = cv_rank_dir + 'fold{}'.format(fold_index) + '/' + familiarity + '_' + role + '_rank.test'
        file_io.write_rank_file(rank_valid_file, rank_valid_rels, rank_valid_features)
        rank_valid_query_file = cv_rank_dir + 'fold{}'.format(fold_index) + '/' + familiarity + '_' + role + '_rank.test.query'
        file_io.write_query_file(rank_valid_query_file, rank_valid_query)

        clf_train_labels, clf_train_features = dataset_label.rel_salience_clf_label(train_labels, pd_feature_data)
        clf_train_file = cv_clf_dir + 'fold{}'.format(fold_index) + '/' + familiarity + '_' + role + '_rel_clf_train.csv'
        file_io.write_classification_file(clf_train_file, clf_train_labels, clf_train_features)

        clf_test_labels, clf_test_features = dataset_label.rel_salience_clf_label(test_labels, pd_feature_data)
        clf_test_file = cv_clf_dir + 'fold{}'.format(fold_index) + '/' + familiarity + '_' + role + '_rel_clf_test.csv'
        file_io.write_classification_file(clf_test_file, clf_test_labels, clf_test_features)

        reg_train_labels, reg_train_features = dataset_label.abs_salience_reg_label(train_labels, pd_feature_data)
        abs_train_file = cv_reg_dir + 'fold{}'.format(fold_index) + '/' + familiarity + '_' + role + '_abs_reg_train.csv'
        file_io.write_regression_file(abs_train_file, reg_train_labels, reg_train_features)

        reg_test_labels, reg_test_features = dataset_label.abs_salience_reg_label(X_test, pd_feature_data)
        abs_test_file = cv_reg_dir + 'fold{}'.format(fold_index) + '/' + familiarity + '_' + role + '_abs_reg_test.csv'
        file_io.write_regression_file(abs_test_file, reg_test_labels, reg_test_features)

        fold_index += 1

    # write the train file and test file into fit_dataset

# for split_time in range(1, 21):
familiarity = 'familiar'
role = 'student'
dataset_split(role, familiarity)

# familiarity = 'unfamiliar'
# role = 'student'
# split_default(role, familiarity)