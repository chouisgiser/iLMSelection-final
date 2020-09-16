import csv
import pandas as pd
import geopandas as gpd

def write_csv_file(filename, dicts, headers):
    with open(filename, 'w') as f:
        f_csv = csv.DictWriter(f, headers)
        f_csv.writeheader()
        f_csv.writerows(dicts)


def split4ranking(dataframe, test_size):
    row_num = dataframe.shape[0]
    train_dataframe = dataframe.iloc[0: row_num - test_size, :]
    test_dataframe = dataframe.iloc[row_num - test_size: row_num, :]

    return train_dataframe, test_dataframe

# def split4calssification(dataframe, test_size):
#     row_num = dataframe.shape[0]
#     train_dataframe = dataframe.iloc[0: row_num - test_size, :]
#     test_dataframe = dataframe.iloc[row_num - test_size: row_num, :]
#
#     return train_dataframe, test_dataframe

def genetic_programming_data (pd_label_data, pd_feature_data):
    salience_list = list()
    feature_list = list()

    for index, row in pd_label_data.iterrows():
        place1 = row["place1"]
        place2 = row["place2"]

        rel1 = row['rel1']
        rel2 = row['rel2']

        total_rel = rel1 + rel2

        if total_rel != 0:
            rel1 = rel1 / total_rel
            rel2 = rel2 / total_rel

        else:
            rel1 = 0
            rel2 = 0
        salience_list.append(rel1)
        salience_list.append(rel2)

        place1_feature = pd_feature_data.loc[pd_feature_data["candidate"] == place1]
        place1_feature = place1_feature.drop(['candidate'], axis=1)
        feature_list.append(place1_feature.iloc[0, :])

        place2_feature = pd_feature_data.loc[pd_feature_data["candidate"] == place2]
        place2_feature = place2_feature.drop(['candidate'], axis=1)
        feature_list.append(place2_feature.iloc[0, :])

    return salience_list, feature_list

def write_genetic_programming_file(target, salience_diff_list, feature_diff_list):
    gp_file = open(target, 'a')

    for index in range(0, len(feature_diff_list)):
        gp_file.write(str(salience_diff_list[index]))
        gp_file.write(' ')
        for featur_index in range(0, len(feature_diff_list[index])):
            value = feature_diff_list[index][featur_index]
            gp_file.write(str(value))
            if featur_index is not (len(feature_diff_list[index])-1):
                gp_file.write(' ')

        gp_file.write('\n')

    gp_file.close()

def linear_regression_data(pd_label_data, pd_feature_data):
    salience_diff_list = list()
    feature_diff_list = list()

    for index, row in pd_label_data.iterrows():
        place1 = row["place1"]
        place2 = row["place2"]

        rel1 = row['rel1']
        rel2 = row['rel2']

        if rel1 + rel2 != 0:
            salience_diff = (rel1 - rel2) / (rel1 + rel2)
        else:
            salience_diff = 0
        salience_diff_list.append(salience_diff)

        place1_feature = pd_feature_data.loc[pd_feature_data["candidate"] == place1]
        place1_feature = place1_feature.drop(['candidate'], axis=1)
        place2_feature = pd_feature_data.loc[pd_feature_data["candidate"] == place2]
        place2_feature = place2_feature.drop(['candidate'], axis=1)
        feature_diff = place1_feature.iloc[0, :] - place2_feature.iloc[0, :]

        feature_diff_list.append(feature_diff.values)

    return salience_diff_list, feature_diff_list


def write_regression_file(target, salience_diff_list, feature_diff_list):
    regression_file = open(target, 'a')

    for index in range(0, len(feature_diff_list)):
        regression_file.write(str(salience_diff_list[index]))
        regression_file.write(' ')
        for featur_index in range(0, len(feature_diff_list[index])):
            value = feature_diff_list[index][featur_index]
            regression_file.write(str(value))
            if featur_index is not (len(feature_diff_list[index])-1):
                regression_file.write(' ')

        regression_file.write('\n')

    regression_file.close()



def write_classification_file(target, label_list, feature_diff_list):
    calssification_file = open(target, 'a')

    for index in range(0, len(feature_diff_list)):
        calssification_file.write(str(label_list[index]))
        calssification_file.write(' ')
        for featur_index in range(0, len(feature_diff_list[index])):
            value = feature_diff_list[index][featur_index]
            calssification_file.write(str(value))
            if featur_index is not (len(feature_diff_list[index])-1):
                calssification_file.write(' ')

        calssification_file.write('\n')

    calssification_file.close()


def pair_rank_data(pd_label_data, pd_feature_data):
    paired_features = list()

    rel_list = list()

    for index, row in pd_label_data.iterrows():
        place1 = row["place1"]
        place2 = row["place2"]
        place1_rel = row["pos_num"]
        rel_list.append(place1_rel)
        place2_rel = row["neg_num"]
        rel_list.append(place2_rel)

        place1_feature = pd_feature_data.loc[pd_feature_data["candidate"] == place1]
        place1_feature = place1_feature.drop(['candidate'], axis=1)
        place2_feature = pd_feature_data.loc[pd_feature_data["candidate"] == place2]
        place2_feature = place2_feature.drop(['candidate'], axis=1)
        rank_place1 = dict()
        rank_place2 = dict()

        for column_index in range(0, place1_feature.shape[1]):
            place1_index_value = place1_feature.iat[0, column_index]
            if not pd.isna(place1_index_value):
                rank_place1[column_index] = place1_index_value
        paired_features.append(rank_place1)

        for column_index in range(0, place2_feature.shape[1]):
            place2_index_value = place2_feature.iat[0, column_index]
            if not pd.isna(place2_index_value):
                rank_place2[column_index] = place2_index_value
        paired_features.append(rank_place2)

    return rel_list, paired_features


def triple_rank_data(pd_label_data, pd_feature_data):
    rel_list = list()
    feature_group = list()
    query_group = list(pd_label_data['query_num'])

    for index, row in pd_label_data.iterrows():
        query_number = row['query_num']

        for i in range(query_number):
            rel = row["rel" + str(i+1)]
            rel_list.append(rel)

            place = row["place" + str(i+1)]
            place_feature = pd_feature_data.loc[pd_feature_data["candidate"] == place]
            place_feature = place_feature.drop(['candidate'], axis=1)

            rank_place = dict()
            for column_index in range(0, place_feature.shape[1]):
                place_index_value = place_feature.iat[0, column_index]
                if not pd.isna(place_index_value):
                    rank_place[column_index] = place_index_value

            feature_group.append(rank_place)

    return rel_list, feature_group, query_group


def write_rank_file(target, rel_list, paired_features):
    rank_file = open(target, 'a')

    for index in range(0, len(paired_features)):
        rank_file.write(str(rel_list[index]))
        rank_file.write(' ')
        for key in paired_features[index]:
            rank_file.write(str(key) + ':' + str(paired_features[index][key]) + " ")
        rank_file.write('\n')

    rank_file.close()


def write_pair_query_file(target, size):
    query_file = open(target, 'a')
    for index in range(0, size):
        query_file.write(str(2))
        query_file.write('\n')
    query_file.close()


def write_query_file(target, query_list):
    query_file = open(target, 'a')
    for query in query_list:
        query_file.write(str(query))
        query_file.write('\n')
    query_file.close()


def read_shapefile(file, attributes):
    rows = list()
    shape_data = gpd.read_file(file)
    for idx, row in shape_data.iterrows():
        # id = idx
        geometry = row["geometry"]
        newrow = [idx, geometry]
        for attribute in attributes:
            newrow.append(row[attribute])
        rows.append(newrow)
    return rows