import pandas as pd

def abs_salience_reg_label (pd_label_data, pd_feature_data):
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

def rel_salience_clf_label (pd_label_data, pd_feature_data):
    label_list = list()
    feature_diff_list = list()

    if 'pos_num' in pd_label_data.columns.values.tolist():
        label4place1 = 'pos_num'
        label4place2 = 'neg_num'
    else:
        label4place1 = 'rel1'
        label4place2 = 'rel2'

    for index, row in pd_label_data.iterrows():
        place1 = row["place1"]
        place2 = row["place2"]

        rel1 = row[label4place1]
        rel2 = row[label4place2]

        if rel1 > rel2:
            label = 2
        elif rel1 < rel2:
            label = 1
        else:
            label = 0
        label_list.append(label)

        place1_feature = pd_feature_data.loc[pd_feature_data["candidate"] == place1]
        place1_feature = place1_feature.drop(['candidate'], axis=1)
        place2_feature = pd_feature_data.loc[pd_feature_data["candidate"] == place2]
        place2_feature = place2_feature.drop(['candidate'], axis=1)
        feature_diff = place1_feature.iloc[0, :] - place2_feature.iloc[0, :]

        feature_diff_list.append(feature_diff.values)

    return label_list, feature_diff_list


def label_scaling(rel, scale):
    # 1-2 point label
    if scale == 2:
        if rel > 0.5:
            rel = 1
        else:
            rel = 0
    # 1-5 point label
    if scale == 5:
        if rel >= 0 and rel <= 0.2:
            rel = 1
        elif rel > 0.2 and rel <= 0.4:
            rel = 2
        elif rel > 0.4 and rel <= 0.6:
            rel = 3
        elif rel > 0.6 and rel <= 0.8:
            rel = 4
        elif rel > 0.8 and rel <= 1.0:
            rel = 5
    # 1-10 point label
    if scale == 7:
        if rel >= 0 and rel <= 0.14:
            rel = 1
        elif rel > 0.1 and rel <= 0.28:
            rel = 2
        elif rel > 0.2 and rel <= 0.42:
            rel = 3
        elif rel > 0.3 and rel <= 0.56:
            rel = 4
        elif rel > 0.4 and rel <= 0.70:
            rel = 5
        elif rel > 0.5 and rel <= 0.84:
            rel = 6
        # elif rel > 0.6 and rel <= 0.7:
        #     rel = 7
        # elif rel > 0.7 and rel <= 0.8:
        #     rel = 8
        # elif rel > 0.8 and rel <= 0.9:
        #     rel = 9
        else:
            rel = 7

    return rel

def rel_salience_rank_label(pd_label_data, pd_feature_data, scale):
    rel_list = list()
    feature_group = list()
    query_group = list(pd_label_data['query_num'])

    for index, row in pd_label_data.iterrows():
        query_number = row['query_num']

        tmp_rel_list = []
        for i in range(query_number):
            rel = row["rel" + str(i+1)]
            tmp_rel_list.append(rel)
            # rel_list.append(rel)

            place = row["place" + str(i+1)]
            place_feature = pd_feature_data.loc[pd_feature_data["candidate"] == place]
            place_feature = place_feature.drop(['candidate'], axis=1)

            rank_place = dict()
            for column_index in range(0, place_feature.shape[1]):
                place_index_value = place_feature.iat[0, column_index]
                if not pd.isna(place_index_value):
                    rank_place[column_index] = place_index_value

            feature_group.append(rank_place)

        for rel in tmp_rel_list:
            if sum(tmp_rel_list) != 0:
                rel = label_scaling(rel/sum(tmp_rel_list), scale)
                rel_list.append(rel)
            else:
                rel_list.append(1)

        # if tmp_rel_list[0] > tmp_rel_list[1]:
        #     rel_list.append(1)
        #     rel_list.append(0)
        # elif tmp_rel_list[0] < tmp_rel_list[1]:
        #     rel_list.append(0)
        #     rel_list.append(1)
        # else:
        #     rel_list.append(0)
        #     rel_list.append(0)

    return rel_list, feature_group, query_group

def rel_salience_reg_label(pd_label_data, pd_feature_data):
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