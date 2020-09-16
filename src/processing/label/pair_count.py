import pandas as pd
import csv

def pair4familiar(filename):
    df_data = pd.read_csv(filename)
    pairs = list()
    pairsinfo = list()
    for index, row in df_data.iterrows():
        pair_set = {row['place1'], row['place2']}
        if pair_set not in pairs:
            pairs.append(pair_set)
            pair_dict = {'place1': row['place1'], 'place2': row['place2'], 'pos_num': 0, 'neg_num': 0, 'eql_num': 0}
            if row['familiar'] == '1':
                pair_dict['pos_num'] += 1
            elif row['familiar'] == '-1':
                pair_dict['neg_num'] += 1
            else:
                pair_dict['eql_num'] += 1
            pairsinfo.append(pair_dict)
        else:
            pair_index = pairs.index(pair_set)
            pair_dict = pairsinfo[pair_index]
            if row['place1'] == pair_dict['place1']:
                if row['familiar'] == '1':
                    pair_dict['pos_num'] += 1
                elif row['familiar'] == '-1':
                    pair_dict['neg_num'] += 1
                else:
                    pair_dict['eql_num'] += 1
            else:
                if row['familiar'] == '1':
                    pair_dict['neg_num'] += 1
                elif row['familiar'] == '-1':
                    pair_dict['pos_num'] += 1
                else:
                    pair_dict['eql_num'] += 1

    return pairsinfo

def pair4unfamiliar(filename):
    df_data = pd.read_csv(filename)
    pairs = list()
    pairsinfo = list()
    for index, row in df_data.iterrows():
        pair_set = {row['place1'], row['place2']}
        if pair_set not in pairs:
            pairs.append(pair_set)
            pair_dict = {'place1': row['place1'], 'place2': row['place2'], 'pos_num': 0, 'neg_num': 0, 'eql_num': 0}
            if row['unfamiliar'] == '1':
                pair_dict['pos_num'] += 1
            elif row['unfamiliar'] == '-1':
                pair_dict['neg_num'] += 1
            else:
                pair_dict['eql_num'] += 1
            pairsinfo.append(pair_dict)
        else:
            pair_index = pairs.index(pair_set)
            pair_dict = pairsinfo[pair_index]
            if row['place1'] == pair_dict['place1']:
                if row['unfamiliar'] == '1':
                    pair_dict['pos_num'] += 1
                elif row['unfamiliar'] == '-1':
                    pair_dict['neg_num'] += 1
                else:
                    pair_dict['eql_num'] += 1
            else:
                if row['unfamiliar'] == '1':
                    pair_dict['neg_num'] += 1
                elif row['unfamiliar'] == '-1':
                    pair_dict['pos_num'] += 1
                else:
                    pair_dict['eql_num'] += 1

    return pairsinfo

print("staff for familiar wayfinders")
headers = ['place1', 'place2', 'pos_num', 'neg_num', 'eql_num']

pairs4familiar_staff = pair4familiar("data/fs11-36.csv")
for pairinfo in pairs4familiar_staff :
    print(pairinfo)
with open('staff4familir.csv', 'w') as f:
    f_csv = csv.DictWriter(f, headers)
    f_csv.writeheader()
    f_csv.writerows(pairs4familiar_staff)

print("staff for familiar wayfinders")
pairs4unfamiliar_staff = pair4unfamiliar("data/fs11-36.csv")
for pairinfo in pairs4unfamiliar_staff :
    print(pairinfo)
with open('staff4unfamilir.csv', 'w') as f:
    f_csv = csv.DictWriter(f, headers)
    f_csv.writeheader()
    f_csv.writerows(pairs4unfamiliar_staff)

print("students for familiar wayfinders")
pairs4familiar_student = pair4familiar("data/stu1-23.csv")
for pairinfo in pairs4familiar_student :
    print(pairinfo)
with open('student4familir.csv', 'w') as f:
    f_csv = csv.DictWriter(f, headers)
    f_csv.writeheader()
    f_csv.writerows(pairs4familiar_student)

print("students for unfamiliar wayfinders")
pairs4unfamiliar_student  = pair4unfamiliar("data/stu1-23.csv")
for pairinfo in pairs4familiar_student :
    print(pairinfo)
with open('student4unfamilir.csv', 'w') as f:
    f_csv = csv.DictWriter(f, headers)
    f_csv.writeheader()
    f_csv.writerows(pairs4unfamiliar_student)


