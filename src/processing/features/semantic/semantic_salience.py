# from googleapiclient.discovery import build
#
# api_key = "AIzaSyAuODcj8k2hjpFjLu-oHRt40dl5tMMWCUc"
# cse_key = "013918263173898975413:h634n9tdopf"
#
# def getService(key):
#     service = build("customsearch", "v1", developerKey= key)
#
#     return service
#
#
# def google_search (search_term, cse_id, **kwargs):
#     service = getService(api_key)
#     response = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
#     return response['items']
#
#
# response = google_search('student', cse_key, lr='lang_en')
#
# print(len(response))
import pandas as pd
import csv

def webJaccard(count_p, count_q, count_occ, c):
    if count_occ < c:
        jaccard_index = 0
    else:
        jaccard_index = count_occ/(count_p + count_q - count_occ)

    return jaccard_index


def webDice(count_p, count_q, count_occ, c):
    if count_occ < c:
        dice_index = 0
    else:
        dice_index = 2 * count_occ / (count_p + count_q)

    return dice_index

def webOverlap(count_p, count_q, count_occ, c):
    if count_occ < c:
        overlap_index = 0
    else:
        if count_p < count_q:
            overlap_index = count_occ / count_p
        else:
            overlap_index = count_occ / count_q

    return overlap_index

def webJaccard4file(file):
    rows = list()
    df_data = pd.read_csv(file)
    for index, row in df_data.iterrows():
        count_p = row['count_role']
        count_q = row['count_name']
        count_occ = (row['count_occ_1'] + row['count_occ_2']) / 2
        c = 10
        jaccard = webJaccard(count_p, count_q, count_occ, c)
        newrow = {"role": row['role'], "name": row['name'], "web_jaccard": jaccard}
        rows.append(newrow)

    return rows

def webDice4file(file):
    rows = list()
    df_data = pd.read_csv(file)
    for index, row in df_data.iterrows():
        count_p = row['count_role']
        count_q = row['count_name']
        count_occ = (row['count_occ_1'] + row['count_occ_2']) / 2
        c = 10
        dice = webDice(count_p, count_q, count_occ, c)
        newrow = {"role": row['role'], "name": row['name'], "web_dice": dice}
        rows.append(newrow)

    return rows

def webOverlap4file(file):
    rows = list()
    df_data = pd.read_csv(file)
    for index, row in df_data.iterrows():
        count_p = row['count_role']
        count_q = row['count_name']
        count_occ = (row['count_occ_1'] + row['count_occ_2']) / 2
        c = 10
        overlap = webOverlap(count_p, count_q, count_occ, c)
        newrow = {"role": row['role'], "name": row['name'], "web_overlap": overlap}
        rows.append(newrow)

    return rows

file = "../data/search_count_1.csv"
rows = webJaccard4file(file)
headers = ['role', 'name', 'web_jaccard']
with open('wellknownness_jaccard.csv', 'w') as f:
    f_csv = csv.DictWriter(f, headers)
    f_csv.writeheader()
    f_csv.writerows(rows)

# file = "data/search_count_1.csv"
# rows = webDice4file(file)
# headers = ['role', 'name', 'web_dice']
# with open('wellknownness_dice.csv', 'w') as f:
#     f_csv = csv.DictWriter(f, headers)
#     f_csv.writeheader()
#     f_csv.writerows(rows)

