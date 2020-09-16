import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import os

def plot_by_average_gain():
    feature_dir = '../results/5fold/5folds_familiarity/fold'
    for familiarity in ['familiar', 'unfamiliar']:
        pd_total = pd.DataFrame()
        pd_total['features'] = ['color', 'intensity', 'shape_size', 'integration', 'choice',
                                'visibility', 'proximity2dp', 'proximity2fe', 'proximity2be',
                                'uniqueness', 'wellknownness', 'relevance', 'frequency']
        pd_total['importance'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for fold_index in range(1, 6):
            feature_gain_file = feature_dir + str(fold_index) + '/' + familiarity + '/gain_split_importance.csv'
            pd_feature_gain = pd.read_csv(feature_gain_file)
            pd_total['importance'] = pd_total['importance'] + pd_feature_gain['importance']

        pd_total['importance'] = pd_total['importance'] / 5
        pd_total.sort_values('importance', inplace=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        index = np.arange(len(pd_total['features']))

        features = np.array(list(pd_total['features']))
        importance = np.array(list(pd_total['importance']))
        bar_width = 0.4

        rect1 = plt.barh(index, importance, bar_width, alpha=0.8)
        font_label = {'family': 'Times New Roman',
                      'weight': 'normal',
                      'size': 20,
                      }
        ax.set_ylabel('Salience measures', font_label)
        ax.set_xlabel('Average gain', font_label)

        y_labels = pd_total['features']
        plt.yticks(index, y_labels, fontsize='20', fontfamily='Times New Roman')
        plt.xticks(fontsize='20', fontfamily='Times New Roman')

        ax.legend(handles=[rect1])

        for a, b in zip(index, importance):
            plt.text(b + 0.05, a, '%.3f' % b, ha='left', va='center', fontsize=20, fontfamily='Times New Roman')

        plt.savefig('figures/importance_' + familiarity + '.pdf', bbox_inches='tight')
        # plt.legend()
        plt.show()

        file_name = '5fold/average_' + familiarity + '_feature_importance.csv'
        pd_total.to_csv(file_name, encoding='utf-8')


def plot_by_total_gain(familiarity):
    feature_dir = '../../results/experiments/feature_importance/ranking_' + familiarity + '/'
    figure_dir = '../../results/reports/figures/'
    table_dir = '../../results/reports/tables/'
    # for familiarity in ['familiar', 'unfamiliar']:
    pd_total = pd.DataFrame()
    pd_total['features'] = ['color', 'intensity', 'shape_size', 'integration', 'choice',
                            'visibility', 'proximity2dp', 'proximity2fe', 'proximity2be',
                            'uniqueness', 'wellknownness', 'relevance', 'frequency']

    pd_total['importance'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for fold_index in range(1, 6):
        feature_gain_file = feature_dir + 'fold' + str(fold_index) + '_importance.csv'
        pd_feature_gain = pd.read_csv(feature_gain_file)
        ss = MinMaxScaler()
        scale_features = ['importance_gain']
        pd_feature_gain[scale_features] = ss.fit_transform(pd_feature_gain[scale_features])
        pd_total['importance'] = pd_total['importance'] + pd_feature_gain['importance_gain']

    # pd_total['importance'] = pd_total['importance'] / 5
    pd_total.sort_values('importance', inplace=True)

    variable_dic = {'color': 'vis_col', 'intensity': 'vis_its',
                    'shape_size': 'vis_siz', 'integration': 'str_itg',
                    'choice': 'str_cho', 'visibility': 'str_vbl',
                    'proximity2dp': 'str_ci',
                    'proximity2fe': 'str_fe', 'proximity2be': 'str_be',
                    'uniqueness': 'sem_fun', 'wellknownness': 'sem_nam',
                    'relevance': 'sem_rel', 'frequency': 'sem_fre'}

    for index, row in pd_total.iterrows():
        pd_total.loc[index, 'features'] = variable_dic[row['features']]

    fig, ax = plt.subplots(figsize=(10, 8))
    index = np.arange(len(pd_total['features']))

    # features = np.array(list(pd_total['features']))
    importance = np.array(list(pd_total['importance']))
    bar_width = 0.4

    rect1 = plt.barh(index, importance, bar_width, alpha=0.8)
    font_label = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 20,
                  }
    ax.set_ylabel('Salience measures', font_label)
    ax.set_xlabel('Gain', font_label)

    y_labels = pd_total['features']
    plt.yticks(index, y_labels, fontsize='20', fontfamily='Times New Roman')
    plt.xticks(np.arange(0, math.ceil(max(importance)), 0.5), fontsize='20', fontfamily='Times New Roman')

    # ax.legend(handles=[rect1])

    for a, b in zip(index, importance):
        plt.text(b + 0.05, a, '%.3f' % b, ha='left', va='center', fontsize=20, fontfamily='Times New Roman')

    cumulative_value = 0
    Q3_index = 0
    for index, row in pd_total.iterrows():
        cumulative_value += row['importance']
        if cumulative_value > np.sum(pd_total['importance']) * 0.25:
            break
        Q3_index += 1

    plt.axhline(Q3_index - bar_width / 2 - 0.3, 0, 19, color="r", linestyle="dashed", alpha=0.8,
                linewidth=4)

    plt.savefig(figure_dir + familiarity + '_model_fi.pdf', bbox_inches='tight')
    # plt.legend()
    plt.show()
    file_name = table_dir + familiarity + '_model_fi.csv'
    pd_total.to_csv(file_name, encoding='utf-8')

def plot_iteration_by_fam_role(familiarity, role):
    iters_dir = '../../results/cv_results/fam_role/'
    figure_dir = '../../results/reports/figures/'
    table_dir = '../../results/reports/tables/'

    # df_hr = pd.DataFrame(columns=['fold', 'role', 'familiarity', 'test set size', 'hit number'])
    # iter_list = list()

    pd_total = pd.DataFrame()
    pd_total['features'] = ['color', 'intensity', 'shape_size', 'integration', 'choice',
                            'visibility', 'proximity2dp', 'proximity2fe', 'proximity2be',
                            'uniqueness', 'wellknownness', 'relevance' ]

    pd_total['importance'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    iter_num = 0
    dir_list = os.listdir(iters_dir)
    for dir_name in dir_list:
        if 'iter' in dir_name:
            iter_num += 1
            # print(iter_num)
            iter_dir = os.path.join(iters_dir, dir_name)
            feature_dir = iter_dir + '/experiments/feature_importance/ranking_' + role + '_' + familiarity

            for model_path, model_dir_list, model_file_list in os.walk(feature_dir):
                for feature_imporatnce_file in model_file_list:
                    if 'importance' in feature_imporatnce_file:
                        pd_feature_gain = pd.read_csv(feature_dir + '/' + feature_imporatnce_file, index_col=0)
                        ss = MinMaxScaler()
                        scale_features = ['importance_gain']
                        pd_feature_gain[scale_features] = ss.fit_transform(pd_feature_gain[scale_features])
                        pd_total['importance'] = pd_total['importance'] + pd_feature_gain['importance_gain']

    # for path, dir_list, file_list in os.walk(iters_dir):
    pd_total['importance'] = pd_total['importance'] / 5

    pd_total['importance'] = pd_total['importance'] / iter_num
    pd_total.sort_values('importance', inplace=True)

    variable_dic = {'color': 'vis_col', 'intensity': 'vis_its',
                    'shape_size': 'vis_siz', 'integration': 'str_itg',
                    'choice': 'str_cho', 'visibility': 'str_vbl',
                    'proximity2dp': 'str_ci',
                    'proximity2fe': 'str_fe', 'proximity2be': 'str_be',
                    'uniqueness': 'sem_fun', 'wellknownness': 'sem_nam',
                    'relevance': 'sem_rel', 'frequency': 'sem_fre'}

    for index, row in pd_total.iterrows():
        pd_total.loc[index, 'features'] = variable_dic[row['features']]

    fig, ax = plt.subplots(figsize=(10, 8))
    index = np.arange(len(pd_total['features']))

    # features = np.array(list(pd_total['features']))
    importance = np.array(list(pd_total['importance']))
    bar_width = 0.4

    rect1 = plt.barh(index, importance, bar_width, alpha=0.8)
    font_label = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 20,
                  }
    ax.set_ylabel('Salience measures', font_label)
    ax.set_xlabel('Gain', font_label)

    y_labels = pd_total['features']
    plt.yticks(index, y_labels, fontsize='20', fontfamily='Times New Roman')
    plt.xticks(np.arange(0, math.ceil(max(importance)), 0.5), fontsize='20', fontfamily='Times New Roman')

    # ax.legend(handles=[rect1])

    for a, b in zip(index, importance):
        plt.text(b + 0.05, a, '%.3f' % b, ha='left', va='center', fontsize=20, fontfamily='Times New Roman')

    cumulative_value = 0
    Q3_index = 0
    for index, row in pd_total.iterrows():
        cumulative_value += row['importance']
        if cumulative_value > np.sum(pd_total['importance']) * 0.25:
            break
        Q3_index += 1

    plt.axhline(Q3_index - bar_width / 2 - 0.3, 0, 19, color="r", linestyle="dashed", alpha=0.8,
                linewidth=4)

    plt.savefig(figure_dir + role + '_' + familiarity + '_model_fi.pdf', bbox_inches='tight')
    plt.show()
    file_name = table_dir + role + '_' + familiarity  + '_model_fi.csv'
    pd_total.to_csv(file_name, encoding='utf-8')


def plot_iteration_by_total_gain(type, attribute):

    feature_dir = '../../results/cv_results/familiarity/feature_importance/ranking_' + attribute
    # df_hr = pd.DataFrame(columns=['fold', 'role', 'familiarity', 'test set size', 'hit number'])
    # iter_list = list()

    pd_total = pd.DataFrame()
    pd_total['features'] = ['color', 'intensity', 'shape_size', 'integration', 'choice',
                            'visibility', 'proximity2dp', 'proximity2fe', 'proximity2be',
                            'uniqueness', 'wellknownness', 'relevance']

    pd_total['importance'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    iter_num = 0
    # dir_list = os.listdir(iters_dir)

    for model_path, model_dir_list, model_file_list in os.walk(feature_dir):
        for feature_imporatnce_file in model_file_list:
            if 'importance' in feature_imporatnce_file:
                iter_num += 1
                pd_feature_gain = pd.read_csv(feature_dir + '/' + feature_imporatnce_file, index_col=0)
                ss = MinMaxScaler()
                scale_features = ['importance_gain']
                pd_feature_gain[scale_features] = ss.fit_transform(pd_feature_gain[scale_features])
                pd_total['importance'] = pd_total['importance'] + pd_feature_gain['importance_gain']

    print(iter_num)

    pd_total['importance'] = pd_total['importance'] / iter_num
    pd_total.sort_values('importance', inplace=True)

    variable_dic = {'color': 'vis_col', 'intensity': 'vis_its',
                    'shape_size': 'vis_siz', 'integration': 'str_itg',
                    'choice': 'str_cho', 'visibility': 'str_vbl',
                    'proximity2dp': 'str_ci',
                    'proximity2fe': 'str_fe', 'proximity2be': 'str_be',
                    'uniqueness': 'sem_fun', 'wellknownness': 'sem_nam',
                    'relevance': 'sem_rel', 'frequency': 'sem_fre'}

    for index, row in pd_total.iterrows():
        pd_total.loc[index, 'features'] = variable_dic[row['features']]

    # plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 8))
    index = np.arange(len(pd_total['features']))

    # features = np.array(list(pd_total['features']))
    importance = np.array(list(pd_total['importance']))
    bar_width = 0.4

    rect1 = plt.barh(index, importance, bar_width, alpha=0.8)
    font_label = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 20,
                  }
    ax.set_ylabel('Salience measures', font_label)
    ax.set_xlabel('Gain', font_label)

    y_labels = pd_total['features']
    plt.yticks(index, y_labels, fontsize='20', fontfamily='Times New Roman')
    plt.xticks(np.arange(0, math.ceil(max(importance)), 0.5), fontsize='20', fontfamily='Times New Roman')

    # ax.legend(handles=[rect1])

    for a, b in zip(index, importance):
        plt.text(b + 0.01, a, '%.3f' % b, ha='left', va='center', fontsize=20, fontfamily='Times New Roman')

    cumulative_value = 0
    Q3_index = 0
    for index, row in pd_total.iterrows():
        cumulative_value += row['importance']
        if cumulative_value > np.sum(pd_total['importance']) * 0.25:
            break
        Q3_index += 1

    # plt.axhline(Q3_index - bar_width / 2 - 0.3, 0, 19, color="r", linestyle="dashed", alpha=0.8, linewidth=4)
    figure_dir = '../../results/reports/figures/'
    plt.savefig(figure_dir + type + '/' + attribute + '_model_fi.pdf', bbox_inches='tight')
    # plt.legend()
    plt.show()

    table_dir = '../../results/reports/tables/'
    file_name = table_dir +  type + '/' + attribute + '_model_fi.csv'
    pd_total.to_csv(file_name, encoding='utf-8')


for familiarity in ['familiar', 'unfamiliar']:
    plot_iteration_by_total_gain(type='familiarity', attribute=familiarity)
#
# for role in ['staff', 'student']:
#     plot_iteration_by_total_gain(type='role', attribute=role)

# plot_iteration_by_total_gain(type='overall', attribute= 'overall')


# for familiarity in ['familiar', 'unfamiliar']:
#     for role in ['staff', 'student']:
#         print(role + ' for ' + familiarity)
#         plot_iteration_by_fam_role(familiarity, role)