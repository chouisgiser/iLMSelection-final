import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_iteration_overall_model_4_groups():
    iters_dir = '../../results/cv_results/overall'
    df_hr = pd.DataFrame(columns=['fold', 'role', 'familiarity', 'test set size', 'hit number'])
    iter_list = list()
    dir_list = os.listdir(iters_dir)
    iter_num = 0
    for dir_name in dir_list:
        if 'iter' in dir_name:
            iter_num += 1
            iter_dir = os.path.join(iters_dir, dir_name)
            data_hr = iter_dir + '/experiments/evaluation/ranking_overall'

            for model_path, model_dir_list, model_file_list in os.walk(data_hr):
                for model_file in model_file_list:
                    if 'evaluation' in model_file:
                        df_iter_hr = pd.read_csv(data_hr + '/' + model_file, index_col=0)
                        df_hr = pd.concat([df_hr, df_iter_hr], ignore_index=True)
                        iter_list.extend([iter_num] * 4)

    # for index in range(1, 6):
    #     dir_name = 'iter' + str (index)
    #     if dir_name in dir_list:
    #         iter_num += 1
    #         iter_dir = os.path.join(iters_dir, dir_name)
    #         data_hr = iter_dir + '/experiments/evaluation'
    #
    #         for model_path, model_dir_list, model_file_list in os.walk(data_hr):
    #             for model_file in model_file_list:
    #                 if 'evaluation' in model_file:
    #                     df_iter_hr = pd.read_csv(data_hr + '/' + model_file, index_col=0)
    #                     df_hr = pd.concat([df_hr, df_iter_hr], ignore_index=True)
    #                     iter_list.extend([iter_num] * 4)

    df_hr['hr'] = df_hr['hit number'] / df_hr['test set size']
    df_hr['hr'] = df_hr['hr'].astype(float)
    df_hr['iter'] = iter_list

    fig, ax = plt.subplots(figsize=(60, 40))

    df_fam_role = df_hr.groupby(['iter', 'role', 'familiarity']).mean()
    df_fam_role_model = df_fam_role['hr']
    staff_familiar_wayfinders_list = list()
    staff_unfamiliar_wayfinders_list = list()
    student_familiar_wayfinders_list = list()
    student_unfamiliar_wayfinders_list = list()

    for index, row in df_fam_role_model.items():
        if index[1] == 'staff' and index[2] == 'familiar':
            staff_familiar_wayfinders_list.append(index)
        elif index[1] == 'staff' and index[2] == 'unfamiliar':
            staff_unfamiliar_wayfinders_list.append(index)
        elif index[1] == 'student' and index[2] == 'familiar':
            student_familiar_wayfinders_list.append(index)
        else:
            student_unfamiliar_wayfinders_list.append(index)

    df_staff_familiar_wayfinders = df_fam_role_model[staff_familiar_wayfinders_list]
    df_staff_unfamiliar_wayfinders = df_fam_role_model[staff_unfamiliar_wayfinders_list]
    df_student_familiar_wayfinders = df_fam_role_model[student_familiar_wayfinders_list]
    df_student_unfamiliar_wayfinders = df_fam_role_model[student_unfamiliar_wayfinders_list]

    index = np.arange(len(df_staff_familiar_wayfinders))

    staff_familiar_wayfinders_values = np.array(list(df_staff_familiar_wayfinders))
    staff_unfamiliar_wayfinders_values = np.array(list(df_staff_unfamiliar_wayfinders))
    student_familiar_wayfinders_values = np.array(list(df_student_familiar_wayfinders))
    student_unfamiliar_wayfinders_values = np.array(list(df_student_unfamiliar_wayfinders))
    bar_width = 0.2

    rects1 = ax.bar(index, staff_familiar_wayfinders_values, bar_width, alpha=0.8, label='Staff for familiar', )
    rects2 = ax.bar(index + bar_width, staff_unfamiliar_wayfinders_values, bar_width, alpha=0.5,
                    label='Staff for unfamiliar')
    rects3 = ax.bar(index + 2 * bar_width, student_familiar_wayfinders_values, bar_width, alpha=0.8,
                    label='Student for familiar', )
    rects4 = ax.bar(index + 3 * bar_width, student_unfamiliar_wayfinders_values, bar_width, alpha=0.5,
                    label='Student for unfamiliar')

    font_label = {'family': 'Times New Roman',
                  'weight': 'heavy',
                  'size': 80,
                  }
    ax.set_ylabel('Hit rate', font_label)
    ax.set_xlabel('Number of repeated split 5-Fold dataset', font_label)

    font_xstick = {'family': 'Times New Roman',
                   'weight': 'heavy',
                   'size': 60,
                   }
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(['1', '2', '3', '4', '5',
                        '6', '7', '8', '9', '10'], font_xstick)
    # ax.set_yticklabels(font_xstick)

    # y_labels = pd_total['features']
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=60, fontfamily='Times New Roman')
    # plt.xticks(fontsize='20', fontfamily='Times New Roman')

    font_legend = {'family': 'Times New Roman',
                   'weight': 'normal',
                   'size': 50,
                   }
    ax.legend(handles=[rects1, rects2, rects3, rects4], prop=font_legend, loc='upper right', )
    # plt.legend(l)

    for a, b in zip(index, staff_familiar_wayfinders_values):
        plt.text(a, b + 0.01, '%.2f' % b, ha='center', va='center', fontsize=45, fontfamily='Times New Roman')

    for a, b in zip(index, staff_unfamiliar_wayfinders_values):
        plt.text(a + bar_width, b + 0.01, '%.2f' % b, ha='center', va='center', fontsize=45,
                 fontfamily='Times New Roman')

    for a, b in zip(index, student_familiar_wayfinders_values):
        plt.text(a + 2 * bar_width, b + 0.01, '%.2f' % b, ha='center', va='center', fontsize=45,
                 fontfamily='Times New Roman')

    for a, b in zip(index, student_unfamiliar_wayfinders_values):
        plt.text(a + 3 * bar_width, b + 0.01, '%.2f' % b, ha='center', va='center', fontsize=45,
                 fontfamily='Times New Roman')
    fig_dir = '../../results/reports/figures/'
    plt.savefig(fig_dir + "overall_model_hr.pdf", bbox_inches='tight')
    plt.show()

    table_dir = '../../results/reports/tables/'
    file_name = table_dir + 'overall_model_hr.csv'
    df_fam_role_model.to_csv(file_name, encoding='utf-8')


def plot_iteration_role_model_4_groups(role_attribute):
    iters_dir = '../../results/cv_results/role'
    df_hr = pd.DataFrame(columns=['fold', 'role', 'familiarity', 'test set size', 'hit number'])
    iter_list = list()
    dir_list = os.listdir(iters_dir)
    iter_num = 0
    for dir_name in dir_list:
        if 'iter' in dir_name:
            iter_num += 1
            iter_dir = os.path.join(iters_dir, dir_name)
            data_hr = iter_dir + '/experiments/evaluation/ranking_' + role_attribute

            for model_path, model_dir_list, model_file_list in os.walk(data_hr):
                for model_file in model_file_list:
                    if 'evaluation' in model_file:
                        df_iter_hr = pd.read_csv(data_hr + '/' + model_file, index_col=0)
                        df_hr = pd.concat([df_hr, df_iter_hr], ignore_index=True)
                        iter_list.extend([iter_num] * 4)

    df_hr['hr'] = df_hr['hit number'] / df_hr['test set size']
    df_hr['hr'] = df_hr['hr'].astype(float)
    df_hr['iter'] = iter_list

    fig, ax = plt.subplots(figsize=(60, 40))

    df_fam_role = df_hr.groupby(['iter', 'role', 'familiarity']).mean()
    df_fam_role_model = df_fam_role['hr']
    staff_familiar_wayfinders_list = list()
    staff_unfamiliar_wayfinders_list = list()
    student_familiar_wayfinders_list = list()
    student_unfamiliar_wayfinders_list = list()

    for index, row in df_fam_role_model.items():
        if index[1] == 'staff' and index[2] == 'familiar':
            staff_familiar_wayfinders_list.append(index)
        elif index[1] == 'staff' and index[2] == 'unfamiliar':
            staff_unfamiliar_wayfinders_list.append(index)
        elif index[1] == 'student' and index[2] == 'familiar':
            student_familiar_wayfinders_list.append(index)
        else:
            student_unfamiliar_wayfinders_list.append(index)

    df_staff_familiar_wayfinders = df_fam_role_model[staff_familiar_wayfinders_list]
    df_staff_unfamiliar_wayfinders = df_fam_role_model[staff_unfamiliar_wayfinders_list]
    df_student_familiar_wayfinders = df_fam_role_model[student_familiar_wayfinders_list]
    df_student_unfamiliar_wayfinders = df_fam_role_model[student_unfamiliar_wayfinders_list]

    index = np.arange(len(df_staff_familiar_wayfinders))

    staff_familiar_wayfinders_values = np.array(list(df_staff_familiar_wayfinders))
    staff_unfamiliar_wayfinders_values = np.array(list(df_staff_unfamiliar_wayfinders))
    student_familiar_wayfinders_values = np.array(list(df_student_familiar_wayfinders))
    student_unfamiliar_wayfinders_values = np.array(list(df_student_unfamiliar_wayfinders))
    bar_width = 0.2

    rects1 = ax.bar(index, staff_familiar_wayfinders_values, bar_width, alpha=0.8, label='Staff for familiar', )
    rects2 = ax.bar(index + bar_width, staff_unfamiliar_wayfinders_values, bar_width, alpha=0.5,
                    label='Staff for unfamiliar')
    rects3 = ax.bar(index + 2 * bar_width, student_familiar_wayfinders_values, bar_width, alpha=0.8,
                    label='Student for familiar', )
    rects4 = ax.bar(index + 3 * bar_width, student_unfamiliar_wayfinders_values, bar_width, alpha=0.5,
                    label='Student for unfamiliar')

    font_label = {'family': 'Times New Roman',
                  'weight': 'heavy',
                  'size': 80,
                  }
    ax.set_ylabel('Hit rate', font_label)
    ax.set_xlabel('Number of repeated split 5-Fold dataset', font_label)

    font_xstick = {'family': 'Times New Roman',
                   'weight': 'heavy',
                   'size': 60,
                   }
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(['1', '2', '3', '4', '5',
                        '6', '7', '8', '9', '10'], font_xstick)
    # ax.set_yticklabels(font_xstick)

    # y_labels = pd_total['features']
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=60, fontfamily='Times New Roman')
    # plt.xticks(fontsize='20', fontfamily='Times New Roman')

    font_legend = {'family': 'Times New Roman',
                   'weight': 'normal',
                   'size': 50,
                   }
    ax.legend(handles=[rects1, rects2, rects3, rects4], prop=font_legend, loc='upper right', )
    # plt.legend(l)

    for a, b in zip(index, staff_familiar_wayfinders_values):
        plt.text(a, b + 0.01, '%.2f' % b, ha='center', va='center', fontsize=45, fontfamily='Times New Roman')

    for a, b in zip(index, staff_unfamiliar_wayfinders_values):
        plt.text(a + bar_width, b + 0.01, '%.2f' % b, ha='center', va='center', fontsize=45,
                 fontfamily='Times New Roman')

    for a, b in zip(index, student_familiar_wayfinders_values):
        plt.text(a + 2 * bar_width, b + 0.01, '%.2f' % b, ha='center', va='center', fontsize=45,
                 fontfamily='Times New Roman')

    for a, b in zip(index, student_unfamiliar_wayfinders_values):
        plt.text(a + 3 * bar_width, b + 0.01, '%.2f' % b, ha='center', va='center', fontsize=45,
                 fontfamily='Times New Roman')
    fig_dir = '../../results/reports/figures/'
    plt.savefig(fig_dir + role_attribute + "_model_hr.pdf", bbox_inches='tight')
    plt.show()

    # table_dir = '../../results/reports/tables/'
    # file_name = table_dir + role_attribute + '_model_hr.csv'
    # df_fam_role_model.to_csv(file_name, encoding='utf-8')

def plot_iteartion_familiarity_model_4_groups(familiarity_attribute):
    iters_dir = '../../results/cv_results/familiarity'
    df_hr = pd.DataFrame(columns=['fold', 'role', 'familiarity', 'test set size', 'hit number'])
    iter_list = list()
    dir_list = os.listdir(iters_dir)
    iter_num = 0
    for dir_name in dir_list:
        if 'iter' in dir_name:
            iter_num += 1
            iter_dir = os.path.join(iters_dir, dir_name)
            data_hr = iter_dir + '/experiments/evaluation/ranking_' + familiarity_attribute

            for model_path, model_dir_list, model_file_list in os.walk(data_hr):
                for model_file in model_file_list:
                    if 'evaluation' in model_file:
                        df_iter_hr = pd.read_csv(data_hr + '/' + model_file, index_col=0)
                        df_hr = pd.concat([df_hr, df_iter_hr], ignore_index=True)
                        iter_list.extend([iter_num] * 4)

    df_hr['hr'] = df_hr['hit number'] / df_hr['test set size']
    df_hr['hr'] = df_hr['hr'].astype(float)
    df_hr['iter'] = iter_list

    fig, ax = plt.subplots(figsize=(60, 40))

    df_fam_role = df_hr.groupby(['iter', 'role', 'familiarity']).mean()
    df_fam_role_model = df_fam_role['hr']
    staff_familiar_wayfinders_list = list()
    staff_unfamiliar_wayfinders_list = list()
    student_familiar_wayfinders_list = list()
    student_unfamiliar_wayfinders_list = list()

    for index, row in df_fam_role_model.items():
        if index[1] == 'staff' and index[2] == 'familiar':
            staff_familiar_wayfinders_list.append(index)
        elif index[1] == 'staff' and index[2] == 'unfamiliar':
            staff_unfamiliar_wayfinders_list.append(index)
        elif index[1] == 'student' and index[2] == 'familiar':
            student_familiar_wayfinders_list.append(index)
        else:
            student_unfamiliar_wayfinders_list.append(index)

    df_staff_familiar_wayfinders = df_fam_role_model[staff_familiar_wayfinders_list]
    df_staff_unfamiliar_wayfinders = df_fam_role_model[staff_unfamiliar_wayfinders_list]
    df_student_familiar_wayfinders = df_fam_role_model[student_familiar_wayfinders_list]
    df_student_unfamiliar_wayfinders = df_fam_role_model[student_unfamiliar_wayfinders_list]

    index = np.arange(len(df_staff_familiar_wayfinders))

    staff_familiar_wayfinders_values = np.array(list(df_staff_familiar_wayfinders))
    staff_unfamiliar_wayfinders_values = np.array(list(df_staff_unfamiliar_wayfinders))
    student_familiar_wayfinders_values = np.array(list(df_student_familiar_wayfinders))
    student_unfamiliar_wayfinders_values = np.array(list(df_student_unfamiliar_wayfinders))
    bar_width = 0.2

    rects1 = ax.bar(index, staff_familiar_wayfinders_values, bar_width, alpha=0.8, label='Staff for familiar', )
    rects2 = ax.bar(index + bar_width, staff_unfamiliar_wayfinders_values, bar_width, alpha=0.5,
                    label='Staff for unfamiliar')
    rects3 = ax.bar(index + 2 * bar_width, student_familiar_wayfinders_values, bar_width, alpha=0.8,
                    label='Student for familiar', )
    rects4 = ax.bar(index + 3 * bar_width, student_unfamiliar_wayfinders_values, bar_width, alpha=0.5,
                    label='Student for unfamiliar')

    font_label = {'family': 'Times New Roman',
                  'weight': 'heavy',
                  'size': 80,
                  }
    ax.set_ylabel('Hit rate', font_label)
    ax.set_xlabel('Number of repeated split 5-Fold dataset', font_label)

    font_xstick = {'family': 'Times New Roman',
                   'weight': 'heavy',
                   'size': 60,
                   }
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(['1', '2', '3', '4', '5',
                        '6', '7', '8', '9', '10'], font_xstick)
    # ax.set_yticklabels(font_xstick)

    # y_labels = pd_total['features']
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=60, fontfamily='Times New Roman')
    # plt.xticks(fontsize='20', fontfamily='Times New Roman')

    font_legend = {'family': 'Times New Roman',
                   'weight': 'normal',
                   'size': 50,
                   }
    ax.legend(handles=[rects1, rects2, rects3, rects4], prop=font_legend, loc='upper right', )
    # plt.legend(l)

    for a, b in zip(index, staff_familiar_wayfinders_values):
        plt.text(a, b + 0.01, '%.2f' % b, ha='center', va='center', fontsize=45, fontfamily='Times New Roman')

    for a, b in zip(index, staff_unfamiliar_wayfinders_values):
        plt.text(a + bar_width, b + 0.01, '%.2f' % b, ha='center', va='center', fontsize=45,
                 fontfamily='Times New Roman')

    for a, b in zip(index, student_familiar_wayfinders_values):
        plt.text(a + 2 * bar_width, b + 0.01, '%.2f' % b, ha='center', va='center', fontsize=45,
                 fontfamily='Times New Roman')

    for a, b in zip(index, student_unfamiliar_wayfinders_values):
        plt.text(a + 3 * bar_width, b + 0.01, '%.2f' % b, ha='center', va='center', fontsize=45,
                 fontfamily='Times New Roman')

    fig_dir = '../../results/reports/figures/'
    plt.savefig(fig_dir + familiarity_attribute + "_model_hr.pdf", bbox_inches='tight')
    plt.show()

    # table_dir = '../../results/reports/tables/'
    # file_name = table_dir + familiarity_attribute + '_model_hr.csv'
    # df_fam_role_model.to_csv(file_name, encoding='utf-8')

def plot_iteration_fam_role_model(familiarity_attribute, role_attribute):
    iters_dir = '../../results/cv_results/fam_role'
    df_hr = pd.DataFrame(columns=['fold', 'role', 'familiarity', 'test set size', 'hit number'])
    iter_list = list()
    dir_list = os.listdir(iters_dir)
    iter_num = 0
    for dir_name in dir_list:
        if 'iter' in dir_name:
            iter_num += 1
            iter_dir = os.path.join(iters_dir, dir_name)
            data_hr = iter_dir + '/experiments/evaluation/ranking_' + role_attribute + '_' + familiarity_attribute

            for model_path, model_dir_list, model_file_list in os.walk(data_hr):
                for model_file in model_file_list:
                    if 'evaluation' in model_file:
                        df_iter_hr = pd.read_csv(data_hr + '/' + model_file, index_col=0)
                        df_hr = pd.concat([df_hr, df_iter_hr], ignore_index=True)
                        iter_list.extend([iter_num] * 4)

    df_hr['hr'] = df_hr['hit number'] / df_hr['test set size']
    df_hr['hr'] = df_hr['hr'].astype(float)
    df_hr['iter'] = iter_list

    fig, ax = plt.subplots(figsize=(60, 40))

    df_fam_role = df_hr.groupby(['iter', 'role' , 'familiarity']).mean()
    df_fam_role_model = df_fam_role['hr']
    staff_familiar_wayfinders_list = list()
    staff_unfamiliar_wayfinders_list = list()
    student_familiar_wayfinders_list = list()
    student_unfamiliar_wayfinders_list = list()

    for index, row in df_fam_role_model.items():
        if index[1] == 'staff' and index[2] == 'familiar':
            staff_familiar_wayfinders_list.append(index)
        elif index[1] == 'staff' and index[2] == 'unfamiliar':
            staff_unfamiliar_wayfinders_list.append(index)
        elif index[1] == 'student' and index[2] == 'familiar':
            student_familiar_wayfinders_list.append(index)
        else:
            student_unfamiliar_wayfinders_list.append(index)

    df_staff_familiar_wayfinders = df_fam_role_model[staff_familiar_wayfinders_list]
    df_staff_unfamiliar_wayfinders = df_fam_role_model[staff_unfamiliar_wayfinders_list]
    df_student_familiar_wayfinders = df_fam_role_model[student_familiar_wayfinders_list]
    df_student_unfamiliar_wayfinders = df_fam_role_model[student_unfamiliar_wayfinders_list]

    index = np.arange(len(df_staff_familiar_wayfinders))

    staff_familiar_wayfinders_values = np.array(list(df_staff_familiar_wayfinders))
    staff_unfamiliar_wayfinders_values = np.array(list(df_staff_unfamiliar_wayfinders))
    student_familiar_wayfinders_values = np.array(list(df_student_familiar_wayfinders))
    student_unfamiliar_wayfinders_values = np.array(list(df_student_unfamiliar_wayfinders))
    bar_width = 0.2

    rects1 = ax.bar(index, staff_familiar_wayfinders_values, bar_width, alpha=0.8, label='Staff for familiar', )
    rects2 = ax.bar(index + bar_width, staff_unfamiliar_wayfinders_values, bar_width, alpha=0.5,
                    label='Staff for unfamiliar')
    rects3 = ax.bar(index + 2*bar_width, student_familiar_wayfinders_values, bar_width, alpha=0.8, label='Student for familiar', )
    rects4 = ax.bar(index + 3*bar_width, student_unfamiliar_wayfinders_values, bar_width, alpha=0.5,
                    label='Student for unfamiliar')

    font_label = {'family': 'Times New Roman',
                  'weight': 'heavy',
                  'size': 80,
                  }
    ax.set_ylabel('Hit rate', font_label)
    ax.set_xlabel('Number of repeated split 5-Fold dataset', font_label)

    font_xstick = {'family': 'Times New Roman',
                   'weight': 'heavy',
                   'size': 60,
                   }
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(['1', '2', '3', '4', '5',
                        '6', '7', '8', '9', '10'], font_xstick)
    # ax.set_yticklabels(font_xstick)

    # y_labels = pd_total['features']
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=60, fontfamily='Times New Roman')
    # plt.xticks(fontsize='20', fontfamily='Times New Roman')

    font_legend = {'family': 'Times New Roman',
                   'weight': 'normal',
                   'size': 50,
                   }
    ax.legend(handles=[rects1, rects2, rects3, rects4], prop=font_legend, loc='upper right', )
    # plt.legend(l)

    for a, b in zip(index, staff_familiar_wayfinders_values):
        plt.text(a, b + 0.01, '%.2f' % b, ha='center', va='center', fontsize=45, fontfamily='Times New Roman')

    for a, b in zip(index, staff_unfamiliar_wayfinders_values):
        plt.text(a + bar_width, b + 0.01, '%.2f' % b, ha='center', va='center', fontsize=45,
                 fontfamily='Times New Roman')

    for a, b in zip(index, student_familiar_wayfinders_values):
        plt.text(a + 2*bar_width, b + 0.01, '%.2f' % b, ha='center', va='center', fontsize=45,
                 fontfamily='Times New Roman')

    for a, b in zip(index, student_unfamiliar_wayfinders_values):
        plt.text(a + 3*bar_width, b + 0.01, '%.2f' % b, ha='center', va='center', fontsize=45,
                 fontfamily='Times New Roman')

    fig_dir = '../../results/reports/figures/'
    plt.savefig(fig_dir + role_attribute + '_' + familiarity_attribute + "_model_hr.pdf", bbox_inches='tight')
    plt.show()

    table_dir = '../../results/reports/tables/'
    file_name = table_dir + role_attribute + '_' + familiarity_attribute + '_model_hr.csv'
    df_fam_role_model.to_csv(file_name, encoding='utf-8')


def plot_iteration_role_model(attribute):
    iters_dir = '../../results/cv_results/role'

    df_hr = pd.DataFrame(columns=['fold', 'role', 'familiarity', 'test set size', 'hit number'])
    iter_list = list()
    dir_list = os.listdir(iters_dir)
    iter_num = 0
    for dir_name in dir_list:
        if 'iter' in dir_name:
            iter_num += 1
            iter_dir = os.path.join(iters_dir, dir_name)
            data_hr = iter_dir + '/experiments/evaluation/ranking_' + attribute

            for model_path, model_dir_list, model_file_list in os.walk(data_hr):
                for model_file in model_file_list:
                    if 'evaluation' in model_file:
                        df_iter_hr = pd.read_csv(data_hr + '/' + model_file, index_col=0)
                        df_hr = pd.concat([df_hr, df_iter_hr], ignore_index=True)
                        iter_list.extend([iter_num] * 4)

    df_hr['hr'] = df_hr['hit number'] / df_hr['test set size']
    df_hr['hr'] = df_hr['hr'].astype(float)
    df_hr['iter'] = iter_list

    fig, ax = plt.subplots(figsize=(15, 10))

    df_role = df_hr.groupby(['iter', 'role']).mean()
    df_role_model = df_role['hr']
    staff_wayfinders_list = list()
    student_wayfinders_list = list()

    for index, row in df_role_model.items():
        if index[1] == 'staff':
            staff_wayfinders_list.append(index)
        else:
            student_wayfinders_list.append(index)
    df_staff_wayfinders = df_role_model[staff_wayfinders_list]
    df_student_wayfinders = df_role_model[student_wayfinders_list]

    index = np.arange(len(df_staff_wayfinders))

    staff_wayfinders_values = np.array(list(df_staff_wayfinders))
    student_wayfinders_values = np.array(list(df_student_wayfinders))
    bar_width = 0.35

    rects1 = ax.bar(index, staff_wayfinders_values, bar_width, alpha=0.8, label='Staff', )
    rects2 = ax.bar(index + bar_width, student_wayfinders_values, bar_width, alpha=0.5,
                    label='Student')

    font_label = {'family': 'Times New Roman',
                  'weight': 'heavy',
                  'size': 18,
                  }
    ax.set_ylabel('Hit rate', font_label)
    ax.set_xlabel('Number of repeated split 5-Fold dataset', font_label)

    font_xstick = {'family': 'Times New Roman',
                   'weight': 'heavy',
                   'size': 16,
                   }
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(['1', '2', '3', '4', '5',
                        '6', '7', '8', '9', '10'], font_xstick)
    # ax.set_yticklabels(font_xstick)

    # y_labels = pd_total['features']
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=16, fontfamily='Times New Roman')
    # plt.xticks(fontsize='20', fontfamily='Times New Roman')

    font_legend = {'family': 'Times New Roman',
                   'weight': 'normal',
                   'size': 14,
                   }
    ax.legend(handles=[rects1, rects2], prop=font_legend, loc='upper right', )
    # plt.legend(l)

    for a, b in zip(index, staff_wayfinders_values):
        plt.text(a, b + 0.02, '%.2f' % b, ha='center', va='center', fontsize=14, fontfamily='Times New Roman')

    for a, b in zip(index, student_wayfinders_values):
        plt.text(a + bar_width, b + 0.02, '%.2f' % b, ha='center', va='center', fontsize=14,
                 fontfamily='Times New Roman')

    fig_dir = '../../results/reports/figures/'
    plt.savefig(fig_dir + attribute + "_model_hr.pdf", bbox_inches='tight')
    plt.show()

    table_dir = '../../results/reports/tables/'
    file_name = table_dir + attribute + '_model_hr.csv'
    df_role_model.to_csv(file_name, encoding='utf-8')

def plot_familiarity_model_by_mae(strategy, attribute):
    familiar_model_mae_dir = '../../results/cv_results/familiarity/evaluation/ranking_familiar/'
    unfamiliar_model_mae_dir = '../../results/cv_results/familiarity/evaluation/ranking_unfamiliar/'

    familiar_model_fold_list = list()
    unfamiliar_model_fold_list = list()

    df_familiar_model_mae = pd.DataFrame(columns=['fold', 'role', 'familiarity', 'test set size', 'hit number', 'HR', 'MAE'])
    df_unfamiliar_model_mae = pd.DataFrame(columns=['fold', 'role', 'familiarity', 'test set size', 'hit number', 'HR', 'MAE'])

    for fold_num in range(1, 18):
        df_iter_familiar_mae = pd.read_csv(familiar_model_mae_dir + 'fold' + str(fold_num) + '_model_evaluation.csv',
                                          index_col=0)
        df_familiar_model_mae = pd.concat([df_familiar_model_mae, df_iter_familiar_mae], ignore_index=True)
        familiar_model_fold_list.extend([fold_num] * df_iter_familiar_mae.shape[0])

        df_iter_unfamiliar_mae = pd.read_csv(unfamiliar_model_mae_dir + 'fold' + str(fold_num) + '_model_evaluation.csv',
                                            index_col=0)
        df_unfamiliar_model_mae = pd.concat([df_unfamiliar_model_mae, df_iter_unfamiliar_mae], ignore_index=True)
        unfamiliar_model_fold_list.extend([fold_num] * df_iter_unfamiliar_mae.shape[0])

    # df_familiar_model_hr['hr'] = df_familiar_model_hr['hit number'] / df_familiar_model_hr['test set size']
    df_familiar_model_mae['MAE'] = df_familiar_model_mae['MAE'].astype(float)
    df_familiar_model_mae['iter'] = familiar_model_fold_list

    # df_unfamiliar_model_hr['hr'] = df_unfamiliar_model_hr['hit number'] / df_unfamiliar_model_hr['test set size']
    df_unfamiliar_model_mae['MAE'] = df_unfamiliar_model_mae['MAE'].astype(float)
    df_unfamiliar_model_mae['fold'] = unfamiliar_model_fold_list

    # plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(20, 10))

    df_familiar_model_group = df_familiar_model_mae.groupby(['fold', 'familiarity']).mean()
    df_familiar_model = df_familiar_model_group['MAE']
    familiar_model_index_list = list()
    for index, row in df_familiar_model.items():
        if index[1] == attribute:
            familiar_model_index_list.append(index)
    df_familiar_model = df_familiar_model[familiar_model_index_list]

    df_unfamiliar_model_group = df_unfamiliar_model_mae.groupby(['fold', 'familiarity']).mean()
    df_unfamiliar_model = df_unfamiliar_model_group['MAE']
    unfamiliar_model_index_list = list()
    for index, row in df_unfamiliar_model.items():
        if index[1] == attribute:
            unfamiliar_model_index_list.append(index)
    df_unfamiliar_model = df_unfamiliar_model[unfamiliar_model_index_list]

    index = np.arange(len(df_familiar_model))

    familiar_model_values = np.array(list(df_familiar_model))
    unfamiliar_model_values = np.array(list(df_unfamiliar_model))
    bar_width = 0.35

    rects1 = ax.bar(index, familiar_model_values, bar_width, alpha=0.8, label='Familiar model', )
    rects2 = ax.bar(index + bar_width, unfamiliar_model_values, bar_width, alpha=0.6, label='Unfamiliar model')

    font_label = {'family': 'Times New Roman',
                  'weight': 'heavy',
                  'size': 18,
                  }

    ax.set_ylabel('MAE', font_label)
    ax.set_xlabel('Group', font_label)

    font_xstick = {'family': 'Times New Roman',
                   'weight': 'heavy',
                   'size': 16,
                   }
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(['1', '2', '3', '4', '5',
                        '6', '7', '8', '9', '10',
                        '11', '12', '13', '14', '15',
                        '16', '17', ], font_xstick)
    # ax.set_yticklabels(font_xstick)

    # y_labe
    # ls = pd_total['features']
    y_max = max([max(familiar_model_values), max(unfamiliar_model_values)])
    plt.yticks(np.arange(0, round(y_max, 1), 0.2), fontsize=16, fontfamily='Times New Roman')
    # plt.xticks(fontsize='20', fontfamily='Times New Roman')

    font_legend = {'family': 'Times New Roman',
                   'weight': 'normal',
                   'size': 14,
                   }
    ax.legend(handles=[rects1, rects2], prop=font_legend, loc='upper right', )
    # plt.legend(l)

    for a, b in zip(index, familiar_model_values):
        plt.text(a, b + 0.02, '%.2f' % b, ha='center', va='center', fontsize=14, fontfamily='Times New Roman')

    for a, b in zip(index, unfamiliar_model_values):
        plt.text(a + bar_width, b + 0.02, '%.2f' % b, ha='center', va='center', fontsize=14,
                 fontfamily='Times New Roman')

    fig_dir = '../../results/reports/figures/'
    plt.savefig(fig_dir + strategy + '/' + attribute + "_wayfinders_mae.pdf", bbox_inches='tight')
    plt.show()

    table_dir = '../../results/reports/tables/'
    file_name = table_dir + strategy + '/' + attribute + '_wayfinders_mae.csv'
    df_attribute_wayfinders = pd.concat([df_familiar_model, df_unfamiliar_model])
    df_attribute_wayfinders.to_csv(file_name, encoding='utf-8')

def plot_familiarity_model_by_hr(strategy, attribute):

    # iters_dir = '../../results/dataset_split_iteration/familiarity'
    familiar_model_hr_dir = '../../results/cv_results/familiarity/evaluation/ranking_familiar/'
    unfamiliar_model_hr_dir = '../../results/cv_results/familiarity/evaluation/ranking_unfamiliar/'

    familiar_model_fold_list = list()
    unfamiliar_model_fold_list = list()

    df_familiar_model_hr = pd.DataFrame(columns=['fold', 'role', 'familiarity', 'test set size', 'hit number', 'HR', 'MAE'])
    df_unfamiliar_model_hr = pd.DataFrame(columns=['fold', 'role','familiarity','test set size','hit number', 'HR', 'MAE'])

    for fold_num in range(1, 18):
        df_iter_familiar_hr = pd.read_csv(familiar_model_hr_dir + 'fold' + str(fold_num) + '_model_evaluation.csv', index_col=0)
        df_familiar_model_hr = pd.concat([df_familiar_model_hr, df_iter_familiar_hr], ignore_index=True)
        familiar_model_fold_list.extend([fold_num] * df_iter_familiar_hr.shape[0])

        df_iter_unfamiliar_hr = pd.read_csv(unfamiliar_model_hr_dir + 'fold' + str(fold_num) + '_model_evaluation.csv', index_col=0)
        df_unfamiliar_model_hr = pd.concat([df_unfamiliar_model_hr, df_iter_unfamiliar_hr], ignore_index=True)
        unfamiliar_model_fold_list.extend([fold_num] * df_iter_familiar_hr.shape[0])

    df_familiar_model_hr['hr'] = df_familiar_model_hr['hit number']/df_familiar_model_hr['test set size']
    df_familiar_model_hr['hr'] = df_familiar_model_hr['hr'].astype(float)
    df_familiar_model_hr['fold'] = familiar_model_fold_list

    df_unfamiliar_model_hr['hr'] = df_unfamiliar_model_hr['hit number']/df_unfamiliar_model_hr['test set size']
    df_unfamiliar_model_hr['hr'] = df_unfamiliar_model_hr['hr'].astype(float)
    df_unfamiliar_model_hr['fold'] = unfamiliar_model_fold_list

    # plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(20, 10))

    df_familiar_model_group = df_familiar_model_hr.groupby(['fold', 'familiarity']).mean()
    df_familiar_model = df_familiar_model_group['hr']
    familiar_model_index_list = list()
    for index, row in df_familiar_model.items():
        if index[1] == attribute:
            familiar_model_index_list.append(index)
    df_familiar_model = df_familiar_model[familiar_model_index_list]

    df_unfamiliar_model_group = df_unfamiliar_model_hr.groupby(['fold', 'familiarity']).mean()
    df_unfamiliar_model = df_unfamiliar_model_group['hr']
    unfamiliar_model_index_list = list()
    for index, row in df_unfamiliar_model.items():
        if index[1] == attribute:
            unfamiliar_model_index_list.append(index)
    df_unfamiliar_model = df_unfamiliar_model[unfamiliar_model_index_list]

    index = np.arange(len(df_familiar_model))

    familiar_model_values = np.array(list(df_familiar_model))
    unfamiliar_model_values = np.array(list(df_unfamiliar_model))
    bar_width = 0.35

    rects1 = ax.bar(index, familiar_model_values, bar_width, alpha=0.8, label='Familiar model', )
    rects2 = ax.bar(index + bar_width, unfamiliar_model_values, bar_width, alpha=0.6, label='Familiar model')

    font_label = {'family': 'Times New Roman',
                  'weight': 'heavy',
                  'size': 18,
                  }

    ax.set_ylabel('HR', font_label)
    ax.set_xlabel('Group', font_label)

    font_xstick = {'family': 'Times New Roman',
                   'weight': 'heavy',
                   'size': 16,
                   }
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(['1', '2', '3', '4', '5',
                        '6', '7', '8', '9', '10',
                        '11', '12', '13', '14', '15',
                        '16', '17' ], font_xstick)
    # ax.set_yticklabels(font_xstick)

    # y_labels = pd_total['features']
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=16, fontfamily='Times New Roman')
    # plt.xticks(fontsize='20', fontfamily='Times New Roman')

    font_legend = {'family': 'Times New Roman',
                   'weight': 'normal',
                   'size': 14,
                   }
    ax.legend(handles=[rects1, rects2], prop=font_legend, loc='upper right', )
    # plt.legend(l)

    for a, b in zip(index, familiar_model_values):
        plt.text(a, b + 0.02, '%.2f' % b, ha='center', va='center', fontsize=14, fontfamily='Times New Roman')

    for a, b in zip(index, unfamiliar_model_values):
        plt.text(a + bar_width, b + 0.02, '%.2f' % b, ha='center', va='center', fontsize=14,
                 fontfamily='Times New Roman')

    fig_dir = '../../results/reports/figures/'
    plt.savefig(fig_dir + strategy + '/' + attribute + "_wayfinders_hr.pdf", bbox_inches='tight')
    plt.show()

    table_dir = '../../results/reports/tables/'
    file_name = table_dir + strategy + '/' + attribute + '_wayfinders_hr.csv'
    df_attribute_wayfinders = pd.concat([df_familiar_model, df_unfamiliar_model])
    df_attribute_wayfinders.to_csv(file_name, encoding='utf-8')

def plot_familiar_model():
    ds_dir = '../../results/experiments/evaluation/ranking_familiar'

    df_hr = pd.DataFrame(columns=['fold', 'role', 'familiarity', 'test set size', 'hit number'])
    iter_list = list()
    iter_num = 0

    for model_path, model_dir_list, model_file_list in os.walk(ds_dir):
        for model_file in model_file_list:
            if 'evaluation' in model_file:
                df_iter_hr = pd.read_csv(ds_dir + '/' + model_file, index_col=0)
                df_hr = pd.concat([df_hr, df_iter_hr], ignore_index=True)
                iter_list.extend([iter_num] * 4)

    df_hr['hr'] = df_hr['hit number'] / df_hr['test set size']
    df_hr['hr'] = df_hr['hr'].astype(float)
    # df_hr['iter'] = iter_list

    fig, ax = plt.subplots(figsize=(15, 10))

    df_familiarity = df_hr.groupby(['familiarity']).mean()
    df_familiar_model = df_familiarity['hr']
    familiar_wayfinders_list = list()
    unfamiliar_wayfinders_list = list()

    for index, row in df_familiar_model.items():
        if index == 'familiar':
            familiar_wayfinders_list.append(index)
        else:
            unfamiliar_wayfinders_list.append(index)
    df_familiar_wayfinders = df_familiar_model[familiar_wayfinders_list]
    df_unfamiliar_wayfinders = df_familiar_model[unfamiliar_wayfinders_list]

    index = np.arange(len(df_familiar_wayfinders))

    familiar_wayfinders_values = np.array(list(df_familiar_wayfinders))
    unfamiliar_wayfinders_values = np.array(list(df_unfamiliar_wayfinders))
    bar_width = 0.1

    rects1 = ax.bar(index, familiar_wayfinders_values, bar_width, alpha=0.8, label='For familiar wayfinders', )
    rects2 = ax.bar(index + bar_width, unfamiliar_wayfinders_values, bar_width, alpha=0.5,
                    label='For unfamiliar wayfinders')

    font_label = {'family': 'Times New Roman',
                  'weight': 'heavy',
                  'size': 18,
                  }
    ax.set_ylabel('Hit rate', font_label)
    ax.set_xlabel('Number of repeated split 5-Fold dataset', font_label)

    font_xstick = {'family': 'Times New Roman',
                   'weight': 'heavy',
                   'size': 16,
                   }

    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(['1', '2', '3', '4', '5',
                        '6', '7', '8', '9', '10'], font_xstick)

    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=16, fontfamily='Times New Roman')

    font_legend = {'family': 'Times New Roman',
                   'weight': 'normal',
                   'size': 14,
                   }
    ax.legend(handles=[rects1, rects2], prop=font_legend, loc='upper right', )
    # plt.legend(l)

    for a, b in zip(index, familiar_wayfinders_values):
        plt.text(a, b + 0.02, '%.2f' % b, ha='center', va='center', fontsize=14, fontfamily='Times New Roman')

    for a, b in zip(index, unfamiliar_wayfinders_values):
        plt.text(a + bar_width, b + 0.02, '%.2f' % b, ha='center', va='center', fontsize=14,
                 fontfamily='Times New Roman')

    fig_dir = '../../results/reports/figures/'
    # plt.savefig(fig_dir + "familiar_model_hr.pdf", bbox_inches='tight')
    plt.show()
    #
    # table_dir = '../../results/reports/tables/'
    # file_name = table_dir + 'familiar_model_hr.csv'
    # df_familiar_model.to_csv(file_name, encoding='utf-8')

def plot_unfamiliar_model():
    ds_dir = '../../results/experiments/evaluation/ranking_unfamiliar'

    df_hr = pd.DataFrame(columns=['fold', 'role', 'familiarity', 'test set size', 'hit number'])
    iter_list = list()
    iter_num = 0

    for model_path, model_dir_list, model_file_list in os.walk(ds_dir):
        for model_file in model_file_list:
            if 'evaluation' in model_file:
                df_iter_hr = pd.read_csv(ds_dir + '/' + model_file, index_col=0)
                df_hr = pd.concat([df_hr, df_iter_hr], ignore_index=True)
                iter_list.extend([iter_num] * 4)

    df_hr['hr'] = df_hr['hit number'] / df_hr['test set size']
    df_hr['hr'] = df_hr['hr'].astype(float)
    # df_hr['iter'] = iter_list

    fig, ax = plt.subplots(figsize=(15, 10))

    df_familiarity = df_hr.groupby(['familiarity']).mean()
    df_familiar_model = df_familiarity['hr']
    familiar_wayfinders_list = list()
    unfamiliar_wayfinders_list = list()

    for index, row in df_familiar_model.items():
        if index == 'familiar':
            familiar_wayfinders_list.append(index)
        else:
            unfamiliar_wayfinders_list.append(index)
    df_familiar_wayfinders = df_familiar_model[familiar_wayfinders_list]
    df_unfamiliar_wayfinders = df_familiar_model[unfamiliar_wayfinders_list]

    index = np.arange(len(df_familiar_wayfinders))

    familiar_wayfinders_values = np.array(list(df_familiar_wayfinders))
    unfamiliar_wayfinders_values = np.array(list(df_unfamiliar_wayfinders))
    bar_width = 0.1

    rects1 = ax.bar(index, familiar_wayfinders_values, bar_width, alpha=0.8, label='For familiar wayfinders', )
    rects2 = ax.bar(index + bar_width, unfamiliar_wayfinders_values, bar_width, alpha=0.5,
                    label='For unfamiliar wayfinders')

    font_label = {'family': 'Times New Roman',
                  'weight': 'heavy',
                  'size': 18,
                  }
    ax.set_ylabel('Hit rate', font_label)
    ax.set_xlabel('Number of repeated split 5-Fold dataset', font_label)

    font_xstick = {'family': 'Times New Roman',
                   'weight': 'heavy',
                   'size': 16,
                   }
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(['1', '2', '3', '4', '5',
                        '6', '7', '8', '9', '10'], font_xstick)

    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=16, fontfamily='Times New Roman')

    font_legend = {'family': 'Times New Roman',
                   'weight': 'normal',
                   'size': 14,
                   }
    ax.legend(handles=[rects1, rects2], prop=font_legend, loc='upper right', )
    # plt.legend(l)

    for a, b in zip(index, familiar_wayfinders_values):
        plt.text(a, b + 0.02, '%.2f' % b, ha='center', va='center', fontsize=14, fontfamily='Times New Roman')

    for a, b in zip(index, unfamiliar_wayfinders_values):
        plt.text(a + bar_width, b + 0.02, '%.2f' % b, ha='center', va='center', fontsize=14,
                 fontfamily='Times New Roman')

    fig_dir = '../../results/reports/figures/'
    # plt.savefig(fig_dir + "familiar_model_hr.pdf", bbox_inches='tight')
    plt.show()
    #
    # table_dir = '../../results/reports/tables/'
    # file_name = table_dir + 'familiar_model_hr.csv'
    # df_familiar_model.to_csv(file_name, encoding='utf-8')


for familiarity in ['familiar', 'unfamiliar']:
    plot_familiarity_model_by_mae(strategy='familiarity', attribute=familiarity)
    plot_familiarity_model_by_hr(strategy='familiarity', attribute=familiarity)



# for role in ['staff', 'student']:
#     plot_iteration_role_model_4_groups(role)
    # plot_iteration_role_model(attribute=role)

# plot_iteration_overall_model_4_groups()

# plot_iteration_familiarity_model(strategy='overall', attribute='overall')

# for familiarity in ['familiar', 'unfamiliar']:
#     for role in ['staff', 'student']:
#         print(role + ' for ' + familiarity)
#         plot_iteration_fam_role_model(familiarity, role)



