import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_model_hr_box(familiarity):
    evaluation_dir = '../../results/cv_results/familiarity/evaluation/'

    df_total = pd.DataFrame(columns=['fold','role','familiarity','test set size','hit number', 'HR', 'model type'])

    df_lambda_model = pd.DataFrame(columns=['fold','role','familiarity','test set size','hit number','HR'])
    df_lc_model = pd.DataFrame(columns=['fold','role','familiarity','test set size','hit number','HR'])
    df_gp_model = pd.DataFrame(columns=['fold','role','familiarity','test set size','hit number','HR'])
    df_rf_model = pd.DataFrame(columns=['fold','role','familiarity','test set size','hit number','HR'])
    df_svm_model = pd.DataFrame(columns=['fold','role','familiarity','test set size','hit number','HR'])

    # dir_list = os.listdir(iters_dir)
    # for dir_name in dir_list:
    #     # print(dir_name)
    #     if 'iter' in dir_name:
    #         iter_dir = os.path.join(iters_dir, dir_name)

    data_lambda = evaluation_dir + 'ranking_' + familiarity
    for iter in range(1, 18):
        df_iter_hr = pd.read_csv(data_lambda + '/' + 'fold{}'.format(iter) + '_model_evaluation.csv', index_col=0)
        df_lambda_model = pd.concat([df_lambda_model, df_iter_hr], ignore_index=True)

    data_lc = evaluation_dir + 'linear_combination'
    for iter in range(1, 18):
        df_iter_hr = pd.read_csv(data_lc + '/' + 'fold{}'.format(iter) + '_model_evaluation.csv', index_col=0)
        df_lc_model = pd.concat([df_lc_model, df_iter_hr], ignore_index=True)

    data_gp = evaluation_dir + 'gp_abs_reg_' + familiarity
    for eval_path, eval_dir_list, eval_file_list in os.walk(data_gp):
        for eval_file in eval_file_list:
            if 'evaluation' in eval_file:
                df_iter_hr = pd.read_csv(data_gp + '/' + eval_file, index_col=0)
                df_gp_model = pd.concat([df_gp_model, df_iter_hr], ignore_index=True)

    data_rf = evaluation_dir + 'rfc_rel_clf_' + familiarity
    for iter in range(1, 18):
        df_iter_hr = pd.read_csv(data_rf + '/' + 'fold{}'.format(iter) + '_model_evaluation.csv', index_col=0)
        df_rf_model = pd.concat([df_rf_model, df_iter_hr], ignore_index=True)

    data_svm = evaluation_dir + 'svm_rel_clf_' + familiarity
    for iter in range(1, 18):
        df_iter_hr = pd.read_csv(data_svm + '/' + 'fold{}'.format(iter) + '_model_evaluation.csv', index_col=0)
        df_svm_model = pd.concat([df_svm_model, df_iter_hr], ignore_index=True)

    df_lambda_model = df_lambda_model[df_lambda_model['familiarity'] == familiarity]
    df_lc_model = df_lc_model[df_lc_model['familiarity'] == familiarity]
    df_gp_model = df_gp_model[df_gp_model['familiarity'] == familiarity]
    df_rf_model = df_rf_model[df_rf_model['familiarity'] == familiarity]
    df_svm_model = df_svm_model[df_svm_model['familiarity'] == familiarity]

    df_lambda_model = df_lambda_model.groupby(['fold']).mean()
    df_lc_model = df_lc_model.groupby(['fold']).mean()
    df_gp_model = df_gp_model.groupby(['fold']).mean()
    df_rf_model = df_rf_model.groupby(['fold']).mean()
    df_svm_model = df_svm_model.groupby(['fold']).mean()

    df_lambda_model['model type'] = ['Proposed'] * df_lambda_model.shape[0]
    df_lc_model['model type'] = ['LC'] * df_lc_model.shape[0]
    df_gp_model['model type'] = ['GP'] * df_gp_model.shape[0]
    df_svm_model['model type'] = ['SVM'] * df_svm_model.shape[0]
    df_rf_model['model type'] = ['RF'] * df_rf_model.shape[0]


    df_total = pd.concat([df_total, df_lc_model, df_gp_model,  df_rf_model, df_svm_model, df_lambda_model], ignore_index=True, axis=0)
    # df_total['hr'] = df_total['hit number']/df_total['test set size']
    # df_total['hr'] = df_total['hit rate']
    df_total['HR'] = df_total['HR'].astype(float)
    ax = sns.boxplot(x= df_total['model type'], y=df_total['HR'], palette="Blues")
    # for patch in ax.artists:
    #     r, g, b, a = patch.get_facecolor()
    #     patch.set_facecolor((r, g, b, .3))
    font_label = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 12,
                  }
    ax.set_ylabel('Hit rate', font_label)
    ax.set_xlabel('Models', font_label)

    font_tick = {'family': 'Times New Roman',
                   'weight': 'normal',
                   'size': 10,
                   }
    ax.set_xticklabels(ax.get_xticklabels(), font_tick)

    plt.yticks(np.arange(0.1, 1.1, 0.1), fontsize=10, fontfamily='Times New Roman')

    fig_dir = '../../results/reports/figures/familiarity/'
    plt.savefig(fig_dir + familiarity + "_model_comparison.pdf", bbox_inches='tight')
    plt.show()

    #
    table_dir = '../../results/reports/tables/familiarity/'
    file_name = table_dir + familiarity + '_model_comparison.csv'
    df_total.to_csv(file_name, encoding='utf-8')

for familiarity in ['familiar', 'unfamiliar']:
    plot_model_hr_box(familiarity)
