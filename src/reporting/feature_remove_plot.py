# -*- coding: utf-8 -*-
"""
# @time    : 27.08.20 15:10
# @author  : zhouzy
# @file    : hit_rate_decrease_bar_plot.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns

def plot_iteration_by_total_gain(attribute):

    feature_remove_file = '../../results/reports/tables/feature_remove/' + attribute + '_feature_reduction.csv'

    pd_feature_delta = pd.read_csv(feature_remove_file)

    # plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(10, 6))

    hr = np.array(list(pd_feature_delta['HR']))
    ax1.plot(pd_feature_delta['number'], hr, 's-', color='r', label="Hit rate")

    ax2 = ax1.twinx()
    mae = np.array(list(pd_feature_delta['MAE']))
    ax2.plot(pd_feature_delta['number'], mae, 'o-', color='b', label="Mean absolute error")

    font_label = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 16,
                  }
    ax1.set_ylabel('HR', font_label)
    ax1.set_xlabel('Reduced number of salience measures', font_label)
    ax2.set_ylabel('MAE', font_label)
    # ax1.set_xlabel('Reduced number of salience measures', font_label)
    #
    # y_labels = pd_feature_delta['HR']
    # if attribute == 'familiar':
    #     ax1.set_ylim(0.54, 0.76)
    #     ax2.set_ylim(1.40, 2.0)
    # else:
    #     ax1.set_ylim(0.60, 0.80)
    #     ax2.set_ylim(1.38, 1.70)
    # if attribute == 'familiar':
    #     ax1.set_ylim(0.68, 0.76)
    #     ax2.set_ylim(1.40, 1.60)
    # else:
    #     ax1.set_ylim(0.73, 0.80)
    #     ax2.set_ylim(1.38, 1.50)


    font_tick = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 14,
                  }
    x_ticks = [f"{num:.0f}" for num in ax1.get_xticks()]
    ax1.set_xticklabels(x_ticks, font_tick)

    y_ticks = [f"{num:.2f}" for num in ax1.get_yticks()]
    ax1.set_yticklabels(y_ticks, font_tick)

    # [round(num, 2) for num in ax2.get_yticks()]
    y_ticks = [f"{num:.2f}" for num in ax2.get_yticks()]
    ax2.set_yticklabels(y_ticks, font_tick)

    # plt.yticks( fontsize='14', fontfamily='Times New Roman')
    # plt.xticks(pd_feature_delta['number'], fontsize='14', fontfamily='Times New Roman')

    # index = np.arange(len(list(pd_feature_delta['HR'])))
    # for a, b in zip(pd_feature_delta['number'], mae):
    #     plt.text(a, b + 0.01, '%.3f' % b, ha='center', va='top', fontsize=9, fontfamily='Times New Roman')
    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    figure_dir = '../../results/reports/figures/feature_remove/'
    plt.savefig(figure_dir + attribute + '_model_feature_reduction.pdf', bbox_inches='tight')
    plt.show()

def plot_iteration_4_individual(attribute):
    feature_remove_file = '../../results/reports/tables/feature_remove/individual_' + attribute + '_feature_reduction.csv'

    pd_feature_delta = pd.read_csv(feature_remove_file)

    # plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(10, 6))

    index = np.arange(len(pd_feature_delta['number']))

    hr_delta = np.array(list(pd_feature_delta['HR']))

    bar_width = 0.4
    plt.bar(index, hr_delta, bar_width, alpha=0.8)
    xlabels = pd_feature_delta['number'].tolist()
    plt.xticks(ticks= index, labels= xlabels, fontfamily='Times New Roman')
    plt.yticks(np.arange(0, 0.1, 0.01), fontfamily='Times New Roman')

    for a, b in zip(index, hr_delta):
        plt.text(a, b + 0.003, '%.3f' % b, ha='center', va='center', fontfamily='Times New Roman')

    # ax1.plot(pd_feature_delta['number'], hr, 's-', color='r', label="Hit rate")

    # ax2 = ax1.twinx()
    # mae = np.array(list(pd_feature_delta['MAE']))
    # ax2.plot(pd_feature_delta['number'], mae, 'o-', color='b', label="Mean absolute error")

    font_label = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 16,
                  }
    ax1.set_ylabel('HR', font_label)
    ax1.set_xlabel('the ordering number salience measure from highest to lowest', font_label)

    font_tick = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 14,
                  }
    # x_ticks = [f"{num:.0f}" for num in ax1.get_xticks()]
    # ax1.set_xticklabels(x_ticks, font_tick)

    y_ticks = [f"{num:.2f}" for num in ax1.get_yticks()]
    ax1.set_yticklabels(y_ticks, font_tick)

    figure_dir = '../../results/reports/figures/feature_remove/'
    plt.savefig(figure_dir + attribute + '_model_individual_feature_reduction.pdf', bbox_inches='tight')
    plt.show()

def plot_iteration_by_metrics(metric):

    familiar_feature_remove_file = '../../results/reports/tables/feature_remove/familiar_feature_reduction.csv'

    unfamiliar_feature_remove_file = '../../results/reports/tables/feature_remove/unfamiliar_feature_reduction.csv'

    pd_feature_familiar = pd.read_csv(familiar_feature_remove_file)
    pd_feature_unfamiliar = pd.read_csv(unfamiliar_feature_remove_file)

    # plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(10, 6))

    font_label = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 16,
                  }

    if metric == 'HR':
        plt.ylim(bottom=0, top=1.1)
        hr_familiar = np.array(list(pd_feature_familiar['HR']))
        hr_unfamiliar = np.array(list(pd_feature_unfamiliar['HR']))
        ax1.set_ylabel('HR', font_label)
    if metric == 'MAE':
        plt.ylim(bottom=1.0, top=1.8)
        hr_familiar = np.array(list(pd_feature_familiar['MAE']))
        hr_unfamiliar = np.array(list(pd_feature_unfamiliar['MAE']))
        ax1.set_ylabel('MAE', font_label)
    ax1.plot(pd_feature_familiar['number'], hr_familiar, 's-', color='r', label="Familiar-trained-model")
    ax1.plot(pd_feature_unfamiliar['number'], hr_unfamiliar, 'o-', color='b', label="Unfamiliar-trained-model")

    ax1.set_xlabel('Reduced number of salience measures', font_label)



    # ax2.set_ylabel('MAE', font_label)
    # ax1.set_xlabel('Reduced number of salience measures', font_label)
    #
    # y_labels = pd_feature_delta['HR']
    # if attribute == 'familiar':
    #     ax1.set_ylim(0.54, 0.76)
    #     ax2.set_ylim(1.40, 2.0)
    # else:
    #     ax1.set_ylim(0.60, 0.80)
    #     ax2.set_ylim(1.38, 1.70)
    # if attribute == 'familiar':
    #     ax1.set_ylim(0.68, 0.76)
    #     ax2.set_ylim(1.40, 1.60)
    # else:
    #     ax1.set_ylim(0.73, 0.80)
    #     ax2.set_ylim(1.38, 1.50)


    font_tick = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 14,
                  }
    # x_ticks = [f"{num:.0f}" for num in ax1.get_xticks()]
    # ax1.set_xticklabels(x_ticks, font_tick)
    #
    # y_ticks = [f"{num:.2f}" for num in ax1.get_yticks()]
    # ax1.set_yticklabels(y_ticks, font_tick)

    # y_ticks = np.arange(0, 1, 0.2)
    # ax1.set_yticklabels(y_ticks, font_tick)

    # [round(num, 2) for num in ax2.get_yticks()]
    # y_ticks = [f"{num:.2f}" for num in ax2.get_yticks()]
    # ax2.set_yticklabels(y_ticks, font_tick)

    plt.yticks( fontsize='14', fontfamily='Times New Roman')
    plt.xticks(pd_feature_familiar['number'], fontsize='14', fontfamily='Times New Roman')

    # index = np.arange(len(list(pd_feature_delta['HR'])))
    # for a, b in zip(pd_feature_delta['number'], mae):
    #     plt.text(a, b + 0.01, '%.3f' % b, ha='center', va='top', fontsize=9, fontfamily='Times New Roman')
    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    figure_dir = '../../results/reports/figures/feature_remove/'
    plt.savefig(figure_dir + metric + '_model_feature_reduction.pdf', bbox_inches='tight')
    plt.show()


# for familiarity in ['familiar', 'unfamiliar']:
    # plot_iteration_4_individual(attribute=familiarity)
    # plot_iteration_by_total_gain(attribute=familiarity)

plot_iteration_by_metrics(metric='HR')
plot_iteration_by_metrics(metric='MAE')