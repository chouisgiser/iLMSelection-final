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
    if attribute == 'familiar':
        ax1.set_ylim(0.68, 0.80)
        ax2.set_ylim(1.35, 1.55)
    else:
        ax1.set_ylim(0.68, 0.80)
        ax2.set_ylim(1.35, 1.55)

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


    # for a, b in zip(index, hr):
    #     plt.text(a+1, b + 0.002, '%.3f' % b, ha='center', va='center', fontsize=16, fontfamily='Times New Roman')
    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    figure_dir = '../../results/reports/figures/feature_remove/'
    plt.savefig(figure_dir + attribute + '_model_feature_reduction.pdf', bbox_inches='tight')
    plt.show()

for familiarity in ['familiar', 'unfamiliar']:
    plot_iteration_by_total_gain(attribute=familiarity)