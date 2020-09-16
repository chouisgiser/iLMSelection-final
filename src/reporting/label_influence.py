import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

data_label = '../../results/experiments/label_influence/label_influence.csv'

df_label = pd.read_csv(data_label)

label_scale_1 = df_label['0-1']
label_scale_2 = df_label['1-5']
label_scale_3 = df_label['1-7']


fig, ax = plt.subplots(figsize=(10, 6))
index = np.arange(2)

label_scale_1_values = np.array(list(label_scale_1))
label_scale_2_values = np.array(list(label_scale_2))
label_scale_3_values = np.array(list(label_scale_3))


bar_width = 0.25

rects1 = ax.bar(index, label_scale_1_values, bar_width, color= 'blue', alpha=0.8, label='0-1 point scale', )
rects2 = ax.bar(index + bar_width, label_scale_2_values, bar_width, color= 'blue', alpha=0.6, label='1-5 point scale')
rects3 = ax.bar(index + 2*bar_width, label_scale_3_values, bar_width, color= 'blue', alpha=0.4, label='1-7 point scale')

font_label = {'family': 'Times New Roman',
              'weight': 'heavy',
              'size': 18,
              }
ax.set_ylabel('Hit rate', font_label)
ax.set_xlabel('Model types', font_label)

font_xstick = {'family': 'Times New Roman',
               'weight': 'heavy',
               'size': 16,
               }
ax.set_xticks(index + 3 * bar_width / 2)
ax.set_xticklabels(['Familiar model', 'Unfamiliar model'], font_xstick)

plt.yticks(np.arange(0, 1.1, 0.2), fontsize=16, fontfamily='Times New Roman')

font_legend = {'family': 'Times New Roman',
               'weight': 'normal',
               'size': 14,
               }
ax.legend(handles=[rects1, rects2, rects3], prop=font_legend)

for a, b in zip(index, label_scale_1_values ):
    plt.text(a, b + 0.02, '%.2f' % b, ha='center', va='center', fontsize=14, fontfamily='Times New Roman')

for a, b in zip(index, label_scale_2_values ):
    plt.text(a + bar_width, b + 0.02, '%.2f' % b, ha='center', va='center', fontsize=14,
             fontfamily='Times New Roman')

for a, b in zip(index, label_scale_3_values ):
    plt.text(a +2* bar_width, b + 0.02, '%.2f' % b, ha='center', va='center', fontsize=14,
             fontfamily='Times New Roman')

plt.show()