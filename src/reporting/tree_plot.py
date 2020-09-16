
import lightgbm as lgb
import matplotlib.pyplot as plt
model_dir = '../../results/models/'


for model_familiarity in ['familiar', 'unfamiliar']:
    total_num = 0
    total_true_num = 0
    model_dir = '../../results/dataset_split_iteration/familiarity/iter1/models/ranking_' + model_familiarity + '/'
    for fold_index in range(1, 2):
        ranker = lgb.Booster(model_file=model_dir + 'fold' + str(fold_index) + '_model.txt')
        ax = lgb.plot_tree(ranker, tree_index=0, figsize=(15, 15), show_info=['split_gain'])
        plt.show()
