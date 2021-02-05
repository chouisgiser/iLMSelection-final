
import lightgbm as lgb
import matplotlib.pyplot as plt
model_dir = '../../results/cv_results/familiarity/models/'
fig_dir = '../../results/reports/trees/'

for model_familiarity in ['familiar', 'unfamiliar']:
    # total_num = 0
    # total_true_num = 0
    for fold_index in range(1, 2):
        ranker = lgb.Booster(model_file=model_dir + 'ranking_' + model_familiarity + '/fold' +
                                        str(fold_index) + '_model.txt')
        ax = lgb.plot_tree(ranker, tree_index=0, figsize=(15, 15), show_info=['split_gain'], orientation='vertical')
        plt.savefig(fig_dir + model_familiarity + "_tree.pdf", bbox_inches='tight')
        plt.show()
