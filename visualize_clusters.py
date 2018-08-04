import numpy as np
import h5py
import os
from vis_funcs import plot_feature_selection, stacked_bar, get_feature_names
from cluster_funcs import plot_daily_kdigos
from kdigo_funcs import load_csv, load_dm
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_selection import chi2


data_path = '../DATA/icu/7days_071118/'
base_path = '../RESULTS/icu/7days_071118/'
full_kdigo_path = '../DATA/icu/7days_073118/'

methods = ['composite', 'ward']

feat_sel = 'everything'

tags = ['_norm_norm_a1', '_norm_norm_a2', '_norm_norm_a4',
        '_norm_custcost_a1', '_norm_custcost_a2']

selection_models = [['linear', [0.01, ]],
                    ['univariate', [10, chi2]],
                    ['univariate', [20, chi2]]]

h5_name = 'stats.h5'

f = h5py.File(base_path + '/' + h5_name, 'r')
ids = f['meta']['ids'][:]

all_feat_names = get_feature_names(base_path + 'features/')['everything_clusters']
kdigos = load_csv(full_kdigo_path + 'kdigo.csv', ids, int)

for tag in tags:
    try:
        dm = np.load(base_path + 'kdigo_dm%s.npy' % tag)
    except:
        dm = load_dm(base_path + 'kdigo_dm%s.csv' % tag, ids)
    sqdm = squareform(dm)

    for method in methods:
        for (dirpath, dirnames, filenames) in os.walk(base_path + 'clusters/%s/%s/' % (tag[1:], method)):
            for dirname in dirnames:
                if 'clusters' in dirname:
                    lbls = load_csv(dirpath + '/' + dirname + '/clusters.txt', ids, str)

                    # feats = load_csv(base_path + 'features/%s/%s/%s/%s.csv' %
                    #                  (tag[1:], method, dirname, feat_sel), ids)
                    #
                    # if not os.path.exists(dirpath + '/' + dirname + '/feature_selection/'):
                    #     os.mkdir(dirpath + '/' + dirname + '/feature_selection/')
                    #     for (model, params) in selection_models:
                    #         plot_feature_selection(f, lbls, feats, all_feat_names, lbl_name='died_inp',
                    #                                method=model, parameters=params,
                    #                                outpath=dirpath + '/' + dirname + '/feature_selection/')
                    if not os.path.exists(dirpath + '/' + dirname + '/daily_kdigos_ext/'):
                        os.mkdir(dirpath + '/' + dirname + '/daily_kdigos_ext/')
                        plot_daily_kdigos(full_kdigo_path, ids, base_path + '/' + h5_name, sqdm, lbls,
                                          outpath=dirpath + '/' + dirname + '/daily_kdigos_ext/', max_day=20)
                    #
                    # stacked_bar(dirpath + '/' + dirname + '/cluster_stats.csv',
                    #             fname=dirpath + '/' + dirname + '/mort_vs_kdigo_bar.png',
                    #             title='Cluster Mortality vs. Max KDIGO')
