import numpy as np
import h5py
import os
from vis_funcs import plot_feature_selection, stacked_bar, get_feature_names
from cluster_funcs import plot_daily_kdigos
from kdigo_funcs import load_csv, load_dm
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_selection import chi2


data_path = '../DATA/icu/7days_091718/'
base_path = '../RESULTS/icu/7days_091718/'
full_kdigo_path = '../DATA/icu/7days_091718/'

meta_grp = 'meta'

methods = ['ward',]

feat_sel = 'everything'

tags = ['_custmismatch_extension_a1E+00_normBC', ] # '_custmismatch_normBC',
        # '_absmismatch_extension_a1E+00_normBC', '_absmismatch_normBC']

selection_models = [['linear', [0.01, ]],
                    ['univariate', [10, chi2]],
                    ['univariate', [20, chi2]]]

h5_name = 'stats.h5'

dk_folder_name = 'daily_kdigos'
dk_range = 7
dk_vert = None


# n_clust_sel = np.array(['16_clusters_a', ])
n_clust_sel = None

f = h5py.File(base_path + '/' + h5_name, 'r')
all_ids = f[meta_grp]['ids'][:]
dtd = f[meta_grp]['days_to_death'][:]
mk = f[meta_grp]['max_kdigo'][:]
t_lims = [7, ]


for t_lim in t_lims:
    sel = np.union1d(np.where(np.isnan(dtd)), np.where(dtd > t_lim))

    ids = all_ids[sel]

    # all_feat_names = get_feature_names(base_path + 'features/')['everything_clusters']
    kdigos = load_csv(full_kdigo_path + 'kdigo.csv', ids, int)

    for tag in tags:
        try:
            dm = np.load(base_path + 'dm/%ddays/kdigo_dm%s.npy' % (t_lim, tag))
        except:
            dm = load_dm(base_path + 'dm/%ddays/kdigo_dm%s.csv' % (t_lim, tag), ids)
        sqdm = squareform(dm)

        for method in methods:
            for (dirpath, dirnames, filenames) in os.walk(base_path + 'clusters/%ddays/%s/%s/' % (t_lim, tag[1:], method)):
                for dirname in dirnames:
                    if 'clusters' in dirname:
                        if n_clust_sel is not None and np.all([x not in dirname for x in n_clust_sel]):
                            continue
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
                        if not os.path.exists(os.path.join(dirpath, dirname, dk_folder_name)):
                            os.mkdir(os.path.join(dirpath, dirname, dk_folder_name))
                            plot_daily_kdigos(full_kdigo_path, ids, os.path.join(base_path, h5_name), sqdm, lbls,
                                              outpath=os.path.join(dirpath, dirname, dk_folder_name),
                                              max_day=dk_range, cutoff=dk_vert)
                        #
                        # stacked_bar(dirpath + '/' + dirname + '/cluster_stats.csv',
                        #             fname=dirpath + '/' + dirname + '/mort_vs_kdigo_bar.png',
                        #             title='Cluster Mortality vs. Max KDIGO')
