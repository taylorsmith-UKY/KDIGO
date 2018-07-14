import os
import h5py
import numpy as np
from vis_funcs import plot_feature_selection, get_feature_names
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier as RFC

# PARAMETERS
#################################################################################

dataset = '7days_070118'

models = [['linear', [0.1, ]],
          ['linear', [0.3, ]],
          ['linear', [0.01, ]],
          ['univariate', [10, chi2]],
          ['univariate', [20, chi2]],
          ['recursive', [RFC(), 10]],
          ['variance', [0.1, ]],
          ['variance', [0.2, ]]]

feature_grp_name = 'everything_clusters'
cluster_method = 'composite_new'
n_clusters = 22

dm_tags = ['_norm_norm_a1', ]

eval_targets = ['died_inp', ]




#################################################################################

featpath = '../DATA/icu/' + dataset + '/'
basepath = '../RESULTS/icu/' + dataset + '/'

feat_names = get_feature_names(featpath)
f = h5py.File(basepath + 'kdigo_dm.h5', 'r')

basepath += 'feature_selection/'

for target_name in eval_targets:
    lbls = f['meta'][target_name]
    tpath = basepath + target_name + '/'
    if not os.path.exists(tpath):
        os.mkdir(tpath)
    for dm_tag in dm_tags:
        dpath = tpath + dm_tag[1:] + '/'
        if not os.path.exists(dpath):
            os.mkdir(dpath)
        for m in models:
            try:
                mpath = dpath + m[0] + '_%.0E' % m[1][0] + '/'
            except:
                mpath = dpath + m[0] + '/'
            if not os.path.exists(mpath):
                os.mkdir(mpath)
            model, sel = plot_feature_selection(f, feature_grp_name, cluster_method, n_clusters, feat_names[feature_grp_name],
                               lbl_name='died_inp', method=m[0], parameters=m[1],
                               dm_tag=dm_tag, outpath=mpath)
