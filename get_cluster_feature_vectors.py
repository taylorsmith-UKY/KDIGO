import numpy as np
import h5py
from cluster_funcs import assign_feature_vectors
from kdigo_funcs import cluster_feature_vectors, load_csv, arr2csv
from utility_funcs import get_feats_by_dod
import os

# --------------------------------------------------- PARAMETERS ----------------------------------------------------- #
h5_fname = '../RESULTS/icu/7days_090818/stats.h5'
lbl_path = '../RESULTS/icu/7days_090818/clusters/'
feature_path = '../RESULTS/icu/7days_090818/features/'

meta_grp = 'meta'

methods = ['composite',]

tags = ['_custmismatch_extension_a1E+00_normBC_n', ]

n_days_l = [7, ]

# -------------------------------------------------------------------------------------------------------------------- #

for n_days in n_days_l:
    ids, _ = get_feats_by_dod(h5_fname, n_days=n_days)

    day_str = '%ddays' % n_days

    desc = load_csv(os.path.join(feature_path, day_str, 'individual', 'descriptive_features.csv'), ids, int,
                    skip_header=True)
    temp_norm = load_csv(os.path.join(feature_path, day_str, 'individual', 'template_norm.csv'), ids)
    slope_norm = load_csv(os.path.join(feature_path, day_str, 'individual', 'slope_norm.csv'), ids)
    all_clin = load_csv(os.path.join(feature_path, day_str, 'individual', 'all_clinical.csv'), ids)
    for tag in tags:
        for method in methods:
            for (dirpath, dirnames, filenames) in os.walk(lbl_path + tag[1:] + '/' + method + '/'):
                for dirname in dirnames:
                    try:
                        lbls = load_csv(dirpath + '/' + dirname + '/clusters.txt', ids, str)
                    except IOError:
                        continue
                    desc_c, temp_c, slope_c = cluster_feature_vectors(desc, temp_norm, slope_norm, lbls)
                    all_desc_c = assign_feature_vectors(lbls, desc_c)
                    all_temp_c = assign_feature_vectors(lbls, temp_c)
                    all_slope_c = assign_feature_vectors(lbls, slope_c)

                    all_traj_c = np.hstack((all_desc_c, all_slope_c, all_temp_c))
                    everything_clusters = np.hstack((all_clin, all_traj_c))

                    fpath = os.path.join(feature_path, day_str, tag[1:])
                    if not os.path.exists(fpath):
                        os.mkdir(fpath)
                    fpath = os.path.join(fpath, method)
                    if not os.path.exists(fpath):
                        os.mkdir(fpath)
                    fpath = os.path.join(fpath, dirname)
                    if not os.path.exists(fpath):
                        os.mkdir(fpath)

                    arr2csv(fpath + 'descriptive.csv', all_desc_c, ids, fmt='%.4f')
                    arr2csv(fpath + 'template.csv', all_temp_c, ids, fmt='%.4f')
                    arr2csv(fpath + 'slope.csv', all_slope_c, ids, fmt='%.4f')
                    arr2csv(fpath + 'all_trajectory.csv', all_traj_c, ids, fmt='%.4f')
                    arr2csv(fpath + 'everything.csv', everything_clusters, ids, fmt='%.4f')
