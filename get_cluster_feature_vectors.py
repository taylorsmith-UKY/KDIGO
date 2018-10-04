import numpy as np
import h5py
from cluster_funcs import assign_feature_vectors
from kdigo_funcs import cluster_feature_vectors, load_csv, arr2csv
from utility_funcs import get_feats_by_dod
import os

# --------------------------------------------------- PARAMETERS ----------------------------------------------------- #
h5_fname = '../RESULTS/icu/7days_100218/stats.h5'
lbl_path = '../RESULTS/icu/7days_100218/clusters/'
feature_path = '../RESULTS/icu/7days_100218/features/'

meta_grp = 'meta'

methods = ['dynamic', 'flat']

tags = ['_custmismatch_normBC']

n_days_l = [7, ]

# -------------------------------------------------------------------------------------------------------------------- #

for n_days in n_days_l:
    f = h5py.File(h5_fname, 'r')

    day_str = '%ddays' % n_days
    if not os.path.exists(os.path.join(feature_path, day_str)):
        os.mkdir(os.path.join(feature_path, day_str))

    for tag in tags:
        for method in methods:
            for (dirpath, dirnames, filenames) in os.walk(os.path.join(lbl_path, day_str, tag[1:], method)):
                for dirname in dirnames:
                    try:
                        ids = np.loadtxt(os.path.join(dirpath, dirname, 'clusters.csv'), delimiter=',', usecols=0, dtype=int)
                        lbls = load_csv(os.path.join(dirpath, dirname, 'clusters.csv'), ids, str)
                    except IOError:
                        continue
                    desc = load_csv(os.path.join(feature_path, 'individual', 'descriptive_features.csv'), ids, int,
                                    skip_header=True)
                    temp_norm = load_csv(os.path.join(feature_path, 'individual', 'template_norm.csv'), ids)
                    slope_norm = load_csv(os.path.join(feature_path, 'individual', 'slope_norm.csv'), ids)
                    all_clin = load_csv(os.path.join(feature_path, 'individual', 'all_clinical.csv'), ids)

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

                    arr2csv(os.path.join(fpath, 'descriptive.csv'), all_desc_c, ids, fmt='%.4f')
                    arr2csv(os.path.join(fpath, 'template.csv'), all_temp_c, ids, fmt='%.4f')
                    arr2csv(os.path.join(fpath, 'slope.csv'), all_slope_c, ids, fmt='%.4f')
                    arr2csv(os.path.join(fpath, 'all_trajectory.csv'), all_traj_c, ids, fmt='%.4f')
                    arr2csv(os.path.join(fpath, 'everything.csv'), everything_clusters, ids, fmt='%.4f')
