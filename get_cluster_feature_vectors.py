import numpy as np
import h5py
from cluster_funcs import assign_feature_vectors
from kdigo_funcs import cluster_feature_vectors, load_csv, arr2csv
import os

h5_fname = '../RESULTS/icu/7days_071118/stats.h5'
lbl_path = '../RESULTS/icu/7days_071118/clusters/'
feature_path = '../RESULTS/icu/7days_071118/features/'

methods = ['ward']

tags = ['_norm_norm_a1', '_norm_norm_a2', '_norm_norm_a4',
        '_norm_custcost_a1', '_norm_custcost_a2',  # '_norm_custcost_a4',
        '_custcost_norm_a1', '_custcost_norm_a2',  # '_custcost_norm_a4',
        '_custcost_custcost_a1', '_custcost_custcost_a2']  # , '_custcost_custcost_a4']

f = h5py.File(h5_fname, 'r')
ids = f['meta']['ids'][:]
f.close()

desc = load_csv(feature_path + 'trajectory_individual/descriptive_features.csv', ids, int, skip_header=True)
temp_norm = load_csv(feature_path + 'trajectory_individual/template_norm.csv', ids)
slope_norm = load_csv(feature_path + 'trajectory_individual/slope_norm.csv', ids)
all_clin = load_csv(feature_path + 'clinical/all_clinical.csv', ids)


for tag in tags:
    for method in methods:
        for (dirpath, dirnames, filenames) in os.walk(lbl_path + tag[1:] + '/' + method + '/'):
            for dirname in dirnames:
                lbls = np.loadtxt(dirpath + '/' + dirname + '/clusters.txt', dtype=str)

                desc_c, temp_c, slope_c = cluster_feature_vectors(desc, temp_norm, slope_norm, lbls)
                all_desc_c = assign_feature_vectors(lbls, desc_c)
                all_temp_c = assign_feature_vectors(lbls, temp_c)
                all_slope_c = assign_feature_vectors(lbls, slope_c)

                all_traj_c = np.hstack((all_desc_c, all_slope_c, all_temp_c))
                everything_clusters = np.hstack((all_clin, all_traj_c))

                fpath = feature_path + tag[1:] + '/'
                if not os.path.exists(fpath):
                    os.mkdir(fpath)
                fpath += method + '/'
                if not os.path.exists(fpath):
                    os.mkdir(fpath)
                fpath += dirname + '/'
                if not os.path.exists(fpath):
                    os.mkdir(fpath)

                arr2csv(fpath + 'descriptive.csv', all_desc_c, ids, fmt='%.4f')
                arr2csv(fpath + 'template.csv', all_temp_c, ids, fmt='%.4f')
                arr2csv(fpath + 'slope.csv', all_slope_c, ids, fmt='%.4f')
                arr2csv(fpath + 'all_trajectory.csv', all_traj_c, ids, fmt='%.4f')
                arr2csv(fpath + 'everything.csv', everything_clusters, ids, fmt='%.4f')


f.close()
