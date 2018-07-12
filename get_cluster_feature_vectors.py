import numpy as np
import h5py
from cluster_funcs import assign_feature_vectors
from kdigo_funcs import cluster_feature_vectors

h5_fname = '../RESULTS/icu/7days_070118/features.h5'
lbl_paths = ['../RESULTS/icu/7days_070118/clusters/norm_norm_a1/composite/23_clusters/',
             '../RESULTS/icu/7days_070118/clusters/norm_norm_a2/composite/20_clusters/',
             '../RESULTS/icu/7days_070118/clusters/custcost_custcost_a1/composite/12_clusters/',
             '../RESULTS/icu/7days_070118/clusters/custcost_custcost_a2/composite/12_clusters/',
             '../RESULTS/icu/7days_070118/clusters/custcost_custcost_a4/composite/12_clusters/',
             '../RESULTS/icu/7days_070118/clusters/custcost_norm_a1/composite/14_clusters/',
             '../RESULTS/icu/7days_070118/clusters/custcost_norm_a2/composite/13_clusters/',
             '../RESULTS/icu/7days_070118/clusters/custcost_norm_a4/composite/13_clusters/',
             '../RESULTS/icu/7days_070118/clusters/norm_custcost_a1/composite/16_clusters/',
             '../RESULTS/icu/7days_070118/clusters/norm_custcost_a2/composite/12_clusters/']

tags = ['_nna1', '_nna2', '_cca1', '_cca2', '_cca4', '_cna1', '_cna2', '_cna4', '_nca1', '_nca2']

tags = [tags[x] + '-2' for x in range(len(tags))]

f = h5py.File(h5_fname, 'r+')
fg = f['features']
desc = fg['descriptive_individual'][:]
temp_norm = fg['template_individual_norm']
slope_norm = fg['slope_individual_norm'][:]

for (path, tag) in zip(lbl_paths,  tags):
    lbls = np.loadtxt(path + 'clusters.txt', dtype=str)

    desc_c, temp_c, slope_c = cluster_feature_vectors(desc, temp_norm, slope_norm, lbls)
    all_desc_c = assign_feature_vectors(lbls, desc_c)
    all_temp_c = assign_feature_vectors(lbls, temp_c)
    all_slope_c = assign_feature_vectors(lbls, slope_c)
    fg.create_dataset('descriptive_clusters' + tag, data=all_desc_c, dtype=int)
    fg.create_dataset('template_clusters' + tag, data=all_temp_c, dtype=float)
    fg.create_dataset('slope_clusters' + tag, data=all_slope_c, dtype=float)

    all_traj_c = np.hstack((all_desc_c, all_slope_c, all_temp_c))
    fg.create_dataset('all_trajectory_clusters' + tag, data=all_traj_c)
    everything_clusters = np.hstack((all_traj_c, fg['all_clinical'][:]))
    fg.create_dataset('everything_clusters' + tag, data=everything_clusters)

f.close()
