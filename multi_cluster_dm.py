import numpy as np
from cluster_funcs import dist_cut_cluster, dist_cut_tree, load_csv
from stat_funcs import get_cstats
from kdigo_funcs import load_dm, arr2csv
import h5py
import os
from fastcluster import ward
from scipy.cluster.hierarchy import to_tree
from scipy.spatial.distance import squareform
import datetime

# PARAMETERS
#################################################################################
base_path = '../RESULTS/icu/7days_071118/'
dm_tags = ['_norm_norm_a1', '_norm_norm_a2', '_norm_norm_a4',
        '_norm_custcost_a1', '_norm_custcost_a2',  # '_norm_custcost_a4',
        '_custcost_norm_a1', '_custcost_norm_a2',  # '_custcost_norm_a4',
        '_custcost_custcost_a1', '_custcost_custcost_a2']  # , '_custcost_custcost_a4']

h5_name = 'stats.h5'

eps_l = [0.010, 0.015, 0.020, 0.025, 0.050, 0.075, 0.100]  # Epsilon threshold for DBSCAN
p_thresh_l = [1e-10, 1e-20, 1e-50]                         # NormalTest threshold to stop splitting
hlim_l = [3, 4]                                            # Height limit for cutting dendrogram
min_size_l = [20, 30, 50, 75]                              # Minimum cluster size
# Note: DBSCAN clusters smaller than min_size are grouped into the noise cluster,
# whereas min_size is used as a stopping criteria when splitting ward's method clusters.

max_noise = 0.1                                            # As fraction of total number patients
# If > max_noise patients designated as noise (only applicable for DBSCAN), discard results

# Which clustering algorithm to apply
# Methods
#   - composite     -   First apply DBSCAN, followed by modified Ward's method
#   - ward          -   Apply modified Ward's method on all patients
method = 'ward'


#################################################################################

date_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')

f = h5py.File(base_path + h5_name, 'r')
ids = f['meta']['ids'][:]
if method != 'ward':
    f.close()

for dm_tag in dm_tags:

    if not os.path.exists(base_path + 'clusters/%s/' % dm_tag[1:]):
        os.mkdir(base_path + 'clusters/%s/' % dm_tag[1:])
    try:
        dm = np.load(base_path + 'kdigo_dm%s.npy' % dm_tag)
    except:
        dm = load_dm(base_path + 'kdigo_dm%s.csv' % dm_tag, ids)

    if method == 'composite':
        for eps in eps_l:
            for p_thresh in p_thresh_l:
                for hlim in hlim_l:
                    for min_size in min_size_l:
                        dist_cut_cluster(base_path + h5_name, dm, ids, path=base_path + 'clusters/%s/' % dm_tag[1:],
                                         eps=eps, p_thresh=p_thresh, min_size=min_size, height_lim=hlim,
                                         interactive=False, save=True, max_noise=max_noise)
    elif method == 'ward':
        link = ward(dm)
        sqdm = squareform(dm)
        if not os.path.exists(base_path + 'clusters/%s/ward/' % dm_tag[1:]):
            os.mkdir(base_path + 'clusters/%s/ward/' % dm_tag[1:])
        for p_thresh in p_thresh_l:
            for hlim in hlim_l:
                for min_size in min_size_l:
                    root = to_tree(link)
                    lbls = np.ones(len(ids), dtype=int).astype(str)
                    lbls = dist_cut_tree(root, lbls, '1', sqdm, p_thresh, min_size=min_size, height_lim=hlim)
                    lbl_names = np.unique(lbls)
                    n_clusters = len(lbl_names)
                    if not os.path.exists(base_path + 'clusters/%s/ward/%d_clusters/'
                                          % (dm_tag[1:], n_clusters)):
                        os.mkdir(base_path + 'clusters/%s/ward/%d_clusters/'
                                 % (dm_tag[1:], n_clusters))
                        arr2csv(base_path + 'clusters/%s/ward/%d_clusters/clusters.txt'
                                % (dm_tag[1:], n_clusters), lbls, ids, fmt='%s')
                        get_cstats(f, base_path + 'clusters/%s/ward/%d_clusters/' % (dm_tag[1:], n_clusters))
                    ref = load_csv(base_path + 'clusters/%s/ward/%d_clusters/clusters.txt'
                                   % (dm_tag[1:], n_clusters), ids, str)
                    if np.all(ref == lbls):
                        cont = False
                    else:
                        cont = True
                        tag = 'a'
                    while cont:
                        if os.path.exists(base_path + 'clusters/%s/ward/%d_clusters_%s/'
                                          % (dm_tag[1:], n_clusters, tag)):
                            ref = load_csv(base_path + 'clusters/%s/ward/%d_clusters_%s/clusters.txt'
                                           % (dm_tag[1:], n_clusters, tag), ids, str)
                            if np.all(ref == lbls):
                                cont = False
                            else:
                                tag = chr(ord(tag) + 1)
                        else:
                            os.makedirs(base_path + 'clusters/%s/ward/%d_clusters_%s/'
                                        % (dm_tag[1:], n_clusters, tag))
                            arr2csv(base_path + 'clusters/%s/ward/%d_clusters_%s/clusters.txt'
                                    % (dm_tag[1:], n_clusters, tag), lbls, ids, fmt='%s')
                            get_cstats(f,
                                       base_path + 'clusters/%s/ward/%d_clusters_%s/' % (dm_tag[1:], n_clusters, tag))
                            cont = False
