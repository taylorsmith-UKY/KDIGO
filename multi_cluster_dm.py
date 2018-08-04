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
dm_tags = ['_cDTW_normdist_a2', '_cDTW_cdist_a1', '_cDTW_cdist_a1']

h5_name = 'stats.h5'

eps_l = [0.050, 0.075, 0.100]                              # Epsilon threshold for DBSCAN
p_thresh_l = [1e-10, 1e-20, 1e-50]                         # NormalTest threshold to stop splitting
hlim_l = [3, 4]                                            # Height limit for cutting dendrogram
min_size_l = [40, 75]                                      # Minimum cluster size
# Note: DBSCAN clusters smaller than min_size are grouped into the noise cluster,
# whereas min_size is used as a stopping criteria when splitting ward's method clusters.

max_noise = 0.1                                            # As fraction of total number patients
# If > max_noise patients designated as noise (only applicable for DBSCAN), discard results

max_clust = 30
# Which clustering algorithm to apply
# Methods
#   - composite     -   First apply DBSCAN, followed by modified Ward's method
#   - ward          -   Apply modified Ward's method on all patients
methods = ('composite', 'ward')


#################################################################################

date_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')

f = h5py.File(base_path + h5_name, 'r')
ids = f['meta']['ids'][:]
f.close()

for dm_tag in dm_tags:

    if not os.path.exists(base_path + 'clusters/%s/' % dm_tag[1:]):
        os.mkdir(base_path + 'clusters/%s/' % dm_tag[1:])
    if os.path.isfile(base_path + 'kdigo_dm%s.npy' % dm_tag):
        dm = np.load(base_path + 'kdigo_dm%s.npy' % dm_tag)
    else:
        dm = load_dm(base_path + 'kdigo_dm%s.csv' % dm_tag, ids)

    for method in methods:
        if method == 'composite':
            for eps in eps_l:
                for p_thresh in p_thresh_l:
                    for hlim in hlim_l:
                        for min_size in min_size_l:
                            dist_cut_cluster(base_path + h5_name, dm, ids, path=base_path + 'clusters/%s/' % dm_tag[1:],
                                             eps=eps, p_thresh=p_thresh, min_size=min_size, height_lim=hlim,
                                             interactive=False, save=True, max_noise=max_noise, max_clust=max_clust)
        elif method == 'ward':
            f = h5py.File(base_path + h5_name, 'r')
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
                        if n_clusters > max_clust:
                            continue
                        if not os.path.exists(base_path + 'clusters/%s/ward/%d_clusters/'
                                              % (dm_tag[1:], n_clusters)):
                            os.mkdir(base_path + 'clusters/%s/ward/%d_clusters/'
                                     % (dm_tag[1:], n_clusters))
                            arr2csv(base_path + 'clusters/%s/ward/%d_clusters/clusters.txt'
                                    % (dm_tag[1:], n_clusters), lbls, ids, fmt='%s')
                            log = open(base_path + 'clusters/%s/ward/%d_clusters/cluster_settings.txt'
                                       % (dm_tag[1:], n_clusters), 'w')
                            log.write('NormalTest p-thresh:\t%.2E\n' % p_thresh)
                            log.write('Ward Height Lim:\t%d\n' % hlim)
                            log.write('Min Cluster Size:\t%d\n' % min_size)
                            log.close()
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
                                log = open(base_path + 'clusters/%s/ward/%d_clusters_%s/cluster_settings.txt'
                                           % (dm_tag[1:], n_clusters, tag), 'w')
                                log.write('NormalTest p-thresh:\t%.2E\n' % p_thresh)
                                log.write('Ward Height Lim:\t%d\n' % hlim)
                                log.write('Min Cluster Size:\t%d\n' % min_size)
                                log.close()
                                get_cstats(f,
                                           base_path + 'clusters/%s/ward/%d_clusters_%s/' % (dm_tag[1:], n_clusters, tag))
                                cont = False
