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
dm_tags = ['_absmismatch_normBC', ]
            # '_absmismatch_extension_a5E-01_normBC', '_absmismatch_extension_a5E-01_custBC',
            # '_absmismatch_extension_a1E+00_normBC', '_absmismatch_extension_a1E+00_custBC',
            # '_custmismatch_normBC', '_custmismatch_custBC',
            # '_custmismatch_extension_a2E-01_normBC', '_custmismatch_extension_a2E-01_custBC',
            # '_custmismatch_extension_a5E-01_normBC', '_custmismatch_extension_a5E-01_custBC']

h5_name = 'stats.h5'

t_lims = range(1, 7)

eps_l = [0.01, 0.05, 0.075]                                      # Epsilon threshold for DBSCAN
p_thresh_l = [0.1, 1e-2, 1e-5, 1e-10, ]                    # NormalTest threshold to stop splitting
hlim_l = [3, 4]                                            # Height limit for cutting dendrogram
min_size_l = [15, 30]                                      # Minimum cluster size
# Note: DBSCAN clusters smaller than min_size are grouped into the noise cluster,
# whereas min_size is used as a stopping criteria when splitting ward's method clusters.

max_noise = 0.1                                            # As fraction of total number patients
# If > max_noise patients designated as noise (only applicable for DBSCAN), discard results

max_clust = 15
# Which clustering algorithm to apply
# Methods
#   - composite     -   First apply DBSCAN, followed by modified Ward's method
#   - ward          -   Apply modified Ward's method on all patients
methods = ('composite', 'ward')


#################################################################################

date_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')

f = h5py.File(base_path + h5_name, 'r')
all_ids = f['meta']['ids'][:]
dtd = f['meta']['days_to_death'][:]
f.close()

for t_lim in t_lims:
    sel = np.union1d(np.where(np.isnan(dtd)), np.where(dtd >= t_lim))
    ids = all_ids[sel]
    if not os.path.exists(base_path + 'clusters/%ddays/' % t_lim):
        os.mkdir(base_path + 'clusters/%ddays/' % t_lim)
    for dm_tag in dm_tags:
        if not os.path.exists(base_path + 'clusters/%ddays/%s/' % (t_lim, dm_tag[1:])):
            os.mkdir(base_path + 'clusters/%ddays/%s/' % (t_lim, dm_tag[1:]))
        if os.path.isfile(base_path + '%ddays/kdigo_dm%s.npy' % (t_lim, dm_tag)):
            dm = np.load(base_path + '%ddays/kdigo_dm%s.npy' % (t_lim, dm_tag))
        else:
            dm = load_dm(base_path + 'kdigo_dm%s.csv' % dm_tag, ids)
        interactive = False
        for method in methods:
            if method == 'composite':
                # eps = 0.05
                for eps in eps_l:
                    for p_thresh in p_thresh_l:
                        for hlim in hlim_l:
                            for min_size in min_size_l:
                                try:
                                    lbls, eps = dist_cut_cluster(base_path + h5_name, dm, ids, path=base_path + 'clusters/%ddays/%s/' % (t_lim, dm_tag[1:]),
                                                     eps=eps, p_thresh=p_thresh, min_size=min_size, height_lim=hlim,
                                                     interactive=interactive, save=True, max_noise=max_noise, max_clust=max_clust)
                                    # interactive = False
                                    eps = eps
                                except:
                                    interactive = False
            elif method == 'ward':
                f = h5py.File(base_path + h5_name, 'r')
                link = ward(dm)
                sqdm = squareform(dm)
                if not os.path.exists(base_path + 'clusters/%ddays/%s/ward/' % (t_lim, dm_tag[1:])):
                    os.mkdir(base_path + 'clusters/%ddays/%s/ward/' % (t_lim, dm_tag[1:]))
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
                            if not os.path.exists(base_path + 'clusters/%ddays/%s/ward/%d_clusters/'
                                                  % (t_lim, dm_tag[1:], n_clusters)):
                                os.mkdir(base_path + 'clusters/%ddays/%s/ward/%d_clusters/'
                                         % (t_lim, dm_tag[1:], n_clusters))
                                arr2csv(base_path + 'clusters/%ddays/%s/ward/%d_clusters/clusters.txt'
                                        % (t_lim, dm_tag[1:], n_clusters), lbls, ids, fmt='%s')
                                log = open(base_path + 'clusters/%ddays/%s/ward/%d_clusters/cluster_settings.txt'
                                           % (t_lim, dm_tag[1:], n_clusters), 'w')
                                log.write('NormalTest p-thresh:\t%.2E\n' % p_thresh)
                                log.write('Ward Height Lim:\t%d\n' % hlim)
                                log.write('Min Cluster Size:\t%d\n' % min_size)
                                log.close()
                                get_cstats(f, base_path + 'clusters/%ddays/%s/ward/%d_clusters/' % (t_lim, dm_tag[1:], n_clusters))
                            ref = load_csv(base_path + 'clusters/%ddays/%s/ward/%d_clusters/clusters.txt'
                                           % (t_lim, dm_tag[1:], n_clusters), ids, str)
                            if np.all(ref == lbls):
                                cont = False
                            else:
                                cont = True
                                tag = 'a'
                            while cont:
                                if os.path.exists(base_path + 'clusters/%ddays/%s/ward/%d_clusters_%s/'
                                                  % (t_lim, dm_tag[1:], n_clusters, tag)):
                                    ref = load_csv(base_path + 'clusters/%ddays/%s/ward/%d_clusters_%s/clusters.txt'
                                                   % (t_lim, dm_tag[1:], n_clusters, tag), ids, str)
                                    if np.all(ref == lbls):
                                        cont = False
                                    else:
                                        tag = chr(ord(tag) + 1)
                                else:
                                    os.makedirs(base_path + 'clusters/%ddays/%s/ward/%d_clusters_%s/'
                                                % (t_lim, dm_tag[1:], n_clusters, tag))
                                    arr2csv(base_path + 'clusters/%ddays/%s/ward/%d_clusters_%s/clusters.txt'
                                            % (t_lim, dm_tag[1:], n_clusters, tag), lbls, ids, fmt='%s')
                                    log = open(base_path + 'clusters/%ddays/%s/ward/%d_clusters_%s/cluster_settings.txt'
                                               % (t_lim, dm_tag[1:], n_clusters, tag), 'w')
                                    log.write('NormalTest p-thresh:\t%.2E\n' % p_thresh)
                                    log.write('Ward Height Lim:\t%d\n' % hlim)
                                    log.write('Min Cluster Size:\t%d\n' % min_size)
                                    log.close()
                                    get_cstats(f,
                                               base_path + 'clusters/%ddays/%s/ward/%d_clusters_%s/' % (t_lim, dm_tag[1:], n_clusters, tag))
                                    cont = False
