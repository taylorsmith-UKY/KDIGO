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
base_path = '../RESULTS/icu/7days_091718/'
dm_tags = ['_custmismatch_extension_a1E+00_normBC', ]#'_custmismatch_normBC',
           #'_absmismatch_extension_a1E+00_normBC', '_absmismatch_normBC']
# dm_tags = ['_custmismatch_extension_a1E+00_custBC_new']
cluster_folder = 'clusters'
h5_name = 'stats.h5'
meta_grp = 'meta'

interactive = True

# t_lims = range(1, 8)[::-2]
t_lims = [7, ]
eps = 0.05                                     # Epsilon threshold for DBSCAN
p_thresh = 1e-150                    # NormalTest threshold to stop splitting
hlim = 20                                           # Height limit for cutting dendrogram0
min_size = 20                                      # Minimum cluster size
# Note: DBSCAN clusters smaller than min_size are grouped into the noise cluster,
# whereas min_size is used as a stopping criteria when splitting ward's method clusters.

# As fraction of total number patients
max_noise = 0.1
# If > max_noise patients designated as noise (only applicable for DBSCAN), discard results

# If cluster size is greater than 20% of the total population, split it
max_size_pct = 0.2

# if number patients with max kdigo = mode of the cluster is
# less than hom_thresh % of the cluster, divide it
hom_thresh = 0.9

max_clust = 50
# Which clustering algorithm to apply
# Methods
#   - composite     -   First apply DBSCAN, followed by modified Ward's method
#   - ward          -   Apply modified Ward's method on all patients
methods = ('composite',)



#################################################################################

date_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')

f = h5py.File(base_path + h5_name, 'r')
all_ids = f[meta_grp]['ids'][:]
dtd = f[meta_grp]['days_to_death'][:]
mk = f[meta_grp]['max_kdigo'][:]
f.close()

if not os.path.exists(os.path.join(base_path, cluster_folder)):
    os.mkdir(os.path.join(base_path, cluster_folder))

for t_lim in t_lims:
    sel = np.union1d(np.where(np.isnan(dtd)), np.where(dtd > t_lim))
    dm_sel = np.ix_(sel, sel)
    ids = all_ids[sel]
    max_kdigo = mk[sel]
    if not os.path.exists(base_path + '%s/%ddays/' % (cluster_folder, t_lim)):
        os.mkdir(base_path + '%s/%ddays/' % (cluster_folder, t_lim))
    for dm_tag in dm_tags:
        if not os.path.exists(base_path + '%s/%ddays/%s/' % (cluster_folder, t_lim, dm_tag[1:])):
            os.mkdir(base_path + '%s/%ddays/%s/' % (cluster_folder, t_lim, dm_tag[1:]))
        if os.path.isfile(base_path + 'dm/%ddays/kdigo_dm%s.npy' % (t_lim, dm_tag)):
            dm = np.load(base_path + 'dm/%ddays/kdigo_dm%s.npy' % (t_lim, dm_tag))
        else:
            dm = load_dm(base_path + 'dm/%ddays/kdigo_dm%s.csv' % (t_lim, dm_tag), ids)
        sqdm = squareform(dm)
        if sqdm.shape[0] > len(ids):
            sqdm = sqdm[dm_sel]
            dm = squareform(sqdm)
        for method in methods:
            if method == 'composite':
                (lbls, eps, p_thresh,
                 height_lim, max_size_pct,
                 hom_thresh, min_size) = dist_cut_cluster(base_path + h5_name, sqdm, ids, max_kdigo,
                                                          path=base_path + '%s/%ddays/%s/' % (cluster_folder, t_lim, dm_tag[1:]),
                                                          eps=eps, p_thresh=p_thresh, min_size=min_size,
                                                          height_lim=hlim, interactive=interactive, save=True,
                                                          max_noise=max_noise, max_clust=max_clust,
                                                          hom_thresh=hom_thresh, max_size_pct=max_size_pct)
            elif method == 'ward':
                f = h5py.File(base_path + h5_name, 'r')
                link = ward(dm)
                if not os.path.exists(base_path + '%s/%ddays/%s/ward/' % (cluster_folder, t_lim, dm_tag[1:])):
                    os.mkdir(base_path + '%s/%ddays/%s/ward/' % (cluster_folder, t_lim, dm_tag[1:]))
                    root = to_tree(link)
                    lbls = np.ones(len(ids), dtype=int).astype(str)
                    lbls = dist_cut_tree(root, lbls, '1', sqdm, p_thresh, max_kdigo=max_kdigo, min_size=min_size, height_lim=hlim,
                                         hom_thresh=hom_thresh, max_size=max_size_pct)
                    lbl_names = np.unique(lbls)
                    n_clusters = len(lbl_names)
                    if n_clusters > max_clust:
                        continue
                    if not os.path.exists(base_path + '%s/%ddays/%s/ward/%d_clusters/'
                                          % (cluster_folder, t_lim, dm_tag[1:], n_clusters)):
                        os.mkdir(base_path + '%s/%ddays/%s/ward/%d_clusters/'
                                 % (cluster_folder, t_lim, dm_tag[1:], n_clusters))
                        arr2csv(base_path + '%s/%ddays/%s/ward/%d_clusters/clusters.txt'
                                % (cluster_folder, t_lim, dm_tag[1:], n_clusters), lbls, ids, fmt='%s')
                        log = open(base_path + '%s/%ddays/%s/ward/%d_clusters/cluster_settings.txt'
                                   % (cluster_folder, t_lim, dm_tag[1:], n_clusters), 'w')
                        log.write('NormalTest p-thresh:\t%.2E\n' % p_thresh)
                        log.write('Ward Height Lim:\t%d\n' % hlim)
                        log.write('Min Cluster Size:\t%d\n' % min_size)
                        log.close()
                        if len(lbls) == len(all_ids):
                            get_cstats(f, base_path + '%s/%ddays/%s/ward/%d_clusters/' % (cluster_folder, t_lim, dm_tag[1:], n_clusters), ids=all_ids)
                        else:
                            get_cstats(f, base_path + '%s/%ddays/%s/ward/%d_clusters/' % (
                            cluster_folder, t_lim, dm_tag[1:], n_clusters), ids=ids)
                    ref = load_csv(base_path + '%s/%ddays/%s/ward/%d_clusters/clusters.txt'
                                   % (cluster_folder, t_lim, dm_tag[1:], n_clusters), ids, str)
                    if np.all(ref == lbls):
                        cont = False
                    else:
                        cont = True
                        tag = 'a'
                    while cont:
                        if os.path.exists(base_path + '%s/%ddays/%s/ward/%d_clusters_%s/'
                                          % (cluster_folder, t_lim, dm_tag[1:], n_clusters, tag)):
                            ref = load_csv(base_path + '%s/%ddays/%s/ward/%d_clusters_%s/clusters.txt'
                                           % (cluster_folder, t_lim, dm_tag[1:], n_clusters, tag), ids, str)
                            if np.all(ref == lbls):
                                cont = False
                            else:
                                tag = chr(ord(tag) + 1)
                        else:
                            os.makedirs(base_path + '%s/%ddays/%s/ward/%d_clusters_%s/'
                                        % (cluster_folder, t_lim, dm_tag[1:], n_clusters, tag))
                            arr2csv(base_path + '%s/%ddays/%s/ward/%d_clusters_%s/clusters.txt'
                                    % (cluster_folder, t_lim, dm_tag[1:], n_clusters, tag), lbls, ids, fmt='%s')
                            log = open(base_path + '%s/%ddays/%s/ward/%d_clusters_%s/cluster_settings.txt'
                                       % (cluster_folder, t_lim, dm_tag[1:], n_clusters, tag), 'w')
                            log.write('NormalTest p-thresh:\t%.2E\n' % p_thresh)
                            log.write('Ward Height Lim:\t%d\n' % hlim)
                            log.write('Min Cluster Size:\t%d\n' % min_size)
                            log.close()
                            get_cstats(f,
                                       base_path + '%s/%ddays/%s/ward/%d_clusters_%s/' % (cluster_folder, t_lim, dm_tag[1:], n_clusters, tag), ids=ids)
                            cont = False
