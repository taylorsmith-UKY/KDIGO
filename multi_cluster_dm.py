import numpy as np
from cluster_funcs import cluster_trajectories
from kdigo_funcs import load_dm
import h5py
import os
from scipy.spatial.distance import squareform
import datetime

# PARAMETERS
#################################################################################
data_path = '../DATA/icu/7days_092818/'
base_path = '../RESULTS/icu/7days_092818/'
dm_tags = ['_absmismatch_extension_a1E+00_normBC', '_absmismatch_normBC',
           '_custmismatch_normBC', '_custmismatch_extension_a1E+00_normBC']
h5_name = 'stats.h5'
meta_grp = 'meta'

interactive = False

# t_lims = range(1, 8)[::-2]
t_lims = [7, ]
eps = 0.05                                     # Epsilon threshold for DBSCAN
p_thresh = 1e-200                    # NormalTest threshold to stop splitting
hlim = 20                                           # Height limit for cutting dendrogram
min_size = 50                                      # Minimum cluster size
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
# Methods - Note: regardless of the method chosen, DBSCAN is applied first
#   - dynamic       -   Use a dynamic hierarchical clustering algorithm that considers the distribution of the distances
#                       along with other factors such as the homogeneity of the max KDIGO scores within each cluster,
#                       and the total cluster size.
#   - flat          -   Apply ward's method hierarchical clustering with a flat cut
methods = ['dynamic', 'flat']

#################################################################################

date_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')

f = h5py.File(base_path + h5_name, 'r')
all_ids = f[meta_grp]['ids'][:]
dtd = f[meta_grp]['days_to_death'][:]
mk = f[meta_grp]['max_kdigo'][:]

if not os.path.exists(os.path.join(base_path, 'clusters')):
    os.mkdir(os.path.join(base_path, 'clusters'))

for t_lim in t_lims:
    sel = np.logical_not(dtd <= t_lim)
    dm_sel = np.ix_(sel, sel)
    ids = all_ids[sel]
    max_kdigo = mk[sel]
    if not os.path.exists(os.path.join(base_path, 'clusters', '%ddays' % t_lim)):
        os.mkdir(os.path.join(base_path, 'clusters', '%ddays' % t_lim))
    for dm_tag in dm_tags:
        save_path = os.path.join(base_path, 'clusters', '%ddays' % t_lim, dm_tag[1:])
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if os.path.isfile(base_path + 'dm/%ddays/kdigo_dm%s.npy' % (t_lim, dm_tag)):
            dm = np.load(base_path + 'dm/%ddays/kdigo_dm%s.npy' % (t_lim, dm_tag))
        else:
            dm = load_dm(base_path + 'dm/%ddays/kdigo_dm%s.csv' % (t_lim, dm_tag), ids)
        sqdm = squareform(dm)
        if sqdm.shape[0] > len(ids):
            sqdm = sqdm[dm_sel]
            dm = squareform(sqdm)
        lbls = cluster_trajectories(f, ids, max_kdigo, dm, meta_grp=meta_grp, eps=eps, p_thresh=p_thresh,
                                    min_size=min_size, hom_thresh=hom_thresh, max_size_pct=max_size_pct,
                                    height_lim=hlim, hmethod=methods, data_path=data_path, save=save_path,
                                    interactive=interactive)
