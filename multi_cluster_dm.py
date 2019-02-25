import numpy as np
from cluster_funcs import cluster_trajectories
import h5py
import os
from scipy.spatial.distance import squareform
import datetime
import shutil

# PARAMETERS
#################################################################################
basePath = "/Volumes/GoogleDrive/My Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/"

dataPath = os.path.join(basePath, 'DATA', 'icu', '7days_021319/')
resPath = os.path.join(basePath, 'RESULTS', 'icu', '7days_021319/')
# dm_tags = ['zeropad_normBC', 'zeropad_classicEuclidean', 'zeropad_classicCityblock',
#            'absmismatch_normBC', 'absmismatch_normBC_popcoord',
#            'absmismatch_normBC_popcoord_shift25E-02', 'absmismatch_normBC_popcoord_shift5E-01',
#            'absmismatch_normBC_popcoord_shift1',
#            'custmismatch_extension_a1E+00_normBC', 'custmismatch_extension_a1E+00_normBC_popcoord',
#            'custmismatch_extension_a1E+00_normBC_popcoord_shift25E-02',
#            'custmismatch_extension_a1E+00_normBC_popcoord_shift5E-01',
#            'custmismatch_extension_a1E+00_normBC_popcoord_shift1',
#            'custmismatch_extension_a1E+00_euclidean',
#            'custmismatch_extension_a1E+00_euclidean_popcoord',
#            'custmismatch_extension_a1E+00_cityblock',
#            'custmismatch_extension_a1E+00_cityblock_popcoord',]
dm_tags = ['absmismatch_cityblock', 'absmismatch_cityblock_popcoord', ] #  'absmismatch_euclidean', 'absmismatch_euclidean_popcoord']

h5_name = 'stats.h5'
meta_grp = 'meta'

interactive = False
skip_db = True
verbose = False
plot_daily = False
only_flat = True

# t_lims = range(1, 8)[::-2]
t_lim = 7
eps = 0.1                                   # Epsilon threshold for DBSCAN
p_threshs = [1E-200]                    # NormalTest threshold to stop splitting
hlims = [6,]                                           # Height limit for cutting dendrogram
min_sizes = [50, ]                                      # Minimum cluster size
hom_threshs = [0.8, ]
# Note: DBSCAN clusters smaller than min_size are grouped into the noise cluster,
# whereas min_size is used as a stopping criteria when splitting ward's method clusters.

# As fraction of total number patients
max_noise = 0.1
# If > max_noise patients designated as noise (only applicable for DBSCAN), discard results

# If cluster size is greater than 20% of the total population, split it
max_size_pct = 0.2

# number of clusters to extract if method is flat
n_clusters = range(2, 19)

max_clust = 50
# Which clustering algorithm to apply
# Methods - Note: regardless of the method chosen, DBSCAN is applied first
#   - dynamic       -   Use a dynamic hierarchical clustering algorithm that considers the distribution of the distances
#                       along with other factors such as the homogeneity of the max KDIGO scores within each cluster,
#                       and the total cluster size.
#   - flat          -   Apply ward's method hierarchical clustering with a flat cut

#################################################################################

date_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')

f = h5py.File(resPath + h5_name, 'r')
ids = f[meta_grp]['ids'][:]
max_kdigo = f[meta_grp]['max_kdigo_7d'][:]
dtd = f[meta_grp]['days_to_death'][:]
pt_sel = np.union1d(np.where(np.isnan(dtd))[0], np.where(dtd >= 2)[0])
dm_sel = np.ix_(pt_sel, pt_sel)

ids = ids[pt_sel]
max_kdigo = max_kdigo[pt_sel]

if not os.path.exists(os.path.join(resPath, 'clusters')):
    os.mkdir(os.path.join(resPath, 'clusters'))

if not os.path.exists(os.path.join(resPath, 'clusters', '%ddays' % t_lim)):
    os.mkdir(os.path.join(resPath, 'clusters', '%ddays' % t_lim))
for dm_tag in dm_tags:
    save_path = os.path.join(resPath, 'clusters', '%ddays' % t_lim, dm_tag)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, 'split_by_kdigo')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if os.path.isfile(resPath + 'dm/%ddays/kdigo_dm_%s.npy' % (t_lim, dm_tag)):
        try:
            dm = np.load(resPath + 'dm/%ddays/kdigo_dm_%s.npy' % (t_lim, dm_tag))[:, 2]
        except IndexError:
            dm = np.load(resPath + 'dm/%ddays/kdigo_dm_%s.npy' % (t_lim, dm_tag))
    else:
        dm = np.loadtxt(resPath + 'dm/%ddays/kdigo_dm_%s.csv' % (t_lim, dm_tag), delimiter=',', usecols=2)
    sqdm = squareform(dm)[dm_sel]
    idx = np.where(max_kdigo == 1)[0]
    tdm_sel = np.ix_(idx, idx)
    tpath = os.path.join(save_path, 'kdigo_1')
    if not os.path.exists(tpath):
        os.mkdir(tpath)
    # if os.path.exists(os.path.join(tpath, 'flat', '1_dbscan_clusters')):
    #     continue
    eps = cluster_trajectories(f, ids[idx], max_kdigo[idx], sqdm[tdm_sel], meta_grp=meta_grp, eps=eps, p_thresh_l=p_threshs,
                                min_size_l=min_sizes, hom_thresh_l=hom_threshs, max_size_pct=max_size_pct,
                                n_clusters_l=n_clusters,
                                height_lim_l=hlims, data_path=dataPath, save=tpath,
                                interactive=interactive, v=verbose, plot_daily=plot_daily, only_flat=only_flat,
                                skip_db=skip_db)
