import os
import numpy as np
from kdigo_funcs import load_csv, extension_penalty_func, mismatch_penalty_func, \
    pairwise_dtw_dist, get_custom_braycurtis, get_euclidean_norm, get_cityblock_norm
import h5py

# ------------------------------- PARAMETERS ----------------------------------#
basePath = "/Volumes/GoogleDrive/My Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/"

dataPath = os.path.join(basePath, 'DATA', 'icu', '7days_041719/')
resPath = os.path.join(basePath, 'RESULTS', 'icu', '7days_041719/')

# DTW parameters
use_mismatch_penalty = True
use_extension_penalty = True
ext_alpha = 1.0               # weight of the extension penalty relative to mismatch

# Distance parameters
dfunc = 'braycurtis'    # 'braycurtis', 'euclidean'
# Coordinates
popvals = True  # False uses normal KDIGO scores

# The transition costs are obtained from the population dynamics for the relative frequency of
# transitions between different KDIGO scores
transition_costs = [1.00,   # [0 - 1]
                    2.73,   # [1 - 2]
                    4.36,   # [2 - 3]
                    6.74]   # [3 - 4]

# Duration of data to use for calculation
t_lim = 7

# -----------------------------------------------------------------------------#
for popvals in [False, True]:
    if popvals:
        bc_coords = np.array([np.sum(transition_costs[:i]) for i in range(len(transition_costs) + 1)], dtype=float)
        coord_tag = '_popcoord'
    else:
        bc_coords = np.array([x for x in range(len(transition_costs) + 1)], dtype=float)
        coord_tag = ''

    # Load patient IDs
    f = h5py.File(resPath + 'stats.h5', 'r')
    ids = f['meta']['ids'][:]
    f.close()

    dm_path = os.path.join(resPath, 'dm')
    dtw_path = os.path.join(resPath, 'dtw')
    if not os.path.exists(dm_path):
        os.mkdir(dm_path)
    if not os.path.exists(dtw_path):
        os.mkdir(dtw_path)

    dm_path = os.path.join(dm_path, '%ddays' % t_lim)
    dtw_path = os.path.join(dtw_path, '%ddays' % t_lim)
    if not os.path.exists(dm_path):
        os.mkdir(dm_path)
    if not os.path.exists(dtw_path):
        os.mkdir(dtw_path)

    # load corresponding KDIGO scores and their associated days post admission
    kdigos = load_csv(dataPath + 'kdigo.csv', ids, int)
    days = load_csv(dataPath + 'days_interp.csv', ids, int)

    # Build filename to distinguish different matrices and specify the
    # mismatch and extension penalties
    if use_extension_penalty:
        alpha = ext_alpha
    else:
        alpha = 0
    if use_mismatch_penalty:
        # mismatch penalty derived from population dynamics
        mismatch = mismatch_penalty_func(*transition_costs)
        dm_tag = '_custmismatch'
    else:
        # absolute difference in KDIGO scores
        mismatch = lambda x, y: abs(x - y)
        dm_tag = '_absmismatch'
    if use_extension_penalty:
        # mismatch penalty derived from population dynamics
        extension = extension_penalty_func(*transition_costs)
        dm_tag += '_extension_a%.0E' % alpha
    else:
        # no extension penalty
        extension = lambda x: 0

    dtw_tag = dm_tag
    if dfunc == 'braycurtis':
        dist = get_custom_braycurtis(bc_coords)
        dm_tag += '_normBC'
    elif dfunc == 'euclidean':
        dist = get_euclidean_norm(bc_coords)
        dm_tag += '_euclidean'
    elif dfunc == 'cityblock':
        dist = get_cityblock_norm(bc_coords)
        dm_tag += '_cityblock'
    dm_tag += coord_tag

    # Compute pair-wise DTW and distance calculation for the included
    # patients. If specified, also saves the results of the pair-wise
    # DTW alignment, which can then save time for subsequent distance
    # calculation.
    if not os.path.exists(dm_path + '/kdigo_dm' + dm_tag + '.npy'):
        # if dfunc == 'braycurtis':
        dm = pairwise_dtw_dist(kdigos, days, ids, dm_path + '/kdigo_dm' + dm_tag + '.csv',
                               dtw_path + '/kdigo_dtwlog' + dtw_tag + '.csv',
                               mismatch=mismatch,
                               extension=extension,
                               dist=dist,
                               alpha=alpha, t_lim=t_lim)
        # elif dfunc == 'euclidean':
        #     dm = pairwise_parallel_dist_euclidean(ids, kdigos, days, dtw_path + '/kdigo_dtwlog' + dtw_tag + '.csv',
        #                                           t_lim, bc_coords)
        # elif dfunc == 'cityblock':
        #     dm = pairwise_parallel_dist_cityblock(ids, kdigos, days, dtw_path + '/kdigo_dtwlog' + dtw_tag + '.csv',
        #                                           t_lim, bc_coords)
        np.save(dm_path + '/kdigo_dm' + dm_tag, dm)
    else:
        print(dm_tag + ' already completed.')
