import os
import numpy as np
from kdigo_funcs import load_csv, extension_penalty_func, mismatch_penalty_func,\
                        pairwise_dtw_dist, get_custom_braycurtis
from scipy.spatial import distance
import h5py

# ------------------------------- PARAMETERS ----------------------------------#
basePath = "../"
t_analyze = 'ICU'
xl_file = "KDIGO_full.xlsx"
timescale = 6  # in hours
incl_0 = False
folder_name = '/7days_092818/'
ext_alpha = 1.0               # weight of the extension penalty relative to mismatch

# The transition costs are obtained from the population dynamics for the relative frequency of
# transitions between different KDIGO scores
transition_costs = [1.00,   # [0 - 1]
                    2.95,   # [1 - 2]
                    4.71,   # [2 - 3]
                    7.62]   # [3 - 4]

# Coordinates to use for Bray-Curtis distance (if custom)
bc_coords = np.array([1, 2, 3, 4, 5])

# Duration of data to use for calculation
t_lims = range(1, 8)[::-1]

bc_shift = 1        # Value to add to coordinates for BC distance
# With bc_shift=0, the distance from KDIGO 3D to all
# other KDIGO stages is maximal (ie. 1.0). Increaseing
# bc_shift allows discrimination between KDIGO 3D and
# the other KDIGO scores
# -----------------------------------------------------------------------------#

# Build path-names
dataPath = basePath + "DATA/" + t_analyze.lower() + folder_name
resPath = basePath + 'RESULTS/' + t_analyze.lower() + folder_name
baseline_file = dataPath + 'baselines_1_7-365_mdrd.csv'

# Load patient IDs and days-to-death (in case any patients died within the
# analysis windows defined by t_lims so that they can be excluded
f = h5py.File(resPath + 'stats.h5', 'r')
all_ids = f['meta']['ids'][:]
dtd = f['meta']['days_to_death'][:]
f.close()

for t_lim in t_lims:
    tpath = resPath + '%ddays/' % t_lim
    if not os.path.exists(tpath):
        os.mkdir(tpath)
    # remove any patients who died < t_lim days after admission
    pt_sel = np.union1d(np.where(np.isnan(dtd))[0], np.where(dtd > t_lim)[0])
    ids = all_ids[pt_sel]

    # load corresponding KDIGO scores and their associated days post admission
    kdigos = load_csv(dataPath + 'kdigo.csv', ids, int)
    days = load_csv(dataPath + 'days_interp.csv', ids, int)

    for use_mismatch_penalty in [True, False]:
        for use_extension_penalty in [True, False,]:
            # Build filename to distinguish different matrices and specify the
            # mismatch and extension penalties
            if use_extension_penalty:
                alpha = ext_alpha
            else:
                alpha = 0
            for dist_flag in ['norm_bc', ]:
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
                if dist_flag == 'cust_bc':
                    # dist = kf.get_custom_braycurtis(*transition_costs, shift=bc_shift)
                    dist = get_custom_braycurtis(bc_coords)
                    dm_tag += '_custBC'
                elif dist_flag == 'norm_bc':
                    dist = distance.braycurtis
                    dm_tag += '_normBC'

                # Compute pair-wise DTW and distance calculation for the included
                # patients. If specified, also saves the results of the pair-wise
                # DTW alignment, which can then save time for subsequent distance
                # calculation.
                if not os.path.exists(tpath + 'kdigo_dm' + dm_tag + '.npy'):
                    dm = pairwise_dtw_dist(kdigos, days, ids, tpath + 'kdigo_dm2' + dm_tag + '.csv',
                                           tpath + 'dtw_log' + dm_tag + '.csv',
                                           mismatch=mismatch,
                                           extension=extension,
                                           dist=dist,
                                           alpha=alpha, t_lim=t_lim)

                    np.save(tpath + 'kdigo_dm' + dm_tag, dm)
                else:
                    print(dm_tag + ' already completed.')
