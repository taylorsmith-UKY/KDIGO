import os
import numpy as np
import kdigo_funcs as kf
from scipy.spatial import distance
import h5py

# ------------------------------- PARAMETERS ----------------------------------#
basePath = "../"
t_analyze = 'ICU'
xl_file = "KDIGO_full.xlsx"
timescale = 6  # in hours
id_ref = 'icu_valid_ids.csv'  # specify different file with subset of IDs if desired
incl_0 = False
folder_name = '/7days_071118/'
alphas = [0.25, 0.5, 1.0]
transition_costs = [1.00,   # [0 - 1]
                    2.95,   # [1 - 2]
                    4.71,   # [2 - 3]
                    7.62]   # [3 - 4]
t_lims = range(1, 8)[::-1]

bc_shift = 1        # Value to add to coordinates for BC distance
# With bc_shift=0, the distance from KDIGO 3D to all
# other KDIGO stages is maximal (ie. 1.0). Increaseing
# bc_shift allows discrimination between KDIGO 3D and
# the other KDIGO scores


sort_id = 'STUDY_PATIENT_ID'
sort_id_date = 'SCR_ENTERED'
dataPath = basePath + "DATA/"
outPath = dataPath + t_analyze.lower() + folder_name
resPath = basePath + 'RESULTS/' + t_analyze.lower() + folder_name
inFile = dataPath + xl_file
id_ref = outPath + id_ref
baseline_file = dataPath + 'baselines_1_7-365_mdrd.csv'

f = h5py.File(resPath + 'stats.h5', 'r')
all_ids = f['meta']['ids'][:]
dtd = f['meta']['days_to_death'][:]
f.close()

for t_lim in t_lims:
    tpath = resPath + '%ddays/' % t_lim
    if not os.path.exists(tpath):
        os.mkdir(tpath)
    pt_sel = np.union1d(np.where(np.isnan(dtd))[0], np.where(dtd > t_lim)[0])
    ids = all_ids[pt_sel]

    kdigos = kf.load_csv(outPath + 'kdigo.csv', ids, int)
    days = kf.load_csv(outPath + 'days_interp.csv', ids, int)

    for use_mismatch_penalty in [True, ]:
        for use_extension_penalty in [True, ]:
            if use_extension_penalty:
                alpha_list = alphas
            else:
                alpha_list = [0.0, ]
            for alpha in alphas:
                for dist_flag in ['abs_pop', ]:
                    if use_mismatch_penalty:
                        mismatch = kf.mismatch_penalty_func(*transition_costs)
                        dm_tag = '_custmismatch'
                    else:
                        mismatch = lambda x, y: abs(x - y)
                        dm_tag = '_absmismatch'
                    if use_extension_penalty:
                        extension = kf.extension_penalty_func(*transition_costs)
                        dm_tag += '_extension_a%.0E' % alpha
                    else:
                        extension = lambda x: 0
                    if dist_flag == 'cust_bc':
                        dist = kf.get_custom_braycurtis(*transition_costs, shift=bc_shift)
                        dm_tag += '_custBC'
                    elif dist_flag == 'norm_bc':
                        dist = distance.braycurtis
                        dm_tag += '_normBC'
                    elif dist_flag == 'abs_pop':
                        dist = kf.get_pop_dist(*transition_costs)
                        dm_tag += '_abspop'

                    if not os.path.exists(tpath + 'kdigo_dm' + dm_tag + '.npy'):
                        dm = kf.pairwise_dtw_dist(kdigos, days, ids, tpath + 'kdigo_dm' + dm_tag + '.csv', None,
                                                  mismatch=mismatch,
                                                  extension=extension,
                                                  dist=dist,
                                                  alpha=alpha,
                                                  desc=dm_tag[1:], t_lim=t_lim)
                        np.save(tpath + 'kdigo_dm' + dm_tag, dm)
                    else:
                        print(dm_tag + ' already completed.')
