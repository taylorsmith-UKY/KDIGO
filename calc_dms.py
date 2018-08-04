import os
import numpy as np
import kdigo_funcs as kf
from scipy.spatial import distance

# ------------------------------- PARAMETERS ----------------------------------#
basePath = "../"
t_analyze = 'ICU'
xl_file = "KDIGO_full.xlsx"
timescale = 6  # in hours
id_ref = 'icu_valid_ids.csv'  # specify different file with subset of IDs if desired
incl_0 = False
h5_name = 'kdigo_dm.h5'
folder_name = '/7days_071118/'
alphas = [0.25, 0.5, 1.0]
transition_costs = [1.00,   # [0 - 1]
                    2.95,   # [1 - 2]
                    4.71,   # [2 - 3]
                    7.62]   # [3 - 4]

use_extension_penalty = True
use_mismatch_penalty = True
use_custom_braycurtis = True

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

ids = np.loadtxt(id_ref, dtype=int)
kdigos = kf.load_csv(outPath + 'kdigo.csv', ids, int)

mk = np.zeros(len(ids))
for i in range(len(ids)):
    mk[i] = np.max(kdigos[i])

aki_idx = np.where(mk)[0]

aki_ids = ids[aki_idx]
aki_kdigos = kf.load_csv(outPath + 'kdigo.csv', aki_ids, int)

for alpha in alphas:
    for use_dic_dtw in [dtw_dic, None]:
        for use_dic_dist in [bc_dic, None]:
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
            if use_custom_braycurtis:
                bc_dist = kf.get_custom_braycurtis(*transition_costs, shift=bc_shift)
                dm_tag += '_custBC'
            else:
                bc_dist = distance.braycurtis
                dm_tag += '_normBC'

            if not os.path.exists(resPath + 'kdigo_dm' + dm_tag + '.csv'):
                dm = kf.pairwise_dtw_dist(aki_kdigos, aki_ids, resPath + 'kdigo_dm' + dm_tag + '.csv', None,
                                          incl_0=False, mismatch=mismatch,
                                          extension=extension,
                                          bc_dist=bc_dist,
                                          alpha=alpha)
                np.save(resPath + 'kdigo_dm' + dm_tag, dm)


