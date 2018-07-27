import os
import numpy as np
import kdigo_funcs as kf

# ------------------------------- PARAMETERS ----------------------------------#
basePath = "../"
t_analyze = 'ICU'
xl_file = "KDIGO_full.xlsx"
timescale = 6  # in hours
id_ref = 'icu_valid_ids.csv'  # specify different file with subset of IDs if desired
incl_0 = False
h5_name = 'kdigo_dm.h5'
folder_name = '/7days_071118/'
alphas = [2.0, 4.0]
transition_costs = [1.5,    # [0 - 1]
                    1.8,    # [1 - 2]
                    2.75,   # [2 - 3]
                    4.25]   # [3 - 4]

sort_id = 'STUDY_PATIENT_ID'
sort_id_date = 'SCR_ENTERED'
dataPath = basePath + "DATA/"
outPath = dataPath + t_analyze.lower() + folder_name
resPath = basePath + 'RESULTS/' + t_analyze.lower() + folder_name
inFile = dataPath + xl_file
id_ref = outPath + id_ref
baseline_file = dataPath + 'baselines_1_7-365_mdrd.csv'
h5_fname = resPath + '/' + h5_name

ids = np.loadtxt(id_ref, dtype=int)
_, kdigos = kf.load_csv(outPath + 'kdigo.csv', ids, int)

mk = np.zeros(len(ids))
for i in range(len(ids)):
    mk[i] = np.max(kdigos[i])

aki_idx = np.where(mk)[0]

aki_ids = ids[aki_idx]
_, aki_kdigos = kf.load_csv(outPath + 'kdigo.csv', aki_ids, int)

dtw_dic = {}
bc_dic = {}

s = 0
dtw_dic[0] = 0
for i in range(len(transition_costs)):
    s += transition_costs[i]
    dtw_dic[i + 1] = s

bc_dic[0] = s
for i in range(len(transition_costs) - 1):
    s -= transition_costs[i]
    bc_dic[i + 1] = s
bc_dic[len(transition_costs)] = 0

for alpha in alphas:
    for use_dic_dtw in [dtw_dic, None]:
        for use_dic_dist in [bc_dic, None]:
            if use_dic_dtw:
                dm_tag = '_cDTW'
            else:
                dm_tag = '_normDTW'
            if use_dic_dist:
                dm_tag += 'cdist'
            else:
                dm_tag += '_normdist'
            dm_tag += '_a%d' % alpha

            if not os.path.exists(resPath + 'kdigo_dm' + dm_tag + '.csv'):
                dm = kf.pairwise_dtw_dist(aki_kdigos, aki_ids, resPath + 'kdigo_dm' + dm_tag + '.csv', None,
                                          incl_0=False, alpha=alpha, dtw_dic=use_dic_dtw, bc_dic=use_dic_dist)
                np.save(resPath + 'kdigo_dm' + dm_tag, dm)


