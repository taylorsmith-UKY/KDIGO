import os
import numpy as np
import json
import h5py
from utility_funcs import arr2csv, load_csv
from sklearn.preprocessing import OneHotEncoder
from dtw_distance import continuous_mismatch, continuous_extension, get_custom_distance_discrete, \
    mismatch_penalty_func, extension_penalty_func
from cluster_funcs import get_dm_tag, merge_clusters


class args:
    alpha = 0.65
    malpha = 0.65
    pdtw = True
    sf = "kdigo_icu_2ptAvg.csv"
    df = "days_interp_icu_2ptAvg.csv"
    dfunc = "braycurtis"
    popcoords = True
    lapType = "none"
    lapVal = 0.0
    nClusters = 26
    seedType = "mean"
    cat = "none"
    clen = 9
    dbaiter = 5
    t_lim = 14


grp_name = 'meta'
statFileName = 'stats.h5'

nFinal = {"1-Im": 6, "1-St": 10,  # "1-Ws": 3,
          "2-Im": 16, "2-St": 8,  # "2-Ws": 8,
          "3-Im": 8, "3-Ws": 4, }  # "3-St": 4, "3D-Ws": 7}

fp = open('kdigo_conf.json', 'r')
conf = json.load(fp)
fp.close()
basePath = conf['basePath']  # Path containing the DATA and RESULTS directories
cohortName = conf['cohortName']  # Will be used for folder name
t_lim = conf['analysisDays']  # How long to consider in the analysis
tRes = conf['timeResolutionHrs']  # Resolution after imputation in hours
analyze = conf['analyze']  # Time period to analyze (hospital/ICU/all)

# Build paths and create if don't already exist
baseDataPath = os.path.join(basePath, 'DATA', 'all_sheets')  # folder containing all raw data
dataPath = os.path.join(basePath, 'DATA', analyze, cohortName)
resPath = os.path.join(basePath, 'RESULTS', analyze, cohortName)
f = h5py.File(os.path.join(resPath, statFileName), 'r')
ids = f[grp_name]['ids'][:]

kdigos = load_csv(os.path.join(dataPath, 'kdigo_icu_2ptAvg.csv'), ids, int)
days = load_csv(os.path.join(dataPath, 'days_interp_icu_2ptAvg.csv'), ids, int)
scrs = load_csv(os.path.join(dataPath, 'scr_interp_icu_2ptAvg.csv'), ids, float)
icu_windows = load_csv(os.path.join(dataPath, 'icu_admit_discharge.csv'), ids, 'date', struct='dict')
hosp_windows = load_csv(os.path.join(dataPath, 'hosp_admit_discharge.csv'), ids, 'date', struct='dict')

tcost_fname = "%s_transition_weights.csv" % args.sf.split('.')[0]
transition_costs = np.loadtxt(os.path.join(dataPath, tcost_fname), delimiter=',', usecols=1, skiprows=1)

if args.popcoords:
    coords = np.array([np.sum(transition_costs[:i]) for i in range(len(transition_costs) + 1)], dtype=float)
else:
    coords = np.array([x for x in range(len(transition_costs) + 1)], dtype=float)

extension = continuous_extension(extension_penalty_func(*transition_costs))
mismatch = continuous_mismatch(mismatch_penalty_func(*transition_costs))
dist = get_custom_distance_discrete(coords, dfunc=args.dfunc, lapVal=args.lapVal, lapType=args.lapType)

dm_tag, dtw_tag = get_dm_tag(args.pdtw, args.alpha, False, args.popcoords, args.dfunc, args.lapVal, args.lapType)
folderName = args.sf.split('.')[0]
folderName += "_" + dtw_tag

dm_path = os.path.join(resPath, 'dm')
dtw_path = os.path.join(resPath, 'dtw')
dm_path = os.path.join(dm_path, '%ddays' % t_lim)
dtw_path = os.path.join(dtw_path, '%ddays' % t_lim)
dm_path = os.path.join(dm_path, folderName)
dtw_path = os.path.join(dtw_path, folderName)

dm = np.load(os.path.join(dm_path, 'kdigo_dm_%s.npy' % dm_tag))
if dm.ndim == 2:
    dm = dm[:, 2]

baseClustPath = os.path.join(resPath, "clusters", "%ddays" % t_lim, folderName,
                             dm_tag, 'flat', '%d_clusters' % args.nClusters)
#
# mergeFolder = os.path.join(baseClustPath, "merged_popDTW_a%dE-02" % (100 * args.alpha),
#                            "center_%s" % args.seedType, "noExt")
#
# lbls = load_csv(os.path.join(baseClustPath, "clusters.csv"), ids, str).astype("<U256")
#
# for cat in list(nFinal.keys()):
#     nClust = nFinal[cat]
#     tpath = os.path.join(mergeFolder, cat, args.seedType, "%d_clusters" % nClust)
#     tlbls = np.loadtxt(os.path.join(tpath, "centers.csv"), delimiter=",", usecols=0, dtype=str)
#     for tlbl in tlbls:
#         if len(tlbl.split("-")) > 1:
#             for olbl in tlbl.split("-"):
#                 idx = np.where(lbls == olbl)[0]
#                 lbls[idx] = tlbl
#
# ohe = OneHotEncoder()
# oh = ohe.fit_transform(lbls[:, None]).toarray()
# nlbls = np.zeros(len(ids), dtype=int)
# for i in range(len(oh)):
#     nlbls[i] = np.where(oh[i, :])[0][0]
#
# if not os.path.exists(os.path.join(baseClustPath, "final_merged")):
#     os.mkdir(os.path.join(baseClustPath, "final_merged"))
# if not os.path.exists(os.path.join(baseClustPath, "final_merged", args.seedType)):
#     os.mkdir(os.path.join(baseClustPath, "final_merged", args.seedType))
#
# arr2csv(os.path.join(baseClustPath, "final_merged", args.seedType, "clusters.csv"), nlbls, ids, fmt='%d')
# arr2csv(os.path.join(baseClustPath, "final_merged", args.seedType, "clusters_origLabels.csv"), lbls, ids, fmt='%s')

# merge_clusters(kdigos, days, dm, os.path.join(baseClustPath, "final_merged", args.seedType), f['meta'], args,
#                mismatch, extension, dist=dist)

merge_clusters(kdigos, days, dm, baseClustPath, f['meta'], args,
               mismatch, extension, dist=dist)