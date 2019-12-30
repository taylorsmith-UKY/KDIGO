import numpy as np
import os
import h5py
from utility_funcs import load_csv
from cluster_funcs import clusterCategorizer
from scipy.spatial.distance import pdist
import json

alignment = "popDTW_a65E-02"
dfunc = "braycurtis"
pc = "popCoords"
nClust = 96

merged = True
seedType = "medoid"

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

f = h5py.File(os.path.join(resPath, "stats.h5"), "r")
ids = f['meta']['ids'][:]
died = f['meta']['died_inp'][:]
mk = f['meta']['max_kdigo_win'][:]


cats = ["%s-%s" % (x, y) for x in ["1", "2", "3", "3D"] for y in ["Im", "St", "Ws"]]

kdigos = load_csv(os.path.join(dataPath, 'kdigo_icu_2ptAvg.csv'), ids, int)
days = load_csv(os.path.join(dataPath, 'days_interp_icu_2ptAvg.csv'), ids, int)
scrs = load_csv(os.path.join(dataPath, 'scr_interp_icu_2ptAvg.csv'), ids, float)
icu_windows = load_csv(os.path.join(dataPath, 'icu_admit_discharge.csv'), ids, 'date', struct='dict')
hosp_windows = load_csv(os.path.join(dataPath, 'hosp_admit_discharge.csv'), ids, 'date', struct='dict')

if not merged:
    clusterPath = os.path.join(resPath, "clusters", "14days", "kdigo_icu_2ptAvg_" + alignment,
                               dfunc + "_" + pc, "flat", "%d_clusters" % nClust)
else:
    clusterPath = os.path.join(resPath, "clusters", "14days", "kdigo_icu_2ptAvg_" + alignment,
                               dfunc + "_" + pc, "flat", "%d_clusters" % nClust, "final_merged", seedType)

lbls = load_csv(os.path.join(clusterPath, "clusters.csv"), ids, int)

nlbls, key = clusterCategorizer(mk, kdigos, days, lbls, 14)
print("Category,LowestMortality,HighestMortality,MaxMortDiff,MinMortDiff")
for cat in cats:
    lblgrp = [x for x in np.unique(nlbls) if cat in x]
    morts = []
    for lbl in lblgrp:
        idx = np.where(nlbls == lbl)[0]
        mort = np.sum(died[idx]) / len(idx) * 100
        morts.append(mort)
    diffs = pdist(np.array(morts)[:, None], metric="cityblock")
    maxDiff = np.max(diffs)
    minDiff = np.min(diffs)
    minMort = np.min(morts)
    maxMort = np.max(morts)
    print("%s,%.2f,%.2f,%.2f,%.2f" % (cat, minMort, maxMort, maxDiff, minDiff))
