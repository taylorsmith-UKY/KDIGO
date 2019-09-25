import os
import numpy as np
import json
import h5py
from utility_funcs import arr2csv, load_csv

fp = open('../kdigo_conf.json', 'r')
conf = json.load(fp)
fp.close()
basePath = conf['basePath']         # Path containing the DATA and RESULTS directories
cohortName = conf['cohortName']     # Will be used for folder name
t_lim = conf['analysisDays']        # How long to consider in the analysis
tRes = conf['timeResolutionHrs']    # Resolution after imputation in hours
analyze = conf['analyze']           # Time period to analyze (hospital/ICU/all)

# Build paths and create if don't already exist
baseDataPath = os.path.join(basePath, 'DATA', 'all_sheets')         # folder containing all raw data
dataPath = os.path.join(basePath, 'DATA', analyze, cohortName)
resPath = os.path.join(basePath, 'RESULTS', analyze, cohortName)
f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
ids = f['meta_avg']['ids'][:]

lblPath = os.path.join(resPath, 'clusters', '14days', 'kdigo_icu_2ptAvg_popDTW_a35E-02',
                       'braycurtis_popCoords_aggLap_lap1E+00', 'flat', '96_clusters')
olbls = load_csv(os.path.join(lblPath, 'clusters.csv'), ids, str)
olbls = np.array(olbls, dtype='|S20').astype(str)

centers = load_csv(os.path.join(lblPath, 'dba_centers', 'popDTW_a35E-02', 'mean', 'centers.csv'), None,
                   struct='dict', id_dtype=str)

startPath = os.path.join(lblPath, "merged_popDTW_a35E-02", "center_mean", "extWeight_35E-02_normExt",
                         "extWeight_30E-02_normExt")

mergeStop = {'1-Im': 3,
             '1-St': 7,
             '1-Ws': 3,
             '2-Im': 14,
             '2-St': 11,
             '2-Ws': 8,
             '3-Im': 10,
             '3-St': 5,
             '3-Ws': 8,
             '3D-Im': 3,
             '3D-St': 0,
             '3D-Ws': 5}

for cat in list(mergeStop):
    if mergeStop[cat] == 0:
        continue
    nlbls = np.unique(np.loadtxt(os.path.join(startPath, cat, 'mean', '%d_clusters' % mergeStop[cat], 'clusters.csv',),
                      delimiter=',', usecols=1, dtype=str))
    for lbl in nlbls:
        for part in lbl.split('-'):
            idx = np.where(olbls == part)[0]
            olbls[idx] = lbl

# arr2csv(os.path.join(startPath, 'merged_clusters.csv'), olbls, ids, str)
