import os
import numpy as np
import json
import h5py
import argparse
from utility_funcs import load_csv
from cluster_funcs import clusterCategorizer
from scipy.spatial.distance import pdist

parser = argparse.ArgumentParser(description='Preprocess Data and Construct KDIGO Vectors.')
parser.add_argument('--config_file', action='store', type=str, dest='cfname',
                    default='kdigo_conf.json')
parser.add_argument('--config_path', action='store', type=str, dest='cfpath',
                    default='')
parser.add_argument('--averagePoints', '-agvPts', action='store', type=int, dest='avgpts',
                    default=2)
parser.add_argument('--sequence_file', '-sf', action='store', type=str, dest='sf', default='kdigo_icu.csv')
parser.add_argument('--meta_group', '-meta', action='store', type=str, dest='meta',
                    default='meta')
parser.add_argument('--data_file', '-df', action='store', type=str, dest='df',
                    default='stats.h5')
parser.add_argument('--alignmentTag', '-align', action='store', type=str, dest='align',
                    default='popDTW_a6500E-4')
parser.add_argument('--distanceFunction', '-dfunc', action='store', type=str, dest='dfunc',
                    default='braycurtis')
parser.add_argument('--n_clust', '-n', action='store', type=int, dest='n_clust',
                    default=96)
parser.add_argument('--populationCoords', '-popcoordsw', '-pc', action='store_true', dest='popcoords')
parser.add_argument('--classificationModel', '-class', action='store', type=str, dest='classModel',
                    default='log', choices=['log', 'svm', 'rf', 'mvr', 'xbg'])
parser.add_argument('--engine', '-eng', action='store', type=str, dest='engine',
                    default='sklearn', choices=['statsmodels', 'sklearn'])
args = parser.parse_args()

grp_name = args.meta
statFileName = args.df

configurationFileName = os.path.join(args.cfpath, args.cfname)
fp = open(configurationFileName, 'r')
conf = json.load(fp)
fp.close()
basePath = conf['basePath']         # Path containing the DATA and RESULTS directories
cohortName = conf['cohortName']     # Will be used for folder name
t_lim = conf['analysisDays']        # How long to consider in the analysis
tRes = conf['timeResolutionHrs']    # Resolution after imputation in hours
analyze = conf['analyze']           # Time period to analyze (hospital/ICU/all)

baseDataPath = os.path.join(basePath, 'DATA', 'all_sheets')         # folder containing all raw data
dataPath = os.path.join(basePath, 'DATA', analyze, cohortName)
resPath = os.path.join(basePath, 'RESULTS', analyze, cohortName)

figPath = os.path.join(resPath, "figures")
if not os.path.exists(figPath):
    os.mkdir(figPath)

f = h5py.File(os.path.join(resPath, statFileName), 'r')
ids = f[grp_name]['ids'][:]
died = f['meta']["died_inp"][:]


cats = ["%s-%s" % (x, y) for x in ["1", "2", "3", "3D"] for y in ["Im", "St", "Ws"]]

kdigos = load_csv(os.path.join(dataPath, 'kdigo_icu_2ptAvg.csv'), ids, int)
days = load_csv(os.path.join(dataPath, 'days_interp_icu_2ptAvg.csv'), ids, int)
scrs = load_csv(os.path.join(dataPath, 'scr_interp_icu_2ptAvg.csv'), ids, float)
icu_windows = load_csv(os.path.join(dataPath, 'icu_admit_discharge.csv'), ids, 'date', struct='dict')
hosp_windows = load_csv(os.path.join(dataPath, 'hosp_admit_discharge.csv'), ids, 'date', struct='dict')

if args.pc:
    pcstr = "popCoords"
else:
    pcstr = "absCoords"
# clusterPath = os.path.join(resPath, "clusters", "14days", )
clusterPath = os.path.join(resPath, "clusters", "14days", args.sf.split(".")[0] + "_" + args.align,
                               args.dfunc + "_" + pcstr, "flat", "%d_clusters" % args.nclust)

#
#
# clusterPath = os.path.join(resPath, "clusters", "14days", "kdigo_icu_2ptAvg_" + alignment,
#                                dfunc + "_" + pc, "flat", "%d_clusters" % nClust)
#
lbls = load_csv(os.path.join(clusterPath, "clusters.csv"), ids, int)

nlbls, key = clusterCategorizer(mk, kdigos, days, lbls, 14)
print("Category,LowestMortality,HighestMortality,MaxMortDiff,MinMortDiff")
s = args.sf.split(".")[0] + "_" + args.align + "_" + args.dfunc + "_" + pcstr
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
    # print("%s,%.2f,%.2f,%.2f,%.2f" % (cat, minMort, maxMort, maxDiff, minDiff))
    s += ",%s" % maxDiff
print(s)
