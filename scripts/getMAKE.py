import os
import numpy as np
import json
import h5py
from stat_funcs import get_MAKE90
from utility_funcs import load_csv

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
f = h5py.File(os.path.join(resPath, 'stats_new.h5'), 'r+')
ids = f['meta']['ids'][:]
stats = f['meta']

for npts in [1, 2]:
    for buf in [0, 30]:
        for ref in ['admit', 'disch']:
            # out = open(os.path.join(dataPath, 'make90_%s_%dpt_%ddbuf.csv' % (ref, npts, buf)), 'w')
            # m90, mids = get_MAKE90(ids, stats, baseDataPath, out, ref=ref, buffer=buf, ct=npts,
            #                        label='make90_%s_%dpt_%ddbuf' % (ref, npts, buf), ovwt=True)
            # out.close()
            #
            m90 = load_csv(os.path.join(dataPath, 'make90_%s_%dpt_%ddbuf.csv' % (ref, npts, buf)), ids, dt=int, idxs=[range(1,12)], skip_header=True)
            died = np.max(m90[:, :3], axis=1)
            esrd = np.max(m90[:, 3:6], axis=1)
            ddep = np.max(m90[:, 6:8], axis=1)
            gfr30 = m90[:, 9]
            gfr50 = m90[:, 10]

            diedIdx = np.where(died)[0]
            aliveIdx = np.setdiff1d(range(len(ids)), diedIdx)

            dropIdx = np.setdiff1d(aliveIdx, np.where(esrd)[0])
            dropIdx = np.setdiff1d(dropIdx, np.where(ddep)[0])

            g30 = np.max(m90[:, [0, 1, 2, 3, 4, 5, 6, 7, 9]], axis=1)
            g50 = np.max(m90[:, [0, 1, 2, 3, 4, 5, 6, 7, 10]], axis=1)

            g30a = np.max(m90[aliveIdx][:, [0, 1, 2, 3, 4, 5, 6, 7, 9]], axis=1)
            g50a = np.max(m90[aliveIdx][:, [0, 1, 2, 3, 4, 5, 6, 7, 10]], axis=1)

            print('%s,%d,%d,%d (%.2f),%d (%.2f),%d (%.2f),%d (%.2f),%d (%.2f),%d (%.2f),%d (%.2f),%d (%.2f),%d (%.2f)' %
                  (ref, buf, npts, len(diedIdx), float(len(diedIdx)) / len(ids) * 100, np.sum(esrd),
                   np.sum(esrd) / (len(aliveIdx)) * 100, np.sum(ddep), np.sum(ddep) / (len(aliveIdx)) * 100,
                   np.sum(gfr30), np.sum(gfr30) / len(dropIdx) * 100, np.sum(gfr50), np.sum(gfr50) / len(dropIdx) * 100,
                   np.sum(g30), np.sum(g30) / len(ids) * 100, np.sum(g50), np.sum(g50) / len(ids) * 100,
                   np.sum(g30a), np.sum(g30a) / len(g30a) * 100, np.sum(g50a), np.sum(g50a) / len(g50a) * 100))
f.close()
