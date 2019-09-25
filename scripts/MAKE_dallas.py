import os
import numpy as np
import json
import h5py
from stat_funcs import get_MAKE90_dallas

fp = open('../utsw_conf.json', 'r')
conf = json.load(fp)
fp.close()
basePath = conf['basePath']         # Path containing the DATA and RESULTS directories
cohortName = conf['cohortName']     # Will be used for folder name
t_lim = conf['analysisDays']        # How long to consider in the analysis
tRes = conf['timeResolutionHrs']    # Resolution after imputation in hours
analyze = conf['analyze']           # Time period to analyze (hospital/ICU/all)

# Build paths and create if don't already exist
baseDataPath = os.path.join(basePath, 'DATA', 'dallas', 'csv')         # folder containing all raw data
dataPath = os.path.join(basePath, 'DATA', 'dallas', analyze, cohortName)
resPath = os.path.join(basePath, 'RESULTS', 'dallas', analyze, cohortName)
f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r+')
ids = f['meta_avg']['ids'][:]
stats = f['meta_avg']

s30 = []
s50 = []
for npts in [1, 2]:
    for buf in [0, 30]:
        for ref in ['admit', 'disch']:
            out = open(os.path.join(dataPath, 'make90_%s_%dpt_%ddbuf.csv' % (ref, npts, buf)), 'w')
            m90, mids = get_MAKE90_dallas(ids, stats, baseDataPath, out, ref=ref, buffer=buf, ct=npts, label='make90_%s_%dpt_%ddbuf' % (ref, npts, buf), ovwt=True)
            out.close()
            died = np.max(m90[:, :3], axis=1)
            esrd = np.max(m90[:, 3:5], axis=1)
            ddep = m90[:, 5]
            gfr30 = m90[:, 7]
            gfr50 = m90[:, 8]

            diedIdx = np.where(died)[0]
            aliveIdx = np.setdiff1d(range(len(ids)), diedIdx)

            dropIdx = np.setdiff1d(aliveIdx, np.where(esrd)[0])
            dropIdx = np.setdiff1d(aliveIdx, np.where(ddep)[0])

            g30 = np.max(m90[:, [0, 1, 2, 3, 4, 5, 7]], axis=1)
            g50 = np.max(m90[:, [0, 1, 2, 3, 4, 5, 8]], axis=1)
            s30.append(g30)
            s50.append(g50)

            print('%s,%d,%d,%d (%.2f),%d (%.2f),%d (%.2f),%d (%.2f),%d (%.2f),%d (%.2f),%d (%.2f)' %
                  (ref, buf, npts, len(diedIdx), float(len(diedIdx)) / len(ids), np.sum(esrd),
                   np.sum(esrd) / (len(aliveIdx)), np.sum(ddep), np.sum(ddep) / (len(aliveIdx)),
                   np.sum(gfr30), np.sum(gfr30) / len(dropIdx), np.sum(gfr50), np.sum(gfr50) / len(dropIdx),
                   np.sum(g30), np.sum(g30) / len(ids), np.sum(g50), np.sum(g50) / len(ids)))
            out.close()
f.close()
