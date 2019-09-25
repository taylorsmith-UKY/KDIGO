import h5py
import numpy as np
import os
import json
from utility_funcs import get_dm_tag, load_csv
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cluster_funcs import plotMultiKdigo, plotKdigo

class Args:
    sf = 'kdigo_icu_2ptAvg.csv'
    df = 'days_interp_icu_2ptAvg.csv'
    pdtw = True
    alpha = 0.35
    dfunc = 'braycurtis'
    popcoords = True
    lapType = 'aggregated'
    lapVal = 1.0
    meta = 'meta_avg'
    plot_centers = False
    nClusters = 96
    seedType = 'medoid'
    mergeType = 'mean'
    dbaiter = 5
    extDistWeight = 0.0
    cumExtDist = False
    maxExt = -1
    t_lim = 14
    cfpath = ''
    cfpath = 'kdigo_conf.json'

args = Args()

configurationFileName = os.path.join('kdigo_conf.json')

fp = open(configurationFileName, 'r')
conf = json.load(fp)
fp.close()

basePath = conf['basePath']
cohortName = conf['cohortName']
t_lim = conf['analysisDays']
tRes = conf['timeResolutionHrs']
v = conf['verbose']
analyze = conf['analyze']
meta_grp = args.meta

baseDataPath = os.path.join(basePath, 'DATA', 'all_sheets')
dataPath = os.path.join(basePath, 'DATA', analyze, cohortName)
resPath = os.path.join(basePath, 'RESULTS', analyze, cohortName)

# Load patient data
f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
ids = f[meta_grp]['ids'][:]
kdigos = load_csv(os.path.join(dataPath, args.sf), ids, int)
days = load_csv(os.path.join(dataPath, args.df), ids, int)
for i in range(len(kdigos)):
    kdigos[i] = kdigos[i][np.where(days[i] <= t_lim)]
    days[i] = days[i][np.where(days[i] <= t_lim)]

dm_tag, dtw_tag = get_dm_tag(args.pdtw, args.alpha, False, args.popcoords, args.dfunc, args.lapVal, args.lapType)

folderName = args.sf.split('.')[0]
folderName += "_" + dtw_tag

lblPath = os.path.join(resPath, 'clusters', '%ddays' % t_lim,
                       folderName, dm_tag, 'flat', '%d_clusters' % args.nClusters)

centers, cnames = load_csv(os.path.join(lblPath, 'dba_centers', dtw_tag, 'mean', 'centers.csv'), None, id_dtype=str, struct='dict')

startPath = os.path.join(lblPath, 'merged_%s' % dtw_tag, 'center_mean', 'extWeight_35E-02_normExt')
categories = ['%s-%s' % (x, y) for x in ['1', '2', '3', '3D'] for y in ['Im', 'St', 'Ws']]
for cat in categories:
    tpath = os.path.join(startPath, cat, 'mean')
    if os.path.exists(tpath):
        tct = 2
        while os.path.exists(os.path.join(tpath, '%d_clusters' % (tct + 1))):
            tct += 1
        print('Category %s initially has %d clusters' % (cat, tct))
        tlbls = np.unique(np.loadtxt(os.path.join(os.path.join(tpath, '%d_clusters' % tct, 'clusters.csv')),
                                     delimiter=',', usecols=1, dtype=str))
        tcenters = {}
        for lbl in tlbls:
            tcenters[lbl] = centers[lbl]
        with PdfPages(os.path.join(tpath, 'new_merge_visualization.pdf')) as pdf:
            while tct > 2:
                if tct <= 9:
                    ncol = 3
                else:
                    ncol = 4
                nrow = int(np.ceil(tct / ncol))
                gs = GridSpec(nrow, ncol)
                gsm = GridSpec(1, 2)
                fig = plt.figure(figsize=[4.8*ncol, 4.8*nrow])
                figm = plt.figure(figsize=[4.8 * 2, 4.8])
                col = 0
                row = 0
                ncenters, nlbls = load_csv(os.path.join(tpath, '%d_clusters' % (tct - 1), 'centers.csv'),
                                           None, id_dtype=str, struct='dict')
                mcol = 0
                for lbl in list(tcenters):
                    if lbl not in nlbls:
                        plotKdigo(figm, gsm[0, mcol], tcenters[lbl], title='Cluster %s' % lbl)
                        mcol += 1
                    plotKdigo(fig, gs[row, col], tcenters[lbl], legendEntries=['Cluster %s' % lbl])
                    col += 1
                    if col == ncol:
                        col = 0
                        row += 1
                pdf.savefig(fig, dpi=600)
                pdf.savefig(figm, dpi=600)
                plt.close(fig)
                plt.close(figm)
                tct -= 1
                nlbl = [x for x in nlbls if x not in tlbls]
                olbls = [x for x in tlbls if x not in nlbls]
                tcenters = ncenters
