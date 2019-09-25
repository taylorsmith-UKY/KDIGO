import h5py
import numpy as np
import os
import argparse
import json
from utility_funcs import get_dm_tag, arr2csv, load_csv

# --------------------- Parser Arguments
parser = argparse.ArgumentParser(description='Merge clusters.')
parser.add_argument('--config_file', action='store', type=str, dest='cfname',
                    default='kdigo_conf.json')
parser.add_argument('--config_path', action='store', type=str, dest='cfpath',
                    default='')
parser.add_argument('--sequence_file', '-sf', action='store', type=str, dest='sf', default='kdigo_icu.csv')
parser.add_argument('--day_file', '-df', action='store', type=str, dest='df', default='days_interp_icu.csv')
parser.add_argument('--popDTW', '-pdtw', action='store_true', dest='pdtw')
parser.add_argument('--ext_alpha', '-alpha', action='store', type=float, dest='alpha', default=1.0)
parser.add_argument('--agg_ext', '-agg', action='store_true', dest='aggExt')
parser.add_argument('--distance_function', '-dfunc', '-d', action='store', type=str, dest='dfunc', default='braycurtis')
parser.add_argument('--pop_coords', '-pcoords', '-pc', action='store_true', dest='popcoords')
parser.add_argument('--laplacian_type', '-lt', action='store', type=str, dest='lapType', default='none', choices=['none', 'individual', 'aggregated'])
parser.add_argument('--laplacian_val', '-lv', action='store', type=float, dest='lapVal', default=1.0)
parser.add_argument('--meta_group', '-meta', action='store', type=str, dest='meta',
                    default='meta')
parser.add_argument('--plot_centers', '-plot_c', action='store_true', dest='plot_centers')
parser.add_argument('--overRidePopExt', '-ovpe', action='store_true', dest='overRidePopExt')
parser.add_argument('--baseClustNum', '-n', action='store', type=int, dest='baseClustNum', default=96)
parser.add_argument('--DBA_popDTW', '-dbapdtw', action='store_true', dest='dbapdtw')
parser.add_argument('--DBA_ext_alpha', '-dbaalpha', action='store', type=float, dest='dbaalpha', default=1.0)
parser.add_argument('--DBA_agg_ext', '-dbaagg', action='store_true', dest='dbaaggExt')
parser.add_argument('--seedType', '-seed', action='store', type=str, dest='seedType', default='medoid')
parser.add_argument('--mergeName', '-mname', action='store', type=str, dest='mname', default='none')
parser.add_argument('--centerExtendMethod', '-cemethod', action='store', type=str, dest='cemethod', default='dup-front')
args = parser.parse_args()

configurationFileName = os.path.join(args.cfpath, args.cfname)

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

f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
ids = f[meta_grp]['ids'][:]
f.close()

dm_tag, dtw_tag = get_dm_tag(args.pdtw, args.alpha, args.aggExt, args.popcoords, args.dfunc, args.lapVal, args.lapType)
dbadm_tag, dbadtw_tag = get_dm_tag(args.dbapdtw, args.dbaalpha, args.dbaaggExt, args.popcoords, args.dfunc, args.lapVal, args.lapType)

folderName = args.sf.split('.')[0]
folderName += "_" + dtw_tag
lblPath = os.path.join(resPath, 'clusters', '%ddays' % t_lim, folderName, dm_tag, 'flat', '%d_clusters' % args.baseClustNum)

if args.mname == 'none':
    lblPath1 = os.path.join(lblPath, "dba_centers")
    lblPath1 = os.path.join(lblPath1, dtw_tag, args.seedType)
    lbls = load_csv(os.path.join(lblPath, 'clusters.csv'), ids, str)
    lbls = np.array(lbls, dtype='|S100').astype(str)
    centers = load_csv(os.path.join(lblPath1, 'centers.csv'), np.unique(lbls), struct='dict', id_dtype=str)
    extStr = args.cemethod
    outPath = os.path.join(resPath, 'features', folderName)
    if not os.path.exists(outPath):
        os.mkdir(outPath)
    for nf in [dm_tag, '%d_clusters' % args.baseClustNum, args.seedType, 'noMerge']:
        outPath = os.path.join(outPath, nf)
        if not os.path.exists(outPath):
            os.mkdir(outPath)
    outName = os.path.join(outPath, 'features_%s.csv' % extStr)
    out = np.zeros((len(ids), max([len(centers[x]) for x in np.unique(lbls)])))
    for lbl in np.unique(lbls):
        idx = np.where(lbls == lbl)[0]
        if 'back' in extStr:
            out[idx, :len(centers[lbl])] = centers[lbl]
            if 'dup' in extStr:
                out[idx, len(centers[lbl]):] = centers[lbl][-1]
        elif 'front' in extStr:
            out[idx, out.shape[1] - len(centers[lbl]):] = centers[lbl]
            if 'dup' in extStr:
                out[idx, :out.shape[1] - len(centers[lbl])] = centers[lbl][-1]
    arr2csv(outName, out, ids, fmt='%.3f')
