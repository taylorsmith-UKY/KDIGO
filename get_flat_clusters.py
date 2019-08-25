import numpy as np
from cluster_funcs import cluster_trajectories
from kdigo_funcs import load_csv
import h5py
import os
from scipy.spatial.distance import squareform
import argparse
import json
from utility_funcs import get_dm_tag

# PARAMETERS
parser = argparse.ArgumentParser(description='Preprocess Data and Construct KDIGO Vectors.')
parser.add_argument('--config_file', action='store', nargs=1, type=str, dest='cfname',
                    default='kdigo_conf.json')
parser.add_argument('--config_path', action='store', nargs=1, type=str, dest='cfpath',
                    default='')
parser.add_argument('--sequence_file', '-sf', action='store', type=str, dest='sf', default='kdigo_icu.csv')
parser.add_argument('--day_file', '-df', action='store', type=str, dest='df', default='days_interp_icu.csv')
parser.add_argument('--popDTW', '-pdtw', action='store_true', dest='pdtw')
parser.add_argument('--aggregate_extension', '-agg', action='store_true', dest='aggExt')
parser.add_argument('--ext_alpha', '-alpha', action='store', type=float, dest='alpha', default=1.0)
parser.add_argument('--distance_function', '-dfunc', '-d', action='store', type=str, dest='dfunc', default='braycurtis')
parser.add_argument('--pop_coords', '-pcoords', '-pc', action='store_true', dest='popcoords')
parser.add_argument('--laplacian_type', '-lt', action='store', type=str, dest='lapType', default='none', choices=['none', 'individual', 'aggregated'])
parser.add_argument('--laplacian_val', '-lv', action='store', type=float, dest='lapVal', default=1.0)
parser.add_argument('--interact', action='store_true', dest='interact')
parser.add_argument('--n_clust', '-n', action='store', nargs=1, type=int, dest='n_clust',
                    default=96)
parser.add_argument('--meta_group', '-meta', action='store', nargs=1, type=str, dest='meta',
                    default='meta')
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

baseDataPath = os.path.join(basePath, 'DATA', 'all_sheets')
dataPath = os.path.join(basePath, 'DATA', analyze, cohortName)
resPath = os.path.join(basePath, 'RESULTS', analyze, cohortName)

interactive = args.interact

meta_grp = args.meta[0]

# number of clusters to extract if method is flat
n_clusters = args.n_clust[0]

########################################################
f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')

ids = f[meta_grp]['ids'][:]
max_kdigo = f[meta_grp]['max_kdigo_win'][:]

kdigos = load_csv(os.path.join(dataPath, args.sf), ids, int)
days = load_csv(os.path.join(dataPath, args.df), ids, int)
for i in range(len(kdigos)):
    kdigos[i] = kdigos[i][np.where(days[i] <= 7)]
    days[i] = days[i][np.where(days[i] <= 7)]

if not os.path.exists(os.path.join(resPath, 'clusters')):
    os.mkdir(os.path.join(resPath, 'clusters'))

if not os.path.exists(os.path.join(resPath, 'clusters', '%ddays' % t_lim)):
    os.mkdir(os.path.join(resPath, 'clusters', '%ddays' % t_lim))

dm_tag, dtw_tag = get_dm_tag(args.pdtw, args.alpha, args.aggExt, args.popcoords, args.dfunc, args.lapVal, args.lapType)
folderName = args.sf.split('.')[0]
folderName += "_" + dtw_tag

dm_path = os.path.join(resPath, 'dm', '%ddays' % t_lim, folderName)

save_path = os.path.join(resPath, 'clusters', '%ddays' % t_lim)
if not os.path.exists(save_path):
    os.mkdir(save_path)

save_path = os.path.join(save_path, folderName)
if not os.path.exists(save_path):
    os.mkdir(save_path)

save_path = os.path.join(save_path, dm_tag)
if not os.path.exists(save_path):
    os.mkdir(save_path)

if os.path.isfile(os.path.join(dm_path, 'kdigo_dm_%s.npy' % dm_tag)):
    try:
        dm = np.load(os.path.join(dm_path, 'kdigo_dm_%s.npy' % dm_tag))[:, 2]
    except IndexError:
        dm = np.load(os.path.join(dm_path, 'kdigo_dm_%s.npy' % dm_tag))
else:
    dm = np.loadtxt(os.path.join(dm_path, 'kdigo_dm_%s.csv' % dm_tag), delimiter=',', usecols=2)
sqdm = squareform(dm)
tpath = save_path
eps = cluster_trajectories(f, f[meta_grp], ids, max_kdigo, sqdm, n_clusters=n_clusters, data_path=dataPath, save=tpath,
                           interactive=interactive, kdigos=kdigos, days=days)
