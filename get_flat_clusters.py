import numpy as np
from cluster_funcs import cluster_trajectories
from kdigo_funcs import load_csv
import h5py
import os
from scipy.spatial.distance import squareform
import argparse
import json

# PARAMETERS
parser = argparse.ArgumentParser(description='Preprocess Data and Construct KDIGO Vectors.')
parser.add_argument('--config_file', action='store', nargs=1, type=str, dest='cfname',
                    default='kdigo_conf.json')
parser.add_argument('--config_path', action='store', nargs=1, type=str, dest='cfpath',
                    default='')
parser.add_argument('--dm_tag', action='store', nargs=1, type=str, dest='dm_tag',
                    default='popmismatch_extension_a1E+00_braycurtis_popcoord')
parser.add_argument('--interact', action='store_true', dest='interact')
parser.add_argument('--n_clust', action='store', nargs=1, type=int, dest='n_clust',
                    default=96)
parser.add_argument('--meta', action='store', nargs=1, type=str, dest='meta',
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
meta_grp = args.meta

baseDataPath = os.path.join(basePath, 'DATA', 'all_sheets')
dataPath = os.path.join(basePath, 'DATA', analyze, cohortName)
resPath = os.path.join(basePath, 'RESULTS', analyze, cohortName)

interactive = args.interact

# number of clusters to extract if method is flat
n_clusters = args.n_clust[0]

########################################################
f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
ids = f[meta_grp]['ids'][:]
max_kdigo = f[meta_grp]['max_kdigo_7d'][:]

kdigos = load_csv(os.path.join(dataPath, 'kdigo.csv'), ids, int)
days = load_csv(os.path.join(dataPath, 'days_interp.csv'), ids, int)
for i in range(len(kdigos)):
    kdigos[i] = kdigos[i][np.where(days[i] <= 7)]
    days[i] = days[i][np.where(days[i] <= 7)]

if not os.path.exists(os.path.join(resPath, 'clusters')):
    os.mkdir(os.path.join(resPath, 'clusters'))

if not os.path.exists(os.path.join(resPath, 'clusters', '%ddays' % t_lim)):
    os.mkdir(os.path.join(resPath, 'clusters', '%ddays' % t_lim))

dm_tag = args.dm_tag
save_path = os.path.join(resPath, 'clusters', '%ddays' % t_lim, dm_tag)
if not os.path.exists(save_path):
    os.mkdir(save_path)

if os.path.isfile(os.path.join(resPath, 'dm', '%ddays' % t_lim, 'kdigo_dm_%s.npy' % dm_tag)):
    try:
        dm = np.load(os.path.join(resPath, 'dm', '%ddays' % t_lim, 'kdigo_dm_%s.npy' % dm_tag))[:, 2]
    except IndexError:
        dm = np.load(os.path.join(resPath, 'dm', '%ddays' % t_lim, 'kdigo_dm_%s.npy' % dm_tag))
else:
    dm = np.loadtxt(os.path.join(resPath, 'dm', '%ddays' % t_lim, 'kdigo_dm_%s.csv' % dm_tag), delimiter=',', usecols=2)
sqdm = squareform(dm)
tpath = save_path
eps = cluster_trajectories(f, ids, max_kdigo, sqdm, n_clusters=n_clusters, data_path=dataPath, save=tpath,
                           interactive=interactive, kdigos=kdigos, days=days)
