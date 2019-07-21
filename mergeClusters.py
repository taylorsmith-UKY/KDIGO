import h5py
import numpy as np
from kdigo_funcs import load_csv
from dtw_distance import continuous_mismatch, continuous_extension, get_continuous_laplacian_braycurtis, \
    get_custom_cityblock_continuous, mismatch_penalty_func, extension_penalty_func
import os
import argparse
from cluster_funcs import merge_clusters
import json

# --------------------- Parser Arguments
parser = argparse.ArgumentParser(description='Merge clusters.')
parser.add_argument('--config_file', action='store', nargs=1, type=str, dest='cfname',
                    default='kdigo_conf.json')
parser.add_argument('--config_path', action='store', nargs=1, type=str, dest='cfpath',
                    default='')
parser.add_argument('--dm_tag', action='store', nargs=1, type=str, dest='dm_tag',
                    default='custmismatch_extension_a1E+00_normBC_popcoord')
parser.add_argument('--baseClustNum', action='store', nargs=1, type=int, dest='baseClustNum',
                    required=True)
parser.add_argument('--dist', action='store', nargs=1, type=str, dest='dist',
                    default='braycurtis')
parser.add_argument('--coords', action='store', nargs=1, type=str, dest='coords', choices=['kdigo', 'population'],
                    default='kdigo')
parser.add_argument('--coordShift', action='store', nargs=1, type=float, dest='coordShift',
                    default=0)
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
meta_grp = args.meta[0]

baseDataPath = os.path.join(basePath, 'DATA', 'all_sheets')
dataPath = os.path.join(basePath, 'DATA', analyze, cohortName)
resPath = os.path.join(basePath, 'RESULTS', analyze, cohortName)

transition_costs = np.loadtxt(os.path.join(dataPath, 'kdigo_transition_weights.csv'), delimiter=',', usecols=1, skiprows=1)

# Load patient data
f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
ids = f['meta']['ids'][:]
kdigos = load_csv(os.path.join(dataPath, 'kdigo.csv'), ids, int)
max_kdigos = f['meta']['max_kdigo_7d'][:]
days = load_csv(os.path.join(dataPath, 'days_interp.csv'), ids, int)

dm = np.load(os.path.join(resPath, 'dm', '7days', 'kdigo_dm_%s.npy' % args.dm_tag[0]))
lblPath = os.path.join(resPath, 'clusters', '7days', args.dm_tag[0], 'flat', '%d_clusters' % args.baseClustNum[0])

mismatch = mismatch_penalty_func(*transition_costs)
mismatch = continuous_mismatch(mismatch)
extension = extension_penalty_func(*transition_costs)
extension = continuous_extension(extension)
folderName = 'merged'
if args.coords == 'kdigo':
    coords = np.array([x for x in range(len(transition_costs) + 1)], dtype=float)
    folderName += '_abs'
elif args.coords[0] == 'population':
    coords = np.array([np.sum(transition_costs[:i]) for i in range(len(transition_costs) + 1)], dtype=float)
    folderName += '_pop'

if args.dist[0] == 'braycurtis':
    dist = get_continuous_laplacian_braycurtis(coords)
elif args.dist[0] == 'cityblock':
    dist = get_custom_cityblock_continuous(coords)

coords += args.coordShift
if args.coordShift:
    folderName += '_shift%E' % args.coordShift

merge_clusters(ids, kdigos, max_kdigos, days, dm, lblPath, f['meta'], mismatch=mismatch, extension=extension, dist=dist, t_lim=7, mergeType='mean', folderName=folderName)