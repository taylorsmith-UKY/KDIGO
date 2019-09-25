import h5py
import numpy as np
from dtw_distance import continuous_mismatch, continuous_extension, get_custom_distance_discrete, \
    mismatch_penalty_func, extension_penalty_func
import os
import argparse
from cluster_funcs import merge_clusters
import json
from utility_funcs import get_dm_tag, load_csv

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
parser.add_argument('--distance_function', '-dfunc', '-d', action='store', type=str, dest='dfunc', default='braycurtis')
parser.add_argument('--pop_coords', '-pcoords', '-pc', action='store_true', dest='popcoords')
parser.add_argument('--laplacian_type', '-lt', action='store', type=str, dest='lapType', default='none', choices=['none', 'individual', 'aggregated'])
parser.add_argument('--laplacian_val', '-lv', action='store', type=float, dest='lapVal', default=1.0)
parser.add_argument('--meta_group', '-meta', action='store', type=str, dest='meta',
                    default='meta')
parser.add_argument('--plot_centers', '-plot_c', action='store_true', dest='plot_centers')
parser.add_argument('--baseClustNum', '-n', action='store', type=int, dest='nClusters', default=96)
parser.add_argument('--category', '-cat', action='store', type=str, dest='cat', nargs='*',
                    default='all')
parser.add_argument('--seedType', '-seed', action='store', type=str, dest='seedType', default='medoid')
parser.add_argument('--mergeType', '-mtype', action='store', type=str, dest='mergeType', default='mean')
parser.add_argument('--DBAIterations', '-dbaiter', action='store', type=int, dest='dbaiter', default=10)
parser.add_argument('--extensionDistanceWeight', '-extDistWeight', action='store', type=float, dest='extDistWeight',
                    default=0.0)
parser.add_argument('--scaleExtension', '-scaleExt', action='store_true', dest='scaleExt')
parser.add_argument('--cumulativeExtensionForDistance', '-cumExtDist', action='store_true', dest='cumExtDist')

parser.add_argument('--maxExtension', '-maxExt', action='store', type=float, default=-1., dest='maxExt')
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

tcost_fname = "%s_transition_weights.csv" % args.sf.split('.')[0]
transition_costs = np.loadtxt(os.path.join(dataPath, tcost_fname), delimiter=',', usecols=1, skiprows=1)

if args.popcoords:
    coords = np.array([np.sum(transition_costs[:i]) for i in range(len(transition_costs) + 1)], dtype=float)
else:
    coords = np.array([x for x in range(len(transition_costs) + 1)], dtype=float)

extension = continuous_extension(extension_penalty_func(*transition_costs))
mismatch = continuous_mismatch(mismatch_penalty_func(*transition_costs))

# Load patient data
f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
ids = f[meta_grp]['ids'][:]
kdigos = load_csv(os.path.join(dataPath, args.sf), ids, int)
days = load_csv(os.path.join(dataPath, args.df), ids, int)

dm_path = os.path.join(resPath, 'dm')
dtw_path = os.path.join(resPath, 'dtw')
dm_path = os.path.join(dm_path, '%ddays' % t_lim)
dtw_path = os.path.join(dtw_path, '%ddays' % t_lim)
dm_tag, dtw_tag = get_dm_tag(args.pdtw, args.alpha, False, args.popcoords, args.dfunc, args.lapVal, args.lapType)

folderName = args.sf.split('.')[0]
folderName += "_" + dtw_tag

dm_path = os.path.join(dm_path, folderName)
dtw_path = os.path.join(dtw_path, folderName)

dm = np.load(os.path.join(dm_path, 'kdigo_dm_%s.npy' % dm_tag))

lblPath = os.path.join(resPath, 'clusters', '%ddays' % t_lim,
                       folderName, dm_tag, 'flat', '%d_clusters' % args.nClusters)

dist = get_custom_distance_discrete(coords, dfunc=args.dfunc, lapVal=args.lapVal, lapType=args.lapType)

max_kdigos = np.zeros(len(kdigos))
for i in range(len(kdigos)):
    max_kdigos[i] = np.max(kdigos[i][np.where(days[i] <= t_lim)[0]])

stats = f[meta_grp]
folderName = dtw_tag

args.t_lim = t_lim
merge_clusters(kdigos, max_kdigos, days, dm, lblPath, stats, args, mismatch, extension, folderName, dist)
