import os
import numpy as np
from utility_funcs import load_csv, get_dm_tag
from dtw_distance import extension_penalty_func, mismatch_penalty_func, pairwise_dtw_dist, get_custom_distance_discrete
import h5py
import argparse
import json

# ------------------------------- PARAMETERS ----------------------------------#
parser = argparse.ArgumentParser(description='Preprocess Data and Construct KDIGO Vectors.')
parser.add_argument('--config_file', action='store', nargs=1, type=str, dest='cfname',
                    default='kdigo_conf.json')
parser.add_argument('--config_path', action='store', nargs=1, type=str, dest='cfpath',
                    default='')
parser.add_argument('--use_extension', '-ext', action='store_true', dest='ext')
parser.add_argument('--use_mismatch', '-mism', action='store_true', dest='mism')
parser.add_argument('--aggregate_extension', '-agg', action='store_true', dest='aggext')
parser.add_argument('--ext_alpha', '-alpha', action='store', type=float, dest='alpha', default=1.0)
parser.add_argument('--laplacian_type', '-lt', action='store', type=str, dest='lap', default='none', choices=['none', 'individual', 'aggregated'])
parser.add_argument('--distance_function', '-dfunc', '-d', action='store', type=str, dest='dfunc', default='braycurtis')
parser.add_argument('--pop_coords', '-pcoords', '-pc', action='store_true', dest='popcoords')
parser.add_argument('--overwrite', action='store_true', dest='overwrite')
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
tcost_fname = conf['transition_cost_file']

baseDataPath = os.path.join(basePath, 'DATA', 'all_sheets')
dataPath = os.path.join(basePath, 'DATA', analyze, cohortName)
resPath = os.path.join(basePath, 'RESULTS', analyze, cohortName)

transition_costs = np.loadtxt(os.path.join(dataPath, tcost_fname), delimiter=',', usecols=1)

# -----------------------------------------------------------------------------#
# Use population derived coordinates from transition weights, otherwise use raw KDIGO scores
if args.popcoords:
    coords = np.array([np.sum(transition_costs[:i]) for i in range(len(transition_costs) + 1)], dtype=float)
    coord_tag = '_popcoord'
else:
    coords = np.array([x for x in range(len(transition_costs) + 1)], dtype=float)
    coord_tag = '_abscoord'

# Load patient IDs
f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
ids = f['meta']['ids'][:]
f.close()

dm_path = os.path.join(resPath, 'dm')
dtw_path = os.path.join(resPath, 'dtw')
if not os.path.exists(dm_path):
    os.mkdir(dm_path)
if not os.path.exists(dtw_path):
    os.mkdir(dtw_path)

dm_path = os.path.join(dm_path, '%ddays' % t_lim)
dtw_path = os.path.join(dtw_path, '%ddays' % t_lim)
if not os.path.exists(dm_path):
    os.mkdir(dm_path)
if not os.path.exists(dtw_path):
    os.mkdir(dtw_path)

# load corresponding KDIGO scores and their associated days post admission
kdigos = load_csv(os.path.join(dataPath, 'kdigo.csv'), ids, int)
days = load_csv(os.path.join(dataPath, 'days_interp.csv'), ids, int)

# Build filename to distinguish different matrices and specify the
# mismatch and extension penalties
if args.mism:
    # mismatch penalty derived from population dynamics
    mismatch = mismatch_penalty_func(*transition_costs)
    dm_tag = '_popmismatch'
else:
    # absolute difference in KDIGO scores
    mismatch = lambda x, y: abs(x - y)
    dm_tag = '_absmismatch'
if args.ext:
    # mismatch penalty derived from population dynamics
    extension = extension_penalty_func(*transition_costs)
    dm_tag += '_extension_a%.0E' % args.alpha[0]
else:
    # no extension penalty
    extension = lambda x: 0

# Construct the tags for distance matrix and DTW files
dm_tag, dtw_tag = get_dm_tag(args.mism, args.ext, args.alpha[0], args.aggext, args.popcoords, args.dfunc, args.lap)

# Load the appropriate distance function
dist = get_custom_distance_discrete(coords, args.dfunc)

# Don't overwrite existing data unless specified
if not os.path.exists(os.path.join(dm_path, '/kdigo_dm' + dm_tag + '.npy')) or args.overwrite:
    dm = pairwise_dtw_dist(kdigos, days, ids, os.path.join(dm_path, 'kdigo_dm' + dm_tag + '.csv'),
                           os.path.join(dtw_path, 'kdigo_dtwlog' + dtw_tag + '.csv'),
                           mismatch=mismatch,
                           extension=extension,
                           dist=dist,
                           alpha=args.alpha[0], t_lim=t_lim, aggext=args.aggext)
    np.save(os.path.join(dm_path, 'kdigo_dm' + dm_tag), dm)
else:
    print(dm_tag + ' already completed.')
