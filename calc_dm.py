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
parser.add_argument('--popDTW', '-pdtw', action='store_true', dest='pdtw')
parser.add_argument('--meta_group', '-meta', action='store', type=str, dest='meta', default='meta')
parser.add_argument('--sequence_file', '-sf', action='store', type=str, dest='sf', default='kdigo_icu.csv')
parser.add_argument('--day_file', '-df', action='store', type=str, dest='df', default='days_interp_icu.csv')
parser.add_argument('--aggregate_extension', '-agg', action='store_true', dest='aggext')
parser.add_argument('--ext_alpha', '-alpha', action='store', type=float, dest='alpha', default=1.0)
parser.add_argument('--distance_function', '-dfunc', '-d', action='store', type=str, dest='dfunc', default='braycurtis')
parser.add_argument('--pop_coords', '-pcoords', '-pc', action='store_true', dest='popcoords')
parser.add_argument('--laplacian_value', '-lv', action='store', type=float, dest='lapVal', default=1.0)
parser.add_argument('--laplacian_type', '-lt', action='store', type=str, dest='lapType', default='none', choices=['none', 'individual', 'aggregated'])
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

# -----------------------------------------------------------------------------#
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
dm_tag, dtw_tag = get_dm_tag(args.pdtw, args.alpha, args.aggext, args.popcoords, args.dfunc, args.lapVal, args.lapType)

folderName = args.sf.split('.')[0]
folderName += "_" + dtw_tag
dm_path = os.path.join(dm_path, folderName)
dtw_path = os.path.join(dtw_path, folderName)
if not os.path.exists(dm_path):
    os.mkdir(dm_path)
if not os.path.exists(dtw_path):
    os.mkdir(dtw_path)

tcost_fname = "%s_transition_weights.csv" % args.sf.split('.')[0]
transition_costs = np.loadtxt(os.path.join(dataPath, tcost_fname), delimiter=',', usecols=1, skiprows=1)

# Use population derived coordinates from transition weights, otherwise use raw KDIGO scores
if args.popcoords:
    coords = np.array([np.sum(transition_costs[:i]) for i in range(len(transition_costs) + 1)], dtype=float)
else:
    coords = np.array([x for x in range(len(transition_costs) + 1)], dtype=float)

# Load patient IDs
f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
ids = f[args.meta]['ids'][:]
f.close()

# load corresponding KDIGO scores and their associated days post admission
kdigos = load_csv(os.path.join(dataPath, args.sf), ids, int)
days = load_csv(os.path.join(dataPath, args.df), ids, int)
for i in range(len(kdigos)):
    kdigos[i] = kdigos[i][np.where(days[i] <= t_lim)]
    days[i] = days[i][np.where(days[i] <= t_lim)]

# Build filename to distinguish different matrices and specify the
# mismatch and extension penalties
if args.pdtw:
    # mismatch penalty derived from population dynamics
    mismatch = mismatch_penalty_func(*transition_costs)
    extension = extension_penalty_func(*transition_costs)
else:
    # absolute difference in KDIGO scores
    mismatch = lambda x, y: abs(x - y)
    # no extension penalty
    extension = lambda x: 0

# Load the appropriate distance function
dist = get_custom_distance_discrete(coords, dfunc=args.dfunc, lapVal=args.lapVal, lapType=args.lapType)

# Don't overwrite existing data unless specified
if not os.path.exists(os.path.join(dm_path, 'kdigo_dm_' + dm_tag + '.npy')) or args.overwrite:
    dm = pairwise_dtw_dist(kdigos, days, ids, os.path.join(dm_path, 'kdigo_dm_' + dm_tag + '.csv'),
                           os.path.join(dtw_path, 'dtw_alignment.csv'),
                           mismatch=mismatch,
                           extension=extension,
                           dist=dist,
                           alpha=args.alpha, t_lim=t_lim, aggext=args.aggext)
    np.save(os.path.join(dm_path, 'kdigo_dm_' + dm_tag), dm)
else:
    print(dm_tag + ' already completed.')
