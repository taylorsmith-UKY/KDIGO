import h5py
import numpy as np
from kdigo_funcs import load_csv
from dtw_distance import continuous_mismatch, continuous_extension, get_continuous_laplacian_braycurtis, \
    get_custom_cityblock_continuous, mismatch_penalty_func, extension_penalty_func
import os
import argparse
from cluster_funcs import merge_clusters
import json
from scipy.spatial.distance import braycurtis as bc
from utility_funcs import get_dm_tag

# --------------------- Parser Arguments
parser = argparse.ArgumentParser(description='Merge clusters.')
parser.add_argument('--config_file', action='store', nargs=1, type=str, dest='cfname',
                    default='kdigo_conf.json')
parser.add_argument('--config_path', action='store', nargs=1, type=str, dest='cfpath',
                    default='')
parser.add_argument('--use_extension', '-ext', action='store_true', dest='ext')
parser.add_argument('--use_mismatch', '-mism', action='store_true', dest='mism')
parser.add_argument('--plot_centers', '-plot_c', action='store_true', dest='plot_centers')
parser.add_argument('--agg_ext', '-agg', action='store_true', dest='aggext')
parser.add_argument('--distance_function', '-dfunc', '-d', action='store', type=str, dest='dfunc', default='braycurtis')
parser.add_argument('--pop_coords', '-pcoords', '-pc', action='store_true', dest='popcoords')
parser.add_argument('--normalDTW', '-normDTW', action='store_true', dest='normDTW')
parser.add_argument('--baseClustNum', action='store', nargs=1, type=int, dest='baseClustNum',
                    required=True)
parser.add_argument('--laplacian_type', '-lt', action='store', type=str, dest='lap', default='none', choices=['none', 'individual', 'aggregated'])
parser.add_argument('--ext_alpha', '-alpha', action='store', type=float, dest='alpha', default=1.0)
parser.add_argument('--meta_group', '-meta', action='store', nargs=1, type=str, dest='meta',
                    default='meta')
parser.add_argument('--folder_name', '-fname', action='store', nargs=1, type=str, dest='fname',
                    default='merged')
parser.add_argument('--category', '-cat', action='store', nargs=1, type=str, dest='cat',
                    default='all')
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

transition_costs = np.loadtxt(os.path.join(dataPath, 'kdigo_transition_weights.csv'),
                              delimiter=',', usecols=1, skiprows=1)
folderName = 'merged'
if args.popcoords:
    coords = np.array([np.sum(transition_costs[:i]) for i in range(len(transition_costs) + 1)], dtype=float)
    coord_tag = '_popcoord'
    folderName += '_popcoord'
else:
    coords = np.array([x for x in range(len(transition_costs) + 1)], dtype=float)
    coord_tag = '_abscoord'
    folderName += '_kdigo'

folderName = args.fname[0]

if args.mism:
    # mismatch penalty derived from population dynamics
    mismatch = mismatch_penalty_func(*transition_costs)
    mismatch = continuous_mismatch(mismatch)
else:
    # absolute difference in KDIGO scores
    def mismatch(x, y):
        return abs(x - y)
if args.ext:
    # mismatch penalty derived from population dynamics
    extension = extension_penalty_func(*transition_costs)
    extension = continuous_extension(extension)
else:
    # no extension penalty
    def extension(_):
        return 0

dm_tag, dtw_tag = get_dm_tag(args.mism, args.ext, args.alpha[0], args.aggext, args.popcoords, args.dfunc, args.lap)

# Load patient data
f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
ids = f['meta']['ids'][:]
kdigos = load_csv(os.path.join(dataPath, 'kdigo.csv'), ids, int)
max_kdigos = f['meta']['max_kdigo_7d'][:]
days = load_csv(os.path.join(dataPath, 'days_interp.csv'), ids, int)

dm = np.load(os.path.join(resPath, 'dm', '7days', 'kdigo_dm_%s.npy' % dm_tag))
lblPath = os.path.join(resPath, 'clusters', '7days', dm_tag, 'flat', '%d_clusters' % args.baseClustNum[0])

dist = bc
if args.dfunc[0] == 'braycurtis':
    if args.agglap:
        dist = get_continuous_laplacian_braycurtis(coords, lf=args.lf, lf_type='aggregated')
    else:
        dist = get_continuous_laplacian_braycurtis(coords, lf=args.lf, lf_type='individual')
elif args.dfunc[0] == 'cityblock':
    dist = get_custom_cityblock_continuous(coords)

if hasattr(args.alpha, '__len__'):
    alpha = args.alpha[0]
else:
    alpha = args.alpha
merge_clusters(ids, kdigos, max_kdigos, days, dm, lblPath, f['meta'], mismatch=mismatch, extension=extension, dist=dist,
               t_lim=7, mergeType='mean', folderName=folderName, normalDTW=args.normDTW, alpha=alpha, category=args.cat[0], plot_centers=args.plot_centers)
