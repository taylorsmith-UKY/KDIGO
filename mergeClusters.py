import h5py
import numpy as np
from kdigo_funcs import load_csv
from dtw_distance import continuous_mismatch, continuous_extension, get_custom_distance_discrete, \
    mismatch_penalty_func, extension_penalty_func
import os
import argparse
from cluster_funcs import merge_clusters
import json
from utility_funcs import get_dm_tag

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
parser.add_argument('--normal_dtw', '-ndtw', action='store_true', dest='normDTW')
parser.add_argument('--overRidePopExt', '-ovpe', action='store_true', dest='overRidePopExt')
parser.add_argument('--folder_name', '-fname', action='store', type=str, dest='fname',
                    default='merged')
parser.add_argument('--baseClustNum', '-n', action='store', type=int, dest='baseClustNum', default=96)
parser.add_argument('--category', '-cat', action='store', type=str, dest='cat', nargs='*',
                    default='all')
parser.add_argument('--DBA_popDTW', '-dbapdtw', action='store_true', dest='dbapdtw')
parser.add_argument('--DBA_ext_alpha', '-dbaalpha', action='store', type=float, dest='dbaalpha', default=1.0)
parser.add_argument('--DBA_agg_ext', '-dbaagg', action='store_true', dest='dbaaggExt')
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

if args.overRidePopExt:
    ext = continuous_extension(extension_penalty_func(*transition_costs))
    def valExtension(x):
        return ext(x) + 1
else:
    valExtension = None

if args.dbapdtw:
    ext = continuous_extension(extension_penalty_func(*transition_costs))
    def extension(x):
        return ext(x) + 1
    # mismatch penalty derived from population dynamics
    mismatch = continuous_mismatch(mismatch_penalty_func(*transition_costs))
    # extension = continuous_extension(extension_penalty_func(*transition_costs))
else:
    # absolute difference in KDIGO scores
    mismatch = lambda x, y: abs(x - y)
    # no extension penalty
    extension = lambda x: 0

# Load patient data
f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
ids = f[meta_grp]['ids'][:]
kdigos = load_csv(os.path.join(dataPath, args.sf), ids, int)
max_kdigos = f[meta_grp]['max_kdigo_win'][:]
days = load_csv(os.path.join(dataPath, args.df), ids, int)

dm_path = os.path.join(resPath, 'dm')
dtw_path = os.path.join(resPath, 'dtw')
dm_path = os.path.join(dm_path, '%ddays' % t_lim)
dtw_path = os.path.join(dtw_path, '%ddays' % t_lim)
dm_tag, dtw_tag = get_dm_tag(args.pdtw, args.alpha, args.aggExt, args.popcoords, args.dfunc, args.lapVal, args.lapType)
dbadm_tag, dbadtw_tag = get_dm_tag(args.dbapdtw, args.dbaalpha, args.dbaaggExt, args.popcoords, args.dfunc, args.lapVal, args.lapType)

folderName = args.sf.split('.')[0]
folderName += "_" + dtw_tag

dm_path = os.path.join(dm_path, folderName)
dtw_path = os.path.join(dtw_path, folderName)

dm = np.load(os.path.join(dm_path, 'kdigo_dm_%s.npy' % dm_tag))

lblPath = os.path.join(resPath, 'clusters', '%ddays' % t_lim, folderName, dm_tag, 'flat', '%d_clusters' % args.baseClustNum)

dist = get_custom_distance_discrete(coords, dfunc=args.dfunc, lapVal=args.lapVal, lapType=args.lapType)
folderName = args.sf.split('.')[0]
folderName += "_" + dtw_tag

stats = f[meta_grp]
merge_clusters(ids, kdigos, max_kdigos, days, dm, lblPath, stats, mismatch=mismatch, extension=extension, dist=dist,
               t_lim=7, mergeType='mean', folderName=dtw_tag, dbaPopDTW=args.dbapdtw, alpha=args.alpha, category=args.cat, plot_centers=args.plot_centers, evalExt=valExtension, dbaAlpha=args.dbaalpha, dbaAggExt=args.dbaaggExt)
