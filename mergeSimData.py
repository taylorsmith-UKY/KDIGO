import h5py
import numpy as np
from dtw_distance import continuous_mismatch, continuous_extension, get_custom_distance_discrete, \
    mismatch_penalty_func, extension_penalty_func
import os
import argparse
from cluster_funcs import merge_simulated_sequences
import json
from utility_funcs import get_dm_tag, load_csv

# --------------------- Parser Arguments
parser = argparse.ArgumentParser(description='Merge clusters.')
parser.add_argument('--config_file', action='store', type=str, dest='cfname',
                    default='kdigo_conf.json')
parser.add_argument('--config_path', action='store', type=str, dest='cfpath',
                    default='')
parser.add_argument('--sequence_file', '-sf', action='store', type=str, dest='sf', default='kdigo_icu.csv')
parser.add_argument('--setSelection', '-setSel', action='store', type=str, dest='setSel', default='random')
parser.add_argument('--popDTW', '-pdtw', action='store_true', dest='pdtw')
parser.add_argument('--ext_alpha', '-alpha', action='store', type=float, dest='alpha', default=1.0)
parser.add_argument('--distance_function', '-dfunc', '-d', action='store', type=str, dest='dfunc', default='braycurtis')
parser.add_argument('--pop_coords', '-pcoords', '-pc', action='store_true', dest='popcoords')
parser.add_argument('--laplacian_type', '-lt', action='store', type=str, dest='lapType', default='none', choices=['none', 'individual', 'aggregated'])
parser.add_argument('--laplacian_val', '-lv', action='store', type=float, dest='lapVal', default=1.0)
parser.add_argument('--plot_centers', '-plot_c', action='store_true', dest='plot_centers')
parser.add_argument('--center_length', '-clen', action='store', type=int, dest='clen', default=14)
parser.add_argument('--category', '-cat', action='store', type=str, dest='cat', nargs='*',
                    default='all')
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

if args.cat[0] == 'all':
    cats = ['1-Im', '1-St', '1-Ws', '2-Im', '2-St', '2-Ws', '3-Im', '3-St', '3-Ws', '3D-Im', '3D-St', '3D-Ws']

elif args.cat[0] == 'allk':
    cats = ['1', '2', '3', '3D']
elif args.cat[0] == 'none':
    cats = []

dist = get_custom_distance_discrete(coords, dfunc=args.dfunc, lapVal=args.lapVal, lapType=args.lapType)

# Load patient data
lblPath = os.path.join(resPath, 'clusters', 'simulated', '%ddays' % args.clen)
if args.setSel == "random":
    lblPath = os.path.join(lblPath, "randomSets")
    for cat in cats:
        catPath = os.path.join(lblPath, cat)
        for dirpath, dirnames, fnames in os.walk(catPath):
            for dirname in dirnames:
                if "set" not in dirname:
                    continue





ids = np.loadtxt(os.path.join(lblPath, 'sequences.csv'), delimiter=',', usecols=0, dtype=int)
rawlbls = load_csv(os.path.join(lblPath, 'labels.csv'), ids, str, skip_header=True)
lbls = []
for i in range(len(rawlbls)):
    lbls.append(''.join(rawlbls[i]))

lbls = np.array(lbls)
sequences = load_csv(os.path.join(lblPath, 'sequences.csv'), ids)

merge_simulated_sequences(ids, sequences, lbls, args, mismatch, extension, dist, lblPath, clustersPerCategory=args.clustPerCat)
