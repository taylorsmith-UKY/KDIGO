import numpy as np
import h5py
from cluster_funcs import assign_feature_vectors
from kdigo_funcs import assign_cluster_features, load_csv, arr2csv, get_cluster_features
import os
import argparse
import json

# --------------------------------------------------- PARAMETERS ----------------------------------------------------- #

parser = argparse.ArgumentParser(description='Merge clusters.')
parser.add_argument('--config_file', action='store', nargs=1, type=str, dest='cfname',
                    default='kdigo_conf.json')
parser.add_argument('--config_path', action='store', nargs=1, type=str, dest='cfpath',
                    default='')
parser.add_argument('--n_clust', action='store', nargs=1, type=int, dest='n_clust',
                    default=96)
parser.add_argument('--feature', '-f', action='store', type=str, dest='feature', default='descriptive_norm')
parser.add_argument('--use_extension', '-ext', action='store_true', dest='ext')
parser.add_argument('--use_mismatch', '-mism', action='store_true', dest='mism')
parser.add_argument('--distance_function', '-dfunc', '-d', action='store', type=str, dest='dfunc', default='braycurtis')
parser.add_argument('--pop_coords', '-pcoords', '-pc', action='store_true', dest='popcoords')
parser.add_argument('--aggregate_function', '-afunc', '-d', action='store', type=str, dest='dfunc', default='braycurtis')
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
feature = args.feature

baseDataPath = os.path.join(basePath, 'DATA', 'all_sheets')
dataPath = os.path.join(basePath, 'DATA', analyze, cohortName)
resPath = os.path.join(basePath, 'RESULTS', analyze, cohortName)

folderName = 'merged'
if args.popcoords:
    coord_tag = '_popcoord'
else:
    coord_tag = '_abscoord'
if args.mism:
    dm_tag = '_popmismatch'
else:
    dm_tag = '_absmismatch'
if args.ext:
    dm_tag += '_extension'

dm_tag += '_' + args.dfunc + coord_tag


# -------------------------------------------------------------------------------------------------------------------- #

day_str = '7days'
if not os.path.exists(os.path.join(resPath, 'features', day_str)):
    os.mkdir(os.path.join(resPath, 'features', day_str))

dm = np.load(os.path.join(resPath, 'dm', day_str, 'kdigo_dm%s.npy' % dm_tag))

lblPath = os.path.join(resPath, 'clusters', day_str, dm_tag, '%d_clusters' % args.n_clust)

ids = np.loadtxt(os.path.join(lblPath, 'clusters.csv'), delimiter=',', usecols=0, dtype=int)
lbls = load_csv(os.path.join(lblPath, 'clusters.csv'), ids, str)

try:
    individual = load_csv(
        os.path.join(resPath, 'features', 'individual', feature + '.csv'), ids)
except ValueError:
    individual = load_csv(
        os.path.join(resPath, 'features', 'individual', feature + '.csv'), ids, skip_header=True)

fpath = os.path.join(resPath, 'features', dm_tag, '%d_clusters' % args.n_clust)
if not os.path.exists(fpath):
    os.mkdir(fpath)

for op in features[feature]:
    all_feats, cluster_feats = get_cluster_features(individual, lbls, dm, op=op)
    fname = feature + '_' + op + '.csv'
    arr2csv(os.path.join(fpath, fname), all_feats, ids, fmt='%.4f')
