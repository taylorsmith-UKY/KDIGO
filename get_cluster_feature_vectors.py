import numpy as np
from utility_funcs import load_csv, arr2csv
from classification_funcs import assign_cluster_features, get_cluster_features
import os
import argparse
import json

# --------------------------------------------------- PARAMETERS ----------------------------------------------------- #

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
parser.add_argument('--feature', '-f', action='store', type=str, dest='feature', default='descriptive_norm')
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
