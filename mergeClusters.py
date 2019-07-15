import h5py
import numpy as np
from kdigo_funcs import load_csv, continuous_mismatch, continuous_extension, \
    get_custom_braycurtis_continuous, get_custom_cityblock_continuous
import os
import argparse
from cluster_funcs import merge_clusters

# --------------------- Path Arguments
basePath = "/Volumes/GoogleDrive/My Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/"
dataPath = os.path.join(basePath, 'DATA', 'icu', '7days_052319/')
resPath = os.path.join(basePath, 'RESULTS', 'icu', '7days_052319/')

transition_costs = [1.00,   # [0 - 1]
                    2.73,   # [1 - 2]
                    4.36,   # [2 - 3]
                    6.74]   # [3 - 4]

# --------------------- Parser Arguments
parser = argparse.ArgumentParser(description='Merge clusters.')
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

# Load patient data
f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
ids = f['meta']['ids'][:]
kdigos = load_csv(os.path.join(dataPath, 'kdigo.csv'), ids, int)
max_kdigos = f['meta']['max_kdigo_7d'][:]
days = load_csv(os.path.join(dataPath, 'days_interp.csv'), ids, int)

dm = np.load(os.path.join(resPath, 'dm', '7days', 'kdigo_dm_%s.npy' % args.dm_tag))
lblPath = os.path.join(resPath, 'clusters', '7days', args.dm_tag, 'flat', '%d_clusters' % args.baseClustNum)

mismatch = continuous_mismatch(*transition_costs)
extension = continuous_extension(*transition_costs)
folderName = 'merged'
if args.coords == 'kdigo':
    coords = np.array([x for x in range(len(transition_costs) + 1)], dtype=float)
    folderName += '_abs'
elif args.coords == 'population':
    coords = np.array([np.sum(transition_costs[:i]) for i in range(len(transition_costs) + 1)], dtype=float)
    folderName += '_pop'

if args.dist == 'BrayCurtis':
    dist = get_custom_braycurtis_continuous(coords)
elif args.dist == 'CityBlock':
    dist = get_custom_cityblock_continuous(coords)

coords += args.coordShift
if args.coordShift:
    folderName += '_shift%E' % args.coordShift

merge_clusters(ids, kdigos, max_kdigos, days, dm, lblPath, f['meta'], mismatch=mismatch, extension=extension, dist=dist, t_lim=7, mergeType='mean', folderName=folderName)