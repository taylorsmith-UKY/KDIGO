import h5py
import numpy as np
from kdigo_funcs import load_csv
from dtw_distance import continuous_mismatch, continuous_extension, mismatch_penalty_func, extension_penalty_func
import os
import argparse
import json
from scipy.spatial.distance import squareform
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from utility_funcs import arr2csv, get_dm_tag
from DBA import performDBA

# --------------------- Parser Arguments
parser = argparse.ArgumentParser(description='Merge clusters.')
parser.add_argument('--config_file', action='store', nargs=1, type=str, dest='cfname',
                    default='kdigo_conf.json')
parser.add_argument('--config_path', action='store', nargs=1, type=str, dest='cfpath',
                    default='')
parser.add_argument('--use_extension', '-ext', action='store_true', dest='ext')
parser.add_argument('--use_mismatch', '-mism', action='store_true', dest='mism')
parser.add_argument('--agg_ext', '-agg', action='store_true', dest='agg')
parser.add_argument('--pop_coords', '-pcoords', '-pc', action='store_true', dest='popcoords')
parser.add_argument('--extensionWeight', '-alpha', action='store', nargs=1, type=float, dest='alpha',
                    default=1.0)
parser.add_argument('--laplacian_type', '-lt', action='store', type=str, dest='lap', default='none', choices=['none', 'individual', 'aggregated'])
parser.add_argument('--folder_name', '-fname', action='store', nargs=1, type=str, dest='fname',
                    default='')
parser.add_argument('--distance_function', '-dfunc', action='store', nargs=1, type=str, dest='dfunc',
                    default='braycurtis')
parser.add_argument('--baseClustNum', '-n', action='store', nargs=1, type=int, dest='baseClustNum',
                    required=True)
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

transition_costs = np.loadtxt(os.path.join(dataPath, 'kdigo_transition_weights.csv'),
                              delimiter=',', usecols=1, skiprows=1)
folderName = args.fname[0]

if args.popcoords:
    coords = np.array([np.sum(transition_costs[:i]) for i in range(len(transition_costs) + 1)], dtype=float)
else:
    coords = np.array([x for x in range(len(transition_costs) + 1)], dtype=float)

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

# Load patient data
f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
all_ids = f['meta']['ids'][:]

dm_tag, dtw_tag = get_dm_tag(args.mism, args.ext, args.alpha[0], args.agg, args.popcoords, args.dfunc, args.lap)

dm = np.load(os.path.join(resPath, 'dm', '7days', 'kdigo_dm_%s.npy' % dm_tag))
lblPath = os.path.join(resPath, 'clusters', '7days', dm_tag, 'flat', '%d_clusters' % args.baseClustNum[0], folderName)

ids = np.loadtxt(os.path.join(lblPath, 'clusters.csv'), usecols=0, dtype=int, delimiter=',')
lbls = load_csv(os.path.join(lblPath, 'clusters.csv'), ids, str)
kdigos = load_csv(os.path.join(dataPath, 'kdigo.csv'), ids, int)
days = load_csv(os.path.join(dataPath, 'days_interp.csv'), ids, int)
for i in range(len(kdigos)):
    kdigos[i] = kdigos[i][np.where(days[i] <= t_lim)]
    days[i] = days[i][np.where(days[i] <= t_lim)]

pt_sel = np.array([x in ids for x in all_ids])
max_kdigos = f['meta']['max_kdigo_7d'][:][pt_sel]
dm = squareform(squareform(dm)[np.ix_(pt_sel, pt_sel)])

if hasattr(args.alpha, '__len__'):
    alpha = args.alpha[0]
else:
    alpha = args.alpha

centerName = 'centers'
if args.mism and args.ext:
    centerName += '_popDTW'
    centerName += '_alpha%.0E' % alpha
    if args.agg:
        centerName += '_aggExt'
else:
    if args.mism or args.ext:
        raise RuntimeError('Argument mismatch (use both extension and mismatch penalties or neither)')
    centerName += '_normDTW'

centers = {}
if not os.path.exists(os.path.join(lblPath, centerName + '.csv')):
    with PdfPages(os.path.join(lblPath, centerName + '.pdf')) as pdf:
        for lbl in np.unique(lbls):
            idx = np.where(lbls == lbl)[0]
            dm_sel = np.ix_(idx, idx)
            tkdigos = [kdigos[x] for x in idx]
            tdm = squareform(squareform(dm)[dm_sel])
            center, stds, confs = performDBA(tkdigos, tdm, mismatch=mismatch, extension=extension, extraDesc=' for cluster ' + lbl, alpha=alpha, aggExt=args.agg)
            centers[lbl] = center
            plt.figure()
            plt.plot(center)
            plt.fill_between(range(len(center)), center-confs, center+confs)
            plt.xticks(range(0, len(center), 4), ['%d' % x for x in range(len(center))])
            plt.yticks(range(5), ['0', '1', '2', '3', '3D'])
            plt.ylim((-0.5, 4.5))
            plt.title(lbl)
            pdf.savefig(dpi=600)
        arr2csv(os.path.join(lblPath, centerName + '.csv'), list(centers.values()), list(centers.keys()))

