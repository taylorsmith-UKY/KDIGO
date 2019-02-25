import os
import numpy as np
from kdigo_funcs import load_csv, extension_penalty_func, mismatch_penalty_func, \
    pairwise_dtw_dist, get_custom_braycurtis, get_euclidean_norm, get_cityblock_norm
import h5py
import json
import sys

# ------------------------------- PARAMETERS ----------------------------------#
try:
    configurationFileName = sys.argv[1]
except IndexError:
    configurationFileName = 'kdigo_conf.json'

fp = open(configurationFileName, 'r')
conf = json.load(fp)
fp.close()

basePath = conf['basePath']
cohortName = conf['cohortName']

transition_costs = conf["transitionCosts"]

# DTW parameters
use_mismatch_penalty = conf["populationMismatch"]
use_extension_penalty = conf["populationExtension"]

# Distance parameters
dfunc = conf["distanceFunction"]
use_population_coordinates = conf["populationCoordinates"]
shift = conf["coordinateShift"]

# Duration of data to use for calculation
t_lim = conf["analysisDays"]
# -----------------------------------------------------------------------------#
dataPath = os.path.join(basePath, 'DATA', 'icu', cohortName)
resPath = os.path.join(basePath, 'RESULTS', 'icu', cohortName)

# Build distance function coordinates
if use_population_coordinates:
    dcoords = np.array([np.sum(transition_costs[:i]) for i in range(len(transition_costs) + 1)], dtype=float)
    coord_tag = '_popcoord'
else:
    dcoords = np.array([x for x in range(len(transition_costs) + 1)], dtype=float)
    coord_tag = '_kdigos'

if shift:
    if shift == 1:
        coord_tag += '_shift1'
    else:
        pow = 1
        while (shift * 10 ** pow) % 1 != 0:
            pow += 1
        coord_tag += '_shift%dE-0%d' % (shift * 10 ** pow, pow)
    dcoords += shift

# Load patient IDs and days-to-death (in case any patients died within the
# analysis windows defined by t_lims so that they can be excluded
f = h5py.File(resPath + 'stats.h5', 'r')
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
kdigos = load_csv(dataPath + 'kdigo.csv', ids, int)
days = load_csv(dataPath + 'days_interp.csv', ids, int)

# Build filename to distinguish different matrices and specify the
# mismatch and extension penalties
if use_mismatch_penalty:
    # mismatch penalty derived from population dynamics
    mismatch = mismatch_penalty_func(*transition_costs)
    dm_tag = '_popmismatch'
else:
    # absolute difference in KDIGO scores
    mismatch = lambda x, y: abs(x - y)
    dm_tag = '_absmismatch'
if use_extension_penalty:
    # mismatch penalty derived from population dynamics
    extension = extension_penalty_func(*transition_costs)
    dm_tag += '_popextension' % alpha
else:
    # no extension penalty
    extension = lambda x: 0

dtw_tag = dm_tag
if dfunc == 'braycurtis':
    dist = get_custom_braycurtis(dcoords)
    dm_tag += '_braycurtis'
elif dfunc == 'euclidean':
    dist = get_euclidean_norm(dcoords)
    dm_tag += '_euclidean'
elif dfunc == 'cityblock':
    dist = get_cityblock_norm(dcoords)
    dm_tag += '_cityblock'
dm_tag += coord_tag

# Compute pair-wise DTW and distance calculation for the included
# patients. If specified, also saves the results of the pair-wise
# DTW alignment, which can then save time for subsequent distance
# calculation.
if not os.path.exists(dm_path + '/kdigo_dm' + dm_tag + '.npy'):
    # if dfunc == 'braycurtis':
    dm = pairwise_dtw_dist(kdigos, days, ids, dm_path + '/kdigo_dm' + dm_tag + '.csv',
                           dtw_path + '/kdigo_dtwlog' + dtw_tag + '.csv',
                           mismatch=mismatch,
                           extension=extension,
                           dist=dist,
                           alpha=alpha, t_lim=t_lim)
    # elif dfunc == 'euclidean':
    #     dm = pairwise_parallel_dist_euclidean(ids, kdigos, days, dtw_path + '/kdigo_dtwlog' + dtw_tag + '.csv',
    #                                           t_lim, bc_coords)
    # elif dfunc == 'cityblock':
    #     dm = pairwise_parallel_dist_cityblock(ids, kdigos, days, dtw_path + '/kdigo_dtwlog' + dtw_tag + '.csv',
    #                                           t_lim, bc_coords)
    np.save(dm_path + '/kdigo_dm' + dm_tag, dm)
else:
    print(dm_tag + ' already completed.')
