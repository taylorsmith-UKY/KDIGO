import numpy as np
from cluster_funcs import getFlatClusters
from kdigo_funcs import get_dmTag
import h5py
import os
from scipy.spatial.distance import squareform
import json

# PARAMETERS
#################################################################################
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
######################################

dm_tag = get_dmTag(use_mismatch_penalty, use_extension_penalty, use_population_coordinates, shift, dfunc)

dataPath = os.path.join(basePath, 'DATA', 'icu', cohortName)
resPath = os.path.join(basePath, 'RESULTS', 'icu', cohortName)

f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
ids = f['meta']['ids'][:]
max_kdigo = f['meta']['max_kdigo_7d'][:]

if not os.path.exists(os.path.join(resPath, 'clusters')):
    os.mkdir(os.path.join(resPath, 'clusters'))

if not os.path.exists(os.path.join(resPath, 'clusters', '%ddays' % t_lim)):
    os.mkdir(os.path.join(resPath, 'clusters', '%ddays' % t_lim))

save_path = os.path.join(resPath, 'clusters', '%ddays' % t_lim, dm_tag)
if not os.path.exists(save_path):
    os.mkdir(save_path)
save_path = os.path.join(save_path, 'split_by_kdigo')
if not os.path.exists(save_path):
    os.mkdir(save_path)
if os.path.isfile(resPath + 'dm/%ddays/kdigo_dm_%s.npy' % (t_lim, dm_tag)):
    try:
        dm = np.load(resPath + 'dm/%ddays/kdigo_dm_%s.npy' % (t_lim, dm_tag))[:, 2]
    except IndexError:
        dm = np.load(resPath + 'dm/%ddays/kdigo_dm_%s.npy' % (t_lim, dm_tag))
else:
    dm = np.loadtxt(resPath + 'dm/%ddays/kdigo_dm_%s.csv' % (t_lim, dm_tag), delimiter=',', usecols=2)
sqdm = squareform(dm)
# max KDIGO 1
idx = np.where(max_kdigo == 1)[0]
tdm_sel = np.ix_(idx, idx)
tdm = squareform(sqdm[tdm_sel])
tpath = os.path.join(save_path, 'kdigo_1')
if not os.path.exists(tpath):
    os.mkdir(tpath)
getFlatClusters(f, ids, dm, data_path=dataPath, save=tpath)
# max KDIGO 2+
idx = np.where(max_kdigo > 1)[0]
tdm_sel = np.ix_(idx, idx)
tdm = squareform(sqdm[tdm_sel])
tpath = os.path.join(save_path, 'kdigo_2plus')
if not os.path.exists(tpath):
    os.mkdir(tpath)
getFlatClusters(f, ids, dm, data_path=dataPath, save=tpath)
