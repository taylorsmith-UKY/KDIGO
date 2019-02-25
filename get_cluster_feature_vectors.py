import numpy as np
import h5py
from cluster_funcs import assign_feature_vectors
from kdigo_funcs import cluster_feature_vectors, load_csv, arr2csv, get_cluster_features
import os

# --------------------------------------------------- PARAMETERS ----------------------------------------------------- #
basePath = "/Volumes/GoogleDrive/My Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/"

dataPath = os.path.join(basePath, 'DATA', 'icu', '7days_021319/')
resPath = os.path.join(basePath, 'RESULTS', 'icu', '7days_021319/')

meta_grp = 'meta'

# methods = ['merged', ]
cluster_method = 'flat'

dm_tag = '_custmismatch_extension_a1E+00_normBC_popcoord'

n_days = 7

features = {'descriptive_features': ['mean', 'mean_bin', 'center'],
            'slope_norm': ['mean', 'center'],
            'template_norm': ['mean', 'center'],
            'slope_ratio': ['mean', 'center'],
            'template_ratio': ['mean', 'center']}

# -------------------------------------------------------------------------------------------------------------------- #

f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')

day_str = '%ddays' % n_days
if not os.path.exists(os.path.join(resPath, 'features', day_str)):
    os.mkdir(os.path.join(resPath, 'features', day_str))


dm = np.load(os.path.join(resPath, 'dm', day_str, 'kdigo_dm%s.npy' % dm_tag))

lblPath = os.path.join(resPath, 'clusters', day_str, dm_tag[1:], 'final')

ids = np.loadtxt(os.path.join(lblPath, 'clusters.csv'), delimiter=',', usecols=0, dtype=int)
lbls = load_csv(os.path.join(lblPath, 'clusters.csv'), ids, str)

for feature in list(features):
    try:
        individual = load_csv(
            os.path.join(resPath, 'features', 'individual', feature + '.csv'), ids)
    except ValueError:
        individual = load_csv(
            os.path.join(resPath, 'features', 'individual', feature + '.csv'), ids, skip_header=True)

    fpath = os.path.join(resPath, 'features', day_str, dm_tag[1:])
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    # fpath = os.path.join(fpath, cluster_method)
    # if not os.path.exists(fpath):
    #     os.mkdir(fpath)
    # fpath = os.path.join(fpath, dirname)
    # if not os.path.exists(fpath):
    #     os.mkdir(fpath)
    fpath = os.path.join(fpath, 'final')
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    for op in features[feature]:
        all_feats, cluster_feats = get_cluster_features(individual, lbls, dm, op=op)
        fname = feature + '_' + op + '.csv'
        arr2csv(os.path.join(fpath, fname), all_feats, ids, fmt='%.4f')
