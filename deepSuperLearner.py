# %% Imports
import os
import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from deepSuperLearner import *
# Base-Learner Models
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from stat_funcs import get_even_pos_neg
# %% Config
basePath = "/Volumes/GoogleDrive/My Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/"

dataPath = os.path.join(basePath, 'DATA', 'icu', '7days_030719/')
resPath = os.path.join(basePath, 'RESULTS', 'icu', '7days_030719/')
featPath = os.path.join(resPath, 'features', 'individual')
labelpath = os.path.join(resPath, 'clusters', '7days', 'custmismatch_extension_a1E+00_normBC_popcoord',
                         'combined', '18_clusters', 'merged', '15_clusters')

features = ['sofa_norm', 'apache_norm', 'descriptive_features']
target = 'died_inp'

# %% Load Data
SEED = 123

f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
ids = f['meta']['ids'][:]

if len(features) == 1:
    data = pd.read_csv(os.path.join(featPath, '%s.csv' % features[0]))
    try:
        ref_ids = data.pop('ids')
    except KeyError:
        ref_ids = data.pop('id')
    featName = features[0]
else:
    X = pd.DataFrame()
    X['id'] = ids
    for i in range(len(features)):
        data = pd.read_csv(os.path.join(featPath, '%s.csv' % features[i]))
        X = X.merge(data, on='id')
    featName = '_'.join(features)

_ = X.pop('id')
X = X.values
y = f['meta'][target][:]

# %% Check correlation of features with target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

sel = get_even_pos_neg(y_train, 'rand_over')
X_train = X_train[sel]
y_train = y_train[sel]

# %% Get estimators
ERT_learner = ExtraTreesClassifier(n_estimators=200, max_depth=None, max_features=1)
kNN_learner = KNeighborsClassifier(n_neighbors=11)
LR_learner = LogisticRegression(solver='lbfgs')
RFC_learner = RandomForestClassifier(n_estimators=200, max_depth=None)
XGB_learner = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=1.)
Base_learners = {'ExtremeRandomizedTrees': ERT_learner, 'kNearestNeighbors': kNN_learner,
                 'LogisticRegression': LR_learner,
                 'RandomForestClassifier': RFC_learner, 'XGBClassifier': XGB_learner}

DSL_learner = DeepSuperLearner(Base_learners)
DSL_learner.fit(X_train, y_train)
DSL_learner.get_precision_recall(X_test, y_test, show_graphs=True)
y_pred = DSL_learner.predict(X_test)