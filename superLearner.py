# %% Imports
import os
import h5py
import pandas as pd
from time import time

from sklearn.model_selection import train_test_split
from sklearn.base import clone

# Base-Learner Models
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score
from mlens.utils import safe_print
from mlens.ensemble import Subsemble, SuperLearner

from stat_funcs import get_even_pos_neg
from deepSuperLearner import *
import copy
# %% Config
basePath = "/Volumes/GoogleDrive/My Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/"

dataPath = os.path.join(basePath, 'DATA', 'icu', '7days_030719/')
resPath = os.path.join(basePath, 'RESULTS', 'icu', '7days_030719/')
featPath = os.path.join(resPath, 'features', 'individual')
labelpath = os.path.join(resPath, 'clusters', '7days', 'custmismatch_extension_a1E+00_normBC_popcoord',
                         'combined', '18_clusters', 'merged', '15_clusters')

features = ['raw_static_custom']
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
    X = data
    featName = features[0]
else:
    X = pd.DataFrame()
    X['id'] = ids
    for i in range(len(features)):
        data = pd.read_csv(os.path.join(featPath, '%s.csv' % features[i]))
        X = X.merge(data, on='id')
    featName = '_'.join(features)

try:
    _ = X.pop('id')
except KeyError:
    pass
X = X.values
y = f['meta'][target][:]

# %% Check correlation of features with target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

sel = get_even_pos_neg(y_train, 'rand_over')
X_train = X_train[sel]
y_train = y_train[sel]

# %% Get estimators

ESTIMATORS = {'CART': DecisionTreeClassifier(),
              'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=0, n_jobs=-1),
              'RandomForest': RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
              'Plain SVM' : SVC(C=0.5, probability=True),
              'Nystroem-SVM': make_pipeline(
                   Nystroem(gamma=0.015, n_components=1000),
                   SVC(kernel='linear', C=0.5, probability=True)),
              'SampledRBF-SVM': make_pipeline(
                          RBFSampler(gamma=0.015, n_components=1000),
                          SVC(kernel='linear', C=0.5, probability=True)),
              'LogisticRegression-lbfgs': LogisticRegression(solver='lbfgs'),
              'GradientBoost' : GradientBoostingClassifier()}

def build_rfensemble(cls, **kwargs):
    """Build ML-Ensemble"""
    use = ["ExtraTrees", "RandomForest", "Plain SVM", "SampledRBF-SVM",
           "LogisticRegression-lbfgs", "Nystroem-SVM", "GradientBoost"]#, "XGBoost"]

    meta = RandomForestClassifier(n_estimators=100,
                                  random_state=SEED,
                                  n_jobs=-1)
    ens = cls(**kwargs)
    base_learners = list()
    for est_name, est in ESTIMATORS.items():
        e = clone(est)
        if est_name not in use:
            continue
        elif est_name == "MLP-adam":
            e.verbose = False
        try:
            e.set_params(**{'n_jobs': 1})
        except ValueError:
            pass

        base_learners.append((est_name, e))
    ens.add(base_learners, proba=True, shuffle=True, random_state=SEED)
    ens.add_meta(meta, shuffle=True, random_state=2)
    return ens

def build_svmensemble(cls, **kwargs):
    """Build ML-Ensemble"""
    use = ["ExtraTrees", "RandomForest", "Plain SVM", "SampledRBF-SVM",
           "LogisticRegression-lbfgs", "Nystroem-SVM", "GradientBoost"]#, "XGBoost"]

    meta = SVC()
    ens = cls(**kwargs)
    base_learners = list()
    for est_name, est in ESTIMATORS.items():
        e = clone(est)
        if est_name not in use:
            continue
        elif est_name == "MLP-adam":
            e.verbose = False
        try:
            e.set_params(**{'n_jobs': 1})
        except ValueError:
            pass

        base_learners.append((est_name, e))
    ens.add(base_learners, proba=True, shuffle=True, random_state=SEED)
    ens.add_meta(meta, shuffle=True, random_state=2)
    return ens


dsl = DeepSuperLearner(ESTIMATORS)

ESTIMATORS['SubsembleRF'] = build_rfensemble(
        Subsemble,
        partition_estimator=MiniBatchKMeans(n_clusters=5, random_state=0),
        partitions=5, verbose=1, folds=5, n_jobs=-1)

ESTIMATORS['SubsembleSVM'] = build_svmensemble(
        Subsemble,
        partition_estimator=MiniBatchKMeans(n_clusters=5, random_state=0),
        partitions=5, verbose=1, folds=5, n_jobs=-1)


ESTIMATORS['SuperLearnerRF'] = build_rfensemble(
        SuperLearner, verbose=1, folds=5, n_jobs=-1)

ESTIMATORS['SuperLearnerSVM'] = build_svmensemble(
        SuperLearner, folds=5, n_jobs=-1)

ESTIMATORS['DSL'] = dsl


base_learners = {
              'RandomForest': RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
              'Plain SVM' : SVC(C=0.5, probability=True),
              'Nystroem-SVM': make_pipeline(
                   Nystroem(gamma=0.015, n_components=1000),
                   SVC(kernel='linear', C=0.5, probability=True)),
              'LogisticRegression-lbfgs': LogisticRegression(solver='lbfgs'),
              'GradientBoost' : GradientBoostingClassifier()}

superEstimators = {}
ens = SuperLearner(folds=5, shuffle=True, random_state=SEED)
tbase = []
for est_name, est in base_learners.items():
    e = clone(est)
    tbase.append((est_name, e))
ens.add(tbase, proba=True, shuffle=True, random_state=SEED)
ens.add_meta(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1))
superEstimators['rf_meta'] = ens

ens = SuperLearner(folds=5, shuffle=True, random_state=SEED)
tbase = []
for est_name, est in base_learners.items():
    e = clone(est)
    tbase.append((est_name, e))
ens.add(tbase, proba=True, shuffle=True, random_state=SEED)
ens.add_meta(SVC(C=0.5, probability=True))
superEstimators['svm_meta'] = ens

ens = SuperLearner(folds=5, shuffle=True, random_state=SEED)
tbase = []
for est_name, est in base_learners.items():
    e = clone(est)
    tbase.append((est_name, e))
ens.add(tbase, proba=True, shuffle=True, random_state=SEED)
ens.add_meta(LogisticRegression(solver='lbfgs'))
superEstimators['log_meta'] = ens


# %% Train and print results
auc, train_time, test_time = {}, {}, {}
for name in list(superEstimators):
    estimator = superEstimators[name]
    safe_print("Training %s ... " % name)

    time_start = time()
    estimator.fit(X_train, y_train)
    train_time[name] = time() - time_start
    time_start = time()
    try:
        pre = estimator.predict_proba(X_test)[:, 1]
    except IndexError:
        pre = estimator.predict_proba(X_test)
    except AttributeError:
        pre = estimator.predict(X_test)[:, 1]
    test_time[name] = time() - time_start
    auc[name] = roc_auc_score(y_test, pre)
    safe_print("done")

safe_print()
safe_print("Classification performance:")
safe_print("===========================")
safe_print("{0: <24} {1: >10} {2: >11} {3: >12}"
           "".format("Classifier  ",
                     "train-time",
                     "test-time",
                     "roc-auc"))

safe_print("-" * 60)
for name in sorted(list(superEstimators)):
    safe_print("{0: <23} {1: >10.2f}s {2: >10.2f}s {3: >12.4f}"
               "".format(name,
                         train_time[name],
                         test_time[name],
                         auc[name]))
    
# for name in list(ESTIMATORS):
#     estimator = ESTIMATORS[name]
#     safe_print("Training %s ... " % name)
# 
#     time_start = time()
#     estimator.fit(X_train, y_train)
#     train_time[name] = time() - time_start
#     time_start = time()
#     try:
#         pre = estimator.predict_proba(X_test)[:, 1]
#     except IndexError:
#         pre = estimator.predict_proba(X_test)
#     except AttributeError:
#         pre = estimator.predict(X_test)[:, 1]
#     test_time[name] = time() - time_start
#     auc[name] = roc_auc_score(y_test, pre)
#     safe_print("done")
# 
# safe_print()
# safe_print("Classification performance:")
# safe_print("===========================")
# safe_print("{0: <24} {1: >10} {2: >11} {3: >12}"
#            "".format("Classifier  ",
#                      "train-time",
#                      "test-time",
#                      "roc-auc"))
# 
# safe_print("-" * 60)
# for name in sorted(list(ESTIMATORS)):
#     safe_print("{0: <23} {1: >10.2f}s {2: >10.2f}s {3: >12.4f}"
#                "".format(name,
#                          train_time[name],
#                          test_time[name],
#                          auc[name]))
