import os
from kdigo_funcs import load_csv, arr2csv
from classification_funcs import classify, feature_selection
import h5py
import numpy as np
from joblib import dump
import argparse
import json
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

##############
# --------------------- Parser Arguments
parser = argparse.ArgumentParser(description='Merge clusters.')
parser.add_argument('--config_file', action='store', type=str, dest='cfname',
                    default='kdigo_conf.json')
parser.add_argument('--config_path', action='store', type=str, dest='cfpath',
                    default='')
parser.add_argument('--features', '-f', action='store', type=str, dest='feats',
                    default='sofa_norm')
parser.add_argument('--cv_num', '-cv', action='store', type=int, dest='cv',
                    default=5)
parser.add_argument('--target', '-t', action='store', type=str, dest='target',
                    default='died_inp')
parser.add_argument('--gridsearch', '-g', action='store_true', dest='grid')
parser.add_argument('--selectionModel', '-sel', action='store', type=str, dest='selModel',
                    default=None, choices=['extratrees', 'svm', 'logreg', 'xbg', 'recursive'])
parser.add_argument('--classificationModel', '-class', action='store', type=str, dest='classModel',
                    default='log', choices=['log', 'svm', 'rf', 'mvr', 'xbg'])
parser.add_argument('--meta_group', '-meta', action='store', type=str, dest='meta',
                    default='meta')
parser.add_argument('--data_file', '-df', action='store', type=str, dest='df',
                    default='stats.h5')
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
meta_grp = args.meta

baseDataPath = os.path.join(basePath, 'DATA', 'all_sheets')
dataPath = os.path.join(basePath, 'DATA', analyze, cohortName)
resPath = os.path.join(basePath, 'RESULTS', analyze, cohortName)

f = h5py.File(os.path.join(resPath, args.df), 'r')
ids = f[meta_grp]['ids'][:]

indFeatPath = os.path.join(resPath, 'features', 'individual')

if not os.path.exists(os.path.join(resPath, 'classification')):
    os.mkdir(os.path.join(resPath, 'classification'))

if not os.path.exists(os.path.join(resPath, 'classification', '%ddays' % t_lim)):
    os.mkdir(os.path.join(resPath, 'classification', '%ddays' % t_lim))

target = args.target
if not os.path.exists(os.path.join(resPath, 'classification', '%ddays' % t_lim, target)):
    os.mkdir(os.path.join(resPath, 'classification', '%ddays' % t_lim, target))
y = f['meta'][target][:]

feat = args.feats
# if len(feat[0]) == 1:
X, hdr = load_csv(os.path.join(indFeatPath, '%s.csv' % feat), ids, skip_header='keep')
if X.ndim == 1:
    X = X[:, None]
featName = feat
assert len(hdr) == X.shape[1]

classPath = os.path.join(resPath, 'classification', '%ddays' % t_lim, target, featName)
if not os.path.exists(classPath):
    os.mkdir(classPath)

selectionModel = args.selModel

if selectionModel == 'recursive':
    params = [ExtraTreesClassifier(), 5]
    smodel, tX = feature_selection(X, y, selectionModel, params)
    selectionPath = os.path.join(classPath, 'RFECV')
    if not os.path.exists(selectionPath):
        os.mkdir(selectionPath)
    thdr = 'STUDY_PATIENT_ID,'+','.join([hdr[x] for x in np.where(smodel.support_)[0]])
    arr2csv(os.path.join(selectionPath, "selected_features.csv"), tX, ids, header=thdr)

else:
    tX = X
    selectionPath = classPath
if not os.path.exists(selectionPath):
    os.mkdir(selectionPath)
classificationModel = args.classModel
modelPath = os.path.join(selectionPath, classificationModel)
if not os.path.exists(modelPath):
    os.mkdir(modelPath)
# else:
#     if os.path.exists(os.path.join(modelPath, "evaluation_summary.png")):
#         exit()
# Feature Selection
clf, probas, fold_probas, fold_lbls, imports = classify(tX, y, classification_model=classificationModel, out_path=modelPath,
                                               feature_name=featName, gridsearch=args.grid, sample_method='rand_over',
                                               cv_num=args.cv)
np.savetxt(os.path.join(modelPath, 'predicted_probas.txt'), probas)
arr2csv(os.path.join(modelPath, 'predicted_probas_fold.csv'), fold_probas, None)
arr2csv(os.path.join(modelPath, 'true_labels_fold.csv'), fold_lbls, None)
arr2csv(os.path.join(modelPath, "feature_importances_fold.csv"), imports, None)
dump(clf, os.path.join(modelPath, 'classifier.joblib'))

model = SelectFromModel(clf, prefit=True)
selected = model.transform(tX)
sel_feats = list(np.array(hdr)[model.get_support()])
sel_header = ','.join(['STUDY_PATIENT_ID', ] + sel_feats)
arr2csv(os.path.join(modelPath, 'selectedFeatures.csv'), selected, ids, header=sel_header)

if hasattr(clf, "coef_"):
    np.savetxt(os.path.join(modelPath, 'coefficients.csv'), clf.coef_[0])
    np.savetxt(os.path.join(modelPath, 'odds_ratios.csv'), np.exp(clf.coef_[0]))

elif hasattr(clf, "feature_importances_"):
    np.savetxt(os.path.join(modelPath, 'feature_importances.csv'), clf.feature_importances_)


