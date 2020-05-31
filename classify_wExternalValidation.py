import os
from kdigo_funcs import load_csv, arr2csv
from classification_funcs import classify, classify_ie
import h5py
import numpy as np
from joblib import dump
import argparse
import json

modelStrs = {"rf": "Random Forest", "log": "Logistic Regression", "svm": "SVM"}
targStrs = {"died_inp": "Hospital Mortality", "make90_d50": "MAKE-90 (50% Drop)"}

##############
# --------------------- Parser Arguments
parser = argparse.ArgumentParser(description='Merge clusters.')
parser.add_argument('--config_file', action='store', type=str, dest='cfname',
                    default='kdigo_conf.json')
parser.add_argument('--ext_config_file', action='store', type=str, dest='ecfname',
                    default='utsw_conf.json')
parser.add_argument('--config_path', action='store', type=str, dest='cfpath',
                    default='')
parser.add_argument('--features', '-f', action='store', type=str, dest='feats',
                    default='sofa_norm')
parser.add_argument('--topNFeatures', '-topN', action='store', type=int, dest='topN',
                    default=-1)
parser.add_argument('--cv_num', '-cv', action='store', type=int, dest='cv',
                    default=5)
parser.add_argument('--target', '-t', action='store', type=str, dest='target',
                    default='died_inp')
parser.add_argument('--gridsearch', '-g', action='store_true', dest='grid')
parser.add_argument('--classificationModel', '-class', action='store', type=str, dest='classModel',
                    default='log', choices=['log', 'svm', 'rf', 'mvr', 'xbg'])
parser.add_argument('--engine', '-eng', action='store', type=str, dest='engine',
                    default='sklearn', choices=['statsmodels', 'sklearn'])
parser.add_argument('--meta_group', '-meta', action='store', type=str, dest='meta',
                    default='meta')
parser.add_argument('--data_file', '-df', action='store', type=str, dest='df',
                    default='stats.h5')
parser.add_argument('--sub_folder', '-sf', action='store', type=str, dest='sf',
                    default='')
parser.add_argument('--only_survivors', '-survivors', action='store_true', dest='survivors')
parser.add_argument('--utsw_external', '-ext', action='store_true', dest='utsw')
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

if args.survivors:
    sel = np.where(f["meta"]['died_inp'][:] == 0)[0]
    y = y[sel]
    ids = ids[sel]

configurationFileName = os.path.join(args.cfpath, args.ecfname)
fp = open(configurationFileName, 'r')
econf = json.load(fp)
fp.close()
extCohortName = econf['cohortName']

extBaseDataPath = os.path.join(basePath, 'DATA', 'dallas', 'csv')
extDataPath = os.path.join(basePath, 'DATA', 'dallas', analyze, extCohortName)
extResPath = os.path.join(basePath, 'RESULTS', 'dallas', analyze, extCohortName)

extf = h5py.File(os.path.join(extResPath, args.df), 'r')
extids = extf[meta_grp]['ids'][:]

extIndFeatPath = os.path.join(extResPath, 'features', 'individual')

if not os.path.exists(os.path.join(extResPath, 'classification')):
    os.mkdir(os.path.join(extResPath, 'classification'))

if not os.path.exists(os.path.join(extResPath, 'classification', '%ddays' % t_lim)):
    os.mkdir(os.path.join(extResPath, 'classification', '%ddays' % t_lim))

target = args.target
if not os.path.exists(os.path.join(extResPath, 'classification', '%ddays' % t_lim, target)):
    os.mkdir(os.path.join(extResPath, 'classification', '%ddays' % t_lim, target))
exty = extf['meta'][target][:]

if args.survivors:
    extsel = np.where(extf["meta"]['died_inp'][:] == 0)[0]
    exty = exty[extsel]
    extids = extids[extsel]

featName = args.feats

X, featNames = load_csv(os.path.join(indFeatPath, '%s.csv' % featName), ids, skip_header='keep')

if X.ndim == 1:
    X = X[:, None]

assert len(featNames) == X.shape[1]

classPath = os.path.join(resPath, 'classification', '%ddays' % t_lim, target, featName)
if not os.path.exists(classPath):
    os.mkdir(classPath)

if args.survivors:
    classPath = os.path.join(classPath, 'survivors')
    if not os.path.exists(classPath):
        os.mkdir(classPath)

classificationModel = args.classModel

modelPath = os.path.join(classPath, classificationModel)
if not os.path.exists(modelPath):
    os.mkdir(modelPath)

modelPath = os.path.join(modelPath, args.engine)
if not os.path.exists(modelPath):
    os.mkdir(modelPath)

extX, featNames = load_csv(os.path.join(extIndFeatPath, '%s.csv' % featName), extids, skip_header='keep')

if extX.ndim == 1:
    extX = extX[:, None]

assert len(featNames) == extX.shape[1]

extclassPath = os.path.join(extResPath, 'classification', '%ddays' % t_lim, target, featName)
if not os.path.exists(extclassPath):
    os.mkdir(extclassPath)

if args.survivors:
    extclassPath = os.path.join(extclassPath, 'survivors')
    if not os.path.exists(extclassPath):
        os.mkdir(extclassPath)

classificationModel = args.classModel

extModelPath = os.path.join(extclassPath, classificationModel)
if not os.path.exists(extModelPath):
    os.mkdir(extModelPath)

extModelPath = os.path.join(extModelPath, args.engine)
if not os.path.exists(extModelPath):
    os.mkdir(extModelPath)


fold_feat_header = "Fold_#," + ",".join(featNames)
slope_header = "Fold_#,Slope"
intercept_header = "Fold_#,Intercept"


# clf, probas, fold_probas, fold_lbls, imports,
# coeffs, efold_probas, intercepts, slopes = classify(X, y,
#                                                     classification_model=classificationModel,
#                                                     out_path=modelPath,
#                                                     feature_name=featName,
#                                                     gridsearch=args.grid,
#                                                     sample_method='rand_over',
#                                                     cv_num=args.cv,
#                                                     engine=args.engine,
#                                                     extX=extX, exty=exty,
#                                                     extPath=extModelPath)

clf, probas, fold_probas, fold_lbls, coeffs,\
efold_probas, intercepts, slopes, extintercepts, extslopes = classify_ie(X, y, classification_model=classificationModel,
                                                                      out_path=modelPath, feature_name=featName,
                                                                      cv_num=args.cv, engine=args.engine, extX=extX,
                                                                      exty=exty,extPath=extModelPath)

hdr = "STUDY_PATIENT_ID," + ",".join(featNames)

# np.savetxt(os.path.join(modelPath, 'predicted_probas.txt'), probas)

arr2csv(os.path.join(modelPath, 'slopes_fold.csv'), slopes, None, header=slope_header)
arr2csv(os.path.join(modelPath, 'intercepts_fold.csv'), intercepts, None, header=intercept_header)

arr2csv(os.path.join(modelPath, 'predicted_probas.csv'), probas, ids,
        header="STUDY_PATIENT_ID,PredictedProbability")
arr2csv(os.path.join(modelPath, 'predicted_probas_fold.csv'), fold_probas, None)
arr2csv(os.path.join(modelPath, 'true_labels_fold.csv'), fold_lbls, None)
# arr2csv(os.path.join(modelPath, "feature_importances_fold.csv"), imports, None)
dump(clf, os.path.join(modelPath, 'classifier.joblib'))

arr2csv(os.path.join(modelPath, 'features.csv'), X, ids, header=hdr)
arr2csv(os.path.join(modelPath, 'true_labels.csv'), y, ids, fmt="%d", header="STUDY_PATIENT_ID,%s" % target)

arr2csv(os.path.join(modelPath, 'coefficients_fold.csv'), coeffs, None, header=fold_feat_header)
arr2csv(os.path.join(modelPath, 'coefficients_avg.csv'), np.mean(coeffs, axis=0), featNames,
        header="Feature,Coefficient")
arr2csv(os.path.join(modelPath, 'odds_ratios_fold.csv'), np.exp(coeffs), None, header=fold_feat_header)
arr2csv(os.path.join(modelPath, 'odds_ratios_avg.csv'), np.mean(np.exp(coeffs), axis=0), featNames,
        header="Feature,Odds-Ratio")


# Save external validation
arr2csv(os.path.join(extModelPath, 'predicted_probas.csv'), np.mean(efold_probas, axis=0), extids,
        header="STUDY_PATIENT_ID,PredictedProbability")
arr2csv(os.path.join(extModelPath, 'predicted_probas_fold.csv'), efold_probas, None)
arr2csv(os.path.join(extModelPath, 'true_labels.csv'), exty, extids, fmt="%d",
        header="STUDY_PATIENT_ID,%s" % target)
arr2csv(os.path.join(extModelPath, 'features.csv'), extX, extids, header=hdr)

arr2csv(os.path.join(extModelPath, 'coefficients_fold.csv'), coeffs, None, header=fold_feat_header)
arr2csv(os.path.join(extModelPath, 'coefficients_avg.csv'), np.mean(coeffs, axis=0), featNames,
        header="Feature,Coefficient")
arr2csv(os.path.join(extModelPath, 'odds_ratios_fold.csv'), np.exp(coeffs), None, header=fold_feat_header)
arr2csv(os.path.join(extModelPath, 'odds_ratios_avg.csv'), np.mean(np.exp(coeffs), axis=0), featNames,
        header="Feature,Odds-Ratio")

arr2csv(os.path.join(extModelPath, 'slopes_fold.csv'), extslopes, None, header=slope_header)
arr2csv(os.path.join(extModelPath, 'intercepts_fold.csv'), extintercepts, None, header=intercept_header)
