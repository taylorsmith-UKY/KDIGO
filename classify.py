import os
from kdigo_funcs import load_csv
from classification_funcs import classify
from sklearn.feature_selection import SelectKBest, VarianceThreshold, RFECV  # RFE, chi2,
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
import h5py
import numpy as np

##############
basePath = "/Volumes/GoogleDrive/My Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/"

dataPath = os.path.join(basePath, 'DATA', 'icu', '7days_021319/')
resPath = os.path.join(basePath, 'RESULTS', 'icu', '7days_021319/')
dm_tag = 'custmismatch_extension_a1E+00_normBC_popcoord'

# featNames = ['sofa_norm', 'apache_norm']
featNames = ['everything_mean', 'everything_center']

# ['VarianceThreshold', vThresh]
# ['UnivariateSelection', ScoringFunction, k]
# ['RFECV', ExtraTreesClassifier(n_estimators=1000,
#                                max_features=128,
#                                n_jobs=n_jobs,
#                                random_state=0)]
# ['RFECV', SVC(kernel='linear')]
featureSelectionModels = [['RFECV', ExtraTreesClassifier(n_estimators=100,
                                                         n_jobs=-1,
                                                         random_state=0), 'ExtraTrees'],
                          ['RFECV', SVC(kernel='linear'), 'SVM']]
gridsearch = False
###############

f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
t_lim = 7
ids = f['meta']['ids'][:]
dtd = f['meta']['days_to_death'][:]
pt_sel = np.union1d(np.where(np.isnan(dtd))[0], np.where(dtd >= 2)[0])
died = f['meta']['died_inp'][:][pt_sel]
ids = ids[pt_sel]

indFeatPath = os.path.join(resPath, 'features', 'individual')
clustFeatPath = os.path.join(resPath, 'features', '7days', dm_tag, 'final')

for featName in featNames:
    X, hdr = load_csv(os.path.join(clustFeatPath, '%s.csv' % featName), ids, skip_header='keep')
    y = died
    if len(hdr) == X.shape[1] + 1:
        hdr = hdr[1:]
    assert len(hdr) == X.shape[1]
    classPath = os.path.join(resPath, 'classification', '7days', 'died_inp', dm_tag)
    if not os.path.exists(classPath):
        os.mkdir(classPath)
    classPath = os.path.join(classPath, featName)
    if not os.path.exists(classPath):
        os.mkdir(classPath)
    for selectionModel in featureSelectionModels:
        modelName = selectionModel[0]
        tX = X
        if 'everything' in featName:
            # Variance threshold
            if modelName == 'VarianceThreshold':
                vThresh = selectionModel[1]
                sel = VarianceThreshold(threshold=(vThresh * (1 - vThresh)))
                sel.fit(tX)
                tX = sel.transform(tX)
                selectionPath = os.path.join(classPath, 'vthresh_%d' % (100 * vThresh))
                df = open(os.path.join(selectionPath, 'feature_scores.txt'), 'w')
                df.write('feature_name,variance\n')
                for i in range(len(hdr)):
                    df.write('%s,%f\n' % (hdr[i], sel.variances_[i]))
                df.close()
            # Univariate Selection
            elif modelName == 'UnivariateSelection':
                scoringFunction = selectionModel[1]
                k = selectionModel[2]
                sel = SelectKBest(scoringFunction, k=k)
                sel.fit(tX, y)
                tX = sel.transform(tX)
                selectionPath = os.path.join(classPath, 'uni_%s_%d' % (scoringFunction.__name__, k))
                df = open(os.path.join(selectionPath, 'feature_scores.txt'), 'w')
                df.write('feature_name,score\n')
                for i in range(len(hdr)):
                    df.write('%s,%f\n' % (hdr[i], sel.scores_[i]))
                df.close()
            elif modelName == 'RFECV':
                estimator = selectionModel[1]
                rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(4),
                              scoring='f1', verbose=1, n_jobs=-1)
                rfecv.fit(tX, y)
                print("Optimal number of features : %d" % rfecv.n_features_)
                tX = tX[:, rfecv.support_]
                selectionPath = os.path.join(classPath, 'RFECV')
                selectedFeats = hdr[rfecv.support_]
                if not os.path.exists(selectionPath):
                    os.mkdir(selectionPath)
                selectionPath = os.path.join(selectionPath, selectionModel[2])
                if not os.path.exists(selectionPath):
                    os.mkdir(selectionPath)
                df = open(os.path.join(selectionPath, 'feature_ranking.txt'), 'w')
                df.write('feature_name,rank\n')
                for i in range(len(hdr)):
                    df.write('%s,%d\n' % (hdr[i], rfecv.ranking_[i]))
                df.close()
            else:
                selectionPath = classPath
            if not os.path.exists(selectionPath):
                os.mkdir(selectionPath)
            else:
                selectionPath = classPath
                if not os.path.exists(selectionPath):
                    os.mkdir(selectionPath)
        for classificationModel in ['rf', 'svm']:
            modelPath = os.path.join(selectionPath, classificationModel)
            if not os.path.exists(modelPath):
                os.mkdir(modelPath)
            # Feature Selection
            classify(tX, y, classification_model='svm', out_path=modelPath, feature_name=featName,
                     gridsearch=gridsearch, sample_method='rand_over')
