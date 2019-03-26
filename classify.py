import os
from kdigo_funcs import load_csv
from classification_funcs import classify
from sklearn.feature_selection import SelectKBest, VarianceThreshold, RFECV  # RFE, chi2,
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
import h5py
import numpy as np
from joblib import dump

##############
basePath = "/Volumes/GoogleDrive/My Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/"

dataPath = os.path.join(basePath, 'DATA', 'icu', '7days_030719/')
resPath = os.path.join(basePath, 'RESULTS', 'icu', '7days_030719/')

# Individual clinical
feats = [#[['sofa_norm', ], [0, ]],
         # [['apache_norm', ], [0, ]],
         #
         # # Individual trajectory
         # [['descriptive_features', ], [0, 1]],
         # [['slope_features_ratio', ], [0, 1]],
         # [['template_features_ratio', ], [0, 1]],
         #
         # # SOFA + trajectory
         # [['sofa_norm', 'descriptive_features', ], [0, 1]],
         # [['sofa_norm', 'slope_features_ratio', ], [0, 1]],
         # [['sofa_norm', 'template_features_ratio', ], [0, 1]],
         # [['sofa_norm', 'descriptive_features', 'slope_features_ratio', 'template_features_ratio'], [0, 1]],

         # APACHE + trajectory
         [['apache_norm', 'descriptive_features', ], [0, 1]],
         [['apache_norm', 'slope_features_ratio', ], [0, 1]],
         [['apache_norm', 'template_features_ratio', ], [0, 1]],
         [['apache_norm', 'descriptive_features', 'slope_features_ratio', 'template_features_ratio'], [0, 1]]]

# ['VarianceThreshold', vThresh]
# ['UnivariateSelection', ScoringFunction, k]
# ['RFECV', ExtraTreesClassifier(n_estimators=1000,
#                                max_features=128,
#                                n_jobs=n_jobs,
#                                random_state=0)]
# ['RFECV', SVC(kernel='linear')]
featureSelectionModels = [[None],
                          ['RFECV', ExtraTreesClassifier(n_estimators=150,
                                                         n_jobs=-1,
                                                         random_state=0), 'ExtraTrees'],
                          ['RFECV', SVC(kernel='linear'), 'SVM'],
                          ['RFECV', LogisticRegression(), 'LogReg']]
gridsearch = False
featureEliminationScore = 'f1'
###############

f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
t_lim = 7
ids = f['meta']['ids'][:]
dtd = f['meta']['days_to_death'][:]
pt_sel = np.union1d(np.where(np.isnan(dtd))[0], np.where(dtd >= 2)[0])
died = f['meta']['died_inp'][:][pt_sel]
ids = ids[pt_sel]

indFeatPath = os.path.join(resPath, 'features', 'individual')

if not os.path.exists(os.path.join(resPath, 'classification')):
    os.mkdir(os.path.join(resPath, 'classification'))

if not os.path.exists(os.path.join(resPath, 'classification', '7days')):
    os.mkdir(os.path.join(resPath, 'classification', '7days'))

if not os.path.exists(os.path.join(resPath, 'classification', '7days', 'died_inp')):
    os.mkdir(os.path.join(resPath, 'classification', '7days', 'died_inp'))

for i in range(len(feats)):
    feat = feats[i]
    if len(feat[0]) == 1:
        X, hdr = load_csv(os.path.join(indFeatPath, '%s.csv' % feat[0][0]), ids, skip_header='keep')
        featName = feat[0][0]
    else:
        data = []
        hdrs = []
        for j in range(len(feat[0])):
            X, hdr = load_csv(os.path.join(indFeatPath, '%s.csv' % feat[0][j]), ids, skip_header='keep')
            data.append(X)
            hdrs.append(hdr)
        X = np.hstack(data)
        hdr = np.concatenate(hdrs)
        featName = '_'.join(feat[0])
    y = died
    assert len(hdr) == X.shape[1]
    classPath = os.path.join(resPath, 'classification', '7days', 'died_inp', featName)
    if not os.path.exists(classPath):
        os.mkdir(classPath)
    for selectionModel in [featureSelectionModels[x] for x in feat[1]]:
        modelName = selectionModel[0]
        tX = X
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
            selectionPath = os.path.join(classPath, 'RFECV')
            if not os.path.exists(selectionPath):
                os.mkdir(selectionPath)
            selectionPath = os.path.join(selectionPath, selectionModel[2])
            if not os.path.exists(selectionPath):
                os.mkdir(selectionPath)
            selectionPath = os.path.join(selectionPath, featureEliminationScore)
            if os.path.exists(os.path.join(selectionPath, 'feature_ranking.txt')):
                rankData = np.loadtxt(os.path.join(selectionPath, 'feature_ranking.txt'), delimiter=',', usecols=1, skiprows=1)
                support = np.array([x == 1 for x in rankData])
            else:
                estimator = selectionModel[1]
                rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(4),
                              scoring=featureEliminationScore, verbose=1, n_jobs=-1)
                rfecv.fit(tX, y)
                print("Optimal number of features : %d" % rfecv.n_features_)
                tX = tX[:, rfecv.support_]

                if not os.path.exists(selectionPath):
                    os.mkdir(selectionPath)
                selectedFeats = hdr[rfecv.support_]
                df = open(os.path.join(selectionPath, 'feature_ranking.txt'), 'w')
                df.write('feature_name,rank\n')
                for i in range(len(hdr)):
                    df.write('%s,%d\n' % (hdr[i], rfecv.ranking_[i]))
                df.close()
        else:
            selectionPath = classPath
        if not os.path.exists(selectionPath):
            os.mkdir(selectionPath)
        for classificationModel in ['rf', 'svm', 'log']:
            modelPath = os.path.join(selectionPath, classificationModel)
            if not os.path.exists(modelPath):
                os.mkdir(modelPath)
            # Feature Selection
            clf = classify(tX, y, classification_model=classificationModel, out_path=modelPath, feature_name=featName,
                           gridsearch=gridsearch, sample_method='rand_over')
