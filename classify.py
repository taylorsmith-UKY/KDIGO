import os
from kdigo_funcs import load_csv, arr2csv
from classification_funcs import classify, feature_selection
from sklearn.model_selection import StratifiedKFold
import h5py
import numpy as np
from joblib import dump

##############
basePath = "/Volumes/GoogleDrive/My Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/"

dataPath = os.path.join(basePath, 'DATA', 'icu', '7days_052319/')
resPath = os.path.join(basePath, 'RESULTS', 'icu', '7days_052319/')

# dataPath = os.path.join(basePath, 'DATA', 'dallas', 'icu', '7days_051319/')
# resPath = os.path.join(basePath, 'RESULTS', 'dallas', 'icu', '7days_051319/')

# Individual clinical
feats = [[['sofa_norm', ], [0, ]],
         [['apache_norm', ], [0, ]],

         [['max_kdigo'], [0, ]],
         [['max_kdigo', 'new_descriptive_norm'], [0, ]],

         # Individual trajectory
         # [['descriptive_features', ], [0, ]],
         #[['new_descriptive_norm', ], [0, ]],
# [['descriptive_features_052319_2d_norm', ], [0, ]],
# [['descriptive_features_052319_3d_norm', ], [0, ]],
         # [['slope_features_ratio', ], [0, ]],
         # [['template_features_ratio', ], [0, ]],]

         # SOFA + trajectory
         # [['sofa_norm', 'descriptive_features', ], [0, ]],
         [['sofa_norm', 'new_descriptive_norm', ], [0, ]],
# [['sofa_norm', 'descriptive_features_052319_2d_norm', ], [0, ]],
# [['sofa_norm', 'descriptive_features_052319_3d_norm', ], [0, ]],
         # # [['sofa_norm', 'slope_features_ratio', ], [0, ]],
         # # [['sofa_norm', 'template_features_ratio', ], [0, ]],
         # # [['sofa_norm', 'descriptive_features', 'slope_features_ratio', 'template_features_ratio'], [0, 1, 4]],
         #
         # # APACHE + trajectory
    # [['apache_norm', 'descriptive_features_052319_2d_norm', ], [0, ]],
    # [['apache_norm', 'descriptive_features_052319_3d_norm', ], [0, ]], ]
    #      [['apache_norm', 'descriptive_features', ], [0, ]],
         [['apache_norm', 'new_descriptive_norm', ], [0, ]],]
         # [['sofa_norm', 'apache_norm', 'descriptive_features', ], [0, ]]]
         # [['apache_norm', 'slope_features_ratio', ], [0, ]],
         # [['apache_norm', 'template_features_ratio', ], [0, ]],
         # [['apache_norm', 'descriptive_features', 'slope_features_ratio', 'template_features_ratio'], [0, 1, 4]],
         # [['descriptive_features', 'slope_features_ratio', 'template_features_ratio'], [0, 1, 4]]]
         #
         # [['apache_norm', 'sofa_norm', 'descriptive_features', ], [0, ]]]
         #   [['raw_static_custom', ], [4, ]]]

models = ['XBG']


targets = [#'m90_admit_30dbuf_2pts_25', 'm90_admit_30dbuf_2pts_50',
           #'m90_disch_30dbuf_2pts_25', 'm90_disch_30dbuf_2pts_50']
            'make90_50',]
# for ref in ['admit', 'disch']:
#     for buf in [0, 15, 30]:
#         for ct in [1, 2]:
#             targets.append('m90_%s_%ddbuf_%dpts_25' % (ref, buf, ct))
#             targets.append('m90_%s_%ddbuf_%dpts_50' % (ref, buf, ct))

# ['VarianceThreshold', vThresh]
# ['UnivariateSelection', ScoringFunction, k]
# ['RFECV', ExtraTreesClassifier(n_estimators=1000,
#                                max_features=128,
#                                n_jobs=n_jobs,
#                                random_state=0)]
# ['RFECV', SVC(kernel='linear')]
featureSelectionModels = [None,
                          ['RFECV', 'ExtraTrees', 'f1'],
                          ['RFECV', 'SVM', 'f1'],
                          ['RFECV', 'LogReg', 'f1'],
                          ['RFECV', 'XBG', 'f1']]
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

for target in targets:
    if not os.path.exists(os.path.join(resPath, 'classification', '7days', target)):
        os.mkdir(os.path.join(resPath, 'classification', '7days', target))
    y = f['meta'][target][:][pt_sel]
    for i in range(len(feats)):
        feat = feats[i]
        if len(feat[0]) == 1:
            X, hdr = load_csv(os.path.join(indFeatPath, '%s.csv' % feat[0][0]), ids, skip_header='keep')
            if X.ndim == 1:
                X = X[:, None]
            featName = feat[0][0]
        else:
            data = []
            hdrs = []
            for j in range(len(feat[0])):
                X, hdr = load_csv(os.path.join(indFeatPath, '%s.csv' % feat[0][j]), ids, skip_header='keep')
                if X.ndim == 1:
                    X = X[:, None]
                data.append(X)
                hdrs.append(hdr)
            X = np.hstack(data)
            hdr = np.concatenate(hdrs)
            featName = '_'.join(feat[0])
        assert len(hdr) == X.shape[1]
        # for i in range(X.shape[1]):
        #     X[np.where(np.isnan(X[:, i]))] = 0

        classPath = os.path.join(resPath, 'classification', '7days', target, featName)
        if not os.path.exists(classPath):
            os.mkdir(classPath)
        for selectionModel in [featureSelectionModels[x] for x in feat[1]]:
            if selectionModel is not None:
                tX, selectionPath = feature_selection(X, y, hdr, selectionModel, classPath)
            else:
                tX = X
                selectionPath = classPath
            if not os.path.exists(selectionPath):
                os.mkdir(selectionPath)
            for classificationModel in models:
                modelPath = os.path.join(selectionPath, classificationModel)
                if not os.path.exists(modelPath):
                    os.mkdir(modelPath)
                # Feature Selection
                clf, probas, fold_probas, fold_lbls = classify(tX, y, classification_model=classificationModel, out_path=modelPath,
                                       feature_name=featName, gridsearch=gridsearch, sample_method='rand_over', cv_num=10)
                np.savetxt(os.path.join(modelPath, 'predicted_probas.txt'), probas)
                fids = range(1, 11)
                arr2csv(os.path.join(modelPath, 'predicted_probas_fold.csv'), fold_probas, fids)
                arr2csv(os.path.join(modelPath, 'true_labels_fold.csv'), fold_lbls, fids)
                dump(clf, os.path.join(modelPath, 'classifier.joblib'))

