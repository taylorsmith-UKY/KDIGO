from __future__ import print_function

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, \
    average_precision_score, roc_curve, auc
from sklearn.metrics.classification import classification_report
from stat_funcs import perf_measure, get_even_pos_neg
import os
from scipy import interp

# --------------------------------------------------- PARAMETERS ----------------------------------------------------- #
base_path = '../RESULTS/icu/7days_100218/'

# features used for classification.
# Options: sofa, apache, all_clinical, all_trajectory, everything
# features = ['sofa', 'apache', 'all_clinical', 'all_trajectory_individual', 'everything_individual'
#             'all_trajectory', 'everything',
#             'all_trajectory_bin', 'everything_bin',
#             'all_trajectory_centers', 'everything_centers']
individual_features = []
cluster_features = ['clusters']
# targets for classification.
# Options: died_inp, MAKE_90_25_admit, MAKE_90_50_admit,
#          MAKE_90_25_disch, MAKE_90_50_disch
lbl_list = ['died_inp', ]
# lbl_list = ['died_inp', 'MAKE_90_25_ADMIT', 'MAKE_90_50_ADMIT', 'MAKE_90_25_DISCH', 'MAKE_90_50_DISCH']

# Specify distance matrices
tags = ['_custmismatch_extension_a1E+00_normBC', ]
test_size = 0.2

sample_methods = ['rand_over', ]

# Note: There will be an equal number of positive and negative examples in the training set, however ALL
#       remaining examples will be used for testing
cv_num = 10

models = ['rf', ]  # 'svm', 'mvr']   # mvr = multi-variate regression
# models = ['mvr', ]

# which cluster method to use. Options: flat, dynamic
cluster_methods = ['merged', ]  # 'dynamic', 'flat']

n_days = 7

meta_grp = 'meta_new'

# tuple indicating when patients should be removed based on their mortality date
# In days, e.g. (0, 2) will exclude patients who die in the first 48 hrs

svm_tuned_parameters = [{'kernel': ['rbf', 'linear'],
                         'C': [0.05, 0.1, 0.25, 0.5, 0.75],
                         'gamma': [0.05, 0.1, 0.25, 0.5, 0.75]}]

rf_tuned_parameters = [{'n_estimators': [50, 100, 250, 500], }]
                        # 'criterion': ['gini', ],
                        # 'max_features': ['sqrt', ]}]

mvr_tuned_parameters = []

gridsearch = False

svm_params = {'kernel': 'linear',
              'C': 0.5,
              'gamma': 0.01}

rf_params = {'n_estimators': 500,
             'criterion': 'gini',
             'max_features': 'sqrt'}

mvr_params = {}
# -------------------------------------------------------------------------------------------------------------------- #

def classify(X, y, classification_model, out_path, feature_name, gridsearch=gridsearch, sample_method='under'):
    svm_tuned_parameters = [{'kernel': ['rbf', 'linear'],
                             'C': [0.05, 0.1, 0.25, 0.5, 0.75],
                             'gamma': [0.05, 0.1, 0.25, 0.5, 0.75]}]

    rf_tuned_parameters = [{'n_estimators': [50, 100, 250, 500], }]
    # 'criterion': ['gini', ],
    # 'max_features': ['sqrt', ]}]

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if classification_model == 'rf':
        clf = RandomForestClassifier()
        params = rf_params
        tuned_params = rf_tuned_parameters
    elif classification_model == 'svm':
        clf = SVC()
        params = svm_params
        tuned_params = svm_tuned_parameters
    elif classification_model == 'mvr':
        clf = LinearRegression()
        coef = []

    sel = get_even_pos_neg(y, sample_method)
    vX = X[sel]
    vy = y[sel]

    if gridsearch and classification_model != 'mvr':
        params = param_gridsearch(clf, X, y, tuned_params, out_path)

    if classification_model != 'mvr':
        clf.set_params(**params)
        log_file = open(os.path.join(out_path, 'classification_log.txt'), 'w')
        log_file.write('Fold_#,Accuracy,Precision,Recall,F1-Score,TP,FP,TN,FN,ROC_AUC\n')

        skf = StratifiedKFold(n_splits=cv_num, shuffle=True, random_state=1)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        for i, (train_idx, val_idx) in enumerate(skf.split(vX, vy)):
            print('Evaluating on Fold ' + str(i + 1) + ' of ' + str(cv_num) + '.')
            # Get the training and test sets
            X_train = vX[train_idx]
            y_train = vy[train_idx]
            X_val = vX[val_idx]
            y_val = vy[val_idx]

            # Load and fit the model
            if classification_model == 'rf':
                clf = RandomForestClassifier()
                clf.set_params(**params)
            elif classification_model == 'svm':
                clf = SVC(probability=True)
                clf.set_params(**params)
            elif classification_model == 'mvr':
                clf = LinearRegression()
                # clf.set_params(**params)
            clf.fit(X_train, y_train)

            if classification_model == 'mvr':
                coef.append(clf.coef_)
            else:
                # Plot the precision vs. recall curve
                probas = clf.predict_proba(X_val)[:, 1]
                pred = clf.predict(X_val)
                pcurve, rcurve, _ = precision_recall_curve(y_val, probas)
                fig = plt.figure(figsize=(8, 4))
                plt.subplot(121)
                plt.step(rcurve, pcurve, color='b', alpha=0.2,
                         where='post')
                plt.fill_between(rcurve, pcurve, where=None, alpha=0.2,
                                 color='b')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.ylim([0.0, 1.05])
                plt.xlim([0.0, 1.0])
                plt.title(
                    'Precision-Recall Curve: AP=%0.2f' % average_precision_score(y_val,
                                                                                 probas))

                # Plot ROC curve
                fpr, tpr, thresholds = roc_curve(y_val, probas)
                roc_auc = auc(fpr, tpr)
                plt.subplot(122)
                plt.plot(fpr, tpr)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve (area = %0.2f)' % roc_auc)
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(os.path.join(out_path, 'fold' + str(i + 1) + '_evaluation.png'))
                plt.close(fig)
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                aucs.append(roc_auc)

                acc = accuracy_score(y_val, pred)
                prec = precision_score(y_val, pred)
                rec = recall_score(y_val, pred)
                f1 = f1_score(y_val, pred)

                tp, fp, tn, fn = perf_measure(y_val, pred)

                log_file.write('%d,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d,%.4f\n' % (
                    i + 1, acc, prec, rec, f1, tp, fp, tn, fn, roc_auc))

        log_file.close()

        fig = plt.figure()
        for i in range(cv_num):
            plt.plot(mean_fpr, tprs[i], lw=1, alpha=0.3)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Luck', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.title(classification_model.upper() + ' Classification Performance\n' + feature_name)
        plt.savefig(os.path.join(out_path, 'evaluation_summary.png'))
        plt.close(fig)
    else:
        np.savetxt(os.path.join(out_path, 'mvr_coefficients.csv'), coef, delimiter=',')


def param_gridsearch(m, X, y, tuned_parameters, out_path):
    clf = GridSearchCV(m, tuned_parameters, cv=StratifiedKFold(cv_num),
                       scoring='roc_auc', verbose=1, n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    clf.fit(X_train, y_train)
    results = clf.cv_results_

    print("Best score found on development set:")
    print()
    print(clf.best_score_)
    print()
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    log_file = open(os.path.join(out_path, 'gridsearch.txt'), 'w')
    log_file.write("Best parameters set found on development set:\n\n")
    log_file.write(str(clf.best_params_))
    log_file.write('\n\n')
    log_file.write("Grid scores on development set:\n\n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
        log_file.write("%0.3f (+/-%0.03f) for %r\n"
                       % (mean, std * 2, params))
    print()
    log_file.write("\n\nDetailed classification report:\n")
    log_file.write("The model is trained on the full development set.\n")
    log_file.write("The scores are computed on the full evaluation set.\n")
    y_true, y_pred = y_test, clf.predict(X_test)
    log_file.write(classification_report(y_true, y_pred))
    log_file.write('\n')
    log_file.close()

    bp = clf.best_params_

    #
    # for base_feat in range(len(list(results['params'])[0])):
    #
    #
    # for param in list(bp):
    #     fig = plt.figure(figsize=(13, 13))
    #     plt.title("GridSearchCV evaluating using ROC-AUC",
    #               fontsize=16)
    #
    #     plt.xlabel("param")
    #     plt.ylabel("AUC")
    #
    #     # Get the regular numpy array from the MaskedArray
    #     X_axis = np.array(results['param_' + param].data, dtype=float)
    #
    #     # Set axis limits based on value range
    #     xr = [np.min(X_axis), np.max(X_axis)]
    #     buf = (xr[1] - xr[0]) * 0.05
    #
    #     ax = plt.gca()
    #     ax.set_xlim(xr[0] - buf, xr[1] + buf)
    #     ax.set_ylim(0.5, 1)
    #
    #     for sample, style in (('train', '--'), ('test', '-')):
    #         sample_score_mean = results['mean_%s_score' % sample]
    #         sample_score_std = results['std_%s_score' % sample]
    #         ax.fill_between(X_axis, sample_score_mean - sample_score_std,
    #                         sample_score_mean + sample_score_std,
    #                         alpha=0.1 if sample == 'test' else 0, color='blue')
    #         ax.plot(X_axis, sample_score_mean, style, color='blue',
    #                 alpha=1 if sample == 'test' else 0.7,
    #                 label="AUC (%s)" % sample)
    #
    #     best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]
    #     best_score = results['mean_test_score'][best_index]
    #
    #     # Plot a dotted vertical line at the best score for that scorer marked by x
    #     ax.plot([X_axis[best_index], ] * 2, [0, best_score],
    #             linestyle='-.', color='black', marker='x', markeredgewidth=3, ms=8)
    #
    #     # Annotate the best score for that scorer
    #     ax.annotate("%0.2f" % best_score,
    #                 (X_axis[best_index], best_score + 0.005))
    #
    #     plt.legend(loc="best")
    #     plt.grid('off')
    #     plt.savefig(os.path.join(out_path, 'gridsearch_%s.png' % param))
    #     plt.close(fig)
    return bp


f = h5py.File(base_path + 'stats.h5', 'r')
if not os.path.exists(base_path + 'classification'):
    os.mkdir(base_path + 'classification')

day_str = '%ddays' % n_days
if not os.path.exists(base_path + 'classification/%s/' % day_str):
    os.mkdir(base_path + 'classification/%s/' % day_str)

all_ids = f[meta_grp]['ids'][:]

# all_ids, labels = get_feats_by_dod(f, n_days=n_days, features=lbl_list)

# build path for input features and where to save results
# for (label_name, y) in zip(lbl_list, labels):
for label_name in lbl_list:
    y = f[meta_grp][label_name][:]
    if len(np.unique(y)) > 2:
        y[np.where(y)] = 1
    # output path
    lblpath = base_path + 'classification/%s/%s/' % (day_str, label_name)
    if not os.path.exists(lblpath):
        os.mkdir(lblpath)
    source_paths = []
    out_paths = []
    feat_l = []
    for feature in cluster_features:
        feature_class_path = os.path.join(lblpath, feature)
        if not os.path.exists(feature_class_path):
            os.mkdir(feature_class_path)
        for dm_tag in tags:
            dm_class_path = os.path.join(feature_class_path, dm_tag[1:])
            if not os.path.exists(dm_class_path):
                os.mkdir(dm_class_path)
            for cluster_method in cluster_methods:
                cm_class_path = os.path.join(dm_class_path, cluster_method)
                if not os.path.exists(cm_class_path):
                    os.mkdir(cm_class_path)
                cluster_feature_base_path = os.path.join(base_path, 'features', day_str, dm_tag[1:], cluster_method)
                for (dirpath, dirnames, filenames) in os.walk(cluster_feature_base_path):
                    for dirname in dirnames:
                        try:
                            n_clust = int(dirname.split('_')[0])
                        except ValueError:
                            continue
                        source_paths.append(os.path.join(cluster_feature_base_path, dirname))
                        cluster_class_path = os.path.join(cm_class_path, dirname)
                        out_paths.append(cluster_class_path)
                        if not os.path.exists(cluster_class_path):
                            os.mkdir(cluster_class_path)
                        feat_l.append(feature)
    for feature in individual_features:
        feature_class_path = os.path.join(lblpath, feature)
        if not os.path.exists(feature_class_path):
            os.mkdir(feature_class_path)
        source_paths = [os.path.join(base_path, 'features', 'individual'), ]
        out_paths = [feature_class_path, ]
        feat_l.append(feature)

    for (feature, source_path, out_path) in zip(feat_l, source_paths, out_paths):
        try:
            X = np.loadtxt(os.path.join(source_path, feature + '.csv'), delimiter=',')
        except ValueError:
            X = np.loadtxt(os.path.join(source_path, feature + '.csv'), delimiter=',', skiprows=1)
        ids = np.array(X[:, 0], dtype=int)
        X = X[:, 1:]
        id_sel = np.array([x in ids for x in all_ids])
        ty = y[id_sel]
        # X = load_csv(os.path.join(source_path, feature + '.csv'), ids)
        # output path
        for classification_model in models:
            model_path = os.path.join(out_path, classification_model)
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            for sample_method in sample_methods:
                save_path = os.path.join(model_path, sample_method)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                # if not os.path.exists(os.path.join(save_path, 'evaluation_summary.png')) and not \
                #         os.path.exists(os.path.join(save_path, 'mvr_coefficients.csv')):
                classify(X, ty, classification_model, out_path=save_path, feature_name=feature,
                             gridsearch=gridsearch)
