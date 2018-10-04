from __future__ import print_function

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, \
    average_precision_score, roc_curve, auc
from utility_funcs import load_csv, perf_measure, get_feats_by_dod, get_even_pos_neg
import os
from scipy import interp

# --------------------------------------------------- PARAMETERS ----------------------------------------------------- #
base_path = '../RESULTS/icu/7days_100218/'

# features used for classification.
# Options: sofa, apache, all_clinical, all_trajectory, everything
features = ['sofa', 'apache', 'all_clinical', 'all_trajectory', 'everything', ]

# targets for classification.
# Options: died_inp, MAKE_90_25_admit, MAKE_90_50_admit,
#          MAKE_90_25_disch, MAKE_90_50_disch
lbl_list = ['died_inp', ]

# Specify distance matrices
tags = ['_custmismatch_extension_a1E+00_normBC', '_custmismatch_normBC']
test_size = 0.2

# Note: There will be an equal number of positive and negative examples in the training set, however ALL
#       remaining examples will be used for testing
cv_num = 10

models = ['svm', 'rf', 'mvr']   # mvr = multi-variate regression
# models = ['mvr', ]

# which cluster method to use. Options: flat, dynamic
cluster_methods = ['dynamic', 'flat']

n_days_l = [7, ]

# tuple indicating when patients should be removed based on their mortality date
# In days, e.g. (0, 2) will exclude patients who die in the first 48 hrs

svm_tuned_parameters = [{'kernel': ['rbf', 'linear'],
                         'C': [0.75, 0.25, 0.01],
                         'gamma': [0.01, 0.75]}]

rf_tuned_parameters = [{'n_estimators': [5, 500],
                        'criterion': ['gini', ],
                        'max_features': ['sqrt', ]}]

mvr_tuned_parameters = []

gridsearch = False

svm_params = {'kernel': 'linear',
              'C': 0.75,
              'gamma': 0.01}

rf_params = {'n_estimators': 500,
             'criterion': 'gini',
             'max_features': 'sqrt'}

mvr_params = {}
# -------------------------------------------------------------------------------------------------------------------- #


def main():
    f = h5py.File(base_path + 'stats.h5', 'r')
    if not os.path.exists(base_path + 'classification'):
        os.mkdir(base_path + 'classification')

    for n_days in n_days_l:
        day_str = '%ddays' % n_days
        if not os.path.exists(base_path + 'classification/%s/' % day_str):
            os.mkdir(base_path + 'classification/%s/' % day_str)

        all_ids, labels = get_feats_by_dod(f, n_days=n_days, features=lbl_list)

        # build path for input features and where to save results
        for (label_name, y) in zip(lbl_list, labels):
            if len(np.unique(y)) > 2:
                y[np.where(y)] = 1
            # output path
            lblpath = base_path + 'classification/%s/%s/' % (day_str, label_name)
            if not os.path.exists(lblpath):
                os.mkdir(lblpath)

            for feature in features:
                feature_class_path = os.path.join(lblpath, feature)
                if not os.path.exists(feature_class_path):
                    os.mkdir(feature_class_path)

                if feature not in ['sofa', 'apache', 'all_clinical']:
                    # output path
                    source_paths = []
                    out_paths = []
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
                else:
                    source_paths = [os.path.join(base_path, 'features', 'individual'), ]
                    out_paths = [feature_class_path, ]

                for (source_path, out_path) in zip(source_paths, out_paths):
                    X = np.loadtxt(os.path.join(source_path, feature + '.csv'), delimiter=',')
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
                        classify(X, ty, classification_model, out_path=model_path, feature_name=feature, gridsearch=gridsearch)


def classify(X, y, classification_model, out_path, feature_name, gridsearch=gridsearch):
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

    sel = get_even_pos_neg(y)
    vX = X[sel]
    vy = y[sel]

    if gridsearch and classification_model != 'mvr':
        params = param_gridsearch(clf, X[sel], y[sel], tuned_params, out_path)

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
            pred = clf.predict(X_val)
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

    if classification_model != 'mvr':
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
                       scoring='f1_macro')
    clf.fit(X, y)

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
        if log:
            log_file.write("%0.3f (+/-%0.03f) for %r\n"
                           % (mean, std * 2, params))
    print()
    log_file.close()

    bp = clf.best_params_

    return bp


main()
