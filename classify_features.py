from __future__ import print_function

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, \
    average_precision_score, roc_curve, auc
from kdigo_funcs import load_csv
import os
from scipy import interp

# --------------------------------------------------- PARAMETERS ----------------------------------------------------- #
base_path = '../RESULTS/icu/7days_071118/'
# features = ['max_kdigo']
features = ['everything_clusters', 'all_trajectory_clusters']
h5_name = 'stats.h5'

lbl_list = ['died_inp', ]

# tags = ['_norm_norm_a1', '_norm_norm_a2', '_norm_norm_a4',
#         '_norm_custcost_a1', '_norm_custcost_a2',  # '_norm_custcost_a4',
#         '_custcost_norm_a1', '_custcost_norm_a2',  # '_custcost_norm_a4',
#         '_custcost_custcost_a1', '_custcost_custcost_a2']  # , '_custcost_custcost_a4']

# tags = ['clinical', ]
# tags = ['_norm_norm_a1', '_norm_norm_a2', '_norm_norm_a4',
#         '_norm_custcost_a1', '_norm_custcost_a2']

# tags = ['_absmismatch_extension_a5E-01_normBC', '_absmismatch_extension_a5E-01_custBC',
#            '_absmismatch_extension_a1E+00_normBC', '_absmismatch_extension_a1E+00_custBC',
#            '_custmismatch_normBC', '_custmismatch_custBC',
#            '_custmismatch_extension_a2E-01_normBC', '_custmismatch_extension_a2E-01_custBC',
#            '_custmismatch_extension_a5E-01_normBC', '_custmismatch_extension_a5E-01_custBC']  # ,
tags = ['_custmismatch_extension_a1E+00_normBC', '_custmismatch_extension_a1E+00_custBC']
test_size = 0.2

# Note: There will be an equal number of positive and negative examples in the training set, however ALL
#       remaining examples will be used for testing
cv_num = 10

models = ['svm', 'rf']

cluster_methods = ['composite', 'ward']

params = [['exc_7days', (0, 7)]]

# tuple indicating when patients should be removed based on their mortality date
# In days, e.g. (0, 2) will exclude patients who die in the first 48 hrs

svm_tuned_parameters = [{'kernel': ['rbf', 'linear'],
                         'C': [0.75, 0.25, 0.01],
                         'gamma': [0.01, 0.75]}]

rf_tuned_parameters = [{'n_estimators': [5, 500],
                        'criterion': ['gini', ],
                        'max_features': ['sqrt', ]}]

gridsearch = False

svm_params = {'kernel': 'linear',
              'C': 0.75,
              'gamma': 0.01}

rf_params = {'n_estimators': 500,
             'criterion': 'gini',
             'max_features': 'sqrt'}

max_clust = 18
# -------------------------------------------------------------------------------------------------------------------- #


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    return TP, FP, TN, FN


f = h5py.File(base_path + h5_name, 'r')
if not os.path.exists(base_path + 'classification'):
    os.mkdir(base_path + 'classification')

all_ids = f['meta']['ids'][:]
for (sel_str, mort_exc) in params:
    dtd = f['meta']['days_to_death'][:]
    nan_sel = np.where(np.isnan(dtd))[0]
    non_nan = np.setdiff1d(np.arange(len(dtd)), nan_sel)
    dtd_nonan = dtd[non_nan]
    pt_sel = np.union1d(np.union1d(non_nan[np.where(dtd_nonan < mort_exc[0])], nan_sel),
                        np.union1d(non_nan[np.where(dtd_nonan >= mort_exc[1])], nan_sel))
    ids = all_ids[pt_sel]
    if not os.path.exists(base_path + 'classification/%s/' % sel_str):
        os.mkdir(base_path + 'classification/%s/' % sel_str)
    for lbls in lbl_list:
        y = f['meta'][lbls][:][pt_sel]
        n_samples = len(y)

        ptpath = base_path + 'classification/%s/%s/' % (sel_str, lbls)
        if not os.path.exists(ptpath):
            os.mkdir(ptpath)

        for dm_tag in tags:
            if dm_tag == 'clinical':
                dpath = ptpath + dm_tag + '/'
            else:
                dpath = ptpath + dm_tag[1:] + '/'
            if not os.path.exists(dpath):
                os.mkdir(dpath)

            for method in cluster_methods:
                cmpath = dpath + method + '/'
                if not os.path.exists(cmpath):
                    os.mkdir(cmpath)

                for model in models:
                    mpath = cmpath + model + '/'
                    if not os.path.exists(mpath):
                        os.mkdir(mpath)
                    # Set the parameters by cross-validation
                    if gridsearch:
                        if model == 'svm':
                            tuned_parameters = svm_tuned_parameters
                        elif model == 'rf':
                            tuned_parameters = rf_tuned_parameters

                    for feature in features:
                        fpath = mpath + feature + '/'
                        if not os.path.exists(fpath):
                            os.mkdir(fpath)
                        if 'clusters' in feature:
                            for (dirpath, dirnames, filenames) in os.walk(
                                    base_path + 'features/%s/%s/' % (dm_tag[1:], method)):
                                fdpath = base_path + 'features/%s/%s/' % (dm_tag[1:], method)
                                for dirname in dirnames:
                                    n_clust = int(dirname.split('_')[0])
                                    if n_clust > max_clust:
                                        continue
                                    try:
                                        if 'everything' in feature:
                                            X = load_csv(fdpath + dirname + '/everything.csv', ids)
                                        elif 'all_trajectory' in feature:
                                            X = load_csv(fdpath + dirname + '/all_trajectory.csv', ids)

                                        cpath = fpath + dirname + '/'
                                        if not os.path.exists(cpath):
                                            os.mkdir(cpath)
                                        tstr = feature

                                        if X.ndim == 1:
                                            X = X.reshape(-1, 1)

                                        print('Evaluating classification performance using: %s\t%s' % (tstr, dirname))
                                        # Load features and labels
                                        if os.path.exists(cpath + 'evaluation_summary.png'):
                                            continue
                                        # If even distribution of pos/neg for training
                                        pos_idx = np.where(y == 1)[0]
                                        neg_idx = np.where(y == 0)[0]
                                        n_pos = len(pos_idx)
                                        n_neg = len(neg_idx)
                                        if n_pos < n_neg:
                                            n_train = int(n_pos * (1 - test_size))
                                        else:
                                            n_train = int(n_neg * (1 - test_size))
                                        # Total number of negative examples is determined to be equal to the total number of positive examples.
                                        # Appropriate percentage of the positive examples and negative examples chosen at random.
                                        # train_idx = np.sort(np.concatenate((np.random.permutation(pos_idx)[:n_train],
                                        #                                     np.random.permutation(neg_idx)[:n_train])))
                                        # test_idx = np.setdiff1d(np.arange(n_samples), train_idx)

                                        pos_train = np.random.permutation(pos_idx)[:n_train]
                                        neg_train = np.random.permutation(neg_idx)[:n_train]

                                        if n_pos < n_neg:
                                            pos_test = np.random.permutation(np.setdiff1d(pos_idx, pos_train))
                                            neg_test = np.random.permutation(np.setdiff1d(neg_idx, neg_train))[
                                                       :len(pos_test)]
                                        else:
                                            neg_test = np.random.permutation(np.setdiff1d(neg_idx, neg_train))
                                            pos_test = np.random.permutation(np.setdiff1d(pos_idx, pos_train))[
                                                       :len(neg_test)]

                                        train_idx = np.sort(np.concatenate((pos_train, neg_train)))
                                        test_idx = np.sort(np.concatenate((pos_test, neg_test)))

                                        X_test = X[test_idx]
                                        X_train = X[train_idx]
                                        y_test = y[test_idx]
                                        y_train = y[train_idx]

                                        print("Number Positive Training Examples: " + str(len(pos_train)))
                                        print("Number Negative Training Examples: " + str(len(neg_train)))
                                        print("Number Positive Testing Examples: " + str(len(pos_test)))
                                        print("Number Negative Testing Examples: " + str(len(neg_test)))
                                        print()

                                        log_file = open(cpath + 'training_log.txt', 'w')
                                        log = True
                                        log_file.write(
                                            "Number Positive Training Examples: " + str(len(pos_train)) + "\n")
                                        log_file.write(
                                            "Number Negative Training Examples: " + str(len(neg_train)) + "\n")
                                        log_file.write("Number Positive Testing Examples: " + str(len(pos_test)) + "\n")
                                        log_file.write(
                                            "Number Negative Testing Examples: " + str(len(neg_test)) + "\n\n")

                                        if model == 'rf':
                                            m = RandomForestClassifier()
                                        elif model == 'svm':
                                            m = SVC()
                                        if gridsearch:
                                            print("# Tuning hyper-parameters for f1")
                                            print()
                                            if log:
                                                log_file.write("# Tuning hyper-parameters for f1\n\n")

                                            clf = GridSearchCV(m, tuned_parameters, cv=StratifiedKFold(cv_num),
                                                               scoring='f1_macro')
                                            clf.fit(X_train, y_train)

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
                                            if log:
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

                                            print("Detailed classification report:")
                                            print()
                                            print("The model is trained on the full development set.")
                                            print("The scores are computed on the full evaluation set.")
                                            print()
                                            y_true, probas = y_test, clf.predict(X_test)
                                            print(classification_report(y_true, probas))
                                            if log:
                                                log_file.write('\n')
                                                log_file.write("Detailed classification report:\n\n")
                                                log_file.write("The model is trained on the full development set.\n")
                                                log_file.write("The scores are computed on the full evaluation set.\n")
                                                log_file.write(classification_report(y_true, probas))

                                            bp = clf.best_params_

                                        else:
                                            if model == 'svm':
                                                bp = svm_params
                                            elif model == 'rf':
                                                bp = rf_params

                                        log_file.write('\n\nCross Validation - Even Pos/Neg for Eval\n')
                                        skf = StratifiedKFold(n_splits=cv_num, shuffle=True, random_state=1)

                                        if n_pos < n_neg:
                                            neg_sel = np.random.permutation(neg_idx)[:n_pos]
                                            sel = np.sort(np.concatenate((pos_idx, neg_sel)))
                                        else:
                                            pos_sel = np.random.permutation(pos_idx)[:n_neg]
                                            sel = np.sort(np.concatenate((neg_idx, pos_sel)))
                                        vX = X[sel]
                                        vy = y[sel]
                                        log_file.write('Fold_#,Accuracy,Precision,Recall,F1-Score,TP,FP,TN,FN,ROC_AUC\n')

                                        tprs = []
                                        aucs = []
                                        mean_fpr = np.linspace(0, 1, 100)
                                        for i, (train_idx, val_idx) in enumerate(skf.split(vX, vy)):
                                            print('Evaluating on Fold ' + str(i + 1) + ' of ' + str(cv_num) + '.')
                                            if not os.path.exists(base_path + tstr):
                                                os.mkdir(base_path + tstr)
                                            # Get the training and test sets
                                            X_train = vX[train_idx]
                                            y_train = vy[train_idx]
                                            X_val = vX[val_idx]
                                            y_val = vy[val_idx]

                                            # Load and fit the model
                                            if model == 'rf':
                                                clf = RandomForestClassifier()
                                                clf.set_params(**bp)
                                            elif model == 'svm':
                                                clf = SVC(probability=True)
                                                clf.set_params(**bp)
                                            clf.fit(X_train, y_train)

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
                                            plt.savefig(cpath + 'fold' + str(i + 1) + '_evaluation.png')
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
                                        plt.title(model.upper() + ' Classification Performance\n' + feature)
                                        plt.savefig(cpath + 'evaluation_summary.png')
                                        plt.close('all')
                                    except:
                                        print('Feature %s not found in directory %s%s' % (feature, fdpath, dirname))
                        else:
                            fdpath = base_path + 'features/%s/' % dm_tag
                            X = load_csv(fdpath + '%s.csv' % feature, ids)
                            tstr = feature

                            if X.ndim == 1:
                                X = X.reshape(-1, 1)

                            print('Evaluating classification performance using: %s' % tstr)
                            # Load features and labels
                            if os.path.exists(fpath + 'evaluation_summary.png'):
                                continue
                            # If even distribution of pos/neg for training
                            pos_idx = np.where(y == 1)[0]
                            neg_idx = np.where(y == 0)[0]
                            n_pos = len(pos_idx)
                            n_neg = len(neg_idx)
                            if n_pos < n_neg:
                                n_train = int(n_pos * (1 - test_size))
                            else:
                                n_train = int(n_neg * (1 - test_size))
                            # Total number of negative examples is determined to be equal to the total number of positive examples.
                            # Appropriate percentage of the positive examples and negative examples chosen at random.
                            # train_idx = np.sort(np.concatenate((np.random.permutation(pos_idx)[:n_train],
                            #                                     np.random.permutation(neg_idx)[:n_train])))
                            # test_idx = np.setdiff1d(np.arange(n_samples), train_idx)

                            pos_train = np.random.permutation(pos_idx)[:n_train]
                            neg_train = np.random.permutation(neg_idx)[:n_train]

                            if n_pos < n_neg:
                                pos_test = np.random.permutation(np.setdiff1d(pos_idx, pos_train))
                                neg_test = np.random.permutation(np.setdiff1d(neg_idx, neg_train))[
                                           :len(pos_test)]
                            else:
                                neg_test = np.random.permutation(np.setdiff1d(neg_idx, neg_train))
                                pos_test = np.random.permutation(np.setdiff1d(pos_idx, pos_train))[
                                           :len(neg_test)]

                            train_idx = np.sort(np.concatenate((pos_train, neg_train)))
                            test_idx = np.sort(np.concatenate((pos_test, neg_test)))

                            X_test = X[test_idx]
                            X_train = X[train_idx]
                            y_test = y[test_idx]
                            y_train = y[train_idx]

                            print("Number Positive Training Examples: " + str(len(pos_train)))
                            print("Number Negative Training Examples: " + str(len(neg_train)))
                            print("Number Positive Testing Examples: " + str(len(pos_test)))
                            print("Number Negative Testing Examples: " + str(len(neg_test)))
                            print()

                            log_file = open(fpath + 'training_log.txt', 'w')
                            log = True
                            log_file.write(
                                "Number Positive Training Examples: " + str(len(pos_train)) + "\n")
                            log_file.write(
                                "Number Negative Training Examples: " + str(len(neg_train)) + "\n")
                            log_file.write("Number Positive Testing Examples: " + str(len(pos_test)) + "\n")
                            log_file.write(
                                "Number Negative Testing Examples: " + str(len(neg_test)) + "\n\n")

                            if model == 'rf':
                                m = RandomForestClassifier()
                            elif model == 'svm':
                                m = SVC()
                            if gridsearch:
                                print("# Tuning hyper-parameters for f1")
                                print()
                                if log:
                                    log_file.write("# Tuning hyper-parameters for f1\n\n")

                                clf = GridSearchCV(m, tuned_parameters, cv=StratifiedKFold(cv_num),
                                                   scoring='f1_macro')
                                clf.fit(X_train, y_train)

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
                                if log:
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

                                print("Detailed classification report:")
                                print()
                                print("The model is trained on the full development set.")
                                print("The scores are computed on the full evaluation set.")
                                print()
                                y_true, probas = y_test, clf.predict(X_test)
                                print(classification_report(y_true, probas))
                                if log:
                                    log_file.write('\n')
                                    log_file.write("Detailed classification report:\n\n")
                                    log_file.write("The model is trained on the full development set.\n")
                                    log_file.write("The scores are computed on the full evaluation set.\n")
                                    log_file.write(classification_report(y_true, probas))

                                bp = clf.best_params_

                            else:
                                if model == 'svm':
                                    bp = svm_params
                                elif model == 'rf':
                                    bp = rf_params

                            log_file.write('\n\nCross Validation - Even Pos/Neg for Eval\n')
                            skf = StratifiedKFold(n_splits=cv_num, shuffle=True, random_state=1)

                            if n_pos < n_neg:
                                neg_sel = np.random.permutation(neg_idx)[:n_pos]
                                sel = np.sort(np.concatenate((pos_idx, neg_sel)))
                            else:
                                pos_sel = np.random.permutation(pos_idx)[:n_neg]
                                sel = np.sort(np.concatenate((neg_idx, pos_sel)))
                            vX = X[sel]
                            vy = y[sel]
                            log_file.write('Fold_#,Accuracy,Precision,Recall,F1-Score,TP,FP,TN,FN,ROC_AUC\n')

                            tprs = []
                            aucs = []
                            mean_fpr = np.linspace(0, 1, 100)
                            for i, (train_idx, val_idx) in enumerate(skf.split(vX, vy)):
                                print('Evaluating on Fold ' + str(i + 1) + ' of ' + str(cv_num) + '.')
                                if not os.path.exists(base_path + tstr):
                                    os.mkdir(base_path + tstr)
                                # Get the training and test sets
                                X_train = vX[train_idx]
                                y_train = vy[train_idx]
                                X_val = vX[val_idx]
                                y_val = vy[val_idx]

                                # Load and fit the model
                                if model == 'rf':
                                    clf = RandomForestClassifier()
                                    clf.set_params(**bp)
                                elif model == 'svm':
                                    clf = SVC(probability=True)
                                    clf.set_params(**bp)
                                clf.fit(X_train, y_train)

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
                                plt.savefig(fpath + 'fold' + str(i + 1) + '_evaluation.png')
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
                            plt.title(model.upper() + ' Classification Performance\n' + feature)
                            plt.savefig(fpath + 'evaluation_summary.png')
                            plt.close('all')
