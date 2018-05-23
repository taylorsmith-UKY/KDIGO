from __future__ import print_function

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve,\
     average_precision_score, roc_curve, auc
import os
from scipy import interp


# --------------------------------------------------- PARAMETERS ----------------------------------------------------- #
data_file = '../RESULTS/icu/7days_inc_death_051918/kdigo_dm.h5'
features = ['sofa_norm', 'apache_norm', 'clinical_norm', 'all_trajectory_norm', 'all_trajectory_individual_norm', 'everything_individual', 'everything_clusters']
lbls = '/meta/died_inp'
basepath = '../RESULTS/icu/7days_inc_death_051918/classification/'

test_size = 0.2

# Note: There will be an equal number of positive and negative examples in the training set, however ALL
#       remaining examples will be used for testing
cv_num = 10

models = ['rf', 'svm']

scores = ['precision', 'recall', 'f1']

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
        if y_hat[i] == 0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return TP, FP, TN, FN


f = h5py.File(data_file, 'r')

if not os.path.exists(basepath):
    os.mkdir(basepath)

for model in models:
    basepath += model + '/'
    if not os.path.exists(basepath):
        os.mkdir(basepath)

    # Set the parameters by cross-validation
    if model == 'svm':
        tuned_parameters = [{'kernel': ['rbf', 'linear', 'sigmoid'],
                             'C': [0.5, 0.4, 0.25, 0.1, 0.01, 0.001, 0.0001],
                             'gamma': [0.01, 0.1, 0.5, 0.75, 0.85, 0.95]}]
    elif model == 'rf':
        tuned_parameters = [{'n_estimators': [2, 5, 10, 100, 250, 500],
                             'criterion': ['gini', 'entropy'],
                             'max_features': ['sqrt', 'log2']}]

    for score in scores:
        tpath = basepath + score + '/'
        if not os.path.exists(tpath):
            os.mkdir(tpath)

        for feature in features:
            print('Evaluating classification performance using: %s' % feature)
            # Load features and labels
            X = f['/features/' + feature][:]
            y = f['/meta/died_inp'][:]
            n_samples = len(y)

            # If even distribution of pos/neg for training
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            n_pos = len(pos_idx)
            n_train = int(n_pos * (1 - test_size))
            # Total number of negative examples is determined to be equal to the total number of positive examples.
            # Appropriate percentage of the positive examples and negative examples chosen at random.
            # train_idx = np.sort(np.concatenate((np.random.permutation(pos_idx)[:n_train],
            #                                     np.random.permutation(neg_idx)[:n_train])))
            # test_idx = np.setdiff1d(np.arange(n_samples), train_idx)

            pos_train = np.random.permutation(pos_idx)[:n_train]
            neg_train = np.random.permutation(neg_idx)[:n_train]

            pos_test = np.random.permutation(np.setdiff1d(pos_idx, pos_train))
            neg_test = np.random.permutation(np.setdiff1d(neg_idx, neg_train))[:len(pos_test)]

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

            log_file = open(tpath + feature + '_log.txt', 'w')
            log = True
            log_file.write("Number Positive Training Examples: "+str(len(pos_train))+"\n")
            log_file.write("Number Negative Training Examples: " + str(len(neg_train)) + "\n")
            log_file.write("Number Positive Testing Examples: " + str(len(pos_test)) + "\n")
            log_file.write("Number Negative Testing Examples: " + str(len(neg_test)) + "\n\n")

            if model == 'rf':
                m = RandomForestClassifier()
            elif model == 'svm':
                m = SVC()

            print("# Tuning hyper-parameters for %s" % score)
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

            log_file.write('\n\nCross Validation - Even Pos/Neg for Eval\n')
            skf = StratifiedKFold(n_splits=cv_num, shuffle=True, random_state=1)

            neg_sel = np.random.permutation(neg_idx)[:n_pos]
            sel = np.sort(np.concatenate((pos_idx, neg_sel)))
            vX = X[sel]
            vy = y[sel]
            log_file.write('Fold_#,Accuracy,Precision,Recall,F1-Score,TP,FP,TN,FN\n')

            if not os.path.exists(tpath + feature + '/'):
                os.mkdir(tpath + feature + '/')

            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)
            for i, (train_idx, val_idx) in enumerate(skf.split(vX, vy)):
                print('Evaluating on Fold ' + str(i+1) + ' of ' + str(cv_num) + '.')
                if not os.path.exists(basepath + feature):
                    os.mkdir(basepath + feature)
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
                plt.title('Precision-Recall Curve: AP=%0.2f' % average_precision_score(y_val, probas))

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
                plt.savefig(tpath + feature + '/fold' + str(i + 1) + '_evaluation.png')
                plt.close(fig)
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                aucs.append(roc_auc)

                acc = accuracy_score(y_val, pred)
                prec = precision_score(y_val, pred)
                rec = recall_score(y_val, pred)
                f1 = f1_score(y_val, pred)

                tp, fp, tn, fn = perf_measure(y_val, pred)

                log_file.write('%d,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d\n' % (i+1, acc, prec, rec, f1, tp, fp, tn, fn))
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
            plt.savefig(tpath + feature + '/evaluation_summary.png')
