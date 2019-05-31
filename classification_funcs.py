import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils import resample
from sklearn.tree import export_graphviz
from sklearn.feature_selection import SelectKBest, VarianceThreshold, RFECV  # RFE, chi2,
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, \
    average_precision_score, roc_curve, auc
from sklearn.metrics.classification import classification_report
from xgboost import XGBClassifier
from stat_funcs import perf_measure, get_even_pos_neg
import os
from scipy import interp
from subprocess import call

svm_tuned_parameters = [{'kernel': ['rbf', 'linear'],
                         'C': [0.05, 0.1, 0.25, 0.5, 0.75],
                         'gamma': [0.05, 0.1, 0.25, 0.5, 0.75]}]

rf_tuned_parameters = [{'n_estimators': [50, 100, 250, 500], }]

svm_params = {'kernel': 'linear',
              'C': 0.5,
              'gamma': 0.01}

rf_params = {'n_estimators': 100,
             'criterion': 'gini',
             'max_features': 'sqrt'}


def classify(X, y, classification_model, out_path, feature_name, gridsearch=False, sample_method='under', cv_num=5):

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
    elif classification_model == 'log':
        clf = LogisticRegression(n_jobs=-1)
        params = {}
    elif classification_model == 'XBG':
        clf = XGBClassifier()
        params = {}

    if gridsearch and classification_model != 'mvr':
        params = param_gridsearch(clf, X, y, tuned_params, out_path, cv_num=cv_num)

    if classification_model != 'mvr':
        clf.set_params(**params)
        log_file = open(os.path.join(out_path, 'classification_log.txt'), 'w')
        log_file.write('Fold_#,Accuracy,Precision,Recall,F1-Score,TP,FP,TN,FN,ROC_AUC\n')

        skf = StratifiedKFold(n_splits=cv_num, shuffle=True, random_state=1)

        tprs = []
        aucs = []
        probs = np.zeros(len(y))
        mean_fpr = np.linspace(0, 1, 100)
        pre_probs = []
        val_lbls = []
        for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print('Evaluating on Fold ' + str(i + 1) + ' of ' + str(cv_num) + '.')
            # Get the training and test sets
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

            sel = get_even_pos_neg(y_train, sample_method)
            X_train = X_train[sel]
            y_train = y_train[sel]

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
            elif classification_model == 'log':
                clf = LogisticRegression(solver='lbfgs')
            elif classification_model == 'xbg':
                clf = XGBClassifier()

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
                # plt.legend(loc="lower right")
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

                probs[val_idx] = probas
                pre_probs.append(probas)
                val_lbls.append(y_val)

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
        clf = clf.fit(X, y)
        return clf, probs, pre_probs, val_lbls
    else:
        np.savetxt(os.path.join(out_path, 'mvr_coefficients.csv'), coef, delimiter=',')
    return clf



def param_gridsearch(m, X, y, tuned_parameters, out_path, cv_num=8):
    clf = GridSearchCV(m, tuned_parameters, cv=StratifiedKFold(cv_num),
                       scoring='roc_auc', verbose=1)
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
    return bp


def feature_selection(X, y, featNames, selectionModel, path):
    modelName = selectionModel[0]
    if modelName == 'VarianceThreshold':
        vThresh = selectionModel[1]
        sel = VarianceThreshold(threshold=(vThresh * (1 - vThresh)))
        sel.fit(X)
        tX = sel.transform(X)
        selectionPath = os.path.join(path, 'vthresh_%d' % (100 * vThresh))
        df = open(os.path.join(selectionPath, 'feature_scores.txt'), 'w')
        df.write('feature_name,variance\n')
        for i in range(len(featNames)):
            df.write('%s,%f\n' % (featNames[i], sel.variances_[i]))
        df.close()
    # Univariate Selection
    elif modelName == 'UnivariateSelection':
        scoringFunction = selectionModel[1]
        k = selectionModel[2]
        sel = SelectKBest(scoringFunction, k=k)
        sel.fit(X, y)
        tX = sel.transform(X)
        selectionPath = os.path.join(path, 'uni_%s_%d' % (scoringFunction.__name__, k))
        df = open(os.path.join(selectionPath, 'feature_scores.txt'), 'w')
        df.write('feature_name,score\n')
        for i in range(len(featNames)):
            df.write('%s,%f\n' % (featNames[i], sel.scores_[i]))
        df.close()
    elif modelName == 'RFECV':
        selectionPath = os.path.join(path, 'RFECV')
        if not os.path.exists(selectionPath):
            os.mkdir(selectionPath)
        selectionPath = os.path.join(selectionPath, selectionModel[1])
        if not os.path.exists(selectionPath):
            os.mkdir(selectionPath)
        featureEliminationScore = selectionModel[2]
        selectionPath = os.path.join(selectionPath, featureEliminationScore)
        if os.path.exists(os.path.join(selectionPath, 'feature_ranking.txt')):
            rankData = np.loadtxt(os.path.join(selectionPath, 'feature_ranking.txt'), delimiter=',', usecols=1,
                                  skiprows=1)
            support = np.array([x == 1 for x in rankData])
            tX = X[:, support]
            print('Loaded previous feature selection.')
        else:
            estimator = selectionModel[1]
            if estimator == 'ExtraTrees':
                estimator = ExtraTreesClassifier(n_estimators=100,
                                     n_jobs=-1,
                                     random_state=0)
            elif estimator == 'SVM':
                estimator = SVC(kernel='linear')
            elif estimator == 'LogReg':
                estimator = LogisticRegression(solver='lbfgs')
            elif estimator == 'XBG':
                estimator = XGBClassifier()
            rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(4),
                          scoring=featureEliminationScore, verbose=1, n_jobs=-1)
            rfecv.fit(X, y)
            print("Optimal number of features : %d" % rfecv.n_features_)
            tX = X[:, rfecv.support_]

            if not os.path.exists(selectionPath):
                os.mkdir(selectionPath)
            df = open(os.path.join(selectionPath, 'feature_ranking.txt'), 'w')
            df.write('feature_name,rank\n')
            for i in range(len(featNames)):
                df.write('%s,%d\n' % (featNames[i], rfecv.ranking_[i]))
            df.close()
    return tX, selectionPath
