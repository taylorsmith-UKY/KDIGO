import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import SelectKBest, VarianceThreshold, RFECV, SelectFromModel
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, \
    average_precision_score, roc_curve, auc
from sklearn.metrics.classification import classification_report
from xgboost import XGBClassifier
from stat_funcs import perf_measure, get_even_pos_neg, count_eps, eval_classification
import os
from scipy import interp
from utility_funcs import arr2csv, cartesian, get_date, get_array_dates, load_csv
from scipy.spatial.distance import squareform
from copy import copy
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
import statsmodels.api as sm
from joblib import load, dump

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


# %%
def descriptive_trajectory_features(kdigos, ids, days, t_lim=None, filename='descriptive_features.csv', tRes=6):
    npts = len(kdigos)
    features = np.zeros((npts, 17))
    header = 'STUDY_PATIENT_ID,peak_KDIGO,KDIGO_at_admit,KDIGO_at_EndOfWindow,Min_KDIGO,AKI_first_3days,AKI_after_3days,' + \
             'multiple_hits,KDIGO1+_gt_24hrs,KDIGO2+_gt_24hrs,KDIGO3+_gt_24hrs,KDIGO4_gt_24hrs,' + \
             'flat,strictly_increase,strictly_decrease,slope_posTOneg,slope_negTOpos,numPeaks'
    PEAK_KDIGO = 0
    KDIGO_ADMIT = 1
    KDIGO_EndOfWin = 2
    MIN_KDIGO = 3
    AKI_FIRST_3D = 4
    AKI_AFTER_3D = 5
    NUM_HITS = 6
    KDIGO1_GT1D = 7
    KDIGO2_GT1D = 8
    KDIGO3_GT1D = 9
    KDIGO4_GT1D = 10
    FLAT = 11
    ONLY_INC = 12
    ONLY_DEC = 13
    POStoNEG = 14
    NEGtoPOS = 15
    NUM_PEAKS = 16
    # LONG_START = 17
    # LONG_STOP = 18

    for i in range(len(kdigos)):
        kdigo = kdigos[i]
        tdays = days[i]
        if t_lim is not None:
            sel = np.where(tdays <= t_lim)[0]
            kdigo = kdigo[sel]
            tdays = tdays[sel]
        # No AKI
        features[i, PEAK_KDIGO] = max(kdigo)
        features[i, MIN_KDIGO] = min(kdigo)
        features[i, KDIGO_ADMIT] = kdigo[0]
        features[i, KDIGO_EndOfWin] = kdigo[-1]
        if min(tdays) <= 3:
            features[i, AKI_FIRST_3D] = max(kdigo[np.where(tdays <= 3)])
        if max(tdays) > 3:
            features[i, AKI_AFTER_3D] = max(kdigo[np.where(tdays > 3)])

        ct = 0
        for i in range(1, len(kdigo)):
            if kdigo[i] == kdigo[i - 1]:
                ct += 1
            else:
                break

        # if ct >= ((9 * 4) / 5):
        #     features[i, LONG_START] = 1

        ct = 0
        for i in range(len(kdigo)-1)[::-1]:
            if kdigo[i] == kdigo[i + 1]:
                ct += 1
            else:
                break
        # if ct >= ((9 * 4) / 5):
        #     features[i, LONG_STOP] = 1

        # Multiple hits separated by >= 24 hrs
        numHits, numPeaks = count_eps(kdigo, t_gap=24, timescale=6)
        features[i, NUM_HITS] = numHits
        features[i, NUM_PEAKS] = numPeaks

        # >=24 hrs at KDIGO 1
        if len(np.where(kdigo == 1)[0]) >= (24 / tRes):
            features[i, KDIGO1_GT1D] = 1
        # >=24 hrs at KDIGO 2
        if len(np.where(kdigo == 2)[0]) >= (24 / tRes):
            features[i, KDIGO2_GT1D] = 1
        # >=24 hrs at KDIGO 3
        if len(np.where(kdigo == 3)[0]) >= (24 / tRes):
            features[i, KDIGO3_GT1D] = 1
        # >=24 hrs at KDIGO 3D
        if len(np.where(kdigo == 4)[0]) >= (24 / tRes):
            features[i, KDIGO4_GT1D] = 1
        # Flat trajectory
        if np.all(kdigo == kdigo[0]):
            features[i, FLAT] = 1

        # KDIGO strictly increases
        diff = kdigo[1:] - kdigo[:-1]
        if np.any(diff > 0):
            if np.all(diff >= 0):
                features[i, ONLY_INC] = 1

        # KDIGO strictly decreases
        if np.any(diff < 0):
            if np.all(diff <= 0):
                features[i, ONLY_DEC] = 1

        # Slope changes sign
        direction = 0
        temp = kdigo[0]
        posCount = 0
        negCount = 0
        for j in range(len(kdigo)):
            if kdigo[j] < temp:
                # Pos to neg
                if direction == 1:
                    posCount += 1
                    # features[i, POStoNEG] += 1
                direction = -1
            elif kdigo[j] > temp:
                # Neg to pos
                if direction == -1:
                    negCount += 1
                    # features[i, NEGtoPOS] += 1
                direction = 1
            temp = kdigo[j]
        features[i, POStoNEG] = posCount
        features[i, NEGtoPOS] = negCount
    if filename is not None:
        arr2csv(filename, features, ids, fmt='%d', header=header)
    return features, header

# %%
def template_trajectory_features(kdigos, ids, days=None, t_lim=None, filename='template_trajectory_features.csv',
                                 scores=np.array([0, 1, 2, 3, 4], dtype=int), npoints=3, ratios=False, gap=0, stride=1):
    combination = scores
    for i in range(npoints - 1):
        combination = np.vstack((combination, scores))
    npts = len(kdigos)
    templates = cartesian(combination)
    header = 'id'
    for i in range(len(templates)):
        header += ',' + str(templates[i])
    features = np.zeros((npts, len(templates)))
    for i in range(npts):
        kdigo = kdigos[i]
        if len(kdigo) < npoints:
            continue
        if days is not None:
            tdays = days[i]
            sel = np.where(tdays <= t_lim)[0]
            kdigo = kdigo[sel]
        nwin = np.floor((len(kdigo) - npoints + 1) / stride).astype(int)
        for j in range(nwin):
            start = j * stride
            # tk = kdigo[start:start + npoints]
            tk = [kdigo[x] for x in range(start, start + npoints + (gap * (npoints - 1)), gap + 1)]
            sel = np.where(templates[:, 0] == tk[0])[0]
            loc = [x for x in sel if np.all(templates[x, :] == tk)]
            features[i, loc] += 1
        if ratios:
            features[i, :] = features[i, :] / np.sum(features[i, :])
    if ratios:
        arr2csv(filename, features, ids, fmt='%f', header=header)
    else:
        arr2csv(filename, features, ids, fmt='%d', header=header)
    return features, header

# %%
def slope_trajectory_features(kdigos, ids, days=None,t_lim=None, scores=np.array([0, 1, 2, 3, 4]),
                              filename='slope_features.csv', ratios=False, gap=0, stride=1):
    slopes = []
    header = 'ids'
    for i in range(len(scores)):
        slopes.append(scores[i] - scores[0])
        header += ',%d' % (scores[i] - scores[0])
    for i in range(len(slopes)):
        if slopes[i] > 0:
            slopes.append(-slopes[i])
            header += ',%d' % (-slopes[i])
    slopes = np.array(slopes)
    npts = len(kdigos)
    features = np.zeros((npts, len(slopes)))
    for i in range(npts):
        kdigo = kdigos[i]
        if days is not None:
            tdays = days[i]
            sel = np.where(tdays <= t_lim)[0]
            kdigo = kdigo[sel]
        nwin = np.floor((len(kdigo) - 1) / stride).astype(int)
        for j in range(nwin):
            start = stride * j
            ts = kdigo[start + gap + 1] - kdigo[start]
            loc = np.where(slopes == ts)[0][0]
            features[i, loc] += 1
        if ratios:
            features[i, :] = features[i, :] / np.sum(features[i, :])
    if ratios:
        arr2csv(filename, features, ids, fmt='%f', header=header)
    else:
        arr2csv(filename, features, ids, fmt='%d', header=header)
    return features, header

# %%
def get_cluster_features(individual, lbls, dm, op='mean'):
    clbls = np.unique(lbls)
    n_clust = len(clbls)
    n_feats = individual.shape[1]
    cluster_feats = np.zeros((n_clust, n_feats))
    cluster_feats_ind = np.zeros((len(lbls), n_feats))

    if dm.ndim == 1:
        dm = squareform(dm)
    for i in range(n_clust):
        tlbl = clbls[i]
        sel = np.where(lbls == tlbl)[0]
        cluster_ind = individual[sel, :]

        sqsel = np.ix_(sel, sel)
        cdm = dm[sqsel]
        cdm = np.sum(cdm, axis=0)
        cidx = np.argsort(cdm)[0]

        if type(op) == str:
            if op == 'mean':
                cluster_ind = np.mean(cluster_ind, axis=0)
            elif op == 'mean_bin':
                cluster_ind = np.mean(cluster_ind, axis=0)
                cluster_ind[np.where(cluster_ind >= 0.5)] = 1
                cluster_ind[np.where(cluster_ind < 1)] = 0
            elif op == 'center':
                cluster_ind = cluster_ind[cidx, :]
        else:
            cluster_ind = op(cluster_ind, axis=0)

        cluster_feats[i, :] = cluster_ind
        cluster_feats_ind[sel, :] = cluster_ind

    return cluster_feats_ind, cluster_feats

# %%
def assign_cluster_features(desc, temp, slope, lbls, dm):
    clbls = np.unique(lbls)
    n_clust = len(clbls)
    n_desc = desc.shape[1]
    n_temp = temp.shape[1]
    n_slope = slope.shape[1]
    desc_c = np.zeros((n_clust, n_desc))
    desc_c_bin = np.zeros((n_clust, n_desc))
    temp_c = np.zeros((n_clust, n_temp))
    slope_c = np.zeros((n_clust, n_slope))
    desc_c_center = np.zeros((n_clust, n_desc))
    temp_c_center = np.zeros((n_clust, n_temp))
    slope_c_center = np.zeros((n_clust, n_slope))
    if dm.ndim == 1:
        dm = squareform(dm)
    for i in range(n_clust):
        tlbl = clbls[i]
        sel = np.where(lbls == tlbl)[0]
        tdesc = desc[sel, :]
        ttemp = temp[sel, :]
        tslope = slope[sel, :]

        sqsel = np.ix_(sel, sel)
        cdm = dm[sqsel]
        cdm = np.sum(cdm, axis=0)
        cidx = np.argsort(cdm)[0]

        tdesc = np.mean(tdesc, axis=0)
        tdesc_bin = np.array(tdesc, copy=True)
        tdesc_bin[np.where(tdesc >= 0.5)] = 1
        tdesc_bin[np.where(tdesc < 1)] = 0
        ttemp = np.mean(ttemp, axis=0)
        tslope = np.mean(tslope, axis=0)

        desc_c[i, :] = tdesc
        desc_c_bin[i, :] = tdesc_bin
        temp_c[i, :] = ttemp
        slope_c[i, :] = tslope
        desc_c_center[i, :] = desc[sel[cidx], :]
        temp_c_center[i, :] = temp[sel[cidx], :]
        slope_c_center[i, :] = slope[sel[cidx], :]
    return desc_c_bin, desc_c, temp_c, slope_c, desc_c_center, temp_c_center, slope_c_center


# %%
def getStaticFeatures(ids, scrs, kdigos, days, stats, dataPath):
    header = 'STUDY_PATIENT_ID,AdmitScr,Age,Albumin,Anemia_A,Anemia_B,Anemia_C,Bicarbonate_Low,Bicarbonate_High,Bilirubin,BMI,BUN,Diabetic,' \
             'Dopamine,Epinephrine,FiO2_Low,FiO2_High,FluidOverload,Gender,GCS,Net_Fluid,Gross_Fluid,' \
             'HeartRate_Low,HeartRate_High,Hematocrit_low,Hematocrit_high,Hemoglobin_low,Hemoglobin_High,' \
             'Hypertensive,ECMO,IABP,MechanicalVentilation,VAD,Lactate,MAP_low,MAP_high,AdmitKDIGO,Nephrotox_ct,' \
             'Vasopress_ct,pCO2_low,pCO2_high,Peak_SCr,pH_low,pH_high,Platelets,pO2_low,pO2_high,Potassium_low,' \
             'Potassium_high,Race,Respiration_low,Respiration_high,Septic,Smoker,Sodium_low,Sodium_high,' \
             'Temperature_low,Temperature_high,Urine_flow,Urine_output,WBC_low,WBC_high,Height,Weight,hrsInICU,'

    feats = np.zeros((len(ids), (len(header.split(',')) + 63)))
    admit_scrs = np.zeros(len(ids))
    admit_kdigos = np.zeros(len(ids))
    peak_scrs = np.zeros(len(ids))
    hrsInIcu = np.zeros(len(ids))

    hosp_admits = get_array_dates(stats['hosp_dates'][:, 0].astype(str))
    icu_admits = get_array_dates(stats['icu_dates'][:, 0].astype(str))

    for i in range(len(ids)):
        admit_scrs[i] = scrs[i][0]
        admit_kdigos[i] = kdigos[i][0]
        peak_scrs[i] = np.max(scrs[i][np.where(days[i] <= 1)])
        hrs = (icu_admits[i] - hosp_admits[i]).total_seconds() / (60 * 60)
        if hrs < 24:
            hrsInIcu[i] = 24 - hrs

    tstats = {'AdmitScr': admit_scrs, 'AdmitKDIGO': admit_kdigos,
              'Peak_SCr': peak_scrs, 'Dopamine': stats['dopa'][:],
              'FluidOverload': stats['fluid_overload'][:], 'GCS': stats['glasgow'][:, 0],
              'HeartRate_Low': stats['heart_rate'][:, 0], 'HeartRate_High': stats['heart_rate'][:, 1],
              'MechanicalVentilation': stats['mv_flag_d1'][:], 'Urine_output': stats['urine_out'][:],
              'hrsInIcu': hrsInIcu, 'VAD': stats['vad_d1'][:],
              'ECMO': stats['ecmo_d1'], 'IABP': stats['iabp_d1']}

    col = 0
    ct = 0
    prev = ''
    for k in header.split(',')[1:]:
        if k.split('_')[0].lower() in list(stats):
            if k.split('_')[0] == prev.split('_')[0]:
                ct += 1
            else:
                ct = 0
            feats[:, col] = stats[k.split('_')[0].lower()][:, ct]
            col += 1
        elif k.lower() in list(stats):
            feats[:, col] = stats[k.lower()][:]

        prev = copy(k)

    header += ','.join(['SOFA_%d' % x for x in range(6)])
    header += ',' + ','.join(['APACHE_%d' % x for x in range(13)])
    header += ',' + ','.join(['Charlson_%d' % x for x in range(14)])
    header += ',' + ','.join(['Elixhauser_%d' % x for x in range(31)])

    sofa = load_csv(os.path.join(dataPath, 'sofa.csv'), ids, int, skip_header=True)
    feats[col:col+6] = sofa
    col += 6

    apache = load_csv(os.path.join(dataPath, 'apache.csv'), ids, int, skip_header=True)
    feats[col:col + 13] = apache
    col += 13

    feats[col:col+14] = stats['charlson_components'][:]
    col += 14

    feats[col:col + 31] = stats['elixhauser_components'][:]



# %%
def classify(X, y, classification_model, out_path, feature_name, gridsearch=False, sample_method='under', cv_num=5,
             pretrained=False, engine="sklearn", extX=None, exty=None, extPath=""):

    # 'criterion': ['gini', ],
    # 'max_features': ['sqrt', ]}]

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if not pretrained:
        if engine == "sklearn":
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

            clf.set_params(**params)
        else:
            if classification_model != 'log':
                print("Statsmodels not implemented for anything other than logistic regression.")
                exit()
    else:
        clf = pretrained

    log_file = open(os.path.join(out_path, 'classification_log.csv'), 'w')
    log_file.write('Fold_#,ROC_AUC,Accuracy,Sensitivity,Specificity,Precision,Recall,F1-Score,PPV,NPV,TP,FP,TN,FN\n')

    # auc, acc, sensitivity, specificity, ppv, npv, precision, recall, f1, support, tn, fp, fn, tp

    skf = StratifiedKFold(n_splits=cv_num, shuffle=True, random_state=1)

    tprs = []
    aucs = []
    probs = np.zeros(len(y))
    mean_fpr = np.linspace(0, 1, 100)
    fold_probs = []
    val_lbls = []
    imports = []
    coeffs = []
    intercepts = []
    slopes = []

    testExternal = False
    efold_probs = []
    if extX is not None and exty is not None:
        # eskf = StratifiedKFold(n_splits=cv_num, shuffle=True, random_state=1)
        # extSplitter = eskf.split(n_splits=cv_num, shuffle=True, random_state=1)
        etprs = []
        eaucs = []
        eprobs = np.zeros(len(y))
        emean_fpr = np.linspace(0, 1, 100)
        eval_lbls = []
        eimports = []
        ecoeffs = []
        testExternal = True
        elog_file = open(os.path.join(extPath, 'classification_log.csv'), 'w')
        elog_file.write(
            'Fold_#,ROC_AUC,Accuracy,Sensitivity,Specificity,Precision,Recall,F1-Score,PPV,NPV,TP,FP,TN,FN\n')

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

        if engine == "sklearn":
            if not pretrained:
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
                    clf = LogisticRegression(solver='lbfgs', max_iter=300)
                elif classification_model == 'xbg':
                    clf = XGBClassifier()

                clf.fit(X_train, y_train)
        else:
            clf = sm.Logit(y_train, X_train,).fit(method="lbfgs")

        # Plot the precision vs. recall curve
        if engine == "sklearn":
            probas = clf.predict_proba(X_val)[:, 1]
        else:
            probas = clf.predict(X_val)

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

        roc_auc, acc, sensitivity, specificity, ppv, \
        npv, precision, recall, f1, support, tn, fp, fn, tp = eval_classification(probas, y_val)

        log_file.write('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d\n' % (
            i + 1, roc_auc, acc, sensitivity, specificity, precision, recall, f1, ppv, npv, tp, fp, tn, fn))

        probs[val_idx] = probas
        fold_probs.append(probas)
        val_lbls.append(y_val)
        if hasattr(clf, "coef_"):
            imports.append(clf.coef_[0])

        elif hasattr(clf, "feature_importances_"):
            imports.append(clf.feature_importances_)

        if testExternal:
            if engine == "sklearn":
                probas = clf.predict_proba(extX)[:, 1]
            else:
                probas = clf.predict(extX)

            pos_idx = np.where(exty)[0]
            neg_idx = np.where(exty == 0)[0]

            pos_avg_score = np.nanmean(probas[pos_idx])
            neg_avg_score = np.nanmean(probas[neg_idx])

            slope = pos_avg_score - neg_avg_score
            intercept = clf.intercept_

            slopes.append(slope)
            intercepts.append(intercept)

            pcurve, rcurve, _ = precision_recall_curve(exty, probas)
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
                'Precision-Recall Curve: AP=%0.2f' % average_precision_score(exty,
                                                                             probas))

            # Plot ROC curve
            fpr, tpr, thresholds = roc_curve(exty, probas)
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
            plt.savefig(os.path.join(extPath, 'fold' + str(i + 1) + '_evaluation_external.png'))
            plt.close(fig)
            etprs.append(interp(emean_fpr, fpr, tpr))
            etprs[-1][0] = 0.0
            eaucs.append(roc_auc)

            roc_auc, acc, sensitivity, specificity, ppv, \
            npv, precision, recall, f1, support, tn, fp, fn, tp = eval_classification(probas, exty)

            elog_file.write('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d\n' % (
                i + 1, roc_auc, acc, sensitivity, specificity, precision, recall, f1, ppv, npv, tp, fp, tn, fn))

            efold_probs.append(probas)
        else:
            pos_idx = np.where(y_val)[0]
            neg_idx = np.where(y_val == 0)[0]

            pos_avg_score = np.nanmean(probas[pos_idx])
            neg_avg_score = np.nanmean(probas[neg_idx])

            slope = pos_avg_score - neg_avg_score
            intercept = clf.intercept_

            slopes.append(slope)
            intercepts.append(intercept)

        if engine == "sklearn":
            coeffs.append(clf.coef_[0])
        else:
            coeffs.append(clf.params)

        # dump(clf, os.path.join(out_path, "classifier_fold%d.joblib" % i))

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

    if testExternal:
        fig = plt.figure()
        for i in range(cv_num):
            plt.plot(emean_fpr, tprs[i], lw=1, alpha=0.3)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Luck', alpha=.8)
        emean_tpr = np.mean(etprs, axis=0)
        emean_tpr[-1] = 1.0
        mean_auc = auc(emean_fpr, emean_tpr)
        std_auc = np.std(eaucs)
        plt.plot(emean_fpr, emean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)
        std_tpr = np.std(etprs, axis=0)
        tprs_upper = np.minimum(emean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(emean_tpr - std_tpr, 0)
        plt.fill_between(emean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.title(classification_model.upper() + ' Classification Performance\n' + feature_name)
        plt.savefig(os.path.join(extPath, 'evaluation_summary.png'))
        plt.close(fig)
        elog_file.close()
        if engine == "sklearn":
            clf = clf.fit(extX, exty)
        else:
            clf = sm.Logit(exty, extX).fit(method="lbfgs")
    else:
        if engine == "sklearn":
            clf = clf.fit(X, y)
        else:
            clf = sm.Logit(y, X).fit(method="lbfgs")
    log_file.close()
    return clf, probs, fold_probs, val_lbls, imports, np.array(coeffs), efold_probs, intercepts, slopes


def classify_ie(X, y, classification_model, out_path, feature_name, extX, exty, sample_method='under', cv_num=5,
             engine="sklearn", extPath=""):

    # 'criterion': ['gini', ],
    # 'max_features': ['sqrt', ]}]

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if classification_model == 'rf':
        clf = RandomForestClassifier()
        params = rf_params
    elif classification_model == 'svm':
        clf = SVC()
        params = svm_params
    elif classification_model == 'mvr':
        clf = LinearRegression()
    elif classification_model == 'log':
        clf = LogisticRegression(n_jobs=-1)
        params = {}
    elif classification_model == 'XBG':
        clf = XGBClassifier()
        params = {}

    clf.set_params(**params)

    log_file = open(os.path.join(out_path, 'classification_log.csv'), 'w')
    log_file.write('Fold_#,ROC_AUC,Accuracy,Sensitivity,Specificity,Precision,Recall,F1-Score,PPV,NPV,TP,FP,TN,FN\n')

    # auc, acc, sensitivity, specificity, ppv, npv, precision, recall, f1, support, tn, fp, fn, tp

    skf = StratifiedKFold(n_splits=cv_num, shuffle=True, random_state=1)

    tprs = []
    aucs = []
    probs = np.zeros(len(y))
    mean_fpr = np.linspace(0, 1, 100)
    fold_probs = []
    val_lbls = []
    coeffs = []
    intercepts = []
    slopes = []

    exttprs = []
    extaucs = []
    extmean_fpr = np.linspace(0, 1, 100)
    extfold_probs = []
    extintercepts = []
    extslopes = []

    elog_file = open(os.path.join(extPath, 'classification_log.csv'), 'w')
    elog_file.write(
        'Fold_#,ROC_AUC,Accuracy,Sensitivity,Specificity,Precision,Recall,F1-Score,PPV,NPV,TP,FP,TN,FN\n')

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

        # Instantiate model
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
            clf = LogisticRegression(solver='lbfgs', max_iter=300)
        elif classification_model == 'xbg':
            clf = XGBClassifier()

        # Fit on internal cohort
        clf.fit(X_train, y_train)

        # Plot the precision vs. recall curve
        probas = clf.predict_proba(X_val)[:, 1]

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

        roc_auc, acc, sensitivity, specificity, ppv, \
        npv, precision, recall, f1, support, tn, fp, fn, tp = eval_classification(probas, y_val)

        log_file.write('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d\n' % (
            i + 1, roc_auc, acc, sensitivity, specificity, precision, recall, f1, ppv, npv, tp, fp, tn, fn))

        probs[val_idx] = probas
        fold_probs.append(probas)
        val_lbls.append(y_val)

        pos_idx = np.where(y_val)[0]
        neg_idx = np.where(y_val == 0)[0]

        pos_avg_score = np.nanmean(probas[pos_idx])
        neg_avg_score = np.nanmean(probas[neg_idx])

        slope = pos_avg_score - neg_avg_score

        slopes.append(slope)
        intercepts.append(neg_avg_score)

        if hasattr(clf, "coef_"):
            coeffs.append(clf.coef_[0])
        else:
            coeffs.append(clf.feature_importances_)


        # Save results for external cohort
        extprobas = clf.predict_proba(extX)[:, 1]

        pcurve, rcurve, _ = precision_recall_curve(exty, extprobas)
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
            'Precision-Recall Curve: AP=%0.2f' % average_precision_score(exty,
                                                                         extprobas))

        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(exty, extprobas)
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
        plt.savefig(os.path.join(extPath, 'fold' + str(i + 1) + '_evaluation_external.png'))
        plt.close(fig)
        exttprs.append(interp(extmean_fpr, fpr, tpr))
        exttprs[-1][0] = 0.0
        extaucs.append(roc_auc)

        roc_auc, acc, sensitivity, specificity, ppv, \
        npv, precision, recall, f1, support, tn, fp, fn, tp = eval_classification(extprobas, exty)

        elog_file.write('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d\n' % (
            i + 1, roc_auc, acc, sensitivity, specificity, precision, recall, f1, ppv, npv, tp, fp, tn, fn))

        extfold_probs.append(extprobas)

        pos_idx = np.where(exty)[0]
        neg_idx = np.where(exty == 0)[0]

        extpos_avg_score = np.nanmean(extprobas[pos_idx])
        extneg_avg_score = np.nanmean(extprobas[neg_idx])

        extslope = extpos_avg_score - extneg_avg_score

        extslopes.append(extslope)
        extintercepts.append(extneg_avg_score)

        # dump(clf, os.path.join(out_path, "classifier_fold%d.joblib" % i))

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

    fig = plt.figure()
    for i in range(cv_num):
        plt.plot(extmean_fpr, tprs[i], lw=1, alpha=0.3)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    extmean_tpr = np.mean(exttprs, axis=0)
    extmean_tpr[-1] = 1.0
    mean_auc = auc(extmean_fpr, extmean_tpr)
    std_auc = np.std(extaucs)
    plt.plot(extmean_fpr, extmean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    std_tpr = np.std(exttprs, axis=0)
    tprs_upper = np.minimum(extmean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(extmean_tpr - std_tpr, 0)
    plt.fill_between(extmean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title(classification_model.upper() + ' Classification Performance\n' + feature_name)
    plt.savefig(os.path.join(extPath, 'evaluation_summary.png'))
    plt.close(fig)
    elog_file.close()

    if engine == "sklearn":
        clf = clf.fit(X, y)
    else:
        clf = sm.Logit(y, X).fit(method="lbfgs")
    log_file.close()

    return clf, probs, fold_probs, val_lbls, np.array(coeffs), extfold_probs, intercepts, slopes, extintercepts, extslopes

# %%
def getStatmodelsCoefficients(fileName, nFeatures):
    f = open(fileName, "r")
    for i in range(12):
        _ = f.readline()
    coeffs = []
    lowers = []
    uppers = []
    for i in range(nFeatures):
        tokens = f.readline().rstrip().split()
        coeffs.append(float(tokens[1]))
        lowers.append(float(tokens[-2]))
        uppers.append(float(tokens[-1]))
    return coeffs, lowers, uppers


# %%
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
            estimator = selectionModel[1].lower()
            if estimator == 'extratrees':
                estimator = ExtraTreesClassifier(n_estimators=100,
                                     n_jobs=-1,
                                     random_state=0)
            elif estimator == 'svm':
                estimator = SVC(kernel='linear')
            elif estimator == 'logreg' or estimator == 'log' or estimator == 'lr':
                estimator = LogisticRegression(solver='lbfgs')
            elif estimator == 'xbg':
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


def build_class_vectors(ids, vecs, starts, dods, nDays=8, ptsPerDay=4, pad='zero', fillDir='back'):
    targLen = nDays * ptsPerDay
    out = np.zeros((len(ids), targLen))
    for i in range(len(vecs)):
        if dods[i] != 'nan':
            dodBin = int((get_date(dods[i]) - get_date(starts[i])).total_seconds() / (60 * 60 * 6))
            dodBin = min(len(vecs[i]), dodBin)
        else:
            dodBin = len(vecs[i])
        stop = min(dodBin, targLen)
        if fillDir == 'back':
            out[i, :stop] = vecs[i][:stop]
            if pad == 'fill':
                out[i, stop:] = vecs[i][stop-1]
        else:
            out[i, -stop:] = vecs[i][:stop]
            if pad == 'fill':
                out[i, :(targLen - stop)] = vecs[i][0]
    return out



# %%
def feature_selection(data, lbls, method='univariate', params=[]):
    '''

    :param data: Matrix of the form n_samples x n_features
    :param lbls: n_samples binary features
    :param method: {'variance', 'univariate', 'recursive', 'FromModel'}

    :param params: variance     - [float pct]
                   univariate   - [int n_feats, function scoring]
                   recursive    - [obj estimator, int cv]
                   FromModel
                        linear  - [float c]
                        tree    - []

    :return:
    '''

    if method == 'variance':
        assert len(params) == 1
        pct = params[0]
        sel = VarianceThreshold(threshold=(pct * (1 - pct)))
        sf = sel.fit_transform(data)
        return sel, sf

    elif method == 'univariate':
        assert len(params) == 2
        n_feats = params[0]
        scoring = params[1]
        sel = SelectKBest(scoring, k=n_feats)
        sf = sel.fit_transform(data, lbls)
        return sel, sf

    elif method == 'recursive':
        assert len(params) == 2
        estimator = params[0]
        cv = params[1]
        rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(cv),
                      scoring='accuracy')
        rfecv.fit(data, lbls)
        sel = rfecv.support_
        sf = data[:, sel]
        return rfecv, sf

    elif method == 'linear':
        assert len(params) == 1
        c = params[0]
        lsvc = LinearSVC(C=c, penalty='l1', dual=False).fit(data, lbls)
        model = SelectFromModel(lsvc, prefit=True)
        sf = model.transform(data)
        return model, sf

    elif method == 'tree':
        assert params == []
        clf = RandomForestClassifier()
        clf.fit(data, lbls)
        model = SelectFromModel(clf, prefit=True)
        sf = model.transform(data)
        return model, sf

    else:
        print("Please select one of the following methods:")
        print("[variance, univariate, recursive, linear, tree")


def build_blank_performance_spreadsheets(models, n_predictors=10):
    getCell = lambda col, row: "%s%d" % (chr(ord("A") + col), row + 1)

    wb = Workbook()
    pred_ws = wb.active
    pred_ws.title = "PredictorTable"
    perf_ws = wb.create_sheet(title="PerformanceTable")

    # Set up global alignments and fonts
    perfFont = Font(sz=16)
    predFont = Font(sz=14)
    al = Alignment(horizontal="center", vertical="center")
    # All font sizes the same, columns B+ center alignment
    for colNum in range(len(models) + 1):
        colLet = getCell(colNum, 0)[0]
        col = perf_ws.column_dimensions[colLet]
        col.data_type = "s"
        col.font = perfFont
        if colNum > 0:
            col.alignment = al
        col = pred_ws.column_dimensions[colLet]
        col.data_type = "s"
        col.font = predFont
        col.alignment = al
        if colNum > 0:
            col.alignment = al

    # First row and column both bold
    perfFont = Font(b=True, sz=16)
    predFont = Font(b=True, sz=14)
    col = pred_ws.column_dimensions["A"]
    col.font = predFont
    row = pred_ws.row_dimensions["1"]
    row.font = predFont

    col = perf_ws.column_dimensions["A"]
    col.font = perfFont
    row = perf_ws.row_dimensions["1"]
    row.font = perfFont

    cell = getCell(0, 0)
    pred_ws[cell] = "Model"
    perf_ws[cell] = "Model"
    # Fill in model names
    for colNum, model in enumerate(models):
        cell = getCell(colNum + 1, 0)
        pred_ws[cell] = model
        perf_ws[cell] = model


    # Set up predictor table
    cell = getCell(0, 1)
    pred_ws[cell] = "Predictors"

    row = n_predictors
    cell = getCell(0, row)
    pred_ws[cell] = "Model Performance Measures"

    row += 1
    cell = getCell(0, row)
    pred_ws[cell] = "C statistic (95%CI)"

    row += 1
    cell = getCell(0, row)
    pred_ws[cell] = "  -  Difference"

    row += 1
    cell = getCell(0, row)
    pred_ws[cell] = "  -  P Value"

    row += 1
    cell = getCell(0, row)
    pred_ws[cell] = "Integrated discrimination improvement (IDI) (95%CI)"

    row += 1
    cell = getCell(0, row)
    pred_ws[cell] = "  -  P Value"

    row += 1
    cell = getCell(0, row)
    pred_ws[cell] = "Sensitivity (95%CI)"

    row += 1
    cell = getCell(0, row)
    pred_ws[cell] = "Specificity (95%CI)"

    row += 1
    cell = getCell(0, row)
    pred_ws[cell] = "PPV (95%CI)"

    row += 1
    cell = getCell(0, row)
    pred_ws[cell] = "NPV (95%CI)"

    # Set up performance table
    row = 1
    cell = getCell(0, row)
    perf_ws[cell] = "AUC (95%CI)"

    row += 1
    cell = getCell(0, row)
    perf_ws[cell] = "Difference in AUC (95%CI)"

    row += 1
    cell = getCell(0, row)
    perf_ws[cell] = "  -  P Value"

    row += 1
    cell = getCell(0, row)
    perf_ws[cell] = "IDI (95%CI) %"

    row += 1
    cell = getCell(0, row)
    perf_ws[cell] = "  -  P Value"

    row += 1
    cell = getCell(0, row)
    perf_ws[cell] = "NRI (95%CI) %"

    row += 1
    cell = getCell(0, row)
    perf_ws[cell] = "  - Continuous"

    row += 1
    cell = getCell(0, row)
    perf_ws[cell] = "  -  P Value"

    row += 1
    cell = getCell(0, row)
    perf_ws[cell] = "  - Categorical"

    row += 1
    cell = getCell(0, row)
    perf_ws[cell] = "  -  P Value"

    row += 1
    cell = getCell(0, row)
    perf_ws[cell] = "  - Events No. (%)"

    row += 1
    cell = getCell(0, row)
    perf_ws[cell] = "  - Nonevents No. (%)"
    return wb

def plot_odds_ratios():
    pass
