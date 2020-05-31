import os
import h5py
import numpy as np
import argparse
import json
from sklearn.metrics import confusion_matrix
import pandas as pd
from stat_funcs import read_reclass_eval, iqr
from classification_funcs import build_blank_performance_spreadsheets as bbps

modelStrs = {"rf": "Random Forest", "log": "Logistic Regression", "svm": "SVM"}
targStrs = {"died_inp": "Hospital Mortality", "make90_d50": "MAKE-90 (50% Drop)"}
featStrs = {"sofa_apache_norm": "SOFA + APACHE II","max_kdigo_d03_norm": "Max KDIGO D0-3", "base_model": "Base Model", "base_model_withTrajectory": "Base Model with Trajectory", "clinical_model": "Clinical Model", "clinical_model_mortality": "Clinical Model (Mortality)", "clinical_model_make": "Clinical Model (MAKE)", "clinical_model_mortality_wTrajectory": "Clinical Model (Mortality) + Trajectory", "clinical_model_make_wTrajectory": "Clinical Model (MAKE) + Trajectory"}

##############
# --------------------- Parser Arguments
parser = argparse.ArgumentParser(description='Merge clusters.')
parser.add_argument('--config_file', action='store', type=str, dest='cfname',
                    default='kdigo_conf.json')
parser.add_argument('--config_path', action='store', type=str, dest='cfpath',
                    default='')
parser.add_argument('--features', '-f', action='store', type=str, dest='feats', nargs="*",
                    default=['sofa_norm', ])
parser.add_argument('--topNFeatures', '-topN', action='store', type=int, dest='topN',
                    default=10)
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

outcome = args.target
featureList = args.feats
printList = [featStrs[x] for x in featureList]

nTopFeats = args.topN

survivors = args.survivors

if not args.utsw:
    baseDataPath = os.path.join(basePath, 'DATA', 'all_sheets')
    dataPath = os.path.join(basePath, 'DATA', analyze, cohortName)
    resPath = os.path.join(basePath, 'RESULTS', analyze, cohortName)
    featPath = os.path.join(resPath, "features", "individual")
else:
    baseDataPath = os.path.join(basePath, 'DATA', 'dallas', 'csv')
    dataPath = os.path.join(basePath, 'DATA', 'dallas', analyze, cohortName)
    resPath = os.path.join(basePath, 'RESULTS', 'dallas', analyze, cohortName)
    featPath = os.path.join(resPath, "features", "individual")

f = h5py.File(os.path.join(resPath, "stats.h5"), 'r')
ids = f['meta']['ids'][:]
labels = f['meta'][outcome][:]

model = args.classModel

survsel = np.where(f['meta']['died_inp'][:] == 0)[0]
if survivors:
    sel = survsel
    labels = labels[sel]
    ids = ids[sel]

events = np.sum(labels)
nonevents = len(labels) - events

path = "temp"
os.mkdir(path)

try:
    indEvals = {}

    np.savetxt(os.path.join(path, "labels.txt"), labels, fmt="%d")
    baseFeature = featureList.pop(0)
    outcomePath = os.path.join(resPath, "classification", "%ddays" % t_lim, outcome)

    baseModelPath = os.path.join(outcomePath, baseFeature)

    if survivors:
        baseModelPath = os.path.join(baseModelPath, "survivors")

    # model = args.feats[0]

    baseModelPath = os.path.join(baseModelPath, model)
    baseModelPath = os.path.join(baseModelPath, args.engine)
    baseModelProbs = np.loadtxt(os.path.join(baseModelPath, "predicted_probas.csv"), delimiter=",", usecols=1, skiprows=1)

    indEvals[baseFeature] = {}
    reclassResults = {}
    baseFeatureNames = np.loadtxt(os.path.join(featPath, "%s.csv" % baseFeature), dtype="str", delimiter=",")[0][1:]
    if os.path.exists(os.path.join(baseModelPath, "feature_importances.csv")):
        baseModelImps = np.loadtxt(os.path.join(baseModelPath, "feature_importances.csv"))
    else:
        baseModelImps = np.abs(np.loadtxt(os.path.join(baseModelPath, "coefficients_avg.csv"), delimiter=",", usecols=1, skiprows=1))
    o = np.argsort(baseModelImps)[::-1]

    slopes = np.loadtxt(os.path.join(baseModelPath, "slopes_fold.csv"), delimiter=",", usecols=1, skiprows=1)
    ints = np.loadtxt(os.path.join(baseModelPath, "intercepts_fold.csv"), delimiter=",", usecols=1, skiprows=1)

    indEvals[baseFeature]["slope"] = [np.mean(slopes), np.percentile(slopes, 5), np.percentile(slopes, 95)]
    indEvals[baseFeature]["intercept"] = [np.mean(ints), np.percentile(ints, 5), np.percentile(ints, 95)]

    feats = []
    if len(baseFeatureNames) == 1:
        feats.append((baseFeatureNames[0], float(baseModelImps)))
    else:
        for i in range(nTopFeats):
            feats.append([baseFeatureNames[o[i]], baseModelImps[o[i]]])

    indEvals[baseFeature]["top_feats"] = feats
    basePreds = np.array([x > 0.5 for x in baseModelProbs], dtype=int)
    tn, fp, fn, tp = confusion_matrix(labels, basePreds).ravel()
    indEvals[baseFeature]['tn'] = tn
    indEvals[baseFeature]['fp'] = fp
    indEvals[baseFeature]['fn'] = fn
    indEvals[baseFeature]['tp'] = tp

    indEvals[baseFeature]["performance"] = pd.read_csv(os.path.join(baseModelPath, "classification_log.csv"))
    np.savetxt(os.path.join(path, "model1.txt"), baseModelProbs)

    # os.system("Rscript evalIndClassification.R")
    # res = read_class_eval(path)
    # reclassResults[baseFeature] = res
    # os.remove(os.path.join(path, "class.txt"))

    for feature in featureList:
        # if type(feature == list):

        indEvals[feature] = {}
        if survivors:
            modelPath = os.path.join(outcomePath, feature, "survivors", model)
        else:
           modelPath = os.path.join(outcomePath, feature, model)
        modelPath = os.path.join(modelPath, args.engine)
        featureNames = np.loadtxt(os.path.join(featPath, "%s.csv" % feature), delimiter=",", dtype=str)[0, 1:]
        if os.path.exists(os.path.join(modelPath, "feature_importances.csv")):
            try:
                imps = np.loadtxt(os.path.join(modelPath, "feature_importances.csv"), delimiter=",", usecols=1)
            except IndexError:
                imps = np.loadtxt(os.path.join(modelPath, "feature_importances.csv"), delimiter=",")
        else:
            try:
                imps = np.abs(np.loadtxt(os.path.join(modelPath, "coefficients_avg.csv"), delimiter=",", usecols=1, skiprows=1))
            except IndexError:
                imps = np.abs(np.loadtxt(os.path.join(modelPath, "coefficients_avg.csv"), delimiter=",", skiprows=1))

        o = np.argsort(imps)[::-1]

        feats = []
        for i in range(nTopFeats):
            feats.append([featureNames[o[i]], imps[o[i]]])
        indEvals[feature]["top_feats"] = feats

        try:
            indEval = pd.read_csv(os.path.join(modelPath, "classification_log.csv"))
        except FileNotFoundError:
            indEval = pd.read_csv(os.path.join(modelPath, "classification_log.txt"))
        indEvals[feature]["performance"] = indEval
        modelProbs = np.loadtxt(os.path.join(modelPath, "predicted_probas.csv"), delimiter=",", usecols=1, skiprows=1)
        np.savetxt(os.path.join(path, "model2.txt"), modelProbs)
        os.system("Rscript evalReclassification.R")
        res = read_reclass_eval(path)
        reclassResults[feature] = res

        preds = np.array([x > 0.5 for x in modelProbs], dtype=int)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        indEvals[feature]['tn'] = tn
        indEvals[feature]['fp'] = fp
        indEvals[feature]['fn'] = fn
        indEvals[feature]['tp'] = tp

        slopes = np.loadtxt(os.path.join(modelPath, "slopes_fold.csv"), delimiter=",", usecols=1, skiprows=1)
        ints = np.loadtxt(os.path.join(modelPath, "intercepts_fold.csv"), delimiter=",", usecols=1, skiprows=1)

        indEvals[feature]["slope"] = [np.mean(slopes), np.percentile(slopes, 5), np.percentile(slopes, 95)]
        indEvals[feature]["intercept"] = [np.mean(ints), np.percentile(ints, 5), np.percentile(ints, 95)]

# Remove the temporary directory
finally:
    for dirpath, dirnames, fnames in os.walk(path):
        for fname in fnames:
            os.remove(os.path.join(path, fname))
        os.rmdir(path)


#
# getCell = lambda col, row: "%s%d" % (chr(ord("A") + col), row + 1)
# wb = bbps(models=printList)
#
# # Fill in predictors
# for row in range(1, nTopFeats + 1):
#     col = 1
#     cell = getCell(col, row)
#     i = row - 1
#     if row < len(indEvals[baseFeature]["top_feats"]):
#         wb['PredictorTable'][cell] = indEvals[baseFeature]["top_feats"][i][0]
#     else:
#         wb['PredictorTable'][cell] = "-"
#
#     for feat in featureList:
#         col += 1
#         cell = getCell(col, row)
#         if row < len(indEvals[feat]["top_feats"]):
#             wb['PredictorTable'][cell] = indEvals[feat]["top_feats"][i][0]
#         else:
#             wb['PredictorTable'][cell] = "-"
#
# # AUC
# row = nTopFeats + 1
# col = 1
# cell = getCell(col, row)
# wb['PredictorTable'][cell] = "%.2f (%.2f-%.2f)" % tuple([reclassResults[featureList[0]]["roc"]["before"]["value"], ] +
#                                                         reclassResults[featureList[0]]["roc"]["before"]["ci"])
# for feat in featureList:
#     col += 1
#     cell = getCell(col, row)
#     wb['PredictorTable'][cell] = "%.2f (%.2f-%.2f)" % tuple(
#         [reclassResults[feat]["roc"]["after"]["value"], ] +
#         reclassResults[feat]["roc"]["after"]["ci"])
#
# # AUC Difference
# row += 1
# col = 1
# cell = getCell(col, row)
# wb['PredictorTable'][cell] = "-"
# for feat in featureList:
#     col += 1
#     cell = getCell(col, row)
#     wb['PredictorTable'][cell] = "%.2f" % (reclassResults[feat]["roc"]["after"]["value"] -
#                                            reclassResults[feat]["roc"]["before"]["value"])
# # P-val
# row += 1
# col = 1
# cell = getCell(col, row)
# wb['PredictorTable'][cell] = "-"
# for feat in featureList:
#     col += 1
#     cell = getCell(col, row)
#     wb['PredictorTable'][cell] = "%.2E" % (reclassResults[feat]["roc"]["pval"])
#
# # IDI
# row += 1
# col = 1
# cell = getCell(col, row)
# wb['PredictorTable'][cell] = "-"
# for feat in featureList:
#     col += 1
#     cell = getCell(col, row)
#     wb['PredictorTable'][cell] = "%.2f (%.2f-%.2f)" % tuple(
#         [reclassResults[feat]["idi"]["value"], ] +
#         reclassResults[feat]["idi"]["ci"])
#
# # P-val
# row += 1
# col = 1
# cell = getCell(col, row)
# wb['PredictorTable'][cell] = "-"
# for feat in featureList:
#     col += 1
#     cell = getCell(col, row)
#     wb['PredictorTable'][cell] = "%.2E" % (reclassResults[feat]["idi"]["pval"])
#
# # Sensitivity
# row += 1
# col = 1
# wb['PredictorTable'][cell] = "%.2f (%.2f-%.2f)" % iqr(indEvals[baseFeature]["performance"]["Sensitivity"])
# for feat in featureList:
#     col += 1
#     cell = getCell(col, row)
#     wb['PredictorTable'][cell] = "%.2f (%.2f-%.2f)" % iqr(indEvals[baseFeature]["performance"]["Sensitivity"])
#
# # Specificity
# row += 1
# col = 1
# wb['PredictorTable'][cell] = "%.2f (%.2f-%.2f)" % iqr(indEvals[featureList[0]]["performance"]["Specificity"])
# for feat in featureList:
#     col += 1
#     cell = getCell(col, row)
#     wb['PredictorTable'][cell] = "%.2f (%.2f-%.2f)" % iqr(indEvals[feat]["performance"]["Specificity"])
#
# # PPV
# row += 1
# col = 1
# wb['PredictorTable'][cell] = "%.2f (%.2f-%.2f)" % iqr(indEvals[featureList[0]]["performance"]["PPV"])
# for feat in featureList:
#     col += 1
#     cell = getCell(col, row)
#     wb['PredictorTable'][cell] = "%.2f (%.2f-%.2f)" % iqr(indEvals[feat]["performance"]["PPV"])
#
# # NPV
# row += 1
# col = 1
# wb['PredictorTable'][cell] = "%.2f (%.2f-%.2f)" % iqr(indEvals[featureList[0]]["performance"]["NPV"])
# for feat in featureList:
#     col += 1
#     cell = getCell(col, row)
#     wb['PredictorTable'][cell] = "%.2f (%.2f-%.2f)" % iqr(indEvals[feat]["performance"]["NPV"])
#
# # Peformance Table
# # AUC
# row = 1
# col = 1
# cell = getCell(col, row)
# wb['PerformanceTable'][cell] = "%.2f (%.2f-%.2f)" % tuple([reclassResults[featureList[0]]["roc"]["before"]["value"], ] +
#                                                         reclassResults[featureList[0]]["roc"]["before"]["ci"])
# for feat in featureList:
#     col += 1
#     cell = getCell(col, row)
#     wb['PerformanceTable'][cell] = "%.2f (%.2f-%.2f)" % tuple(
#         [reclassResults[feat]["roc"]["after"]["value"], ] +
#         reclassResults[feat]["roc"]["after"]["ci"])
#
# # AUC Difference
# row += 1
# col = 2
# cell = getCell(col, row)
# wb['PerformanceTable'][cell] = "-"
# for feat in featureList:
#     col += 1
#     cell = getCell(col, row)
#     wb['PerformanceTable'][cell] = "%.2f" % (reclassResults[feat]["roc"]["after"]["value"] -
#                                            reclassResults[feat]["roc"]["before"]["value"])
# # P-val
# row += 1
# col = 1
# cell = getCell(col, row)
# wb['PerformanceTable'][cell] = "-"
# for feat in featureList:
#     col += 1
#     cell = getCell(col, row)
#     wb['PerformanceTable'][cell] = "%.2E" % (reclassResults[feat]["roc"]["pval"])
#
# # IDI
# row += 1
# col = 1
# cell = getCell(col, row)
# wb['PerformanceTable'][cell] = "-"
# for feat in featureList:
#     col += 1
#     cell = getCell(col, row)
#     wb['PerformanceTable'][cell] = "%.2f (%.2f-%.2f)" % tuple(
#         [reclassResults[feat]["idi"]["value"], ] +
#         reclassResults[feat]["idi"]["ci"])
#
# # P-val
# row += 1
# col = 1
# cell = getCell(col, row)
# wb['PerformanceTable'][cell] = "-"
# for feat in featureList:
#     col += 1
#     cell = getCell(col, row)
#     wb['PerformanceTable'][cell] = "%.2E" % (reclassResults[feat]["idi"]["pval"])
#
# # NRI
# row += 1
# # Continuous
# row += 1
# col = 1
# cell = getCell(col, row)
# wb['PerformanceTable'][cell] = "-"
# for feat in featureList:
#     col += 1
#     cell = getCell(col, row)
#     wb['PerformanceTable'][cell] = "%.2f (%.2f-%.2f)" % tuple(
#         [reclassResults[feat]["nri"]["continuous"]["value"], ] +
#         reclassResults[feat]["nri"]["continuous"]["ci"])
#
# # P-val
# row += 1
# col = 1
# cell = getCell(col, row)
# wb['PerformanceTable'][cell] = "-"
# for feat in featureList:
#     col += 1
#     cell = getCell(col, row)
#     wb['PerformanceTable'][cell] = "%.2E" % (reclassResults[feat]["nri"]["continuous"]["pval"])
#
# # Categorical
# row += 1
# col = 1
# cell = getCell(col, row)
# wb['PerformanceTable'][cell] = "-"
# for feat in featureList:
#     col += 1
#     cell = getCell(col, row)
#     wb['PerformanceTable'][cell] = "%.2f (%.2f-%.2f)" % tuple(
#         [reclassResults[feat]["nri"]["categorical"]["value"], ] +
#         reclassResults[feat]["nri"]["categorical"]["ci"])
#
# # P-val
# row += 1
# col = 1
# cell = getCell(col, row)
# wb['PerformanceTable'][cell] = "-"
# for feat in featureList:
#     col += 1
#     cell = getCell(col, row)
#     wb['PerformanceTable'][cell] = "%.2E" % (reclassResults[feat]["nri"]["categorical"]["pval"])
#
# # Events
# row += 1
# col = 1
# cell = getCell(col, row)
# wb['PerformanceTable'][cell] = "-"
# for feat in featureList:
#     col += 1
#     cell = getCell(col, row)
#     btp = indEvals[baseFeature]['tp']
#     ntp = indEvals[feat]['tp']
#     ct = ntp - btp
#     wb['PerformanceTable'][cell] = "%d (%.2f)" % (ct, (ct / events) * 100)
#
# # Non-Events
# row += 1
# col = 1
# cell = getCell(col, row)
# wb['PerformanceTable'][cell] = "-"
# for feat in featureList:
#     col += 1
#     cell = getCell(col, row)
#     btn = indEvals[baseFeature]['tn']
#     ntn = indEvals[feat]['tn']
#     ct = ntn - btn
#     wb['PerformanceTable'][cell] = "%d (%.2f)" % (ct, (ct / nonevents) * 100)
#
# print("Done building Excel spreadsheets...")

# wb.save(os.path.join(outcomePath, "reclassification_tables_%s.xlsx" % model))
#
# Predictor Table
# if survivors:
#     out = open(os.path.join(outcomePath, "predictor_table_%s_survivors_%s.csv" % (model, args.engine)), "w")
# else:
#     out = open(os.path.join(outcomePath, "predictor_table_%s_%s.csv" % (model, args.engine)), "w")
#
# s = "Models," + baseFeature
# for feat in featureList:
#     s += "," + feat
# s += "\n"
# out.write(s)
#
# s = "Predictors"
# for i in range(nTopFeats):
#     if i < len(indEvals[baseFeature]["top_feats"]):
#         s += "," + indEvals[baseFeature]["top_feats"][i][0]
#     else:
#         s += ",-"
#
#     for feat in featureList:
#         if i < len(indEvals[feat]["top_feats"]):
#             s += "," + indEvals[feat]["top_feats"][i][0]
#         else:
#             s += ",-"
#     s += "\n"
#     out.write(s)
#     s = ""
#
# out.write("Model Performance Measures\n")
# # out.write("C statistic (95%CI)\n")
# s = "C statistic (95%CI)"
# s += ",%.2f" % reclassResults[featureList[0]]["roc"]["before"]["value"]
# s += " (%.2f-%.2f)" % tuple(reclassResults[featureList[0]]["roc"]["before"]["ci"])
# for feat in featureList:
#     s += ",%.2f" % reclassResults[feat]["roc"]["after"]["value"]
#     s += " (%.2f-%.2f)" % tuple(reclassResults[feat]["roc"]["after"]["ci"])
# s += "\n"
# out.write(s)
#
# s = "  -  Difference,-"
# for feat in featureList:
#     s += ",%.2f" % (reclassResults[feat]["roc"]["after"]["value"] - reclassResults[feat]["roc"]["before"]["value"])
# s += "\n"
# out.write(s)
#
# s = "  -  P Value,-"
# for feat in featureList:
#     s += ",%.2E" % (reclassResults[feat]["roc"]["pval"])
# s += "\n"
# out.write(s)
#
# s = "Integrated discrimination improvement (IDI) (95%CI),-"
# for feat in featureList:
#     s += ",%.2f" % reclassResults[feat]["idi"]["value"]
#     s += " (%.2f-%.2f)" % tuple(reclassResults[feat]["idi"]["ci"])
# s += "\n"
# out.write(s)
#
# s = "  -  P Value,-"
# for feat in featureList:
#     s += ",%.2E" % (reclassResults[feat]["idi"]["pval"])
# s += "\n"
# out.write(s)
#
#
# s = "Sensitivity (95%CI)"
# s += ",%.2f (%.2f-%.2f)" % iqr(indEvals[featureList[0]]["performance"]["Sensitivity"])
# for feat in featureList:
#     s += ",%.2f (%.2f-%.2f)" % iqr(indEvals[feature]["performance"]["Sensitivity"])
# s += "\n"
# out.write(s)
#
# s = "Specificity (95%CI)"
# s += ",%.2f (%.2f-%.2f)" % iqr(indEvals[featureList[0]]["performance"]["Specificity"])
# for feat in featureList:
#     s += ",%.2f (%.2f-%.2f)" % iqr(indEvals[feature]["performance"]["Specificity"])
# s += "\n"
# out.write(s)
#
# s = "PPV (95%CI)"
# s += ",%.2f (%.2f-%.2f)" % iqr(indEvals[featureList[0]]["performance"]["PPV"])
# for feat in featureList:
#     s += ",%.2f (%.2f-%.2f)" % iqr(indEvals[feature]["performance"]["PPV"])
# s += "\n"
# out.write(s)
#
# s = "NPV (95%CI)"
# s += ",%.2f (%.2f-%.2f)" % iqr(indEvals[featureList[0]]["performance"]["NPV"])
# for feat in featureList:
#     s += ",%.2f (%.2f-%.2f)" % iqr(indEvals[feature]["performance"]["NPV"])
# s += "\n"
# out.write(s)
# out.close()


# predictive_performance.csv
if survivors:
    out = open(os.path.join(outcomePath, "predictive_performance_%s_survivors_%s.csv" % (model, args.engine)), "w")
else:
    out = open(os.path.join(outcomePath, "predictive_performance_%s_%s.csv" % (model, args.engine)), "w")

s = "Models," + baseFeature
for feat in featureList:
    s += "," + feat
s += "\n"
out.write(s)

# Intercepts
s = "Intercept (95%CI)"
s += ",%.2f" % (indEvals[featureList[0]]["intercept"][0])
s += " (%.2f-%.2f)" % tuple([indEvals[featureList[0]]["intercept"][1], indEvals[featureList[0]]["intercept"][2]])
for feat in featureList:
    s += ",%.2f" % indEvals[feat]["intercept"][0]
    s += " (%.2f-%.2f)" % tuple([indEvals[feat]["intercept"][1], indEvals[feat]["intercept"][2]])
s += "\n"
out.write(s)

# Slopes
s = "Slope (95%CI)"
s += ",%.2f" % (indEvals[featureList[0]]["slope"][0])
s += " (%.2f-%.2f)" % tuple([indEvals[featureList[0]]["slope"][1], indEvals[featureList[0]]["slope"][2]])
for feat in featureList:
    s += ",%.2f" % indEvals[feat]["slope"][0]
    s += " (%.2f-%.2f)" % tuple([indEvals[feat]["slope"][1], indEvals[feat]["slope"][2]])
s += "\n"
out.write(s)

# out.write("C statistic (95%CI)\n")
s = "AUC (95%CI)"
s += ",%.2f" % reclassResults[featureList[0]]["roc"]["before"]["value"]
s += " (%.2f-%.2f)" % tuple(reclassResults[featureList[0]]["roc"]["before"]["ci"])
for feat in featureList:
    s += ",%.2f" % reclassResults[feat]["roc"]["after"]["value"]
    s += " (%.2f-%.2f)" % tuple(reclassResults[feat]["roc"]["after"]["ci"])
s += "\n"
out.write(s)

s = "Difference in AUC (95%CI),-"
for feat in featureList:
    s += ",%.2f" % (reclassResults[feat]["roc"]["after"]["value"] - reclassResults[feat]["roc"]["before"]["value"])
s += "\n"
out.write(s)

s = "  -  P Value,-"
for feat in featureList:
    s += ",%.2E" % (reclassResults[feat]["roc"]["pval"])
s += "\n"
out.write(s)

s = "IDI (95%CI) %,-"
for feat in featureList:
    s += ",%.2f" % reclassResults[feat]["idi"]["value"]
    s += " (%.2f-%.2f)" % tuple(reclassResults[feat]["idi"]["ci"])
s += "\n"
out.write(s)

s = "  -  P Value,-"
for feat in featureList:
    s += ",%.2E" % (reclassResults[feat]["idi"]["pval"])
s += "\n"
out.write(s)

s = "NRI (95%CI) %\n"
out.write(s)

s = "  - Continuous,-"
for feat in featureList:
    s += ",%.2f" % reclassResults[feat]["nri"]["continuous"]["value"]
    s += " (%.2f-%.2f)" % tuple(reclassResults[feat]["nri"]["continuous"]["ci"])
s += "\n"
out.write(s)

s = "  -  P Value,-"
for feat in featureList:
    s += ",%.2E" % (reclassResults[feat]["nri"]["continuous"]["pval"])
s += "\n"
out.write(s)

s = "  - Categorical,-"
for feat in featureList:
    s += ",%.2f" % reclassResults[feat]["nri"]["categorical"]["value"]
    s += " (%.2f-%.2f)" % tuple(reclassResults[feat]["nri"]["categorical"]["ci"])
s += "\n"
out.write(s)

s = "  -  P Value,-"
for feat in featureList:
    s += ",%.2E" % (reclassResults[feat]["nri"]["categorical"]["pval"])
s += "\n"
out.write(s)

s = "  - Events No. (%),-"
for feat in featureList:
    btp = indEvals[baseFeature]['tp']
    ntp = indEvals[feat]['tp']
    ct = ntp - btp
    #
    # val = reclassResults[feat]["nri"]["events"]["value"]
    # ct = int(val * events)
    s += ",%d (%.2f)" % (ct, (ct / np.sum(labels))*100)
    # s += " (%.2f-%.2f)" % tuple(reclassResults[feat]["nri"]["events"]["ci"])
s += "\n"
out.write(s)

s = "  - Nonevents No. (%),-"
for feat in featureList:
    btn = indEvals[baseFeature]['tn']
    ntn = indEvals[feat]['tn']
    ct = ntn - btn

    # val = reclassResults[feat]["nri"]["non-events"]["value"]
    # ct = int(val * events)
    s += ",%d (%.2f)" % (ct, (ct / (len(labels) - np.sum(labels)))*100)
    # s += " (%.2f-%.2f)" % tuple(reclassResults[feat]["nri"]["non-events"]["ci"])
s += "\n"
out.write(s)

out.close()
