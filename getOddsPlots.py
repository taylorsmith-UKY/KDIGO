import os
import numpy as np
import json
import h5py
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description='Preprocess Data and Construct KDIGO Vectors.')
parser.add_argument('--config_file', action='store', type=str, dest='cfname',
                    default='kdigo_conf.json')
parser.add_argument('--config_path', action='store', type=str, dest='cfpath',
                    default='')
parser.add_argument('--averagePoints', '-agvPts', action='store', type=int, dest='avgpts',
                    default=2)
parser.add_argument('--meta_group', '-meta', action='store', type=str, dest='meta',
                    default='meta_imputed')
parser.add_argument('--data_file', '-df', action='store', type=str, dest='df',
                    default='stats.h5')
parser.add_argument('--features', '-f', action='store', type=str, dest='feats',
                    default='sofa_norm')
parser.add_argument('--numFeatures', '-nfeats', action='store', type=int, dest='nfeats',
                    default=20)
parser.add_argument('--target', '-t', action='store', type=str, dest='target',
                    default='died_inp')
parser.add_argument('--classificationModel', '-class', action='store', type=str, dest='classModel',
                    default='log', choices=['log', 'svm', 'rf', 'mvr', 'xbg'])
parser.add_argument('--engine', '-eng', action='store', type=str, dest='engine',
                    default='sklearn', choices=['statsmodels', 'sklearn'])
parser.add_argument('--only_survivors', '-survivors', action='store_true', dest='survivors')
args = parser.parse_args()

grp_name = args.meta
statFileName = args.df

configurationFileName = os.path.join(args.cfpath, args.cfname)
fp = open(configurationFileName, 'r')
conf = json.load(fp)
fp.close()
basePath = conf['basePath']         # Path containing the DATA and RESULTS directories
cohortName = conf['cohortName']     # Will be used for folder name
t_lim = conf['analysisDays']        # How long to consider in the analysis
tRes = conf['timeResolutionHrs']    # Resolution after imputation in hours
analyze = conf['analyze']           # Time period to analyze (hospital/ICU/all)

elixIdxs = [0, 4, 5, 6, 9, 10, 11, 14, 15, 16, 18, 20, 30]

# Build paths and create if don't already exist
baseDataPath = os.path.join(basePath, 'DATA', 'all_sheets')         # folder containing all raw data
dataPath = os.path.join(basePath, 'DATA', analyze, cohortName)
resPath = os.path.join(basePath, 'RESULTS', analyze, cohortName)

f = h5py.File(os.path.join(resPath, statFileName), 'r')
ids = f[grp_name]['ids'][:]

indFeatPath = os.path.join(resPath, 'features', 'individual')

target = args.target
featName = args.feats
model = args.classModel
engine = args.engine

classPath = os.path.join(resPath, 'classification', '%ddays' % t_lim, target, featName)
if args.survivors:
    classPath = os.path.join(classPath, 'survivors')

classPath = os.path.join(classPath, model, engine)

df = pd.read_csv(os.path.join(classPath, "odds_ratios_fold.csv"), index_col="Fold_#")
cdf = pd.read_csv(os.path.join(classPath, "coefficients_fold.csv"), index_col="Fold_#")
avg_coeffs = np.zeros(len(cdf.columns))
avg_odds = np.zeros(len(cdf.columns))

for i, k in enumerate(list(cdf)):
    avg_coeffs[i] = np.mean(cdf[k].values)
    avg_odds[i] = np.mean(df[k].values)

mag = np.abs(avg_coeffs)
order = np.argsort(mag)[::-1]

# All Features
of20 = open(os.path.join(classPath, "coefficient_summary_top%d.csv" % args.nfeats), "w")
of20.write("Feature,Coefficient,OddsRatio\n")

ofall = open(os.path.join(classPath, "coefficient_summary_allFeatures.csv"), "w")
ofall.write("Feature,Coefficient,OddsRatio\n")

ct = 0
for k in np.array(list(df))[order]:
    coef = np.mean(cdf[k].values)
    odds = np.mean(df[k].values)
    odds_lower = np.percentile(df[k].values, 5)
    odds_upper = np.percentile(df[k].values, 95)
    coef_lower = np.percentile(cdf[k].values, 5)
    coef_upper = np.percentile(cdf[k].values, 95)
    if ct < 20:
        of20.write("%s,%.2f (%.2f-%.2f),%.2f (%.2f-%.2f)\n" % (k, coef, coef_lower, coef_upper, odds, odds_lower, odds_upper))
    ofall.write("%s,%.2f (%.2f-%.2f),%.2f (%.2f-%.2f)\n" % (k, coef, coef_lower, coef_upper, odds, odds_lower, odds_upper))
    ct += 1
of20.close()
ofall.close()

odf = open(os.path.join(classPath, "odds_plot_allFeats.csv"), "w")
odf.write("Feature,Fold_#,OddsRatio\n")
for k in np.array(list(df))[order]:
    for i in range(10):
        odf.write("%s,%d,%f\n" % (k, i, df[k][i]))
odf.close()

nrows = np.floor(len(order) / 10) + 1
df = pd.read_csv(os.path.join(classPath, "odds_plot_allFeats.csv"))
fig = plt.figure(figsize=[6.4, 4.8 * nrows])
sns.boxplot(x="OddsRatio", y="Feature", data=df)
plt.tight_layout()
plt.savefig(os.path.join(classPath, "odds_ratios_allFeats.png"), dpi=600)
plt.close(fig)

fig = plt.figure(figsize=[6.4, 4.8 * nrows])
sns.boxplot(x="OddsRatio", y="Feature", data=df)
plt.tight_layout()
plt.xlim(0, 1)
plt.savefig(os.path.join(classPath, "odds_ratios_only_positives_allFeats.png"), dpi=600)
plt.close(fig)

fig = plt.figure(figsize=[6.4, 4.8 * nrows])
sns.boxplot(x="OddsRatio", y="Feature", data=df)
plt.tight_layout()
plt.xlim(0, 5)
plt.savefig(os.path.join(classPath, "odds_ratios_middle_allFeats.png"), dpi=600)
plt.close(fig)

df = pd.read_csv(os.path.join(classPath, "odds_ratios_fold.csv"), index_col="Fold_#")
# Top Features
odf = open(os.path.join(classPath, "odds_plot_%d.csv" % args.nfeats), "w")
odf.write("Feature,Fold_#,OddsRatio\n")
for k in np.array(list(df))[order][:args.nfeats]:
    for i in range(10):
        odf.write("%s,%d,%f\n" % (k, i, df[k][i]))
odf.close()

df = pd.read_csv(os.path.join(classPath, "odds_plot_%d.csv" % args.nfeats))
fig = plt.figure()
sns.boxplot(x="OddsRatio", y="Feature", data=df)
plt.tight_layout()
plt.savefig(os.path.join(classPath, "odds_ratios_top%d_all.png" % args.nfeats), dpi=600)
plt.close(fig)

fig = plt.figure()
sns.boxplot(x="OddsRatio", y="Feature", data=df)
plt.tight_layout()
plt.xlim(0, 1)
plt.savefig(os.path.join(classPath, "odds_ratios_top%d_only_positives.png" % args.nfeats), dpi=600)
plt.close(fig)

fig = plt.figure()
sns.boxplot(x="OddsRatio", y="Feature", data=df)
plt.tight_layout()
plt.xlim(0, 5)
plt.savefig(os.path.join(classPath, "odds_ratios_top%d_middle.png" % args.nfeats), dpi=600)
plt.close(fig)