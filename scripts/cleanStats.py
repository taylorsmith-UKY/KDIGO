import os
import numpy as np
import pandas as pd
import json
import h5py
from utility_funcs import arr2csv, load_csv, dict2csv, get_array_dates
from stat_funcs import summarize_stats, formatted_stats, get_uky_urine, get_uky_fluids
from copy import copy
from sklearn.preprocessing import MinMaxScaler
import argparse

parser = argparse.ArgumentParser(description='Preprocess Data and Construct KDIGO Vectors.')
parser.add_argument('--config_file', action='store', type=str, dest='cfname',
                    default='kdigo_conf.json')
parser.add_argument('--averagePoints', '-agvPts', action='store', type=int, dest='avgpts',
                    default=2)
parser.add_argument('--meta_group', '-meta', action='store', type=str, dest='meta',
                    default='meta')
parser.add_argument('--data_file', '-df', action='store', type=str, dest='df',
                    default='stats.h5')
args = parser.parse_args()

configurationFileName = os.path.join(args.cfpath, args.cfname)
fp = open(configurationFileName, 'r')
conf = json.load(fp)
fp.close()
basePath = conf['basePath']         # Path containing the DATA and RESULTS directories
cohortName = conf['cohortName']     # Will be used for folder name
t_lim = conf['analysisDays']        # How long to consider in the analysis
tRes = conf['timeResolutionHrs']    # Resolution after imputation in hours
analyze = conf['analyze']           # Time period to analyze (hospital/ICU/all)

# Build paths and create if don't already exist
baseDataPath = os.path.join(basePath, 'DATA', 'all_sheets')         # folder containing all raw data
dataPath = os.path.join(basePath, 'DATA', analyze, cohortName)
resPath = os.path.join(basePath, 'RESULTS', analyze, cohortName)

grp_name = args.meta
statFileName = args.df
f = h5py.File(os.path.join(resPath, statFileName), 'r+')
ids = f[grp_name]['ids'][:]

kdigos = load_csv(os.path.join(dataPath, 'kdigo_icu_2ptAvg.csv'), ids, int)
days = load_csv(os.path.join(dataPath, 'days_interp_icu_2ptAvg.csv'), ids, int)
scrs = load_csv(os.path.join(dataPath, 'scr_interp_icu_2ptAvg.csv'), ids, float)
icu_windows = load_csv(os.path.join(dataPath, 'icu_admit_discharge.csv'), ids, 'date', struct='dict')
hosp_windows = load_csv(os.path.join(dataPath, 'hosp_admit_discharge.csv'), ids, 'date', struct='dict')

if grp_name not in list(f):
    raise RuntimeError("Group name %s not found in file: %s" % (grp_name, statFileName))

f.copy(f[grp_name], '/%s_cleaned' % grp_name)
stats = f['%s_cleaned' % grp_name]

lims = {'bicarbonate': (5, 50),
        'bun': (5, None),
        'fio2': (21, 100),
        'heart_rate': (25, None),
        'pco2': (0, 125),
        'respiration': (5, 70),
        'temperature': (86, 113),
        'weight': (30, 200),
        'map': (0, None)}

# bothpercents = ['map', 'height']

bottomcents = ['map_low', 'height']
toppercents = ['urine_out', 'wbc_low', 'wbc_high', 'map_high', 'height']

# Remove  bad values
# By provided explicit ranges:
for k in list(lims):
    temp = stats[k][:]
    lowCount = 0
    highCount = 0
    if lims[k][0] is not None:
        lowCount = len(np.where(temp < lims[k][0])[0])
        temp[np.where(temp < lims[k][0])] = np.nan
    if lims[k][1] is not None:
        highCount = len(np.where(temp > lims[k][1])[0])
        temp[np.where(temp > lims[k][1])] = np.nan
    # stats[k][:] = temp
    # print(k, lowCount, highCount)


for k in bottomcents:
    ct = 0
    if k in list(stats):
        temp = stats[k][:]
        m = np.nanmean(temp)
        std = np.nanstd(temp)
        ct = len(np.where(temp < (m - (std * 1.96)))[0])
        temp[np.where(temp < (m - (std * 1.96)))] = np.nan
        # stats[k][:] = temp
    else:
        tk = k.split("_")[0]
        if k.split("_")[1] == "low":
            idx = 0
        elif k.split("_")[1] == "high":
            idx = 1
        temp = stats[tk][:, idx]
        m = np.nanmean(temp)
        std = np.nanstd(temp)
        ct = len(np.where(temp < (m - (std * 1.96)))[0])
        temp[np.where(temp < (m - (std * 1.96)))] = np.nan
        # stats[tk][:, idx] = temp
    # print(k, ct)

for k in toppercents:
    if k in list(stats):
        temp = stats[k][:]
        m = np.nanmean(temp)
        std = np.nanstd(temp)
        ct = len(np.where(temp > (m + (std * 1.96)))[0])
        temp[np.where(temp > (m + (std * 1.96)))] = np.nan
        # stats[k][:] = temp
    else:
        tk = k.split("_")[0]
        if k.split("_")[1] == "low":
            idx = 0
        elif k.split("_")[1] == "high":
            idx = 1
        temp = stats[tk][:, idx]
        m = np.nanmean(temp)
        std = np.nanstd(temp)
        ct = len(np.where(temp > (m + (std * 1.96)))[0])
        temp[np.where(temp > (m + (std * 1.96)))] = np.nan

# It is OK to replace height and weight with imputed values
temp = stats['height'][:]
tm = np.nanmedian(temp)
temp[np.where(np.isnan(temp))] = tm
temp[np.where(temp == 0)[0]] = tm
stats['height'][:] = temp

temp = stats['weight'][:]
tm = np.nanmedian(temp)
temp[np.where(np.isnan(temp))] = tm
stats['weight'][:] = temp

# Recalculate BMI using imputed values
stats['bmi'][:] = stats['weight'][:] / (stats['height'][:] * stats['height'][:])

# Recalculate urine out and urine flow with imputed weights
urine_outs, urine_flows = get_uky_urine(ids, stats['weight'][:], baseDataPath)
temp = urine_outs
m = np.nanmean(temp)
std = np.nanstd(temp)
temp[np.where(temp > (m + (std * 1.96)))] = np.nan
urine_outs = temp

stats['urine_out'][:] = urine_outs
stats['urine_flow'][:] = urine_flows

# Recalculate fluid overloads using imputed weights
nets, tots, fos, cfbs = get_uky_fluids(ids, stats['weight'][:], baseDataPath)

stats['fluid_overload'][:] = fos

if 'sofa' not in list(stats):
    sofa = load_csv(os.path.join(dataPath, 'sofa.csv'), ids, int, skip_header=True)
    sofa = np.sum(sofa, axis=1)
    stats['sofa'] = sofa

if 'apache' not in list(stats):
    apache = load_csv(os.path.join(dataPath, 'apache.csv'), ids, int, skip_header=True)
    apache = np.sum(apache, axis=1)
    stats['apache'] = apache

f.copy(f['%s_cleaned' % grp_name], '/%s_imputed' % grp_name)
nstats = f['%s_imputed' % grp_name]
for k in list(nstats):
    temp = nstats[k][:]
    if "crrt" in k or "hd" in k or "mv" in k:
        temp[np.where(np.isnan(temp))] = 0
        nstats[k][:] = temp
    else:
        try:
            tm = np.nanmedian(temp)
            temp[np.where(np.isnan(temp))] = tm
            nstats[k][:] = temp
        except TypeError:
            pass

f.close()
