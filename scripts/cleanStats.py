import os
import numpy as np
import pandas as pd
import json
import h5py
from utility_funcs import arr2csv, load_csv, dict2csv, get_array_dates
from stat_funcs import summarize_stats, formatted_stats, get_uky_urine, get_uky_fluids
from copy import copy
from sklearn.preprocessing import MinMaxScaler

grp_name = 'meta'
statFileName = 'stats_111219.h5'

fp = open('../kdigo_conf1.json', 'r')
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
f = h5py.File(os.path.join(resPath, statFileName), 'r+')
ids = f[grp_name]['ids'][:]

kdigos = load_csv(os.path.join(dataPath, 'kdigo_icu_2ptAvg.csv'), ids, int)
days = load_csv(os.path.join(dataPath, 'days_interp_icu_2ptAvg.csv'), ids, int)
scrs = load_csv(os.path.join(dataPath, 'scr_interp_icu_2ptAvg.csv'), ids, float)
icu_windows = load_csv(os.path.join(dataPath, 'icu_admit_discharge.csv'), ids, 'date', struct='dict')
hosp_windows = load_csv(os.path.join(dataPath, 'hosp_admit_discharge.csv'), ids, 'date', struct='dict')

if grp_name in list(f):
    stats = f[grp_name]
else:
    stats = summarize_stats(f, ids, kdigos, days, scrs, icu_windows, hosp_windows, baseDataPath, grp_name, t_lim)
f.copy(f[grp_name], '/%s_cleaned' % grp_name)
stats = f['%s_cleaned' % grp_name]

lims = {'bicarbonate': (5, 50),
        'bun': (5, 1000),
        'fio2': (21, 100),
        'heart_rate': (25, 1000),
        'pco2': (0, 125),
        'respiration': (5, 70),
        'temperature': (86, 113),
        'weight': (30, 200),
        'map': (0, 1000)}

bothpercents = ['map', 'height']
toppercents = ['urine_out', 'wbc']

# Remove  bad values
# By provided explicit ranges:
for k in list(lims):
    temp = stats[k][:]
    temp[np.where(temp < lims[k][0])] = np.nan
    temp[np.where(temp > lims[k][1])] = np.nan
    stats[k][:] = temp
    # try:
    #     temp[np.where(np.isnan(temp))] = np.nan
    #     stats[k][:] = temp
    # except TypeError:
    #     pass


# # By applying confidence interval limit to the top
# for k in toppercents:
#     temp = stats[k][:]
#     m = np.nanmean(temp)
#     std = np.nanstd(temp)
#     temp[np.where(temp > (m + (std * 1.96)))] = np.nan
#     stats[k][:] = temp

# It is OK to replace height and weight with imputed values
temp = stats['height'][:]
tm = np.nanmedian(temp)
temp[np.where(np.isnan(temp))] = tm
stats['height'][:] = temp

temp = stats['weight'][:]
tm = np.nanmedian(temp)
temp[np.where(np.isnan(temp))] = tm
stats['weight'][:] = temp

# Recalculate BMI using imputed values
stats['bmi'][:] = stats['weight'][:] / (stats['height'][:] * stats['height'][:])

# Recalculate urine out and urine flow with imputed weights
urine_outs, urine_flows = get_uky_urine(ids, stats['weight'][:], baseDataPath)
stats['urine_out'][:] = urine_outs
stats['urine_flow'][:] = urine_flows

# Recalculate fluid overloads using imputed weights
nets, tots, fos, cfbs = get_uky_fluids(ids, stats['weight'][:], baseDataPath)
stats['fluid_overload'][:] = fos

for k in list(stats):
    temp = stats[k][:]
    m = np.nanmean(temp)
    std = np.nanstd(temp)
    temp[np.where(temp < (m - (std * 1.96)))] = np.nan
    temp[np.where(temp > (m + (std * 1.96)))] = np.nan
    stats[k][:] = temp


if 'sofa' not in list(stats):
    sofa = load_csv(os.path.join(dataPath, 'sofa.csv'), ids, int, skip_header=True)
    sofa = np.sum(sofa, axis=1)
    stats['sofa'] = sofa

if 'apache' not in list(stats):
    apache = load_csv(os.path.join(dataPath, 'apache.csv'), ids, int, skip_header=True)
    apache = np.sum(apache, axis=1)
    stats['apache'] = apache
#
# formatted_stats(stats, os.path.join(resPath, 'clusters', 'max_kdigo'))
# #
f.copy(f['%s_cleaned' % grp_name], '/%s_imputed' % grp_name)
nstats = f['%s_imputed' % grp_name]
for k in list(nstats):
    temp = nstats[k][:]
    try:
        tm = np.nanmedian(temp)
        temp[np.where(np.isnan(temp))] = tm
        nstats[k][:] = temp
    except TypeError:
        pass

f.close()