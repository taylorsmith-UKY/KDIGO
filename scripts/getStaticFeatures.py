import os
import numpy as np
import pandas as pd
import json
import h5py
from utility_funcs import arr2csv, load_csv, dict2csv, get_array_dates
from stat_funcs import summarize_stats
from copy import copy
from sklearn.preprocessing import MinMaxScaler

grp_name = 'meta_cleaned'
statFileName = 'stats_111219.h5'

fp = open('../kdigo_conf1.json', 'r')
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

kdigos = load_csv(os.path.join(dataPath, 'kdigo_icu_2ptAvg.csv'), ids, int)
days = load_csv(os.path.join(dataPath, 'days_interp_icu_2ptAvg.csv'), ids, int)

scrs = load_csv(os.path.join(dataPath, 'scr_interp_icu_2ptAvg.csv'), ids, float)

icu_windows = load_csv(os.path.join(dataPath, 'icu_admit_discharge.csv'), ids, 'date', struct='dict')
hosp_windows = load_csv(os.path.join(dataPath, 'hosp_admit_discharge.csv'), ids, 'date', struct='dict')

# stats = summarize_stats(f, ids, kdigos, days, scrs, icu_windows, hosp_windows, baseDataPath, grp_name, 14)
stats = f[grp_name]

header = 'STUDY_PATIENT_ID,AdmitScr,Age,Albumin,Anemia_A,Anemia_B,Anemia_C,baseline_scr,Bicarbonate_Low,Bicarbonate_High,Bilirubin,BMI,BUN,' \
         'FiO2_Low,FiO2_High,FluidOverload,Gender,' \
         'HeartRate_Low,HeartRate_High,Hematocrit_low,Hematocrit_high,Hemoglobin_low,Hemoglobin_High,' \
         'ECMO,IABP,MechanicalVentilation,VAD,MAP_low,MAP_high,AdmitKDIGO,Nephrotox_ct,' \
         'Vasopress_ct,pCO2_low,pCO2_high,Peak_SCr,pH_low,pH_high,Platelets,pO2_low,pO2_high,Potassium_low,' \
         'Potassium_high,Race,Respiration_low,Respiration_high,Septic,Sodium_low,Sodium_high,' \
         'Temperature_low,Temperature_high,Urine_output,WBC_low,WBC_high,Height,Weight,hrsInICU_48hr,' \
         'hrsInICU_D01,MechHemodynamicSupport,unPlannedAdmission,MaxKDIGO_D01,MaxKDIGO_D03,MaxKDIGO_14d'

feats = np.zeros((len(ids), (len(header.split(',')) + len(elixIdxs) - 1)))
admit_scrs = np.zeros(len(ids))
admit_kdigos = np.zeros(len(ids))
peak_scrs = np.zeros(len(ids))
hrsInIcu = np.zeros(len(ids))
ahrsInIcu = np.zeros(len(ids))
mks = np.zeros((len(ids), 3))

hosp_admits = get_array_dates(stats['hosp_dates'][:, 0].astype(str))
icu_admits = get_array_dates(stats['icu_dates'][:, 0].astype(str))
for i in range(len(ids)):
    admit_scrs[i] = scrs[i][0]
    admit_kdigos[i] = kdigos[i][0]
    peak_scrs[i] = np.max(scrs[i][np.where(days[i] <= 1)])
    mks[i, 0] = np.max(kdigos[i][np.where(days[i] <= 1)])
    mks[i, 1] = np.max(kdigos[i][np.where(days[i] <= 3)])
    mks[i, 2] = np.max(kdigos[i][np.where(days[i] <= 14)])

    icuAdmitDay = icu_admits[i].toordinal() - hosp_admits[i].toordinal()
    if icuAdmitDay == 0:
        hrsInIcu[i] = (24 - icu_admits[i].hour) + 24
    elif icuAdmitDay == 1:
        hrsInIcu[i] = (24 - icu_admits[i].hour)

    hrsToIcu = (icu_admits[i] - hosp_admits[i]).total_seconds() / (60 * 60 * 24)
    if hrsToIcu < 48:
        ahrsInIcu[i] = 48 - hrsToIcu

tstats = {'AdmitScr': admit_scrs, 'AdmitKDIGO': admit_kdigos,
          'Peak_SCr': peak_scrs, 'Dopamine': stats['dopa'][:],
          'FluidOverload': stats['fluid_overload'][:], 'GCS': stats['glasgow'][:, 0],
          'HeartRate_Low': stats['heart_rate'][:, 0], 'HeartRate_High': stats['heart_rate'][:, 1],
          'MechanicalVentilation': stats['mv_flag'][:], 'Urine_output': stats['urine_out'][:],
          'hrsInICU_D01': hrsInIcu, 'hrsInICU_48hr': ahrsInIcu, 'VAD': stats['vad'][:],
          'ECMO': stats['ecmo'], 'IABP': stats['iabp'], 'MechHemodynamicSupport': stats['mhs'],
          'unPlannedAdmission': stats['unPlannedAdmissions'],
          "MaxKDIGO_D01": mks[:, 0], "MaxKDIGO_D03": mks[:, 1], "MaxKDIGO_14d": mks[:, 2]}

col = 0
ct = 0
prev = ''
for k in header.split(',')[1:]:
    if k in list(tstats):
        feats[:, col] = tstats[k]
        col += 1
    elif k.split('_')[0].lower() in list(stats):
        if k.split('_')[0] == prev.split('_')[0]:
            ct += 1
        else:
            ct = 0
        if stats[k.split('_')[0].lower()].ndim > 1:
            feats[:, col] = stats[k.split('_')[0].lower()][:, ct]
        else:
            feats[:, col] = stats[k.split('_')[0].lower()][:]
        col += 1
    elif k.lower() in list(stats):
        feats[:, col] = stats[k.lower()][:]
        col += 1
    else:
        print("Couldn't find feature %s" % k)
    prev = copy(k)

#
# header += ',' + ','.join(['SOFA_%d' % x for x in range(6)])
# header += ',' + ','.join(['APACHE_%d' % x for x in range(13)])
# header += ',' + ','.join(['Charlson_%d' % x for x in range(14)])
# header += ',' + ','.join(['Elixhauser_%d' % x for x in range(31)])
#
# sofa = load_csv(os.path.join(dataPath, 'sofa.csv'), ids, int, skip_header=True)
# feats[:, col:col+6] = sofa
# col += 6
#
# apache = load_csv(os.path.join(dataPath, 'apache.csv'), ids, int, skip_header=True)
# feats[:, col:col + 13] = apache
# col += 13
#
# feats[:, col:col+14] = stats['charlson_components'][:]
# col += 14

elix = stats['elixhauser_components'][:]
for idx in elixIdxs:
    feats[:, col] = elix[:, idx]
    col += 1
    header += "," + "Elixhauser_%d" % idx

contStr = 'FeatureName,Mean,StdDev,Median,IQ-1,IQ-3,Min,Max,Missing_#(%)\n'
catStr = 'FeatureName,n (%),Missing_#(%)\n'
# print('FeatureName,ColNum,Min,Max,MedianmMean,StDev,Missing_#(%)')
for i in range(col):
    # if header.split(',')[i + 1] in ['Anemia_A', 'Anemia_B', 'Anemia_C', 'Diabetic', 'Dopamine', 'Epinephrine', 'Male',
    #                                 'Hypertensive', 'ECMO', 'IABP', 'MechanicalVentilation', 'VAD', 'AdmitKDIGO',
    #                                 'Nephrotox_Ct', 'Vasopress_Ct', 'Black', 'Septic', 'Smoker']:
    #     pass
    if len(np.unique(feats[:, i])) <= 2:
        catStr += '%s,%d (%.2f),%d (%.2f)\n' % (header.split(',')[i + 1], len(np.where(feats[:, i])[0]),
                                                len(np.where(feats[:, i])[0]) / len(ids) * 100,
                                                len(np.where(np.isnan(feats[:, i]))[0]),
                                                len(np.where(np.isnan(feats[:, i]))[0]) / len(ids) * 100)
    elif len(np.unique(feats[:, i])) < 10:
        for val in np.unique(feats[:, i]):
            if np.isnan(val):
                continue
            # print('%s-%d,%d (%.2f),%d (%.2f)' % (header.split(',')[i + 1], val, len(np.where(feats[:, i] == val)[0]),
            #                               len(np.where(feats[:, i] == val)[0]) / len(ids) * 100, len(np.where(np.isnan(feats[:, i]))[0]),
            #                                  len(np.where(np.isnan(feats[:, i]))[0]) / len(ids) * 100))
            catStr += '%s-%d,%d (%.2f),%d (%.2f)\n' % (header.split(',')[i + 1], val, len(np.where(feats[:, i] == val)[0]),
                                                      len(np.where(feats[:, i] == val)[0]) / len(ids) * 100,
                                                      len(np.where(np.isnan(feats[:, i]))[0]),
                                                      len(np.where(np.isnan(feats[:, i]))[0]) / len(ids) * 100)
    else:
        # print('%s,%.4g,%.4g,%.4g,%.4g,%.4g,%.4g,%.4g,%d (%.2f)' % (
        #       header.split(',')[i + 1], np.nanmean(feats[:, i]), np.nanstd(feats[:, i]), np.nanmedian(feats[:, i]), np.nanpercentile(feats[:, i], 25), np.nanpercentile(feats[:, i], 75),
        #       np.nanmin(feats[:, i]), np.nanmax(feats[:, i]), len(np.where(np.isnan(feats[:, i]))[0]),
        #       len(np.where(np.isnan(feats[:, i]))[0]) / len(ids) * 100))
        contStr += '%s,%.4g,%.4g,%.4g,%.4g,%.4g,%.4g,%.4g,%d (%.2f)\n' % \
                   (header.split(',')[i + 1], np.nanmean(feats[:, i]), np.nanstd(feats[:, i]),
                    np.nanmedian(feats[:, i]), np.nanpercentile(feats[:, i], 25), np.nanpercentile(feats[:, i], 75),
                    np.nanmin(feats[:, i]), np.nanmax(feats[:, i]), len(np.where(np.isnan(feats[:, i]))[0]),
                    len(np.where(np.isnan(feats[:, i]))[0]) / len(ids) * 100)

print(contStr)
print(catStr)

# arr2csv(os.path.join(resPath, 'features', 'individual', 'static_features.csv'), feats, ids, fmt='%.5g', header=header)
#
# print('FeatureName,ColNum,Min,Max,MedianmMean,StDev,Missing_#(%)')
# for i in range(feats.shape[1]):
#     if header.split(',')[i + 1] in ['Anemia_A', 'Anemia_B', 'Anemia_C', 'Diabetic', 'Dopamine', 'Epinephrine', 'Male',
#                                     'Hypertensive', 'ECMO', 'IABP', 'MechanicalVentilation', 'VAD', 'AdmitKDIGO',
#                                     'Nephrotox_Ct', 'Vasopress_Ct', 'Black', 'Septic', 'Smoker']:
#         if len(np.unique(feats[:, i])) > 2:
#             for val in np.unique(feats[:, i]):
#                 if np.isnan(val):
#                     continue
#                 print('%s-%d,%d,%d (%.2f)' % (header.split(',')[i + 1], i, val, len(np.where(feats[:, i] == val)[0]),
#                                               len(np.where(feats[:, i] == val)[0]) / len(ids)) * 100)
#         else:
#             print('%s,%d (%.2f)' % (header.split(',')[i + 1], len(np.where(feats[:, i] == 1)[0]),
#                                     len(np.where(feats[:, i] == 1)[0]) / len(ids)) * 100)
#     else:
#         print('%s,%d,%.4g,%.4g,%.4g,%.4g,%d (%.2f)' % (
#               header.split(',')[i + 1], i, np.nanmin(feats[:, i]), np.nanmean(feats[:, i]), np.nanmedian(feats[:, i]),
#               np.nanmax(feats[:, i]), len(np.where(np.isnan(feats[:, i]))[0]),
#               len(np.where(np.isnan(feats[:, i]))[0]) / len(ids) * 100))
#
# mms = MinMaxScaler()
# nfeats = mms.fit_transform(feats)
#
# arr2csv(os.path.join(resPath, 'features', 'individual', 'static_features_norm.csv'), nfeats, ids, fmt='%.5g', header=header)
#
# mfeats = np.array(feats)
# for i in range(feats.shape[1]):
#     mfeats[:, i][np.where(np.isnan(feats[:, i]))] = np.nanmean(feats[:, i])
#
# arr2csv(os.path.join(resPath, 'features', 'individual', 'static_features_meanfill.csv'), mfeats, ids, fmt='%.5g', header=header)
#
# mms = MinMaxScaler()
# mnfeats = mms.fit_transform(mfeats)
#
# arr2csv(os.path.join(resPath, 'features', 'individual', 'static_features_norm_meanfill.csv'), mnfeats, ids, fmt='%.5g', header=header)
#
# cleanFeats = []
# nhdr = []
# for i in range(feats.shape[1]):
#     if len(np.where(np.isnan(feats[:, i]))[0]) < (0.3 * len(feats)):
#         cleanFeats.append(feats[:, i])
#         nhdr.append(header.split(',')[i+1])
#
# cleanFeats = np.vstack(cleanFeats).T
# nhdr = 'STUDY_PATIENT_ID,' + ','.join(nhdr)
# arr2csv(os.path.join(resPath, 'features', 'individual', 'static_features_30pctMissing.csv'), cleanFeats, ids, fmt='%.5g', header=nhdr)
#
# mCleanFeats = np.array(cleanFeats)
# for i in range(mCleanFeats.shape[1]):
#     mCleanFeats[:, i][np.where(np.isnan(mCleanFeats[:, i]))] = np.nanmean(mCleanFeats[:, i])
# mCleanFeats = mms.fit_transform(cleanFeats)
# arr2csv(os.path.join(resPath, 'features', 'individual', 'static_features_30pctMissing_mean.csv'), mCleanFeats, ids,
#         fmt='%.5g', header=nhdr)
