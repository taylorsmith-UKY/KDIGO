import os
import numpy as np
import json
import h5py
from utility_funcs import arr2csv, load_csv, get_array_dates
from copy import copy
from sklearn.preprocessing import MinMaxScaler
import argparse

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
parser.add_argument('--print_statistics', '-ps', action='store_true', dest='ps')
parser.add_argument('--overwrite_results', '-ovwt', action='store_true', dest='ovwt')
args = parser.parse_args()

#
# class args:
#     avgpts = 2
#     df = "stats.h5"
#     meta = "meta_imputed"
#     cfpath = ""
#     cfname = "test_conf3.json"
#     ps = True
#     ovwt = False

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

kdigos = load_csv(os.path.join(dataPath, 'kdigo_icu_2ptAvg.csv'), ids, int)
days = load_csv(os.path.join(dataPath, 'days_interp_icu_2ptAvg.csv'), ids, int)
scrs = load_csv(os.path.join(dataPath, 'scr_interp_icu_2ptAvg.csv'), ids, float)
icu_windows = load_csv(os.path.join(dataPath, 'icu_admit_discharge.csv'), ids, 'date', struct='dict')
hosp_windows = load_csv(os.path.join(dataPath, 'hosp_admit_discharge.csv'), ids, 'date', struct='dict')

stats = f[grp_name]

header = 'STUDY_PATIENT_ID,AdmitScr,Age,Albumin,Anemia,baseline_scr,Bicarbonate_Low,Bicarbonate_High,Bilirubin,BMI,BUN,' \
         'FiO2_Low,FiO2_High,FluidOverload,Gender,' \
         'HeartRate_Low,HeartRate_High,Hematocrit_low,Hematocrit_high,Hemoglobin_low,Hemoglobin_High,' \
         'ECMO,IABP,MechanicalVentilation,VAD,MAP_low,MAP_high,AdmitKDIGO,Nephrotox_exp,' \
         'Vasopress_exp,pCO2_low,pCO2_high,Peak_SCr,pH_low,pH_high,Platelets,pO2_low,pO2_high,Potassium_low,' \
         'Potassium_high,Race,Respiration_low,Respiration_high,Septic,Sodium_low,Sodium_high,' \
         'Temperature_low,Temperature_high,Urine_output,WBC_low,WBC_high,Height,Weight,hrsInICU_48hr,' \
         'MechHemodynamicSupport,unPlannedAdmission,MaxKDIGO'

feats = np.zeros((len(ids), (len(header.split(',')) + len(elixIdxs) - 1)))
admit_scrs = np.zeros(len(ids))
admit_kdigos = np.zeros(len(ids))
peak_scrs = np.zeros(len(ids))

hosp_admits = get_array_dates(stats['hosp_dates'][:, 0].astype(str))
icu_admits = get_array_dates(stats['icu_dates'][:, 0].astype(str))
for i in range(len(ids)):
    admit_scrs[i] = scrs[i][0]
    admit_kdigos[i] = kdigos[i][0]
    peak_scrs[i] = np.max(scrs[i][np.where(days[i] <= 3)])


tstats = {'AdmitScr': admit_scrs, 'AdmitKDIGO': admit_kdigos,
          'Peak_SCr': peak_scrs, 'Dopamine': stats['dopa'][:],
          'FluidOverload': stats['fluid_overload'][:], 'GCS': stats['glasgow'][:, 0],
          'HeartRate_Low': stats['heart_rate'][:, 0], 'HeartRate_High': stats['heart_rate'][:, 1],
          'MechanicalVentilation': stats['mv_flag'][:], 'Urine_output': stats['urine_out'][:],
          'VAD': stats['vad'][:], "Anemia": stats['anemia'][:, 0],
          'ECMO': stats['ecmo'], 'IABP': stats['iabp'], 'MechHemodynamicSupport': stats['mhs'],
          'unPlannedAdmission': stats['unPlannedAdmissions'],
          "MaxKDIGO": stats["max_kdigo_d03"][:], "MaxKDIGO_%dd" % t_lim: stats['max_kdigo_win'][:]}

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
    elif k in list(stats):
        feats[:, col] = stats[k][:]
        col += 1
    else:
        print("Couldn't find feature %s" % k)
    prev = copy(k)

elix = stats['elixhauser_components'][:]
for idx in elixIdxs:
    feats[:, col] = elix[:, idx]
    col += 1
    header += "," + "Elixhauser_%d" % idx

indFeatPath = os.path.join(resPath, 'features', 'individual')
if args.ps or args.ovwt:
    contStr = 'FeatureName,Mean,StdDev,Median,IQ-1,IQ-3,Min,Max,Missing_#(%)\n'
    catStr = 'FeatureName,n (%),Missing_#(%)\n'
    # print('FeatureName,ColNum,Min,Max,MedianmMean,StDev,Missing_#(%)')
    for i in range(col):
        if len(np.unique(feats[:, i])) <= 2:
            catStr += '%s,%d (%.2f),%d (%.2f)\n' % (header.split(',')[i + 1], len(np.where(feats[:, i])[0]),
                                                    len(np.where(feats[:, i])[0]) / len(ids) * 100,
                                                    len(np.where(np.isnan(feats[:, i]))[0]),
                                                    len(np.where(np.isnan(feats[:, i]))[0]) / len(ids) * 100)
        elif len(np.unique(feats[:, i])) < 10:
            for val in np.unique(feats[:, i]):
                if np.isnan(val):
                    continue
                catStr += '%s-%d,%d (%.2f),%d (%.2f)\n' % (header.split(',')[i + 1], val, len(np.where(feats[:, i] == val)[0]),
                                                          len(np.where(feats[:, i] == val)[0]) / len(ids) * 100,
                                                          len(np.where(np.isnan(feats[:, i]))[0]),
                                                          len(np.where(np.isnan(feats[:, i]))[0]) / len(ids) * 100)
        else:
            contStr += '%s,%.4g,%.4g,%.4g,%.4g,%.4g,%.4g,%.4g,%d (%.2f)\n' % \
                       (header.split(',')[i + 1], np.nanmean(feats[:, i]), np.nanstd(feats[:, i]),
                        np.nanmedian(feats[:, i]), np.nanpercentile(feats[:, i], 25), np.nanpercentile(feats[:, i], 75),
                        np.nanmin(feats[:, i]), np.nanmax(feats[:, i]), len(np.where(np.isnan(feats[:, i]))[0]),
                        len(np.where(np.isnan(feats[:, i]))[0]) / len(ids) * 100)

    # print(contStr)
    # print(catStr)
    if args.ovwt:
        tf = open(os.path.join(indFeatPath, "descriptiveStats_baseModel_continuousFeatures.csv"), "w")
        tf.write(contStr + "\n")
        tf.close()

        tf = open(os.path.join(indFeatPath, "descriptiveStats_baseModel_categoricalFeatures.csv"), "w")
        tf.write(catStr + "\n")
        tf.close()
if args.ps:
    print(contStr)
    print(catStr)


# Save feature table before normalization
if not os.path.exists(os.path.join(indFeatPath, 'base_model_NoNorm.csv')) or args.ovwt:
    arr2csv(os.path.join(indFeatPath, 'base_model_NoNorm.csv'), feats, ids, fmt='%.5g', header=header)
else:
    fname = "base_model_NoNorm"
    print("File named: '%s' already saved. To overwrite with new results,\nspecify the"
          "command line argument '-ovwt' when running." % fname)

# Save normalized table
mms = MinMaxScaler()
fnorm = mms.fit_transform(feats)
if not os.path.exists(os.path.join(indFeatPath, 'base_model.csv')) or args.ovwt:
    arr2csv(os.path.join(indFeatPath, 'base_model.csv'), fnorm, ids, fmt='%.5g', header=header)
else:
    fname = "base_model"
    print("File named: '%s' already saved. To overwrite with new results,\nspecify the"
          "command line argument '-ovwt' when running." % fname)

# Add maximum KDIGO up to 14 days
mkfeats = np.hstack([feats, stats['max_kdigo_win'][:][:, None]])
mkheader = header + "," + "MaxKDIGO_%dd" % t_lim
if not os.path.exists(os.path.join(indFeatPath, 'base_model_14dKDIGO_originalValues.csv')) or args.ovwt:
    arr2csv(os.path.join(indFeatPath, 'base_model_14dKDIGO_originalValues.csv'), mkfeats, ids, fmt='%.5g', header=mkheader)
else:
    fname = "base_model_14dKDIGO_originalValues"
    print("File named: '%s' already saved. To overwrite with new results,\nspecify the"
          "command line argument '-ovwt' when running." % fname)

# Save normalized of previous version
mkfnorm = mms.fit_transform(mkfeats)
if not os.path.exists(os.path.join(indFeatPath, 'base_model_14dKDIGO.csv')) or args.ovwt:
    arr2csv(os.path.join(indFeatPath, 'base_model_14dKDIGO.csv'), mkfnorm, ids, fmt='%.5g', header=mkheader)
else:
    fname = "base_model_14dKDIGO"
    print("File named: '%s' already saved. To overwrite with new results,\nspecify the"
          "command line argument '-ovwt' when running." % fname)

# Load trajectory features and add
desc, descHdr = load_csv(os.path.join(indFeatPath, "descriptive_features.csv"), ids, skip_header="keep")

dnorm = mms.fit_transform(desc)
feats = np.hstack([fnorm, dnorm])
header += "," + ",".join(descHdr)

if not os.path.exists(os.path.join(indFeatPath, 'base_model_withTrajectory.csv')) or args.ovwt:
    arr2csv(os.path.join(indFeatPath, 'base_model_withTrajectory.csv'), feats, ids, fmt='%.5g', header=header)
else:
    fname = "base_model_withTrajectory"
    print("File named: '%s' already saved. To overwrite with new results,\nspecify the"
          "command line argument '-ovwt' when running." % fname)
