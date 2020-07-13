import os
import numpy as np
import json
import h5py
from utility_funcs import arr2csv, load_csv, get_array_dates
from stat_funcs import get_uky_rrt_flags
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
parser.add_argument('--clin_features_mort', '-cfm', action='store', type=str, dest='ffname_mort',
                    default='clinical_model_features_mortality.csv')
parser.add_argument('--clin_features_make', '-cfma', action='store', type=str, dest='ffname_make',
                    default='clinical_model_features_make.csv')
args = parser.parse_args()

header = 'STUDY_PATIENT_ID,Admit_Scr,Age,Albumin,Anemia,baseline_scr,Bicarbonate_Low,Bicarbonate_High,Bilirubin,BMI,BUN,' \
         'FiO2_Low,FiO2_High,FluidOverload,Gender,' \
         'HeartRate_Low,HeartRate_High,Hematocrit_low,Hematocrit_high,Hemoglobin_low,Hemoglobin_High,' \
         'ECMO,IABP,MechanicalVentilation,VAD,MAP_low,MAP_high,AdmitKDIGO,Nephrotox_exp,' \
         'Vasopress_exp,pCO2_low,pCO2_high,Peak_SCr,pH_low,pH_high,Platelets,pO2_low,pO2_high,Potassium_low,' \
         'Potassium_high,Race,Respiration_low,Respiration_high,Septic,Sodium_low,Sodium_high,' \
         'Temperature_low,Temperature_high,Urine_output,Urine_flow,WBC_low,WBC_high,Height,Weight,hrsInICU_72hr,' \
         'MechHemodynamicSupport,unPlannedAdmission,MaxKDIGO_D03,RRT_Flag_D03,LastKDIGO_D03'

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
stats = f[grp_name]
dieds = stats["died_inp"][:][:, None]
makes = stats["make90_disch_30dbuf_2pts_d50"][:][:, None]

kdigos = load_csv(os.path.join(dataPath, 'kdigo_icu_2ptAvg.csv'), ids, int)
days = load_csv(os.path.join(dataPath, 'days_interp_icu_2ptAvg.csv'), ids, int)
scrs = load_csv(os.path.join(dataPath, 'scr_interp_icu_2ptAvg.csv'), ids, float)
icu_windows = load_csv(os.path.join(dataPath, 'icu_admit_discharge.csv'), ids, 'date', struct='dict')
hosp_windows = load_csv(os.path.join(dataPath, 'hosp_admit_discharge.csv'), ids, 'date', struct='dict')

rrt_flags, hd_flags, crrt_flags, hd_days, crrt_days, rrt_days = get_uky_rrt_flags(ids, icu_windows, baseDataPath, 3)


feats = np.zeros((len(ids), (len(header.split(',')) + len(elixIdxs) - 1)))
admit_scrs = np.zeros(len(ids))
admit_kdigos = np.zeros(len(ids))
peak_scrs = np.zeros(len(ids))
last_kdigos = np.zeros(len(ids))
mkd03 = np.zeros(len(ids))

hosp_admits = get_array_dates(stats['hosp_dates'][:, 0].astype(str))
icu_admits = get_array_dates(stats['icu_dates'][:, 0].astype(str))
for i in range(len(ids)):
    mkd03[i] = np.max(kdigos[i][np.where(days[i] <= 3)[0]])
    admit_scrs[i] = scrs[i][0]
    admit_kdigos[i] = kdigos[i][0]
    peak_scrs[i] = np.max(scrs[i][np.where(days[i] <= 3)])
    last_kdigos[i] = kdigos[i][np.where(days[i] <= 3)[0][-1]]

baseline_gfr = stats["baseline_gfr"][:]
ckd = np.array(baseline_gfr < 60, dtype=int)

tstats = {'AdmitScr': admit_scrs, 'AdmitKDIGO': admit_kdigos,
          'Peak_SCr': peak_scrs, 'Dopamine': stats['dopa_d03'][:],
          'FluidOverload': stats['fluid_overload'][:], 'GCS': stats['glasgow'][:, 0],
          'HeartRate_Low': stats['heart_rate'][:, 0], 'HeartRate_High': stats['heart_rate'][:, 1],
          'MechanicalVentilation': stats['mv_flag_d03'][:], 'Urine_output': stats['urine_out'][:],
          'VAD': stats['vad_d03'][:], "Anemia": stats['anemia'][:, 0],
          'ECMO': stats['ecmo_d03'], 'IABP': stats['iabp_d03'], 'MechHemodynamicSupport': stats['mhs_d03'],
          'unPlannedAdmission': stats['unPlannedAdmissions'],
          "BUN": np.nanmax(stats["bun"][:], axis=1), "RRT_Flag_D03": rrt_flags,
          "MaxKDIGO_D03": mkd03, "LastKDIGO_D03": last_kdigos,
          "Nephrotox_exp": stats["nephrotox_exp_d03"], "Vasopress_exp": stats["vasopress_exp_d03"]}

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

# Save feature table before normalization
if not os.path.exists(os.path.join(indFeatPath, 'base_model_noNorm.csv')) or args.ovwt:
    arr2csv(os.path.join(indFeatPath, 'base_model_noNorm.csv'), feats, ids, fmt='%.5g', header=header)
else:
    fname = "base_model_noNorm"
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


# Load trajectory features and add
desc, descHdr = load_csv(os.path.join(indFeatPath, "descriptive_features.csv"), ids, skip_header="keep")

dnorm = mms.fit_transform(desc)
feats_noNorm = np.hstack([feats, desc])
feats = np.hstack([fnorm, dnorm])
header += "," + ",".join(descHdr)

if not os.path.exists(os.path.join(indFeatPath, 'base_model_withTrajectory.csv')) or args.ovwt:
    arr2csv(os.path.join(indFeatPath, 'base_model_withTrajectory.csv'), feats, ids, fmt='%.5g', header=header)
    arr2csv(os.path.join(indFeatPath, 'base_model_withTrajectory_noNorm.csv'), feats_noNorm, ids, fmt='%.5g', header=header)
else:
    fname = "base_model_withTrajectory"
    print("File named: '%s' already saved. To overwrite with new results,\nspecify the"
          "command line argument '-ovwt' when running." % fname)


if os.path.exists(args.ffname_mort):

    with open(args.ffname_mort, "r") as tf:
        clinModelFeats = []
        for line in tf:
            clinModelFeats.append(line.rstrip())

    clinFeats = np.zeros((len(ids), len(clinModelFeats)))
    clinFeats_noNorm = np.zeros((len(ids), len(clinModelFeats)))
    ct = 0
    chdr = "STUDY_PATIENT_ID"
    for i, k in enumerate(header.split(",")[1:]):
        if k in clinModelFeats:
            clinFeats[:, ct] = feats[:, i]
            clinFeats_noNorm[:, ct] = feats_noNorm[:, i]
            ct += 1
            chdr += "," + k

    assert ct == len(clinModelFeats)
    arr2csv(os.path.join(indFeatPath, 'clinical_model_mortality.csv'), clinFeats, ids, fmt='%.5g', header=chdr)
    arr2csv(os.path.join(indFeatPath, 'clinical_model_mortality_noNorm.csv'), clinFeats_noNorm, ids, fmt='%.5g', header=chdr)

    dhdr = chdr + "," + ",".join(descHdr)
    arr2csv(os.path.join(indFeatPath, 'clinical_model_mortality_wTrajectory.csv'), np.hstack([clinFeats, dnorm]), ids, fmt='%.5g',
            header=dhdr)

    chdr += ",Died"
    arr2csv(os.path.join(indFeatPath, 'clinical_model_mortality_noNorm_wOutcome.csv'), np.hstack([clinFeats_noNorm, dieds]), ids, fmt='%.5g',
            header=chdr)

if os.path.exists(args.ffname_make):

    with open(args.ffname_make, "r") as tf:
        clinModelFeats = []
        for line in tf:
            clinModelFeats.append(line.rstrip())

    clinFeats = np.zeros((len(ids), len(clinModelFeats)))
    clinFeats_noNorm = np.zeros((len(ids), len(clinModelFeats)))
    ct = 0
    chdr = "STUDY_PATIENT_ID"
    for i, k in enumerate(header.split(",")[1:]):
        if k in clinModelFeats:
            clinFeats[:, ct] = feats[:, i]
            clinFeats_noNorm[:, ct] = feats_noNorm[:, i]
            ct += 1
            chdr += "," + k

    assert ct == len(clinModelFeats)
    arr2csv(os.path.join(indFeatPath, 'clinical_model_make.csv'), clinFeats, ids, fmt='%.5g', header=chdr)
    arr2csv(os.path.join(indFeatPath, 'clinical_model_make_noNorm.csv'), clinFeats_noNorm, ids, fmt='%.5g', header=chdr)

    dhdr = chdr + "," + ",".join(descHdr)
    arr2csv(os.path.join(indFeatPath, 'clinical_model_make_wTrajectory.csv'), np.hstack([clinFeats, dnorm]), ids,
            fmt='%.5g',
            header=dhdr)

    chdr += ",Died,MAKE"
    arr2csv(os.path.join(indFeatPath, 'clinical_model_make_noNorm_wOutcome.csv'), np.hstack([clinFeats_noNorm, dieds, makes]), ids, fmt='%.5g',
            header=chdr)
