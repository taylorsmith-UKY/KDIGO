import os
import numpy as np
import json
import pandas as pd
import h5py
from utility_funcs import arr2csv, load_csv, get_array_dates
from copy import copy
from sklearn.preprocessing import MinMaxScaler
from classification_funcs import descriptive_trajectory_features
import argparse

parser = argparse.ArgumentParser(description='Preprocess Data and Construct KDIGO Vectors.')
parser.add_argument('--config_file', action='store', type=str, dest='cfname',
                    default='utsw_conf1.json')
parser.add_argument('--ref_config_file', action='store', type=str, dest='rcfname',
                    default='manuscript.json')
parser.add_argument('--config_path', action='store', type=str, dest='cfpath',
                    default='')
parser.add_argument('--averagePoints', '-agvPts', action='store', type=int, dest='avgpts',
                    default=2)
parser.add_argument('--data_file', '-df', action='store', type=str, dest='df',
                    default='utsw_allFeats.csv')
parser.add_argument('--print_statistics', '-ps', action='store_true', dest='ps')
parser.add_argument('--overwrite_results', '-ovwt', action='store_true', dest='ovwt')
parser.add_argument('--clin_features_mort', '-cfm', action='store', type=str, dest='ffname_mort',
                    default='clinical_model_features_mortality.csv')
parser.add_argument('--clin_features_make', '-cfma', action='store', type=str, dest='ffname_make',
                    default='clinical_model_features_make.csv')
args = parser.parse_args()

# Load UK parameters
configurationFileName = os.path.join(args.cfpath, args.rcfname)
fp = open(configurationFileName, 'r')
conf = json.load(fp)
fp.close()

basePath = conf['basePath']         # Path containing the DATA and RESULTS directories
cohortName = conf['cohortName']     # Will be used for folder name
t_lim = conf['analysisDays']        # How long to consider in the analysis
tRes = conf['timeResolutionHrs']    # Resolution after imputation in hours
analyze = conf['analyze']           # Time period to analyze (hospital/ICU/all)

obaseDataPath = os.path.join(basePath, 'DATA', 'all_sheets')         # folder containing all raw data
odataPath = os.path.join(basePath, 'DATA', analyze, cohortName)
oresPath = os.path.join(basePath, 'RESULTS', analyze, cohortName)

# Load UTSW Data
configurationFileName = os.path.join(args.cfpath, args.cfname)
fp = open(configurationFileName, 'r')
conf = json.load(fp)
fp.close()
ncohortName = conf['cohortName']     # Will be used for folder name

# Build paths and create if don't already exist
nbaseDataPath = os.path.join(basePath, 'DATA', 'dallas', 'csv')         # folder containing all raw data
ndataPath = os.path.join(basePath, 'DATA', 'dallas', analyze, ncohortName)
nresPath = os.path.join(basePath, 'RESULTS', 'dallas', analyze, ncohortName)

f = h5py.File(os.path.join(nresPath, "stats.h5"), 'r')
ids = f["meta"]['ids'][:]
stats = f["meta"]
dieds = stats["died_inp"][:][:, None]
makes = stats["make90_disch_30dbuf_2pts_d50"][:][:, None]

# Load raw data for UTSW
df = pd.read_csv(os.path.join(nresPath, args.df))
nids = df["STUDY_PATIENT_ID"].values

kdigos = load_csv(os.path.join(ndataPath, 'kdigo_icu_2ptAvg.csv'), ids, int)
sel = np.array([x in ids for x in nids])
nids = np.intersect1d(nids, ids)

assert len(nids) == len(ids)

kdigos = load_csv(os.path.join(ndataPath, 'kdigo_icu_2ptAvg.csv'), nids, int)
days = load_csv(os.path.join(ndataPath, 'days_interp_icu_2ptAvg.csv'), nids, int)

scrs = load_csv(os.path.join(ndataPath, 'scr_interp_icu_2ptAvg.csv'), nids, float)

icu_windows = load_csv(os.path.join(ndataPath, 'icu_admit_discharge.csv'), nids, 'date', struct='dict')
hosp_windows = load_csv(os.path.join(ndataPath, 'hosp_admit_discharge.csv'), nids, 'date', struct='dict')

for i in range(len(kdigos)):
    idx = np.where(days[i] <= 14)[0]
    kdigos[i] = kdigos[i][idx]
    days[i] = days[i][idx]
    scrs[i] = scrs[i][idx]

nstats = {}

mkd03 = np.zeros(len(nids))
admit_kdigo = np.zeros(len(nids))
last_kdigos = np.zeros(len(ids))
for i in range(len(nids)):
    mkd03[i] = np.max(kdigos[i][np.where(days[i] <= 3)[0]])
    admit_kdigo[i] = kdigos[i][0]
    last_kdigos[i] = kdigos[i][np.where(days[i] <= 3)[0][-1]]

uop_tot = df["UOP_wmissing"].values[sel]
hrs = df["ICU_stay_hours"].values[sel]
weights = df["Weight"].values[sel]

uop = uop_tot / hrs
uflow = uop / weights

rrt_flag = np.zeros(len(nids))
hd_flags = df['pt_received_HD3D_F'].values[sel]
crrt_flags = df['pt_received_CRRT3D_F'].values[sel]

rrt_flag[np.where(hd_flags)] = 1
rrt_flag[np.where(crrt_flags)] = 2

nstats["Admit_Scr"] = df['AdmitScr'].values[sel]
nstats["Anemia"] = df['Anemia_A'].values[sel]
nstats["baseline_scr"] = df['baseline_sCr'].values[sel]
nstats["BUN"] = df['BUN_high'].values[sel]
nstats["AdmitKDIGO"] = admit_kdigo
nstats["Nephrotox_exp"] = df['Nephrotox_ct'].values[sel]
nstats["Vasopress_exp"] = df['Vasopress_ct'].values[sel]
nstats["Race"] = df['RACE_BLACK'].values[sel]
nstats["Urine_flow"] = uflow
nstats["Urine_output"] = uop
nstats["MechHemodynamicSupport"] = df['MechanicalHemodynamicSupport'].values[sel]
nstats["MaxKDIGO_D03"] = mkd03
nstats["RRT_Flag_D03"] = rrt_flag
nstats["LastKDIGO_D03"] = last_kdigos

nFeatPath = os.path.join(nresPath, "features")
if not os.path.exists(nFeatPath):
    os.mkdir(nFeatPath)
nFeatPath = os.path.join(nFeatPath, "individual")
if not os.path.exists(nFeatPath):
    os.mkdir(nFeatPath)

# Build base model feature table
ofname = os.path.join(oresPath, "features", "individual", "base_model_noNorm.csv")
featNames, ofeats, oids = load_csv(ofname, None, skip_header="keep")

numFeats = len(featNames)
feats = np.zeros((len(nids), numFeats))
for i, k in enumerate(featNames):
    wf = k + "_F"
    if k in list(nstats):
        feats[:, i] = nstats[k]
    elif k in list(df):
        feats[:, i] = df[k].values[sel]
    elif k.lower() in list(df):
        feats[:, i] = df[k.lower()].values[sel]
    elif wf in list(df):
        feats[:, i] = df[wf].values[sel]
    else:
        print("Feature %s not found" % k)
        break

hdr = "PATIENT_NUM," + ",".join(featNames)
arr2csv(os.path.join(nFeatPath, "base_model_noNorm.csv"), feats, nids, header=hdr)

mms = MinMaxScaler().fit(ofeats)
feats_norm = mms.transform(feats)
arr2csv(os.path.join(nFeatPath, "base_model.csv"), feats_norm, nids, header=hdr)

dFileName = os.path.join(nFeatPath, "descriptive_trajectory_features.csv")
dfeats, dhdr = descriptive_trajectory_features(kdigos, nids, days, t_lim=14, filename=dFileName)
dhdr = dhdr.split(",")[1:]

dfeats_norm = MinMaxScaler().fit_transform(dfeats)
hdr += "," + ",".join(dhdr)

allFeatsNorm = np.hstack([feats_norm, dfeats_norm])
arr2csv(os.path.join(nFeatPath, "base_model_withTrajectory.csv"), allFeatsNorm, nids, header=hdr)

allFeatsNoNorm = np.hstack([feats, dfeats])
arr2csv(os.path.join(nFeatPath, "base_model_withTrajectory_noNorm.csv"), allFeatsNoNorm, nids, header=hdr)


feats = allFeatsNorm
if os.path.exists(args.ffname_mort):
    with open(args.ffname_mort, "r") as tf:
        clinModelFeats = []
        for line in tf:
            clinModelFeats.append(line.rstrip())

    clinFeats = np.zeros((len(nids), len(clinModelFeats)))
    clinFeats_noNorm = np.zeros((len(nids), len(clinModelFeats)))
    ct = 0
    chdr = "STUDY_PATIENT_ID"
    for i, k in enumerate(hdr.split(",")[1:]):
        if k in clinModelFeats:
            clinFeats[:, ct] = feats[:, i]
            clinFeats_noNorm[:, ct] = allFeatsNoNorm[:, i]
            ct += 1
            chdr += "," + k

    assert ct == len(clinModelFeats)
    arr2csv(os.path.join(nFeatPath, 'clinical_model_mortality.csv'), clinFeats, nids, fmt='%.5g', header=chdr)
    arr2csv(os.path.join(nFeatPath, 'clinical_model_mortality_noNorm.csv'), clinFeats_noNorm, nids, fmt='%.5g', header=chdr)

    thdr = chdr + "," + ",".join(dhdr)
    arr2csv(os.path.join(nFeatPath, 'clinical_model_mortality_wTrajectory.csv'), np.hstack([clinFeats, dfeats_norm]), nids,
            fmt='%.5g',
            header=thdr)

    chdr += ",Died"
    arr2csv(os.path.join(nFeatPath, 'clinical_model_mortality_noNorm_wOutcome.csv'), np.hstack([clinFeats_noNorm, dieds]), nids, fmt='%.5g',
            header=chdr)

if os.path.exists(args.ffname_make):
    with open(args.ffname_make, "r") as tf:
        clinModelFeats = []
        for line in tf:
            clinModelFeats.append(line.rstrip())

    clinFeats = np.zeros((len(nids), len(clinModelFeats)))
    clinFeats_noNorm = np.zeros((len(nids), len(clinModelFeats)))
    ct = 0
    chdr = "STUDY_PATIENT_ID"
    for i, k in enumerate(hdr.split(",")[1:]):
        if k in clinModelFeats:
            clinFeats[:, ct] = feats[:, i]
            clinFeats_noNorm[:, ct] = allFeatsNoNorm[:, i]
            ct += 1
            chdr += "," + k

    assert ct == len(clinModelFeats)
    arr2csv(os.path.join(nFeatPath, 'clinical_model_make.csv'), clinFeats, nids, fmt='%.5g', header=chdr)
    arr2csv(os.path.join(nFeatPath, 'clinical_model_make_noNorm.csv'), clinFeats_noNorm, nids, fmt='%.5g', header=chdr)

    thdr = chdr + "," + ",".join(dhdr)
    arr2csv(os.path.join(nFeatPath, 'clinical_model_make_wTrajectory.csv'), np.hstack([clinFeats, dfeats_norm]), nids,
            fmt='%.5g',
            header=thdr)

    chdr += ",Died,MAKE"
    arr2csv(os.path.join(nFeatPath, 'clinical_model_make_noNorm_wOutcome.csv'), np.hstack([clinFeats_noNorm, dieds, makes]), nids, fmt='%.5g',
            header=chdr)
