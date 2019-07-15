import argparse
import os
import numpy as np
import pandas as pd
import json
import h5py
from utility_funcs import arr2csv, load_csv, dict2csv
from kdigo_funcs import get_dialysis_mask, get_t_mask, extract_scr_data, extract_masked_data, get_baselines, \
    get_exclusion_criteria, linear_interpo, scr2kdigo
from classification_funcs import descriptive_trajectory_features
from stat_funcs import get_uky_demographics, summarize_stats, get_sofa, get_apache, formatted_stats
from sklearn.preprocessing import MinMaxScaler


parser = argparse.ArgumentParser(description='Preprocess Data and Construct KDIGO Vectors.')
parser.add_argument('--config_file', action='store', nargs=1, type=str, dest='cfname',
                    default='kdigo_conf.json')
parser.add_argument('--config_path', action='store', nargs=1, type=str, dest='cfpath',
                    default='')
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

baseDataPath = os.path.join(basePath, 'DATA', 'all_sheets')
dataPath = os.path.join(basePath, 'DATA', analyze, cohortName)
resPath = os.path.join(basePath, 'RESULTS', analyze, cohortName)

if not os.path.exists(os.path.join(dataPath, 'rrt_mask.csv')):
    rrt_mask = get_dialysis_mask(baseDataPath)
    np.savetxt(os.path.join(dataPath, 'rrt_mask.csv'), rrt_mask, delimiter=',', fmt='%d')
else:
    print('Loaded previous dialysis mask...')
    rrt_mask = np.loadtxt(os.path.join(dataPath, 'rrt_mask.csv'), dtype=int)

# Get mask indicating whether each point was in hospital or ICU
if not os.path.exists(os.path.join(dataPath, 'hosp_mask.csv')):
    hosp_mask, hosp_windows, icu_windows = get_t_mask(baseDataPath)
    dict2csv(os.path.join(dataPath, 'icu_admit_discharge.csv'), icu_windows, fmt='%s')
    dict2csv(os.path.join(dataPath, 'hosp_admit_discharge.csv'), hosp_windows, fmt='%s')
else:
    print('Loaded previous ICU masks...')
    hosp_mask = np.loadtxt(os.path.join(dataPath, 'icu_mask.csv'), dtype=int)
    icu_windows, iids = load_csv(os.path.join(dataPath, 'icu_admit_discharge.csv'), ids=None, dt='date', struct='dict')
    hosp_windows, hids = load_csv(os.path.join(dataPath, 'hosp_admit_discharge.csv'), ids=None, dt='date', struct='dict')

ids = np.sort(np.array(list(hosp_windows), dtype=int))
# Baselines
if os.path.exists(os.path.join(baseDataPath, 'all_baseline_info.csv')):
    bsln_m = pd.read_csv(os.path.join(baseDataPath, 'all_baseline_info.csv'))
    bsln_scr_loc = bsln_m.columns.get_loc('bsln_val')
    bsln_type_loc = bsln_m.columns.get_loc('bsln_type')
    bsln_m = bsln_m.values
    print('Loaded previously determined baselines.')
    genders = None
else:
    print('Determining all patient baseline SCr.')
    admits = []
    for tid in ids:
        admits.append(hosp_windows[tid][0])
    genders, races, ages = get_uky_demographics(ids, hosp_windows, baseDataPath)
    get_baselines(baseDataPath, hosp_windows, genders, races, ages, outp_rng=(1, 365), inp_rng=(7, 365))

    bsln_m = pd.read_csv(os.path.join(baseDataPath, 'all_baseline_info.csv'))
    bsln_scr_loc = bsln_m.columns.get_loc('bsln_val')
    bsln_type_loc = bsln_m.columns.get_loc('bsln_type')
    bsln_m = bsln_m.values
    print('Finished calculated baselines.')

if not os.path.exists(os.path.join(dataPath, 'scr_raw.csv')):
    scrs, dates, tmasks, hd_masks, crrt_masks, pd_masks, rrt_masks = extract_scr_data(icu_windows, hosp_windows, baseDataPath)
    arr2csv(os.path.join(dataPath, 'scr_raw.csv'), scrs, ids, fmt='%.3f')
    arr2csv(os.path.join(dataPath, 'dates.csv'), dates, ids, fmt='%s')
    arr2csv(os.path.join(dataPath, 'ind_rrt_masks.csv'), rrt_masks, ids, fmt='%d')
    arr2csv(os.path.join(dataPath, 'ind_hosp_masks.csv'), tmasks, ids, fmt='%d')
else:
    scrs = load_csv(os.path.join(dataPath, 'scr_raw.csv'), ids)
    dates = load_csv(os.path.join(dataPath, 'dates.csv'), ids, dt='date')

if not os.path.exists(os.path.join(dataPath, 'scr_raw_icu.csv')):
    (scrs, dates, rrt_masks) = extract_masked_data([scrs, dates, rrt_masks], tmasks)
    arr2csv(os.path.join(dataPath, 'scr_raw_icu.csv'), scrs, ids, fmt='%.3f')
    arr2csv(os.path.join(dataPath, 'dates_icu.csv'), dates, ids, fmt='%s')
    arr2csv(os.path.join(dataPath, 'ind_rrt_masks_icu.csv'), rrt_masks, ids, fmt='%d')
else:
    scrs = load_csv(os.path.join(dataPath, 'scr_raw_icu.csv'), ids)
    dates = load_csv(os.path.join(dataPath, 'dates_icu.csv'), ids, 'date')
    rrt_masks = load_csv(os.path.join(dataPath, 'ind_rrt_masks_icu.csv'), ids, int)

if not os.path.exists(os.path.join(dataPath, 'exclusion_criteria.csv')):
    if genders is None:
        genders, races, ages = get_uky_demographics(ids, hosp_windows, baseDataPath)
    exc, hdr = get_exclusion_criteria(ids, hosp_mask, icu_windows, genders, races, ages, baseDataPath)
    arr2csv(os.path.join(dataPath, 'exclusion_criteria.csv'), exc, ids, fmt='%d', header=hdr)
    excluded = np.max(exc, axis=1)
    keep = np.where(excluded == 0)[0]
else:
    exc = load_csv(os.path.join(dataPath, 'exclusion_criteria.csv'), ids, int)
    excluded = np.max(exc, axis=1)
    keep = np.where(excluded == 0)[0]

if not os.path.exists(os.path.join(dataPath, 'scr_interp.csv')):
    # Interpolate missing values
    print('Interpolating missing values')
    post_interpo, dmasks_interp, days_interp, interp_masks, ids = linear_interpo(scrs, ids, dates, rrt_masks, tRes)
    arr2csv(dataPath + 'scr_interp.csv', post_interpo, ids)
    arr2csv(dataPath + 'days_interp.csv', days_interp, ids, fmt='%d')
    arr2csv(dataPath + 'interp_masks.csv', interp_masks, ids, fmt='%d')
    arr2csv(dataPath + 'dmasks_interp.csv', dmasks_interp, ids, fmt='%d')
    np.savetxt(dataPath + 'post_interp_ids.csv', ids, fmt='%d')
else:
    print('Loaded previously interpolated SCrs')
    post_interpo = load_csv(os.path.join(dataPath, 'scr_interp.csv'), ids, float)
    dmasks_interp = load_csv(os.path.join(dataPath, 'dmasks_interp.csv'), ids, int)
    days_interp = load_csv(os.path.join(dataPath, 'days_interp.csv'), ids, int)
    interp_masks = load_csv(os.path.join(dataPath, 'interp_masks.csv'), ids, int)

if not os.path.exists(os.path.join(dataPath, 'kdigo.csv')):
    # Convert SCr to KDIGO
    baselines = load_csv(dataPath + 'baselines.csv', ids, idxs=[1, ])
    print('Converting to KDIGO')
    kdigos = scr2kdigo(post_interpo, baselines, dmasks_interp, days_interp, interp_masks)
    arr2csv(dataPath + 'kdigo.csv', kdigos, ids, fmt='%d')
else:
    print('Loaded KDIGO scores')
    kdigos = load_csv(os.path.join(dataPath, 'kdigo.csv'), ids, int)


if not os.path.exists(os.path.join(resPath, 'stats.h5')):
    f = h5py.File(os.path.join(resPath, 'stats.h5'), 'w')
else:
    f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r+')
try:
    all_stats = summarize_stats(f, ids, kdigos, days_interp, scrs, icu_windows, hosp_windows,
                                baseDataPath, grp_name='meta_all', tlim=7)
finally:
    f.close()

f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r+')
all_stats = f['meta_all']
max_kdigo = all_stats['max_kdigo_7d'][:]
pt_sel = np.where(max_kdigo > 0)[0]
assert len(max_kdigo) == len(excluded)
pt_sel = np.intersect1d(keep, pt_sel)

if 'meta' not in list(f):
    stats = f.create_group('meta')
    for i in range(len(list(all_stats))):
        name = list(all_stats)[i]
        stats.create_dataset(name, data=all_stats[name][:][pt_sel], dtype=all_stats[name].dtype)

# Calculate clinical mortality prediction scores
sofa = None
if not os.path.exists(dataPath + 'sofa.csv'):
    print('Getting SOFA scores')
    sofa, sofa_hdr, sofa_raw, sofa_raw_hdr = get_sofa(ids, all_stats, post_interpo, days_interp,
                                                      out_name=os.path.join(dataPath, 'sofa.csv'))
    arr2csv(os.path.join(dataPath, 'sofa_raw.csv'), sofa_raw, ids, fmt='%.3f', header=sofa_raw_hdr)
    arr2csv(os.path.join(dataPath, 'sofa.csv'), sofa, ids, fmt='%.3f', header=sofa_hdr)
else:
    sofa, sofa_hdr = load_csv(dataPath + 'sofa.csv', ids, dt=int, skip_header='keep')
    sofa_raw, sofa_raw_hdr = load_csv(dataPath + 'sofa_raw.csv', ids, dt=float, skip_header='keep')

apache = None
if not os.path.exists(dataPath + 'apache.csv'):
    print('Getting APACHE-II Scores')
    apache, apache_hdr, apache_raw, apache_raw_hdr = get_apache(ids, all_stats, post_interpo, days_interp,
                                                                out_name=os.path.join(dataPath, 'apache.csv'))
    arr2csv(os.path.join(dataPath, 'apache_raw.csv'), apache_raw, ids, fmt='%.3f', header=apache_raw_hdr)
    arr2csv(os.path.join(dataPath, 'apache.csv'), apache, ids, fmt='%.3f', header=apache_hdr)
else:
    apache, apache_hdr = load_csv(dataPath + 'apache.csv', ids, dt=int, skip_header='keep')
    apache_raw, apache_raw_hdr = load_csv(dataPath + 'apache_raw.csv', ids, dt=float, skip_header='keep')

if 'sofa' not in list(all_stats):
    sofa_sum = np.sum(sofa, axis=1)
    all_stats.create_dataset('sofa', data=sofa_sum, dtype=int)
    stats.create_dataset('sofa', data=sofa_sum[pt_sel], dtype=int)
if 'apache' not in list(all_stats):
    apache_sum = np.sum(apache, axis=1)
    all_stats.create_dataset('apache', data=apache_sum, dtype=int)
    stats.create_dataset('apache', data=apache_sum[pt_sel], dtype=int)

f.close()

f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
cohort_ids = ids[pt_sel]
cohort_kdigos = load_csv(os.path.join(dataPath, 'kdigo.csv'), ids, int)
cohort_days = load_csv(os.path.join(dataPath, 'days_interp.csv'), ids, int)
os.mkdir(os.path.join(resPath, 'clusters'))
os.mkdir(os.path.join(resPath, 'clusters', 'died_inp'))
died = f['meta']['died_inp'][:]
died[np.where(died)] = 1
arr2csv(os.path.join(resPath, 'clusters', 'died_inp', 'clusters.csv'), died, ids, fmt='%d')
formatted_stats(f['meta'], os.path.join(resPath, 'clusters', 'died_inp'))

# Calculate individual trajectory based features if not already done
if not os.path.exists(os.path.join(resPath, 'features')):
    os.mkdir(os.path.join(resPath, 'features'))

if not os.path.exists(os.path.join(resPath, 'features', 'individual')):
    mms = MinMaxScaler()
    os.mkdir(os.path.join(resPath, 'features', 'individual'))
    desc, desc_hdr = descriptive_trajectory_features(cohort_kdigos, cohort_ids, days=cohort_days, t_lim=t_lim,
                                                     filename=os.path.join(resPath, 'features', 'individual',
                                                                           'descriptive_features.csv'))
    desc_norm = mms.fit_transform(desc)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'new_descriptive_norm.csv'), desc_norm, cohort_ids,
            fmt='%.3f', header=desc_hdr)

else:
    desc = load_csv(os.path.join(resPath, 'features', 'individual', 'descriptive_features.csv'), cohort_ids,
                    skip_header=True)

sofa = sofa[pt_sel, :]
sofa_norm = mms.fit_transform(sofa)
sofa_raw = sofa_raw[pt_sel, :]
sofa_raw_norm = mms.fit_transform(sofa_raw)

apache = apache[pt_sel, :]
apache_norm = mms.fit_transform(apache)
apache_raw = apache_raw[pt_sel, :]
apache_raw_norm = mms.fit_transform(apache_raw)

sofa_hdr = 'STUDY_PATIENT_ID,' + ','.join(sofa_hdr)
apache_hdr = 'STUDY_PATIENT_ID,' + ','.join(apache_hdr)
sofa_raw_hdr = 'STUDY_PATIENT_ID,' + ','.join(sofa_raw_hdr)
apache_raw_hdr = 'STUDY_PATIENT_ID,' + ','.join(apache_raw_hdr)

if not os.path.isfile(os.path.join(resPath, 'features', 'individual', 'sofa.csv')):
    arr2csv(os.path.join(resPath, 'features', 'individual', 'sofa.csv'),
            sofa, cohort_ids, fmt='%d', header=sofa_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'sofa_norm.csv'),
            sofa_norm, cohort_ids, fmt='%.4f', header=sofa_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'sofa_raw.csv'),
            sofa_raw, cohort_ids, fmt='%.4f', header=sofa_raw_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'sofa_raw_norm.csv'),
            sofa_raw_norm, cohort_ids, fmt='%.4f', header=sofa_raw_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'apache.csv'),
            apache, cohort_ids, fmt='%d', header=apache_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'apache_norm.csv'),
            apache_norm, cohort_ids, fmt='%.4f', header=apache_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'apache_raw.csv'),
            apache_raw, cohort_ids, fmt='%.4f', header=apache_raw_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'apache_raw_norm.csv'),
            apache_raw_norm, cohort_ids, fmt='%.4f', header=apache_raw_hdr)

print('Ready for distance matrix calculation. Please run script \'calc_dms.py\'')
