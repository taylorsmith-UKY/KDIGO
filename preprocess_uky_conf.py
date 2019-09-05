import argparse
import os
import numpy as np
import pandas as pd
import json
import h5py
from utility_funcs import arr2csv, load_csv, dict2csv, get_array_dates
from kdigo_funcs import get_admit_disch, extract_scr_data, extract_masked_data, get_baselines, \
    get_exclusion_criteria, linear_interpo, scr2kdigo, get_transition_weights, rolling_average, extract_window_data
from classification_funcs import descriptive_trajectory_features
from stat_funcs import get_uky_demographics, summarize_stats, get_sofa, get_apache, formatted_stats
from sklearn.preprocessing import MinMaxScaler

# Parser takes command line arguments, in this case the location and name of the configuration file.
parser = argparse.ArgumentParser(description='Preprocess Data and Construct KDIGO Vectors.')
parser.add_argument('--config_file', action='store', type=str, dest='cfname',
                    default='kdigo_conf.json')
parser.add_argument('--config_path', action='store', type=str, dest='cfpath',
                    default='')
parser.add_argument('--averagePoints', '-agvPts', action='store', type=int, dest='avgpts',
                    default=2)
args = parser.parse_args()

configurationFileName = os.path.join(args.cfpath, args.cfname)
fp = open(configurationFileName, 'r')
conf = json.load(fp)
fp.close()

# Load base configuration from file
basePath = conf['basePath']         # Path containing the DATA and RESULTS directories
cohortName = conf['cohortName']     # Will be used for folder name
t_lim = conf['analysisDays']        # How long to consider in the analysis
tRes = conf['timeResolutionHrs']    # Resolution after imputation in hours
analyze = conf['analyze']           # Time period to analyze (hospital/ICU/all)

# Build paths and create if don't already exist
baseDataPath = os.path.join(basePath, 'DATA', 'all_sheets')         # folder containing all raw data
dataPath = os.path.join(basePath, 'DATA', analyze, cohortName)
resPath = os.path.join(basePath, 'RESULTS', analyze, cohortName)
if not os.path.exists(dataPath):
    os.mkdir(dataPath)
if not os.path.exists(resPath):
    os.mkdir(resPath)

# Get mask for all SCr indicating whether each point was in hospital or ICU, or neither
if not os.path.exists(os.path.join(dataPath, 'icu_admit_discharge.csv')):
    print('Getting hospitalization mask...')
    hosp_windows, icu_windows = get_admit_disch(baseDataPath)
    dict2csv(os.path.join(dataPath, 'icu_admit_discharge.csv'), icu_windows, fmt='%s')
    dict2csv(os.path.join(dataPath, 'hosp_admit_discharge.csv'), hosp_windows, fmt='%s')
else:
    print('Loaded previous hospitalization mask.')
    icu_windows, iids = load_csv(os.path.join(dataPath, 'icu_admit_discharge.csv'), ids=None, dt='date', struct='dict')
    hosp_windows, hids = load_csv(os.path.join(dataPath, 'hosp_admit_discharge.csv'), ids=None, dt='date', struct='dict')

# We will initially consider all patients who have any hospital admission
ids = np.sort(np.array(list(hosp_windows), dtype=int))

# Get patient baseline SCr if not already computed
genders = None
if not os.path.exists(os.path.join(baseDataPath, 'all_baseline_info.csv')):
    print('Determining all patient baseline SCr.')
    genders, races, ages = get_uky_demographics(ids, hosp_windows, baseDataPath)
    get_baselines(baseDataPath, hosp_windows, genders, races, ages, outp_rng=(1, 365), inp_rng=(7, 365))

    bsln_m = pd.read_csv(os.path.join(baseDataPath, 'all_baseline_info.csv'))
    bsln_scr_loc = bsln_m.columns.get_loc('bsln_val')
    bsln_type_loc = bsln_m.columns.get_loc('bsln_type')
    bsln_m = bsln_m.values
    print('Finished calculated baselines.')

# Extract raw SCr data and corresponding masks for each patient individually
if not os.path.exists(os.path.join(dataPath, 'scr_raw.csv')):
    print('Extracting patient SCr data...')
    scrs, dates, tmasks, hd_masks, crrt_masks, pd_masks, rrt_masks = extract_scr_data(icu_windows, hosp_windows, baseDataPath)
    dates = get_array_dates(dates)
    arr2csv(os.path.join(dataPath, 'scr_raw.csv'), scrs, ids, fmt='%.3f')
    arr2csv(os.path.join(dataPath, 'dates.csv'), dates, ids, fmt='%s')
    arr2csv(os.path.join(dataPath, 'ind_rrt_masks.csv'), rrt_masks, ids, fmt='%d')
    arr2csv(os.path.join(dataPath, 'ind_hosp_masks.csv'), tmasks, ids, fmt='%d')
else:
    if not os.path.exists(os.path.join(dataPath, 'scr_raw_%s.csv' % analyze)):
        print('Loaded previous SCr data.')
        scrs = load_csv(os.path.join(dataPath, 'scr_raw.csv'), ids)
        dates = load_csv(os.path.join(dataPath, 'dates.csv'), ids, dt='date')
        rrt_masks = load_csv(os.path.join(dataPath, 'ind_rrt_masks.csv'), ids, dt=int)
        tmasks = load_csv(os.path.join(dataPath, 'ind_hosp_masks.csv'), ids, dt=int)
    if not os.path.exists(os.path.join(dataPath, 'exclusion_criteria.csv')):
        dates = load_csv(os.path.join(dataPath, 'dates.csv'), ids, dt='date')

# Extract masked data corresponding to the selected analysis windo for each patient
if not os.path.exists(os.path.join(dataPath, 'scr_raw_%s.csv' % analyze)):
    print('Extracting SCr records in ICU...')
    if analyze == 'icu':
        # (scrs, dates, rrt_masks) = extract_masked_data([scrs, dates, rrt_masks], tmasks, sel=2)
        [scrs, dates, rrt_masks] = extract_window_data(ids, [scrs, dates, rrt_masks], dates, icu_windows, 2)
    if analyze == 'hosp':
        # (scrs, dates, rrt_masks) = extract_masked_data([scrs, dates, rrt_masks], tmasks, sel=[1, 2])
        [scrs, dates, rrt_masks] = extract_window_data(ids, [scrs, dates, rrt_masks], dates, hosp_windows, 2)
    arr2csv(os.path.join(dataPath, 'scr_raw_%s.csv' % analyze), scrs, ids, fmt='%.3f')
    arr2csv(os.path.join(dataPath, 'dates_%s.csv' % analyze), dates, ids, fmt='%s')
    arr2csv(os.path.join(dataPath, 'ind_rrt_masks_%s.csv' % analyze), rrt_masks, ids, fmt='%d')
else:
    if not os.path.exists(os.path.join(dataPath, 'scr_interp%s.csv' % analyze)):
        print('Loaded ICU data.')
        scrs = load_csv(os.path.join(dataPath, 'scr_raw_%s.csv' % analyze), ids)
        dates = load_csv(os.path.join(dataPath, 'dates_%s.csv' % analyze), ids, 'date')
        rrt_masks = load_csv(os.path.join(dataPath, 'ind_rrt_masks_%s.csv' % analyze), ids, int)

# If computing rolling average, compute on the raw values
if not os.path.exists(os.path.join(dataPath, 'scr_raw_%s_%dptAvg.csv' % (analyze, args.avgpts))) and args.avgpts > 1:
    avgd = rolling_average(scrs, masks=[rrt_masks], times=dates)
    ascrs = avgd[0]
    admasks = avgd[1][0]
    adates = avgd[2]
    arr2csv(os.path.join(dataPath, 'scr_raw_%s_%dptAvg.csv' % (analyze, args.avgpts)), ascrs, ids, fmt='%f')
    arr2csv(os.path.join(dataPath, 'dates_%s_%dptAvg.csv' % (analyze, args.avgpts)), adates, ids, fmt='%s')
    arr2csv(os.path.join(dataPath, 'ind_rrt_masks_%s_%dptAvg.csv' % (analyze, args.avgpts)), admasks, ids, fmt='%d')
else:
    if not os.path.exists(os.path.join(dataPath, 'scr_interp_%s_%dptAvg.csv' % (analyze, args.avgpts))):
        ascrs = load_csv(os.path.join(dataPath, 'scr_raw_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids)
        adates = load_csv(os.path.join(dataPath, 'dates_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids, 'date')
        admasks = load_csv(os.path.join(dataPath, 'ind_rrt_masks_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids, int)

# Evaluate all exclusion criteria and select patients who don't match any
if not os.path.exists(os.path.join(dataPath, 'exclusion_criteria.csv')):
    print('Evaluating exclusion criteria...')
    if genders is None:
        genders, races, ages = get_uky_demographics(ids, hosp_windows, baseDataPath)
    exc, hdr = get_exclusion_criteria(ids, dates, icu_windows, genders, races, ages, baseDataPath)
    arr2csv(os.path.join(dataPath, 'exclusion_criteria.csv'), exc, ids, fmt='%d', header=hdr)
    excluded = np.max(exc, axis=1)
    keep = np.where(excluded == 0)[0]
else:
    print('Loaded exclusion criteria.')
    exc = load_csv(os.path.join(dataPath, 'exclusion_criteria.csv'), ids, int, skip_header=True)
    excluded = np.max(exc, axis=1)
    keep = np.where(excluded == 0)[0]

# Interpolate raw SCr values in the analysis window
if not os.path.exists(os.path.join(dataPath, 'scr_interp_%s.csv' % analyze)):
    # Interpolate missing values
    print('Interpolating missing values')
    post_interpo, dmasks_interp, days_interp, interp_masks = linear_interpo(scrs, ids, dates, rrt_masks, icu_windows, tRes)
    arr2csv(os.path.join(dataPath, 'scr_interp_%s.csv' % analyze), post_interpo, ids)
    arr2csv(os.path.join(dataPath, 'days_interp_%s.csv' % analyze), days_interp, ids, fmt='%d')
    arr2csv(os.path.join(dataPath, 'interp_masks_%s.csv' % analyze), interp_masks, ids, fmt='%d')
    arr2csv(os.path.join(dataPath, 'dmasks_interp_%s.csv' % analyze), dmasks_interp, ids, fmt='%d')
else:
    print('Loaded previously interpolated SCrs')
    post_interpo = load_csv(os.path.join(dataPath, 'scr_interp_%s.csv' % analyze), ids, float)
    dmasks_interp = load_csv(os.path.join(dataPath, 'dmasks_interp_%s.csv' % analyze), ids, int)
    days_interp = load_csv(os.path.join(dataPath, 'days_interp_%s.csv' % analyze), ids, int)
    interp_masks = load_csv(os.path.join(dataPath, 'interp_masks_%s.csv' % analyze), ids, int)

# Interpolate averaged sequence
if not os.path.exists(os.path.join(dataPath, 'scr_interp_%s_%dptAvg.csv' % (analyze, args.avgpts))) and args.avgpts > 1:
    apost_interpo, admasks_interp, adays_interp, ainterp_masks = linear_interpo(ascrs, ids, adates, admasks, icu_windows, tRes)
    arr2csv(os.path.join(dataPath, 'scr_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), apost_interpo, ids)
    arr2csv(os.path.join(dataPath, 'days_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), adays_interp, ids, fmt='%d')
    arr2csv(os.path.join(dataPath, 'interp_masks_%s_%dptAvg.csv' % (analyze, args.avgpts)), ainterp_masks, ids, fmt='%d')
    arr2csv(os.path.join(dataPath, 'dmasks_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), admasks_interp, ids, fmt='%d')
else:
    apost_interpo = load_csv(os.path.join(dataPath, 'scr_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids)
    admasks_interp = load_csv(os.path.join(dataPath, 'dmasks_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids)
    adays_interp = load_csv(os.path.join(dataPath, 'days_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids)
    ainterp_masks = load_csv(os.path.join(dataPath, 'interp_masks_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids)

# Compute KDIGO scores for the analysis window using interpolated SCr and previously determined baselines
if not os.path.exists(os.path.join(dataPath, 'kdigo_%s.csv' % analyze)):
    # Convert SCr to KDIGO
    baselines = pd.read_csv(os.path.join(baseDataPath, 'all_baseline_info.csv'))['bsln_val'].values
    print('Converting to KDIGO')
    kdigos = scr2kdigo(post_interpo, baselines, dmasks_interp, days_interp, interp_masks)
    arr2csv(os.path.join(dataPath, 'kdigo_%s.csv' % analyze), kdigos, ids, fmt='%d')
    kdigos_noabs = scr2kdigo(post_interpo, baselines, dmasks_interp, days_interp, interp_masks, useAbs=False)
    arr2csv(os.path.join(dataPath, 'kdigo_noAbs_%s.csv' % analyze), kdigos_noabs, ids, fmt='%d')
else:
    print('Loaded KDIGO scores')
    kdigos = load_csv(os.path.join(dataPath, 'kdigo_%s.csv' % analyze), ids, int)

# KDIGO for averaged sequence
if not os.path.exists(os.path.join(dataPath, 'kdigo_%s_%dptAvg.csv') % (analyze, args.avgpts)) and args.avgpts:
    baselines = pd.read_csv(os.path.join(baseDataPath, 'all_baseline_info.csv'))['bsln_val'].values
    akdigos = scr2kdigo(apost_interpo, baselines, admasks_interp, adays_interp, ainterp_masks)
    arr2csv(os.path.join(dataPath, 'kdigo_%s_%dptAvg.csv') % (analyze, args.avgpts), akdigos, ids, fmt='%d')
    akdigos_noabs = scr2kdigo(apost_interpo, baselines, admasks_interp, adays_interp, ainterp_masks, useAbs=False)
    arr2csv(os.path.join(dataPath, 'kdigo_%s_%dptAvg_noAbs.csv') % (analyze, args.avgpts), akdigos_noabs, ids, fmt='%d')
else:
    akdigos = load_csv(os.path.join(dataPath, 'kdigo_%s_%dptAvg.csv') % (analyze, args.avgpts), ids, int)

# File to store non-temporal patient data
if not os.path.exists(os.path.join(resPath, 'stats.h5')):
    f = h5py.File(os.path.join(resPath, 'stats.h5'), 'w')
else:
    f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r+')

# Extract all static data of interest and store in f
# Note: group "meta_all" contains all patients, regardless of exclusion, while "meta" only contains those patients
# who remain following application of exclusion criteria, INCLUDING the requirement of AKI in the analysis window
if 'meta' not in list(f):
    try:
        all_stats = summarize_stats(f, ids, akdigos, adays_interp, apost_interpo, icu_windows, hosp_windows,
                                    baseDataPath, grp_name='meta_all', tlim=7)
        max_kdigo = np.zeros(len(ids))
        max_kdigo_avg = np.zeros(len(ids))
        for i in range(len(kdigos)):
            try:
                max_kdigo[i] = np.max(kdigos[i][np.where(days_interp[i] <= t_lim)[0]])
            except (IndexError, TypeError):
                pass
            try:
                max_kdigo_avg[i] = np.max(akdigos[i][np.where(adays_interp[i] <= t_lim)[0]])
            except (IndexError, TypeError):
                pass
        # max_kdigo = all_stats['max_kdigo_win'][:]
        pt_sel = np.where(max_kdigo > 0)[0]
        pt_sel_avg = np.where(max_kdigo_avg > 0)[0]
        assert len(max_kdigo) == len(excluded)
        pt_sel = np.intersect1d(keep, pt_sel)
        pt_sel_avg = np.intersect1d(keep, pt_sel_avg)
        stats = f.create_group('meta')
        for i in range(len(list(all_stats))):
            name = list(all_stats)[i]
            try:
                stats.create_dataset(name, data=all_stats[name][:][pt_sel], dtype=all_stats[name].dtype)
            except IndexError:
                pass
        stats_avg = f.create_group('meta_avg')
        for i in range(len(list(all_stats))):
            name = list(all_stats)[i]
            try:
                stats_avg.create_dataset(name, data=all_stats[name][:][pt_sel_avg], dtype=all_stats[name].dtype)
            except IndexError:
                pass
    finally:
        f.close()
else:
    f.close()

# Load static data file and create mask indicating patients who don't match any exclusion criteria
f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r+')
all_stats = f['meta_all']
stats = f['meta']
stats_avg = f['meta_avg']
all_ids = all_stats['ids'][:]
cohort_ids = stats['ids'][:]
cohort_ids_avg = stats_avg['ids'][:]
pt_sel = np.array([x in cohort_ids for x in all_ids])
pt_sel_avg = np.array([x in cohort_ids_avg for x in all_ids])

if not os.path.exists(os.path.join(resPath, 'kdigo_%s.csv' % analyze)):
    kdigo_dist = load_csv(os.path.join(dataPath, 'kdigo_%s.csv' % analyze), cohort_ids, int)
    days_dist = load_csv(os.path.join(dataPath, 'days_interp_%s.csv' % analyze), cohort_ids, int)
    for i in range(len(kdigo_dist)):
        if type(kdigo_dist[i]) == np.ndarray and len(days_dist[i]) > 1:
            kdigo_dist[i] = kdigo_dist[i][np.where(days_dist[i] <= t_lim)[0]]
    arr2csv(os.path.join(resPath, 'kdigo_%s.txt' % analyze), kdigo_dist, ids, fmt='%d', delim=' ')

    kdigo_dist = load_csv(os.path.join(dataPath, 'kdigo_%s_%dptAvg.csv' % (analyze, args.avgpts)), cohort_ids_avg, int)
    days_dist = load_csv(os.path.join(dataPath, 'days_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), cohort_ids_avg, int)
    for i in range(len(kdigo_dist)):
        if hasattr(days_dist[i], "__len__") and len(days_dist[i]) > 1:
            kdigo_dist[i] = kdigo_dist[i][np.where(days_dist[i] <= t_lim)[0]]
    arr2csv(os.path.join(resPath, 'kdigo_%s_%dptAvg.txt' % (analyze, args.avgpts)), kdigo_dist, cohort_ids_avg, fmt='%d', delim=' ')


# Calculate clinical mortality prediction scores
# For each, returns the normal categorical scores, as well as the raw values used for each
sofa = None
if not os.path.exists(os.path.join(dataPath, 'sofa.csv')):
    print('Getting SOFA scores')
    sofa, sofa_hdr, sofa_raw, sofa_raw_hdr = get_sofa(ids, all_stats, post_interpo, days_interp,
                                                      out_name=os.path.join(dataPath, 'sofa.csv'))
    arr2csv(os.path.join(dataPath, 'sofa_raw.csv'), sofa_raw, ids, fmt='%.3f', header=sofa_raw_hdr)
    arr2csv(os.path.join(dataPath, 'sofa.csv'), sofa, ids, fmt='%d', header=sofa_hdr)
else:
    sofa, sofa_hdr = load_csv(os.path.join(dataPath, 'sofa.csv'), ids, dt=int, skip_header='keep')
    sofa_raw, sofa_raw_hdr = load_csv(os.path.join(dataPath, 'sofa_raw.csv'), ids, dt=float, skip_header='keep')

apache = None
if not os.path.exists(os.path.join(dataPath, 'apache.csv')):
    print('Getting APACHE-II Scores')
    apache, apache_hdr, apache_raw, apache_raw_hdr = get_apache(ids, all_stats, post_interpo, days_interp,
                                                                out_name=os.path.join(dataPath, 'apache.csv'))
    arr2csv(os.path.join(dataPath, 'apache_raw.csv'), apache_raw, ids, fmt='%.3f', header=apache_raw_hdr)
    arr2csv(os.path.join(dataPath, 'apache.csv'), apache, ids, fmt='%d', header=apache_hdr)
else:
    apache, apache_hdr = load_csv(os.path.join(dataPath, 'apache.csv'), ids, dt=int, skip_header='keep')
    apache_raw, apache_raw_hdr = load_csv(os.path.join(dataPath, 'apache_raw.csv'), ids, dt=float, skip_header='keep')

# Make sure composite scores are stored in the static data file
if 'sofa' not in list(all_stats):
    sofa_sum = np.sum(sofa, axis=1)
    all_stats.create_dataset('sofa', data=sofa_sum, dtype=int)
    stats.create_dataset('sofa', data=sofa_sum[pt_sel], dtype=int)
    stats_avg.create_dataset('sofa', data=sofa_sum[pt_sel_avg], dtype=int)
if 'apache' not in list(all_stats):
    apache_sum = np.sum(apache, axis=1)
    all_stats.create_dataset('apache', data=apache_sum, dtype=int)
    stats.create_dataset('apache', data=apache_sum[pt_sel], dtype=int)
    stats_avg.create_dataset('apache', data=apache_sum[pt_sel_avg], dtype=int)

# Close now to make sure that it is saved.
f.close()

# Load KDIGO in the analysis window for the cohort
f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
all_stats = f['meta_all']
stats = f['meta']
stats_avg = f['meta_avg']

cohort_kdigos = load_csv(os.path.join(dataPath, 'kdigo_%s.csv' % analyze), cohort_ids, int)
cohort_days = load_csv(os.path.join(dataPath, 'days_interp_%s.csv' % analyze), cohort_ids, int)
cohort_kdigos_avg = load_csv(os.path.join(dataPath, 'kdigo_%s.csv' % analyze), cohort_ids_avg, int)
cohort_days_avg = load_csv(os.path.join(dataPath, 'days_interp_%s.csv' % analyze), cohort_ids_avg, int)
if not os.path.exists(os.path.join(resPath, 'clusters')):
    os.mkdir(os.path.join(resPath, 'clusters'))
    os.mkdir(os.path.join(resPath, 'clusters', 'died_inp'))
    os.mkdir(os.path.join(resPath, 'clusters', 'died_inp', 'raw'))
    os.mkdir(os.path.join(resPath, 'clusters', 'died_inp', 'avg'))
died = stats['died_inp'][:]
died[np.where(died)] = 1
arr2csv(os.path.join(resPath, 'clusters', 'died_inp', 'raw', 'clusters.csv'), died, cohort_ids, fmt='%d')

# Construct a table of patient characteristics grouped by mortality
formatted_stats(stats, os.path.join(resPath, 'clusters', 'died_inp', 'raw'))

died = stats_avg['died_inp'][:]
died[np.where(died)] = 1
arr2csv(os.path.join(resPath, 'clusters', 'died_inp', 'avg', 'clusters.csv'), died, cohort_ids_avg, fmt='%d')

# Construct a table of patient characteristics grouped by mortality
formatted_stats(stats_avg, os.path.join(resPath, 'clusters', 'died_inp', 'avg'))

# Calculate individual trajectory based features
if not os.path.exists(os.path.join(resPath, 'features')):
    os.mkdir(os.path.join(resPath, 'features'))

if not os.path.exists(os.path.join(resPath, 'features', 'individual')):
    mms = MinMaxScaler()
    os.mkdir(os.path.join(resPath, 'features', 'individual'))
    os.mkdir(os.path.join(resPath, 'features', 'individual', 'raw'))
    desc, desc_hdr = descriptive_trajectory_features(cohort_kdigos, cohort_ids, days=cohort_days, t_lim=t_lim,
                                                     filename=os.path.join(resPath, 'features', 'individual',
                                                                           'raw', 'descriptive_features.csv'))
    desc_norm = mms.fit_transform(desc)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'raw', 'new_descriptive_norm.csv'), desc_norm, cohort_ids,
            fmt='%.3f', header=desc_hdr)
    
    os.mkdir(os.path.join(resPath, 'features', 'individual', 'avg'))
    desc, desc_hdr = descriptive_trajectory_features(cohort_kdigos_avg, cohort_ids_avg, days=cohort_days_avg, t_lim=t_lim,
                                                     filename=os.path.join(resPath, 'features', 'individual',
                                                                           'avg', 'descriptive_features.csv'))
    desc_norm = mms.fit_transform(desc)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'avg', 'new_descriptive_norm.csv'), desc_norm, cohort_ids,
            fmt='%.3f', header=desc_hdr)

# Normalize values in SOFA and APACHE scores for classification
sofa_na = sofa[pt_sel, :]
sofa_norm_na = mms.fit_transform(sofa_na)
sofa_raw_na = sofa_raw[pt_sel, :]
sofa_raw_norm_na = mms.fit_transform(sofa_raw_na)

apache_na = apache[pt_sel, :]
apache_norm_na = mms.fit_transform(apache_na)
apache_raw_na = apache_raw[pt_sel, :]
apache_raw_norm_na = mms.fit_transform(apache_raw_na)

sofa_avg = sofa[pt_sel_avg, :]
sofa_norm_avg = mms.fit_transform(sofa_avg)
sofa_raw_avg = sofa_raw[pt_sel_avg, :]
sofa_raw_norm_avg = mms.fit_transform(sofa_raw_avg)

apache_avg = apache[pt_sel_avg, :]
apache_norm_avg = mms.fit_transform(apache_avg)
apache_raw_avg = apache_raw[pt_sel_avg, :]
apache_raw_norm_avg = mms.fit_transform(apache_raw_avg)


sofa_hdr = 'STUDY_PATIENT_ID,' + ','.join(sofa_hdr)
apache_hdr = 'STUDY_PATIENT_ID,' + ','.join(apache_hdr)
sofa_raw_hdr = 'STUDY_PATIENT_ID,' + ','.join(sofa_raw_hdr)
apache_raw_hdr = 'STUDY_PATIENT_ID,' + ','.join(apache_raw_hdr)

# Save SOFA and APACHE features for use in classification
if not os.path.isfile(os.path.join(resPath, 'features', 'individual', 'raw', 'sofa.csv')):
    arr2csv(os.path.join(resPath, 'features', 'individual', 'raw', 'sofa.csv'),
            sofa_na, cohort_ids, fmt='%d', header=sofa_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'raw', 'sofa_norm.csv'),
            sofa_norm_na, cohort_ids, fmt='%.4f', header=sofa_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'raw', 'sofa_raw.csv'),
            sofa_raw_na, cohort_ids, fmt='%.4f', header=sofa_raw_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'raw', 'sofa_raw_norm.csv'),
            sofa_raw_norm_na, cohort_ids, fmt='%.4f', header=sofa_raw_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'raw', 'apache.csv'),
            apache_na, cohort_ids, fmt='%d', header=apache_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'raw', 'apache_norm.csv'),
            apache_norm_na, cohort_ids, fmt='%.4f', header=apache_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'raw', 'apache_raw.csv'),
            apache_raw_na, cohort_ids, fmt='%.4f', header=apache_raw_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'raw', 'apache_raw_norm.csv'),
            apache_raw_norm_na, cohort_ids, fmt='%.4f', header=apache_raw_hdr)
    
if not os.path.isfile(os.path.join(resPath, 'features', 'individual', 'avg', 'sofa.csv')):
    arr2csv(os.path.join(resPath, 'features', 'individual', 'avg', 'sofa.csv'),
            sofa_avg, cohort_ids_avg, fmt='%d', header=sofa_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'avg', 'sofa_norm.csv'),
            sofa_norm_avg, cohort_ids_avg, fmt='%.4f', header=sofa_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'avg', 'sofa_raw.csv'),
            sofa_raw_avg, cohort_ids_avg, fmt='%.4f', header=sofa_raw_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'avg', 'sofa_raw_norm.csv'),
            sofa_raw_norm_avg, cohort_ids_avg, fmt='%.4f', header=sofa_raw_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'avg', 'apache.csv'),
            apache_avg, cohort_ids_avg, fmt='%d', header=apache_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'avg', 'apache_norm.csv'),
            apache_norm_avg, cohort_ids_avg, fmt='%.4f', header=apache_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'avg', 'apache_raw.csv'),
            apache_raw_avg, cohort_ids_avg, fmt='%.4f', header=apache_raw_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'avg', 'apache_raw_norm.csv'),
            apache_raw_norm_avg, cohort_ids_avg, fmt='%.4f', header=apache_raw_hdr)

# Compute the 'transition probabiliy' between adjacent KDIGO scores for the cohort and construct the
# corresponding transition weights.
# Tw_unscaled(k) = log(Total # transitions between any scores / # transitions between k and k-1)
if not os.path.exists(os.path.join(dataPath, 'kdigo_transition_weights.csv')):
    tweights = get_transition_weights(cohort_kdigos)
    arr2csv(os.path.join(dataPath, 'kdigo_transition_weights.csv'), tweights,
            ['0-1', '1-2', '2-3', '3-3D'], header='Transition,Weight')

print('Ready for distance matrix calculation. Please run script \'calc_dms.py\'')
