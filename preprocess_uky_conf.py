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
    scrs, dates, tmasks, hd_masks, crrt_masks, pd_masks, dmasks = extract_scr_data(icu_windows, hosp_windows, baseDataPath)
    dates = get_array_dates(dates)
    arr2csv(os.path.join(dataPath, 'scr_raw.csv'), scrs, ids, fmt='%.3f')
    arr2csv(os.path.join(dataPath, 'dates.csv'), dates, ids, fmt='%s')
    arr2csv(os.path.join(dataPath, 'ind_rrt_masks.csv'), dmasks, ids, fmt='%d')
    arr2csv(os.path.join(dataPath, 'ind_hosp_masks.csv'), tmasks, ids, fmt='%d')
else:
    if not os.path.exists(os.path.join(dataPath, 'scr_raw_%s.csv' % analyze)):
        print('Loaded previous SCr data.')
        scrs = load_csv(os.path.join(dataPath, 'scr_raw.csv'), ids)
        dates = load_csv(os.path.join(dataPath, 'dates.csv'), ids, dt='date')
        dmasks = load_csv(os.path.join(dataPath, 'ind_rrt_masks.csv'), ids, dt=int)
        tmasks = load_csv(os.path.join(dataPath, 'ind_hosp_masks.csv'), ids, dt=int)
    if not os.path.exists(os.path.join(dataPath, 'exclusion_criteria.csv')):
        dates = load_csv(os.path.join(dataPath, 'dates.csv'), ids, dt='date')

# Extract data corresponding to ICU
if not os.path.exists(os.path.join(dataPath, 'scr_raw_%s.csv' % analyze)):
    if analyze == 'icu':
        (scrs, dates, dmasks) = extract_masked_data([scrs, dates, dmasks], tmasks, sel=2)
    elif analyze == 'hosp':
        (scrs, dates, dmasks) = extract_masked_data([scrs, dates, dmasks], tmasks, sel=[1, 2])
    arr2csv(os.path.join(dataPath, 'scr_raw_%s.csv' % analyze), scrs, ids, fmt='%.3f')
    arr2csv(os.path.join(dataPath, 'dates_%s.csv' % analyze), dates, ids, fmt='%s')
    arr2csv(os.path.join(dataPath, 'ind_rrt_masks_%s.csv' % analyze), dmasks, ids, fmt='%d')
else:
    scrs = load_csv(os.path.join(dataPath, 'scr_raw_%s.csv' % analyze), ids)
    dates = load_csv(os.path.join(dataPath, 'dates_%s.csv' % analyze), ids, dt='date')
    dmasks = load_csv(os.path.join(dataPath, 'ind_rrt_masks_%s.csv' % analyze), ids, dt=int)

# If computing rolling average, compute on the raw values
if not os.path.exists(os.path.join(dataPath, 'scr_raw_%s_%dptAvg.csv' % (analyze, args.avgpts))) and args.avgpts > 1:
    avgd = rolling_average(scrs, masks=[dmasks], times=dates)
    scrs = avgd[0]
    dmasks = avgd[1][0]
    dates = avgd[2]
    arr2csv(os.path.join(dataPath, 'scr_raw_%s_%dptAvg.csv' % (analyze, args.avgpts)), scrs, ids, fmt='%f')
    arr2csv(os.path.join(dataPath, 'dates_%s_%dptAvg.csv' % (analyze, args.avgpts)), dates, ids, fmt='%s')
    arr2csv(os.path.join(dataPath, 'ind_rrt_masks_%s_%dptAvg.csv' % (analyze, args.avgpts)), dmasks, ids, fmt='%d')
else:
    scrs = load_csv(os.path.join(dataPath, 'scr_raw_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids)
    dates = load_csv(os.path.join(dataPath, 'dates_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids, dt='date')
    dmasks = load_csv(os.path.join(dataPath, 'ind_rrt_masks_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids, dt=int)

# Evaluate all exclusion criteria and select patients who don't match any
if not os.path.exists(os.path.join(dataPath, 'exclusion_criteria_avg.csv')):
    print('Evaluating exclusion criteria for rolling average...')
    if genders is None:
        genders, races, ages = get_uky_demographics(ids, hosp_windows, baseDataPath)
    exc, hdr = get_exclusion_criteria(ids, dates, icu_windows, genders, races, ages, baseDataPath)
    arr2csv(os.path.join(dataPath, 'exclusion_criteria_avg.csv'), exc, ids, fmt='%d', header=hdr)
    excluded = np.max(exc, axis=1)
    keep = np.where(excluded == 0)[0]
else:
    print('Loaded exclusion criteria.')
    exc = load_csv(os.path.join(dataPath, 'exclusion_criteria_avg.csv'), ids, int, skip_header=True)
    excluded = np.max(exc, axis=1)
    keep = np.where(excluded == 0)[0]

# Interpolate SCr sequences
if not os.path.exists(os.path.join(dataPath, 'scr_interp_%s_%dptAvg.csv' % (analyze, args.avgpts))) and args.avgpts > 1:
    post_interpo, dmasks_interp, days_interp, interp_masks = linear_interpo(scrs, ids, dates, dmasks, icu_windows, tRes)
    arr2csv(os.path.join(dataPath, 'scr_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), post_interpo, ids)
    arr2csv(os.path.join(dataPath, 'days_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), days_interp, ids, fmt='%d')
    arr2csv(os.path.join(dataPath, 'interp_masks_%s_%dptAvg.csv' % (analyze, args.avgpts)), interp_masks, ids, fmt='%d')
    arr2csv(os.path.join(dataPath, 'dmasks_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), dmasks_interp, ids, fmt='%d')
else:
    post_interpo = load_csv(os.path.join(dataPath, 'scr_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids)
    dmasks_interp = load_csv(os.path.join(dataPath, 'dmasks_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids)
    days_interp = load_csv(os.path.join(dataPath, 'days_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids)
    interp_masks = load_csv(os.path.join(dataPath, 'interp_masks_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids)

# KDIGO for averaged sequence
if not os.path.exists(os.path.join(dataPath, 'kdigo_%s_%dptAvg.csv') % (analyze, args.avgpts)) and args.avgpts:
    baselines = pd.read_csv(os.path.join(baseDataPath, 'all_baseline_info.csv'))['bsln_val'].values
    kdigos = scr2kdigo(post_interpo, baselines, dmasks_interp, days_interp, interp_masks)
    arr2csv(os.path.join(dataPath, 'kdigo_%s_%dptAvg.csv') % (analyze, args.avgpts), kdigos, ids, fmt='%d')
else:
    kdigos = load_csv(os.path.join(dataPath, 'kdigo_%s_%dptAvg.csv') % (analyze, args.avgpts), ids, int)

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
        all_stats = summarize_stats(f, ids, kdigos, days_interp, post_interpo, icu_windows, hosp_windows,
                                    baseDataPath, grp_name='meta_all', tlim=t_lim)

        # Select patients who do not match any exclusion criteria AND have max KDIGO > 0 during the analysis
        # window and save to separate group (w/ or w/out rolling average on raw SCr)
        # max_kdigo = all_stats['max_kdigo_win'][:]
        max_kdigo = all_stats['max_kdigo_win'][:]
        pt_sel = np.where(max_kdigo > 0)[0]
        pt_sel = np.intersect1d(keep, pt_sel)

        stats = f.create_group('meta')
        for i in range(len(list(all_stats))):
            name = list(all_stats)[i]
            try:
                stats.create_dataset(name, data=all_stats[name][:][pt_sel], dtype=all_stats[name].dtype)
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
all_ids = all_stats['ids'][:]
cohort_ids = stats['ids'][:]
pt_sel = np.array([x in cohort_ids for x in all_ids])

if not os.path.exists(os.path.join(resPath, 'kdigo_%s.csv' % analyze)):
    kdigo_dist = load_csv(os.path.join(dataPath, 'kdigo_%s_%dptAvg.csv' % (analyze, args.avgpts)), cohort_ids, int)
    days_dist = load_csv(os.path.join(dataPath, 'days_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), cohort_ids, int)
    for i in range(len(kdigo_dist)):
        kdigo_dist[i] = kdigo_dist[i][np.where(days_dist[i] <= t_lim)[0]]
    arr2csv(os.path.join(resPath, 'kdigo_%s_%dptAvg.txt' % (analyze, args.avgpts)), kdigo_dist, cohort_ids, fmt='%d')

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
if 'apache' not in list(all_stats):
    apache_sum = np.sum(apache, axis=1)
    all_stats.create_dataset('apache', data=apache_sum, dtype=int)
    stats.create_dataset('apache', data=apache_sum[pt_sel], dtype=int)

# Close now to make sure that it is saved.
f.close()

# Load KDIGO in the analysis window for the cohort
f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
all_stats = f['meta_all']
stats = f['meta']

cohort_kdigos = load_csv(os.path.join(dataPath, 'kdigo_%s_%dptAvg.csv' % (analyze, args.avgpts)), cohort_ids, int)
cohort_days = load_csv(os.path.join(dataPath, 'days_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), cohort_ids, int)
if not os.path.exists(os.path.join(resPath, 'clusters')):
    os.mkdir(os.path.join(resPath, 'clusters'))
    os.mkdir(os.path.join(resPath, 'clusters', 'died_inp'))
    os.mkdir(os.path.join(resPath, 'clusters', 'max_kdigo'))

died = stats['died_inp'][:]
died[np.where(died)] = 1
arr2csv(os.path.join(resPath, 'clusters', 'died_inp', 'clusters.csv'), died, cohort_ids, fmt='%d')

mk = stats['max_kdigo_win'][:]
arr2csv(os.path.join(resPath, 'clusters', 'max_kdigo', 'clusters.csv'), mk, cohort_ids, fmt='%d')

# Construct a table of patient characteristics grouped by mortality
formatted_stats(stats, os.path.join(resPath, 'clusters', 'died_inp'))
formatted_stats(stats, os.path.join(resPath, 'clusters', 'max_kdigo'))

# Calculate individual trajectory based features
if not os.path.exists(os.path.join(resPath, 'features')):
    os.mkdir(os.path.join(resPath, 'features'))

mms = MinMaxScaler()
if not os.path.exists(os.path.join(resPath, 'features', 'individual')):
    os.mkdir(os.path.join(resPath, 'features', 'individual'))
    desc, desc_hdr = descriptive_trajectory_features(cohort_kdigos, cohort_ids, days=cohort_days, t_lim=t_lim,
                                                     filename=os.path.join(resPath, 'features', 'individual',
                                                                           'descriptive_features.csv'))
    desc_norm = mms.fit_transform(desc)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'descriptive_norm.csv'), desc_norm, cohort_ids,
            fmt='%.3f', header=desc_hdr)

# Normalize values in SOFA and APACHE scores for classification
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

# Save SOFA and APACHE features for use in classification
if not os.path.isfile(os.path.join(resPath, 'features', 'individual', 'raw', 'sofa.csv')):
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

# Compute the 'transition probabiliy' between adjacent KDIGO scores for the cohort and construct the
# corresponding transition weights.
# Tw_unscaled(k) = log(Total # transitions between any scores / # transitions between k and k-1)
if not os.path.exists(os.path.join(dataPath, 'kdigo_transition_weights.csv')):
    tweights = get_transition_weights(cohort_kdigos)
    arr2csv(os.path.join(dataPath, 'kdigo_transition_weights.csv'), tweights,
            ['0-1', '1-2', '2-3', '3-3D'], header='Transition,Weight')
    arr2csv(os.path.join(resPath, 'kdigo_transition_weights.csv'), tweights,
            ['0-1', '1-2', '2-3', '3-3D'], header='Transition,Weight')

print('Ready for distance matrix calculation. Please run script \'calc_dms.py\'')
