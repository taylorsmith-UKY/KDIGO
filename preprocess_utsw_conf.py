import argparse
import os
import numpy as np
import pandas as pd
import json
import h5py
from utility_funcs import arr2csv, load_csv, dict2csv, get_array_dates
from kdigo_funcs import get_admit_disch, extract_scr_data, get_baselines, extract_masked_data, \
    get_exclusion_criteria_dallas, linear_interpo, scr2kdigo, rolling_average
from stat_funcs import get_utsw_demographics, summarize_stats_dallas, get_sofa, get_apache, get_MAKE90_dallas
from sklearn.preprocessing import MinMaxScaler

# Parser takes command line arguments, in this case the location and name of the configuration file.
parser = argparse.ArgumentParser(description='Preprocess Data and Construct KDIGO Vectors.')
parser.add_argument('--config_file', '-cf', action='store', type=str, dest='cfname',
                    default='final.json')
parser.add_argument('--config_path', '-cfp', action='store', type=str, dest='cfpath',
                    default='')
parser.add_argument('--averagePoints', '-agvPts', action='store', type=int, dest='avgpts',
                    default=2)
args = parser.parse_args()

configurationFileName = os.path.join(args.cfpath, args.cfname)
fp = open(configurationFileName, 'r')
conf = json.load(fp)
fp.close()

# Load base configuration from file
basePath = conf['basePath']  # Path containing the DATA and RESULTS directories
cohortName = conf['cohortName']  # Will be used for folder name
t_lim = conf['analysisDays']  # How long to consider in the analysis
tRes = conf['timeResolutionHrs']  # Resolution after imputation in hours
analyze = conf['analyze']  # Time period to analyze (hospital/ICU/all)

# Build paths and create if don't already exist
baseDataPath = os.path.join(basePath, 'DATA', 'dallas', 'csv')  # folder containing all raw data
dataPath = os.path.join(basePath, 'DATA', 'dallas', analyze, cohortName)
resPath = os.path.join(basePath, 'RESULTS', 'dallas', analyze, cohortName)
if not os.path.exists(dataPath):
    os.mkdir(dataPath)
if not os.path.exists(resPath):
    os.mkdir(resPath)

# Get mask for all SCr indicating whether each point was in hospital or ICU, or neither
if not os.path.exists(os.path.join(dataPath, 'icu_admit_discharge.csv')):
    print('Getting Indexed Hospital and ICU admissions...')
    # ddf = pd.read_csv(os.path.join(baseDataPath, ""))

    hosp_windows, icu_windows = get_admit_disch(baseDataPath, scr_fname='all_scr_data.csv',
                                                date_fname='tIndexedIcuAdmission.csv', scr_dcol='SPECIMN_TEN_TIME',
                                                id_col='PATIENT_NUM')

    dict2csv(os.path.join(dataPath, 'icu_admit_discharge.csv'), icu_windows, fmt='%s')
    dict2csv(os.path.join(dataPath, 'hosp_admit_discharge.csv'), hosp_windows, fmt='%s')
else:
    print('Loaded previously determined Indexed Hospital and ICU admissions.')
    icu_windows, iids = load_csv(os.path.join(dataPath, 'icu_admit_discharge.csv'), ids=None, dt='date', struct='dict')
    hosp_windows, hids = load_csv(os.path.join(dataPath, 'hosp_admit_discharge.csv'), ids=None, dt='date',
                                  struct='dict')

# We will initially consider all patients who have any hospital admission
ids = np.sort(np.array(list(hosp_windows), dtype=int))

# Get patient baseline SCr if not already computed
males = None
if not os.path.exists(os.path.join(baseDataPath, 'all_baseline_info.csv')):
    print('Determining all patient baseline SCr.')
    males, races, ages, dods, dtds = get_utsw_demographics(ids, hosp_windows, baseDataPath)
    get_baselines(baseDataPath, hosp_windows, males, races, ages, outp_rng=(1, 365), inp_rng=(7, 365),
                  scr_fname='all_scr_data.csv', scr_dcol='SPECIMN_TEN_TIME', scr_vcol='ORD_VALUE',
                  scr_typecol='IP_FLAG', id_col='PATIENT_NUM')

    bsln_m = pd.read_csv(os.path.join(baseDataPath, 'all_baseline_info.csv'))
    bsln_scr_loc = bsln_m.columns.get_loc('bsln_val')
    bsln_type_loc = bsln_m.columns.get_loc('bsln_type')
    bsln_m = bsln_m.values
    print('Finished calculated baselines.')


# Extract raw SCr data and corresponding masks for each patient individually
if not os.path.exists(os.path.join(dataPath, 'scr_raw.csv')):
    print('Extracting patient SCr data...')
    scrs, dates, tmasks, hd_masks, crrt_masks, pd_masks, rrt_masks = extract_scr_data(icu_windows, hosp_windows,
                                                                                      baseDataPath,
                                                                                      scr_fname='all_scr_data.csv',
                                                                                      rrt_fname='tDialysis.csv',
                                                                                      scr_dcol='SPECIMN_TEN_TIME',
                                                                                      scr_vcol='ORD_VALUE',
                                                                                      id_col='PATIENT_NUM',
                                                                                      rrtSep='row')
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
    (iscrs, idates, idmasks) = extract_masked_data([scrs, dates, rrt_masks], tmasks, sel=2)

    arr2csv(os.path.join(dataPath, 'scr_raw_icu.csv'), iscrs, ids, fmt='%.3f')
    arr2csv(os.path.join(dataPath, 'dates_icu.csv'), idates, ids, fmt='%s')
    arr2csv(os.path.join(dataPath, 'ind_rrt_masks_icu.csv'), idmasks, ids, fmt='%d')

    (hscrs, hdates, hdmasks) = extract_masked_data([scrs, dates, rrt_masks], tmasks, sel=[1, 2])
    arr2csv(os.path.join(dataPath, 'scr_raw_hosp.csv'), hscrs, ids, fmt='%.3f')
    arr2csv(os.path.join(dataPath, 'dates_hosp.csv'), hdates, ids, fmt='%s')
    arr2csv(os.path.join(dataPath, 'ind_rrt_masks_hosp.csv'), hdmasks, ids, fmt='%d')

    scrs = iscrs
    dates = idates
    dmasks = idmasks


    # print('Extracting SCr records in ICU...')
    # if analyze == 'icu':
    #     (scrs, dates, rrt_masks) = extract_masked_data([scrs, dates, rrt_masks], tmasks, sel=2)
    #     # [scrs, dates, rrt_masks] = extract_window_data(ids, [scrs, dates, rrt_masks], dates, icu_windows, 2)
    # if analyze == 'hosp':
    #     (scrs, dates, rrt_masks) = extract_masked_data([scrs, dates, rrt_masks], tmasks, sel=[1, 2])
    #     # [scrs, dates, rrt_masks] = extract_window_data(ids, [scrs, dates, rrt_masks], dates, hosp_windows, 2)
    # arr2csv(os.path.join(dataPath, 'scr_raw_%s.csv' % analyze), scrs, ids, fmt='%.3f')
    # arr2csv(os.path.join(dataPath, 'dates_%s.csv' % analyze), dates, ids, fmt='%s')
    # arr2csv(os.path.join(dataPath, 'ind_rrt_masks_%s.csv' % analyze), rrt_masks, ids, fmt='%d')
else:
    if not os.path.exists(os.path.join(dataPath, 'scr_interp%s.csv' % analyze)):
        print('Loaded ICU data.')
        scrs = load_csv(os.path.join(dataPath, 'scr_raw_%s.csv' % analyze), ids)
        dates = load_csv(os.path.join(dataPath, 'dates_%s.csv' % analyze), ids, 'date')
        rrt_masks = load_csv(os.path.join(dataPath, 'ind_rrt_masks_%s.csv' % analyze), ids, int)

# If computing rolling average, compute on the raw values
if not os.path.exists(os.path.join(dataPath, 'scr_raw_%s_%dptAvg.csv' % (analyze, args.avgpts))) and args.avgpts > 1:
    avgd = rolling_average(scrs, masks=[rrt_masks], times=dates)
    scrs = avgd[0]
    dmasks = avgd[1][0]
    dates = avgd[2]
    arr2csv(os.path.join(dataPath, 'scr_raw_%s_%dptAvg.csv' % (analyze, args.avgpts)), scrs, ids, fmt='%f')
    arr2csv(os.path.join(dataPath, 'dates_%s_%dptAvg.csv' % (analyze, args.avgpts)), dates, ids, fmt='%s')
    arr2csv(os.path.join(dataPath, 'ind_rrt_masks_%s_%dptAvg.csv' % (analyze, args.avgpts)), dmasks, ids, fmt='%d')
else:
    if not os.path.exists(os.path.join(dataPath, 'scr_interp_%s_%dptAvg.csv' % (analyze, args.avgpts))):
        scrs = load_csv(os.path.join(dataPath, 'scr_raw_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids)
        dates = load_csv(os.path.join(dataPath, 'dates_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids, 'date')
        dmasks = load_csv(os.path.join(dataPath, 'ind_rrt_masks_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids, int)

# Interpolate averaged sequence
if not os.path.exists(os.path.join(dataPath, 'scr_interp_%s_%dptAvg.csv' % (analyze, args.avgpts))) and args.avgpts > 1:
    post_interpo, dmasks_interp, days_interp, interp_masks = linear_interpo(scrs, ids, dates, dmasks,
                                                                            icu_windows, tRes)
    arr2csv(os.path.join(dataPath, 'scr_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), post_interpo, ids)
    arr2csv(os.path.join(dataPath, 'days_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), days_interp, ids, fmt='%d')
    arr2csv(os.path.join(dataPath, 'interp_masks_%s_%dptAvg.csv' % (analyze, args.avgpts)), interp_masks, ids,
            fmt='%d')
    arr2csv(os.path.join(dataPath, 'dmasks_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), dmasks_interp, ids,
            fmt='%d')
else:
    if not os.path.exists(os.path.join(dataPath, 'kdigo_%s_%dptAvg.csv' % (analyze, args.avgpts))):
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
    post_interpo = load_csv(os.path.join(dataPath, 'scr_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids)
    dmasks_interp = load_csv(os.path.join(dataPath, 'dmasks_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids)
    days_interp = load_csv(os.path.join(dataPath, 'days_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids)
    interp_masks = load_csv(os.path.join(dataPath, 'interp_masks_%s_%dptAvg.csv' % (analyze, args.avgpts)), ids)

if not os.path.exists(os.path.join(dataPath, 'exclusion_criteria.csv')):
    print('Evaluating exclusion criteria...')
    if males is None:
        males, races, ages, dods, dtds = get_utsw_demographics(ids, hosp_windows, baseDataPath)
    exc, hdr = get_exclusion_criteria_dallas(ids, kdigos, days_interp, dates, icu_windows, males, races, ages,
                                             baseDataPath)
    arr2csv(os.path.join(dataPath, 'exclusion_criteria.csv'), exc, ids, fmt='%d', header=hdr)
    excluded = np.max(exc, axis=1)
    keep = np.where(excluded == 0)[0]
else:
    print('Loaded exclusion criteria.')
    exc = load_csv(os.path.join(dataPath, 'exclusion_criteria.csv'), ids, int, skip_header=True)
    excluded = np.max(exc, axis=1)
    keep = np.where(excluded == 0)[0]

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
        # (ids, kdigos, days, scrs, icu_windows, hosp_windows,
        #                            f, base_path, grp_name='meta', tlim=7, v=False):
        f = summarize_stats_dallas(ids, kdigos, days_interp, post_interpo, icu_windows, hosp_windows,
                                   f, baseDataPath, grp_name='meta_all', tlim=t_lim)
        all_stats = f['meta_all']
        # mk = all_stats['max_kdigo_d03'][:]

        pt_sel = keep
        # pt_sel = np.intersect1d(keep, pt_sel)
        stats = f.create_group('meta')
        for i in range(len(list(all_stats))):
            name = list(all_stats)[i]
            if all_stats[name].ndim == 1:
                stats.create_dataset(name, data=all_stats[name][:][pt_sel], dtype=all_stats[name].dtype)
            elif all_stats[name].ndim == 2:
                stats.create_dataset(name, data=all_stats[name][:][pt_sel, :], dtype=all_stats[name].dtype)
            elif all_stats[name].ndim == 3:
                stats.create_dataset(name, data=all_stats[name][:][pt_sel, :, :], dtype=all_stats[name].dtype)
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
mkd03 = stats['max_kdigo_d03'][:]

# Calculate clinical mortality prediction scores
# For each, returns the normal categorical scores, as well as the raw values used for each
sofa = None
if not os.path.exists(os.path.join(dataPath, 'sofa.csv')):
    print('Getting SOFA scores')
    sofa, sofa_hdr, sofa_raw, sofa_raw_hdr = get_sofa(all_ids, all_stats, post_interpo, days_interp,
                                                      out_name=os.path.join(dataPath, 'sofa.csv'))
    arr2csv(os.path.join(dataPath, 'sofa_raw.csv'), sofa_raw, all_ids, fmt='%.3f', header=sofa_raw_hdr)
    arr2csv(os.path.join(dataPath, 'sofa.csv'), sofa, all_ids, fmt='%d', header=sofa_hdr)
    sofa_hdr = sofa_hdr.split(',')[1:]
    sofa_raw_hdr = sofa_raw_hdr.split(',')[1:]

else:
    sofa, sofa_hdr = load_csv(os.path.join(dataPath, 'sofa.csv'), all_ids, dt=int, skip_header='keep')
    sofa_raw, sofa_raw_hdr = load_csv(os.path.join(dataPath, 'sofa_raw.csv'), all_ids, dt=float, skip_header='keep')

apache = None
if not os.path.exists(os.path.join(dataPath, 'apache.csv')):
    print('Getting APACHE-II Scores')
    apache, apache_hdr, apache_raw, apache_raw_hdr = get_apache(all_ids, all_stats, post_interpo, days_interp,
                                                                out_name=os.path.join(dataPath, 'apache.csv'))
    arr2csv(os.path.join(dataPath, 'apache_raw.csv'), apache_raw, all_ids, fmt='%.3f', header=apache_raw_hdr)
    arr2csv(os.path.join(dataPath, 'apache.csv'), apache, all_ids, fmt='%d', header=apache_hdr)
    apache_hdr = apache_hdr.split(',')[1:]
    apache_raw_hdr = apache_raw_hdr.split(',')[1:]
else:
    apache, apache_hdr = load_csv(os.path.join(dataPath, 'apache.csv'), all_ids, dt=int, skip_header='keep')
    apache_raw, apache_raw_hdr = load_csv(os.path.join(dataPath, 'apache_raw.csv'), all_ids, dt=float, skip_header='keep')

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
cohort_kdigos = load_csv(os.path.join(dataPath, 'kdigo_%s_%dptAvg.csv' % (analyze, args.avgpts)), cohort_ids, int)
cohort_days = load_csv(os.path.join(dataPath, 'days_interp_%s_%dptAvg.csv' % (analyze, args.avgpts)), cohort_ids, int)

if not os.path.exists(os.path.join(resPath, 'features')):
    os.mkdir(os.path.join(resPath, 'features'))

if not os.path.exists(os.path.join(resPath, 'features', 'individual')):
    os.mkdir(os.path.join(resPath, 'features', 'individual'))
    arr2csv(os.path.join(resPath, 'features', 'individual', "max_kdigo_d03.csv"), mkd03, cohort_ids,
            fmt="%d", header="STUDY_PATIENT_ID,MaxKDIGOd03")
    arr2csv(os.path.join(resPath, 'features', 'individual', "max_kdigo_d03_norm.csv"), mkd03 / 4, cohort_ids,
            fmt="%.2f", header="STUDY_PATIENT_ID,MaxKDIGOd03")

if not os.path.isfile(os.path.join(resPath, 'features', 'individual', 'sofa.csv')):
    # Normalize values in SOFA and APACHE scores for classification
    mms = MinMaxScaler()

    sofa = sofa[pt_sel, :]
    sofa_norm = mms.fit_transform(sofa)
    sofa_raw = sofa_raw[pt_sel, :]
    sofa_raw_norm = mms.fit_transform(sofa_raw)

    apache = apache[pt_sel, :]
    apache_norm = mms.fit_transform(apache)
    apache_raw = apache_raw[pt_sel, :]
    apache_raw_norm = mms.fit_transform(apache_raw)

    sa = np.hstack([sofa, apache])
    sa_norm = np.hstack([sofa_norm, apache_norm])

    sa_hdr = 'STUDY_PATIENT_ID,' + ','.join(sofa_hdr) + "," + ','.join(apache_hdr)

    sofa_hdr = 'STUDY_PATIENT_ID,' + ','.join(sofa_hdr)
    apache_hdr = 'STUDY_PATIENT_ID,' + ','.join(apache_hdr)
    sofa_raw_hdr = 'STUDY_PATIENT_ID,' + ','.join(sofa_raw_hdr)
    apache_raw_hdr = 'STUDY_PATIENT_ID,' + ','.join(apache_raw_hdr)

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

    arr2csv(os.path.join(resPath, 'features', 'individual', 'sofa_apache.csv'),
            sa, cohort_ids, fmt='%d', header=sa_hdr)
    arr2csv(os.path.join(resPath, 'features', 'individual', 'sofa_apache_norm.csv'),
            sa_norm, cohort_ids, fmt='%.4f', header=sa_hdr)

if not os.path.exists(os.path.join(dataPath, "make90_disch_30dbuf_2pts.csv")):
    print("Evaluating MAKE-90 outcome variants...")
    f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r+')
    stats = f['meta']
    # for ref in ["admit", "disch"]:
    #     for buf in [0, 30]:
    #         for npts in [1, 2]:
    #             print("Reference: %s\tDay Buffer: %d\t GFR Drop Number of Records: %d" % (ref, buf, npts))
    #             baseName = "make90_%s_%ddbuf_%dpts" % (ref, buf, npts)
    #             outName = os.path.join(dataPath, "%s.csv" % baseName)
    #             # outFile = open(outName, "w")
    #             m90 = get_MAKE90_dallas(cohort_ids, stats, baseDataPath, dataPath, ref=ref, buffer=buf, ct=npts,
    #                                     label=baseName)
    ref = "disch"
    buf = 30
    npts = 2
    print("Reference: %s\tDay Buffer: %d\t GFR Drop Number of Records: %d" % (ref, buf, npts))
    baseName = "make90_%s_%ddbuf_%dpts" % (ref, buf, npts)
    outName = os.path.join(dataPath, "%s.csv" % baseName)
    # outFile = open(outName, "w")
    m90 = get_MAKE90_dallas(cohort_ids, stats, baseDataPath, dataPath, ref=ref, buffer=buf, ct=npts,
                            label=baseName)
    f.close()
