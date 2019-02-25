import os
import h5py
import numpy as np
import pandas as pd
import kdigo_funcs as kf
import stat_funcs as sf
import json
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------------------------------------------#
# Load Configuration
configurationFileName = 'kdigo_conf.json'
fp = open(configurationFileName, 'r')
conf = json.load(fp)
fp.close()

basePath = conf['basePath']
cohortName = conf['cohortName']
t_lim = conf['analysisDays']
tRes = conf['timeResolutionHrs']
v = conf['verbose']
# -----------------------------------------------------------------------------#

dataPath = os.path.join(basePath, 'DATA', 'icu', cohortName)
resPath = os.path.join(basePath, 'RESULTS', 'icu', cohortName)

if not os.path.exists(dataPath):
    os.makedirs(dataPath)
if not os.path.exists(resPath):
    os.makedirs(resPath)

# Load raw data from individual CSV files
((date_m, hosp_locs, icu_locs, adisp_loc,
 surg_m, surg_des_loc,
 diag_m, diag_loc, diag_nb_loc,
 scm_esrd_m, scm_esrd_before, scm_esrd_at, scm_esrd_during, scm_esrd_after,
 dia_m, crrt_locs, hd_locs, pd_locs, hd_trt_loc,
 scr_all_m, scr_date_loc, scr_val_loc, scr_desc_loc,
 dem_m, sex_loc, eth_loc,
 dob_m, birth_loc,
 mort_m, mdate_loc,
 io_m, charl_m, charl_loc, elix_m, elix_loc,
 blood_m, pao2_locs, paco2_locs, ph_locs,
 clinical_oth, resp_locs, fio2_locs, gcs_loc, weight_loc, height_loc,
 clinical_vit, temp_locs, map_locs, cuff_locs, hr_locs,
 labs1_m, bili_loc, pltlt_loc, na_locs, pk_locs, hemat_locs, wbc_locs, hemo_locs, bun_loc,
 labs2_m, alb_loc, lac_loc,
 med_m, med_type, med_name, med_date, med_dur,
 organ_sup_mv, mech_vent_dates, mech_vent_days,
 organ_sup_ecmo, ecmo_dates, ecmo_days,
 organ_sup_iabp, iabp_dates, iabp_days,
 organ_sup_vad, vad_dates, vad_days,
 urine_m, urine_locs,
 smoke_m, former_smoke, current_smoke,
 usrds_esrd_m, usrds_esrd_date_loc,
 esrd_man_rev, man_rev_bef, man_rev_dur, man_rev_rrt)) = kf.load_all_csv(os.path.join(basePath, 'DATA/'))

# Get mask inidicating which points are during dialysis
if not os.path.exists(os.path.join(dataPath, 'rrt_mask.csv')):
    dia_mask = kf.get_dialysis_mask(scr_all_m, scr_date_loc, dia_m, crrt_locs, hd_locs, pd_locs)
    np.savetxt(os.path.join(dataPath, 'rrt_mask.csv'), dia_mask, delimiter=',', fmt='%d')
else:
    print('Loaded previous dialysis mask...')
    dia_mask = np.loadtxt(os.path.join(dataPath, 'rrt_mask.csv'), dtype=int)

# Get mask indicating whether each point was in hospital or ICU
if not os.path.exists(os.path.join(dataPath, 'icu_mask.csv')):
    icu_mask, icu_windows = kf.get_t_mask(scr_all_m, scr_date_loc, date_m, icu_locs)
    icu_window_l = []
    np.savetxt(os.path.join(dataPath, 'icu_mask.csv'), icu_mask, delimiter=',', fmt='%d')
    ids = sorted(list(icu_windows))
    for tid in ids:
        icu_window_l.append((str(icu_windows[tid][0]), str(icu_windows[tid][1])))
    kf.arr2csv(dataPath + 'icu_admit_discharge.csv', icu_window_l, ids, fmt='%s')
else:
    print('Loaded previous ICU masks...')
    icu_mask = np.loadtxt(os.path.join(dataPath, 'icu_mask.csv'), dtype=int)
    window_l, ids = kf.load_csv(os.path.join(dataPath, 'icu_admit_discharge.csv'), ids=None, dt='date')
    icu_windows = {}
    for i in range(len(ids)):
        tid = ids[i]
        icu_windows[ids[i]] = window_l[i]

if not os.path.exists(os.path.join(dataPath, 'hosp_mask.csv')):
    hosp_mask, hosp_windows = kf.get_t_mask(scr_all_m, scr_date_loc, date_m, hosp_locs)
    np.savetxt(os.path.join(dataPath, 'hosp_mask.csv'), hosp_mask, delimiter=',', fmt='%d')
    hosp_window_l = []
    ids = sorted(list(hosp_windows))
    for tid in ids:
        hosp_window_l.append((str(hosp_windows[tid][0]), str(hosp_windows[tid][1])))
    kf.arr2csv(dataPath + 'hosp_admit_discharge.csv', hosp_window_l, ids, fmt='%s')
else:
    print('Loaded previous hospitalilzation masks...')
    hosp_mask = np.loadtxt(os.path.join(dataPath, 'hosp_mask.csv'), dtype=int)
    window_l, ids = kf.load_csv(os.path.join(dataPath, 'hosp_admit_discharge.csv'), ids=None, dt=str)
    hosp_windows = {}
    for i in range(len(ids)):
        tid = ids[i]
        hosp_windows[ids[i]] = window_l[i]

bslnFilename = os.path.join(dataPath, 'all_baseline_info.csv')
# Baselines
if os.path.exists(bslnFilename):
    bsln_m = pd.read_csv(bslnFilename)
    bsln_scr_loc = bsln_m.columns.get_loc('bsln_val')
    bsln_type_loc = bsln_m.columns.get_loc('bsln_type')
    bsln_m = bsln_m.values
    print('Loaded previously determined baselines.')
else:
    print('Determining all patient baseline SCr.')
    genders, races, ages = sf.get_uky_demographics(ids, hosp_windows, dem_m, sex_loc, eth_loc, dob_m, birth_loc)
    kf.get_baselines(ids, hosp_windows, scr_all_m, scr_val_loc, scr_date_loc, scr_desc_loc,
                     genders, races, ages, bslnFilename)

    bsln_m = pd.read_csv(bslnFilename)
    bsln_scr_loc = bsln_m.columns.get_loc('bsln_val')
    bsln_type_loc = bsln_m.columns.get_loc('bsln_type')
    bsln_m = bsln_m.values
    print('Finished calculated baselines.', bslnFilename)

if not os.path.exists(os.path.join(dataPath, 'scr_raw.csv')):
    count_log = open(dataPath + 'patient_summary.csv', 'w')
    exc_log = open(dataPath + 'excluded_patients.csv', 'w')
    # Extract patients into separate list elements
    (ids, scrs, dates, days, masks, dmasks, bsln_scr,
     bsln_gfr, btypes, d_disp, t_range, ages) = kf.get_patients(scr_all_m, scr_val_loc, scr_date_loc, adisp_loc,
                                                                icu_mask, dia_mask, icu_windows,
                                                                diag_m, diag_loc,
                                                                scm_esrd_m, scm_esrd_before, scm_esrd_at,
                                                                esrd_man_rev, man_rev_bef,
                                                                bsln_m, bsln_scr_loc, bsln_type_loc,
                                                                date_m,
                                                                surg_m, surg_des_loc,
                                                                dem_m, sex_loc, eth_loc,
                                                                dob_m, birth_loc,
                                                                mort_m, mdate_loc,
                                                                count_log, exc_log)
    count_log.close()
    exc_log.close()
    kf.arr2csv(dataPath + 'scr_raw.csv', scrs, ids, fmt='%.3f')
    kf.arr2csv(dataPath + 'dates.csv', dates, ids, fmt='%s')
    kf.arr2csv(dataPath + 'dialysis.csv', dmasks, ids, fmt='%d')
    kf.arr2csv(dataPath + 'days.csv', days, ids, fmt='%d')
    kf.arr2csv(dataPath + 'baselines.csv', bsln_scr, ids, fmt='%.3f')
    kf.arr2csv(dataPath + 'baseline_gfr.csv', bsln_gfr, ids, fmt='%.3f')
    kf.arr2csv(dataPath + 'baseline_types.csv', bsln_gfr, ids, fmt='%s')
    kf.arr2csv(dataPath + 'time_ranges.csv', t_range, ids, fmt='%.3f')
    kf.arr2csv(dataPath + 'ages.csv', ages, ids, fmt='%.2f')
    kf.arr2csv(dataPath + 'disch_disp.csv', d_disp, ids, fmt='%s')
    np.savetxt(os.path.join(dataPath, 'valid_ids.csv'), ids, fmt='%d')

else:
    ids = np.loadtxt(os.path.join(dataPath, 'valid_ids.csv'), dtype=int)
    scrs = kf.load_csv(os.path.join(dataPath, 'scr_raw.csv'), ids, float)
    dates = kf.load_csv(os.path.join(dataPath, 'dates.csv'), ids, str)
    dmasks = kf.load_csv(os.path.join(dataPath, 'dialysis.csv'), ids, int)

if not os.path.exists(os.path.join(dataPath, 'scr_interp.csv')):
    # Interpolate missing values
    print('Interpolating missing values')
    scrs, dmasks, days, interp_masks, ids = kf.linear_interpo(scrs, ids, dates, dmasks, tRes)
    kf.arr2csv(dataPath + 'scr_interp.csv', scrs, ids)
    kf.arr2csv(dataPath + 'days_interp.csv', days, ids, fmt='%d')
    kf.arr2csv(dataPath + 'interp_masks.csv', interp_masks, ids, fmt='%d')
    kf.arr2csv(dataPath + 'dmasks_interp.csv', dmasks, ids, fmt='%d')
    np.savetxt(dataPath + 'post_interp_ids.csv', ids, fmt='%d')
else:
    print('Loaded previously interpolated SCrs')
    ids = np.loadtxt(os.path.join(dataPath, 'post_interp_ids.csv'), dtype=int)
    scrs = kf.load_csv(os.path.join(dataPath, 'scr_interp.csv'), ids, float)
    dmasks = kf.load_csv(os.path.join(dataPath, 'dmasks_interp.csv'), ids, int)
    days = kf.load_csv(os.path.join(dataPath, 'days_interp.csv'), ids, int)
    interp_masks = kf.load_csv(os.path.join(dataPath, 'interp_masks.csv'), ids, int)
    
if not os.path.exists(os.path.join(dataPath, 'kdigo.csv')):
    # Convert SCr to KDIGO
    bsln_scr = kf.load_csv(dataPath + 'baselines.csv', ids, idxs=1)
    print('Converting to KDIGO')
    kdigos = kf.scr2kdigo(scrs, bsln_scr, dmasks, days, interp_masks)
    kf.arr2csv(dataPath + 'kdigo.csv', kdigos, ids, fmt='%d')
else:
    print('Loaded KDIGO scores')
    kdigos = kf.load_csv(os.path.join(dataPath, 'kdigo.csv'), ids, int)

if not os.path.exists(os.path.join(resPath, 'stats.h5')):
    icu_windows = kf.load_csv(os.path.join(dataPath, 'icu_admit_discharge.csv'), ids, 'date')
    hosp_windows = kf.load_csv(os.path.join(dataPath, 'hosp_admit_discharge.csv'), ids, 'date')
    f = sf.summarize_stats(ids, kdigos, days, scrs, icu_windows, hosp_windows,
                           dem_m, sex_loc, eth_loc,
                           dob_m, birth_loc,
                           diag_m, diag_loc,
                           charl_m, charl_loc, elix_m, elix_loc,
                           organ_sup_mv, mech_vent_dates,
                           organ_sup_ecmo, ecmo_dates,
                           organ_sup_iabp, iabp_dates,
                           organ_sup_vad, vad_dates,
                           io_m,
                           mort_m, mdate_loc,
                           clinical_vit, map_locs, cuff_locs, temp_locs, hr_locs,
                           clinical_oth, height_loc, weight_loc, fio2_locs, gcs_loc, resp_locs,
                           dia_m, crrt_locs, hd_locs, hd_trt_loc,
                           med_m, med_type, med_date, med_name, med_dur,
                           labs1_m, hemat_locs, hemo_locs, bili_loc, bun_loc, pltlt_loc, na_locs, pk_locs, wbc_locs,
                           labs2_m, alb_loc, lac_loc,
                           blood_m, ph_locs, pao2_locs, paco2_locs,
                           urine_m, urine_locs,
                           smoke_m, former_smoke, current_smoke,
                           os.path.join(resPath, 'stats.h5'), dataPath, grp_name='meta_all', tlim=7)
    all_stats = f['meta_all']
    max_kdigo = all_stats['max_kdigo_7d'][:]
    dtd = all_stats['days_to_death']
    aki_sel = np.where(max_kdigo > 0)[0]
    dtd_sel = np.union1d(np.where(np.isnan(dtd))[0], np.where(dtd >= 2)[0])
    pt_sel = np.intersect1d(aki_sel, dtd_sel)
    stats = f.create_group('meta')
    print('Copying without all KDIGO 0 and patients who died in first %d days' % t_lim)
    for i in range(len(list(all_stats))):
        name = list(all_stats)[i]
        stats.create_dataset(name, data=all_stats[name][:][pt_sel], dtype=all_stats[name].dtype)
else:
    f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r+')

all_stats = f['meta_all']
all_ids = all_stats['ids'][:]
stats = f['meta']
ids = stats['ids'][:]
pt_sel = np.array([x in ids for x in all_ids])

# Calculate clinical mortality prediction scores
sofa = None
if not os.path.exists(dataPath + 'sofa.csv'):
    print('Getting SOFA scores')
    sofa = sf.get_sofa(all_ids, all_stats, scrs, days, out_name=os.path.join(dataPath, 'sofa.csv'))
else:
    sofa = kf.load_csv(dataPath + 'sofa.csv', ids, dt=int)
    sofa = np.array(sofa)

apache = None
if not os.path.exists(dataPath + 'apache.csv'):
    print('Getting APACHE-II Scores')
    apache = sf.get_apache(all_ids, all_stats, scrs, days, out_name=os.path.join(dataPath, 'apache.csv'))
else:
    apache = kf.load_csv(dataPath + 'apache.csv', ids, dt=int)
    apache = np.array(apache)

if 'sofa' not in list(all_stats):
    sofa_sum = np.sum(sofa, axis=1)
    all_stats.create_dataset('sofa', data=sofa_sum, dtype=int)
    stats.create_dataset('sofa', data=sofa_sum[pt_sel], dtype=int)
if 'apache' not in list(all_stats):
    apache_sum = np.sum(apache, axis=1)
    all_stats.create_dataset('apache', data=apache_sum, dtype=int)
    stats.create_dataset('apache', data=apache_sum[pt_sel], dtype=int)

f.close()

kdigos = kf.load_csv(os.path.join(dataPath, 'kdigo.csv'), ids, int)
days = kf.load_csv(os.path.join(dataPath, 'days_interp.csv'), ids, int)

# Calculate individual trajectory based features if not already done
if not os.path.exists(os.path.join(resPath, 'features')):
    os.mkdir(os.path.join(resPath, 'features'))

mms = MinMaxScaler()
if not os.path.exists(os.path.join(resPath, 'features', 'individual')):
    os.mkdir(os.path.join(resPath, 'features', 'individual'))
    desc, desc_hdr = kf.descriptive_trajectory_features(kdigos, ids, days=days, t_lim=t_lim,
                                                        filename=os.path.join(resPath, 'features', 'individual',
                                                                              'descriptive_features.csv'))

    slope, slope_hdr = kf.slope_trajectory_features(kdigos, ids, days=days, t_lim=t_lim,
                                                    filename=os.path.join(resPath, 'features', 'individual',
                                                                          'slope_features.csv'))
    slope_norm = mms.fit_transform(slope)
    kf.arr2csv(os.path.join(resPath, 'features', 'individual', 'slope_norm.csv'), slope_norm, ids)

    temp, temp_hdr = kf.template_trajectory_features(kdigos, ids, days=days, t_lim=t_lim,
                                                     filename=os.path.join(resPath, 'features', 'individual',
                                                                           'template_features.csv'))
    temp_norm = kf.normalize_features(temp)
    kf.arr2csv(os.path.join(resPath, 'features', 'individual', 'template_norm.csv'), temp_norm, ids)

    all_traj_ind = np.hstack((desc, slope_norm, temp_norm))
    kf.arr2csv(os.path.join(resPath, 'features', 'individual', 'all_trajectory.csv'),
               all_traj_ind, ids, fmt='%.4f')
else:
    desc = kf.load_csv(os.path.join(resPath, 'features', 'individual', 'descriptive_features.csv'), ids,
                       skip_header=True)
    temp_norm = kf.load_csv(os.path.join(resPath, 'features', 'individual', 'template_norm.csv'), ids,
                            skip_header=False)
    slope_norm = kf.load_csv(os.path.join(resPath, 'features', 'individual', 'slope_norm.csv'), ids,
                             skip_header=False)
    all_traj_ind = np.hstack((desc, slope_norm, temp_norm))

sofa = sofa[pt_sel, :]
sofa_norm = kf.normalize_features(sofa)

apache = np.array(apache[pt_sel, :])
apache_norm = kf.normalize_features(apache)

if not os.path.isfile(os.path.join(resPath, 'features', 'individual', 'sofa.csv')):
    kf.arr2csv(os.path.join(resPath, 'features', 'individual', 'sofa.csv'),
               sofa, ids, fmt='%d')
    kf.arr2csv(os.path.join(resPath, 'features', 'individual', 'sofa_norm.csv'),
               sofa_norm, ids, fmt='%.4f')
    kf.arr2csv(os.path.join(resPath, 'features', 'individual', 'apache.csv'),
               apache, ids, fmt='%d')
    kf.arr2csv(os.path.join(resPath, 'features', 'individual', 'apache_norm.csv'),
               apache_norm, ids, fmt='%.4f')
    all_clin = np.hstack((sofa_norm, apache_norm))
    kf.arr2csv(os.path.join(resPath, 'features', 'individual', 'all_clinical.csv'),
               all_clin, ids, fmt='%.4f')
else:
    all_clin = kf.load_csv(os.path.join(resPath, 'features', 'individual', 'all_clinical.csv'), ids)

if not os.path.exists(os.path.join(resPath, 'features', 'individual', 'everything.csv')):
    everything_ind = np.hstack((all_clin, all_traj_ind))
    kf.arr2csv(os.path.join(resPath, 'features', 'individual', 'everything.csv'),
               everything_ind, ids, fmt='%.4f')

print('Ready for distance matrix calculation. Please run script \'calc_dms.py\'')
