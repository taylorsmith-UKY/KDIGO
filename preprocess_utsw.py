# %%
import os
import h5py
import numpy as np
import pandas as pd
import kdigo_funcs as kf
import stat_funcs as sf
from sklearn.preprocessing import MinMaxScaler

# %%
# ------------------------------- PARAMETERS ----------------------------------#
basePath = "/Volumes/GoogleDrive/My Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/"

dataPath = os.path.join(basePath, 'DATA', 'dallas', 'icu', '7days_052519/')
resPath = os.path.join(basePath, 'RESULTS', 'dallas', 'icu', '7days_052519/')

baseline_file = os.path.join(dataPath, 'all_baseline_info.csv')

t_lim = 7  # in days
timescale = 6  # in hours
v = True
# -----------------------------------------------------------------------------#
# %%
if not os.path.exists(dataPath):
    os.makedirs(dataPath)
if not os.path.exists(resPath):
    os.makedirs(resPath)

# Load raw data from individual CSV files
((date_m, hosp_locs, icu_locs,
  esrd_m, esrd_bef_loc, esrd_during_loc, esrd_after_loc, esrd_date_loc,
  scr_all_m, scr_date_loc, scr_val_loc, scr_ip_loc, rrt_locs,
  dem_m, sex_loc, eth_loc, dob_loc, dod_locs,
  lab_m, lab_col, lab_day, lab_min, lab_max,
  flw_m, flw_col, flw_day, flw_min, flw_max, flw_sum,
  medications, amino_loc, nsaid_loc, acei_loc, arb_loc, press_loc,
  organ_sup, mech_vent_dates, iabp_dates, ecmo_dates, vad_dates,
  diag_m, icd9_code_loc, icd10_code_loc, diag_desc_loc,
  rrt_m, rrt_start_loc, rrt_stop_loc, rrt_type_loc,
  hosp_m, hosp_icu_locs, hosp_hosp_locs,
  usrds_m, usrds_mort_loc, usrds_esrd_loc)) = kf.load_all_csv_dallas(os.path.join(basePath, 'DATA', 'dallas/'))

# %%
ids = np.unique(date_m[:, 0]).astype(int)
# Get mask inidicating which points are during dialysis
if not os.path.exists(os.path.join(dataPath, 'rrt_mask.csv')):
    rrt_mask = kf.get_dialysis_mask_dallas(scr_all_m, scr_date_loc, rrt_m, rrt_start_loc, rrt_stop_loc)
    np.savetxt(os.path.join(dataPath, 'rrt_mask.csv'), rrt_mask, delimiter=',', fmt='%d')
else:
    print('Loaded previous dialysis mask...')
    rrt_mask = np.loadtxt(os.path.join(dataPath, 'rrt_mask.csv'), dtype=int)

# Get mask indicating whether each point was in hospital or ICU
if not os.path.exists(os.path.join(dataPath, 'icu_mask.csv')):
    icu_mask, icu_windows = kf.get_t_mask(scr_all_m, scr_date_loc, date_m, icu_locs)
    icu_window_l = []
    np.savetxt(os.path.join(dataPath, 'icu_mask.csv'), icu_mask, delimiter=',', fmt='%d')
    ids = np.sort(np.array(list(icu_windows), dtype=int))
    for tid in ids:
        icu_window_l.append((str(icu_windows[tid][0]), str(icu_windows[tid][1])))
    kf.arr2csv(dataPath + 'icu_admit_discharge.csv', icu_window_l, ids, fmt='%s')
else:
    print('Loaded previous ICU masks...')
    icu_mask = np.loadtxt(os.path.join(dataPath, 'icu_mask.csv'), dtype=int)
    icu_window_l, wids = kf.load_csv(os.path.join(dataPath, 'icu_admit_discharge.csv'), ids=None, dt='date')
    icu_windows = {}
    for i in range(len(wids)):
        tid = wids[i]
        icu_windows[wids[i]] = icu_window_l[i]

if not os.path.exists(os.path.join(dataPath, 'hosp_mask.csv')):
    hosp_mask, hosp_windows = kf.get_t_mask(scr_all_m, scr_date_loc, date_m, hosp_locs)
    np.savetxt(os.path.join(dataPath, 'hosp_mask.csv'), hosp_mask, delimiter=',', fmt='%d')
    hosp_window_l = []
    ids = np.sort(np.array(list(hosp_windows), dtype=int))
    for tid in ids:
        hosp_window_l.append((str(hosp_windows[tid][0]), str(hosp_windows[tid][1])))
    kf.arr2csv(dataPath + 'hosp_admit_discharge.csv', hosp_window_l, ids, fmt='%s')
else:
    print('Loaded previous hospitalilzation masks...')
    hosp_mask = np.loadtxt(os.path.join(dataPath, 'hosp_mask.csv'), dtype=int)
    hosp_window_l, wids = kf.load_csv(os.path.join(dataPath, 'hosp_admit_discharge.csv'), ids=None, dt=str)
    hosp_windows = {}
    for i in range(len(wids)):
        tid = wids[i]
        hosp_windows[wids[i]] = hosp_window_l[i]

# %%
ids = np.sort(np.array(list(hosp_windows), dtype=int))
# Baselines
if os.path.exists(baseline_file):
    bsln_m = pd.read_csv(baseline_file)
    bsln_scr_loc = bsln_m.columns.get_loc('bsln_val')
    bsln_type_loc = bsln_m.columns.get_loc('bsln_type')
    bsln_m = bsln_m.values
    print('Loaded previously determined baselines.')
else:
    print('Determining all patient baseline SCr.')
    admits = []
    for tid in ids:
        admits.append(hosp_windows[tid][0])
    genders, races, ages, dods, dtds = sf.get_utsw_demographics(ids, dem_m, dob_loc, sex_loc, eth_loc, dod_locs, hosp_windows)
    kf.get_baselines_dallas(ids, hosp_windows, scr_all_m, scr_val_loc, scr_date_loc, scr_ip_loc,
                            genders, races, ages, baseline_file, outp_rng=(1, 365), inp_rng=(7, 365))

    bsln_m = pd.read_csv(baseline_file)
    bsln_scr_loc = bsln_m.columns.get_loc('bsln_val')
    bsln_type_loc = bsln_m.columns.get_loc('bsln_type')
    bsln_m = bsln_m.values
    print('Finished calculated baselines.', baseline_file)

 # %%
if not os.path.exists(os.path.join(dataPath, 'scr_raw.csv')):
    count_log = open(dataPath + 'patient_summary.csv', 'w')
    exc_log = open(dataPath + 'excluded_patients.csv', 'w')
    exc_list = open(dataPath + 'primary_exclusions.csv', 'w')
    # Extract patients into separate list elements
    (ids, scrs, dates, days, masks, dmasks, baselines, bsln_gfr,
     btypes, d_disp, t_range, ages) = kf.get_patients_dallas(scr_all_m, scr_val_loc, scr_date_loc, icu_mask, rrt_mask,
                                                             icu_windows, hosp_windows, diag_m,
                                                             icd9_code_loc, icd10_code_loc, diag_desc_loc,
                                                             esrd_m, esrd_bef_loc, bsln_m, bsln_scr_loc, bsln_type_loc,
                                                             date_m,
                                                             dem_m, sex_loc, eth_loc, dob_loc, dod_locs,
                                                             usrds_m, usrds_mort_loc, usrds_esrd_loc,
                                                             count_log, exc_log, exc_list)

    count_log.close()
    exc_log.close()
    exc_list.close()
    kf.arr2csv(dataPath + 'scr_raw.csv', scrs, ids, fmt='%.3f')
    kf.arr2csv(dataPath + 'dates.csv', dates, ids, fmt='%s')
    kf.arr2csv(dataPath + 'dialysis.csv', dmasks, ids, fmt='%d')
    kf.arr2csv(dataPath + 'days.csv', days, ids, fmt='%d')
    kf.arr2csv(dataPath + 'baselines.csv', baselines, ids, fmt='%.3f')
    kf.arr2csv(dataPath + 'baseline_types.csv', btypes, ids, fmt='%s')
    kf.arr2csv(dataPath + 'baseline_gfr.csv', bsln_gfr, ids, fmt='%.3f')
    kf.arr2csv(dataPath + 'time_ranges.csv', t_range, ids, fmt='%.3f')
    kf.arr2csv(dataPath + 'ages.csv', ages, ids, fmt='%.2f')
    kf.arr2csv(dataPath + 'disch_disp.csv', d_disp, ids, fmt='%s')
    np.savetxt(os.path.join(dataPath, 'valid_ids.csv'), ids, fmt='%d')
    kf.arr2csv(dataPath + 'baseline_types.csv', btypes, ids, fmt='%s')

    # Get the ICU and Hospital admit/discharge dates for the patients kept and save them
    icu_window_l = []
    hosp_window_l = []
    for tid in ids:
        icu_window_l.append(icu_windows[tid])
        hosp_window_l.append(hosp_windows[tid])
    icu_windows = np.array(icu_window_l, dtype=str)
    hosp_windows = np.array(hosp_window_l, dtype=str)
else:
    ids = np.loadtxt(os.path.join(dataPath, 'valid_ids.csv'), dtype=int)
    scrs = kf.load_csv(os.path.join(dataPath, 'scr_raw.csv'), ids, float)
    dates = kf.load_csv(os.path.join(dataPath, 'dates.csv'), ids, str)
    dmasks = kf.load_csv(os.path.join(dataPath, 'dialysis.csv'), ids, int)
    icu_windows = kf.load_csv(os.path.join(dataPath, 'icu_admit_discharge.csv'), ids, 'date')
    hosp_windows = kf.load_csv(os.path.join(dataPath, 'hosp_admit_discharge.csv'), ids, 'date')

# %%
if not os.path.exists(os.path.join(dataPath, 'scr_interp.csv')):
    # Interpolate missing values
    print('Interpolating missing values')
    post_interpo, dmasks_interp, days_interp, interp_masks, ids = kf.linear_interpo(scrs, ids, dates,
                                                                                    dmasks, timescale)
    kf.arr2csv(dataPath + 'scr_interp.csv', post_interpo, ids)
    kf.arr2csv(dataPath + 'days_interp.csv', days_interp, ids, fmt='%d')
    kf.arr2csv(dataPath + 'interp_masks.csv', interp_masks, ids, fmt='%d')
    kf.arr2csv(dataPath + 'dmasks_interp.csv', dmasks_interp, ids, fmt='%d')
    np.savetxt(dataPath + 'post_interp_ids.csv', ids, fmt='%d')
else:
    print('Loaded previously interpolated SCrs')
    ids = np.loadtxt(os.path.join(dataPath, 'post_interp_ids.csv'), dtype=int)
    post_interpo = kf.load_csv(os.path.join(dataPath, 'scr_interp.csv'), ids, float)
    dmasks_interp = kf.load_csv(os.path.join(dataPath, 'dmasks_interp.csv'), ids, int)
    days_interp = kf.load_csv(os.path.join(dataPath, 'days_interp.csv'), ids, int)
    interp_masks = kf.load_csv(os.path.join(dataPath, 'interp_masks.csv'), ids, int)

if not os.path.exists(os.path.join(dataPath, 'kdigo.csv')):
    # Convert SCr to KDIGO
    baselines = kf.load_csv(dataPath + 'baselines.csv', ids, idxs=[1, ])
    print('Converting to KDIGO')
    kdigos = kf.scr2kdigo(post_interpo, baselines, dmasks_interp, days_interp, interp_masks)
    kf.arr2csv(dataPath + 'kdigo.csv', kdigos, ids, fmt='%d')
else:
    print('Loaded KDIGO scores')
    kdigos = kf.load_csv(os.path.join(dataPath, 'kdigo.csv'), ids, int)

if not os.path.exists(os.path.join(resPath, 'stats.h5')):
    all_ids = np.loadtxt(os.path.join(dataPath, 'valid_ids.csv'), dtype=int)
    print('Summarizing all patient stats')
    pt_sel = np.array([x in ids for x in all_ids])
    icu_windows = icu_windows[pt_sel]
    hosp_windows = hosp_windows[pt_sel]
    f = sf.summarize_stats_dallas(ids, kdigos, days_interp, post_interpo, icu_windows, hosp_windows,
                                  dem_m, sex_loc, eth_loc, dob_loc, dod_locs,
                                  organ_sup, mech_vent_dates, iabp_dates, ecmo_dates, vad_dates,
                                  flw_m, flw_col, flw_day, flw_min, flw_max, flw_sum,
                                  lab_m, lab_col, lab_day, lab_min, lab_max,
                                  date_m, hosp_locs, icu_locs,
                                  os.path.join(resPath, 'stats.h5'), dataPath,
                                  os.path.join(basePath, 'DATA', 'dallas', 'csv'), grp_name='meta_all', tlim=7)

    all_stats = f['meta_all']
    max_kdigo = all_stats['max_kdigo_7d'][:]
    pt_sel = np.where(max_kdigo > 0)[0]
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
    sofa, sofa_hdr, sofa_raw, sofa_raw_hdr = sf.get_sofa(all_ids, all_stats, post_interpo, days_interp, out_name=os.path.join(dataPath, 'sofa.csv'))
    kf.arr2csv(os.path.join(dataPath, 'sofa_raw.csv'), sofa_raw, all_ids, fmt='%.3f', header=sofa_raw_hdr)
else:
    sofa, sofa_hdr = kf.load_csv(dataPath + 'sofa.csv', all_ids, dt=int, skip_header='keep')
    sofa_raw = kf.load_csv(dataPath + 'sofa_raw.csv', all_ids, dt=float, skip_header=True)

apache = None
if not os.path.exists(dataPath + 'apache.csv'):
    print('Getting APACHE-II Scores')
    apache, apache_hdr, apache_raw, apache_raw_hdr = sf.get_apache(all_ids, all_stats, post_interpo, days_interp, out_name=os.path.join(dataPath, 'apache.csv'))
    kf.arr2csv(os.path.join(dataPath, 'apache_raw.csv'), apache_raw, all_ids, fmt='%.3f', header=apache_raw_hdr)
else:
    apache, apache_hdr = kf.load_csv(dataPath + 'apache.csv', all_ids, dt=int, skip_header='keep')
    apache_raw = kf.load_csv(dataPath + 'apache_raw.csv', all_ids, dt=float, skip_header=True)

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
aki_ids = ids
aki_kdigos = kf.load_csv(os.path.join(dataPath, 'kdigo.csv'), ids, int)
aki_days = kf.load_csv(os.path.join(dataPath, 'days_interp.csv'), ids, int)
os.mkdir(os.path.join(resPath, 'clusters'))
os.mkdir(os.path.join(resPath, 'clusters', 'died_inp'))
died = f['meta']['died_inp'][:]
died[np.where(died)] = 1
kf.arr2csv(os.path.join(resPath, 'clusters', 'died_inp', 'clusters.csv'), died, ids, fmt='%d')
sf.formatted_stats(f['meta'], os.path.join(resPath, 'clusters', 'died_inp'))
# assert len(kdigos) == len(pt_sel)
# for i in range(len(kdigos)):
#     if pt_sel[i]:
#         aki_kdigos.append(kdigos[i])
#         aki_days.append(days_interp[i])
# assert len(aki_kdigos) == len(ids)

# Calculate individual trajectory based features if not already done
if not os.path.exists(os.path.join(resPath, 'features')):
    os.mkdir(os.path.join(resPath, 'features'))

mms = MinMaxScaler()
if not os.path.exists(os.path.join(resPath, 'features', 'individual')):
    os.mkdir(os.path.join(resPath, 'features', 'individual'))
    desc = kf.descriptive_trajectory_features(aki_kdigos, aki_ids, days=aki_days, t_lim=t_lim,
                                              filename=os.path.join(resPath, 'features', 'individual',
                                                                    'descriptive_features.csv'))
    new_desc, new_desc_hdr = kf.new_descriptive_trajectory_features(aki_kdigos, aki_ids, days=aki_days, t_lim=t_lim,
                                                                    filename=os.path.join(resPath, 'features',
                                                                                          'individual',
                                                                                          'new_descriptive_features.csv'))
    new_desc_norm = mms.fit_transform(new_desc)
    kf.arr2csv(os.path.join(resPath, 'features', 'individual', 'new_descriptive_norm.csv'), new_desc_norm, aki_ids,
               fmt='%.3f', header=new_desc_hdr)

    # slope = kf.slope_trajectory_features(aki_kdigos, aki_ids, days=aki_days, t_lim=t_lim,
    #                                      filename=os.path.join(resPath, 'features', 'individual', 'slope_features.csv'))
    # slope_norm = kf.normalize_features(slope)
    # kf.arr2csv(os.path.join(resPath, 'features', 'individual', 'slope_norm.csv'), slope_norm, aki_ids)
    #
    # temp = kf.template_trajectory_features(aki_kdigos, aki_ids, days=aki_days, t_lim=t_lim,
    #                                        filename=os.path.join(resPath, 'features', 'individual',
    #                                                              'template_features.csv'))
    # temp_norm = kf.normalize_features(temp)
    # kf.arr2csv(os.path.join(resPath, 'features', 'individual', 'template_norm.csv'), temp_norm, aki_ids)
    #
    # all_traj_ind = np.hstack((desc, slope_norm, temp_norm))
    # kf.arr2csv(os.path.join(resPath, 'features', 'individual', 'all_trajectory.csv'),
    #            all_traj_ind, aki_ids, fmt='%.4f')
else:
    desc = kf.load_csv(os.path.join(resPath, 'features', 'individual', 'descriptive_features.csv'), aki_ids,
                       skip_header=True)
    # temp_norm = kf.load_csv(os.path.join(resPath, 'features', 'individual', 'template_norm.csv'), aki_ids,
    #                         skip_header=False)
    # slope_norm = kf.load_csv(os.path.join(resPath, 'features', 'individual', 'slope_norm.csv'), aki_ids,
    #                          skip_header=False)
    # all_traj_ind = np.hstack((desc, slope_norm, temp_norm))

sofa = sofa[pt_sel, :]
sofa_norm = mms.fit_transform(sofa)

apache = np.array(apache[pt_sel, :])
apache_norm = mms.fit_transform(apache)

sofa_hdr = 'STUDY_PATIENT_ID,' + ','.join(sofa_hdr)
apache_hdr = 'STUDY_PATIENT_ID,' + ','.join(apache_hdr)

if not os.path.isfile(os.path.join(resPath, 'features', 'individual', 'sofa.csv')):
    kf.arr2csv(os.path.join(resPath, 'features', 'individual', 'sofa.csv'),
               sofa, aki_ids, fmt='%d', header=sofa_hdr)
    kf.arr2csv(os.path.join(resPath, 'features', 'individual', 'sofa_norm.csv'),
               sofa_norm, aki_ids, fmt='%.4f', header=sofa_hdr)
    kf.arr2csv(os.path.join(resPath, 'features', 'individual', 'apache.csv'),
               apache, aki_ids, fmt='%d', header=apache_hdr)
    kf.arr2csv(os.path.join(resPath, 'features', 'individual', 'apache_norm.csv'),
               apache_norm, aki_ids, fmt='%.4f', header=apache_hdr)
    # all_clin = np.hstack((sofa_norm, apache_norm))
    # kf.arr2csv(os.path.join(resPath, 'features', 'individual', 'all_clinical.csv'),
    #            all_clin, aki_ids, fmt='%.4f')
# else:
    # all_clin = kf.load_csv(os.path.join(resPath, 'features', 'individual', 'all_clinical.csv'), aki_ids)

# if not os.path.exists(os.path.join(resPath, 'features', 'individual', 'everything.csv')):
#     everything_ind = np.hstack((all_clin, all_traj_ind))
#     kf.arr2csv(os.path.join(resPath, 'features', 'individual', 'everything.csv'),
#                everything_ind, aki_ids, fmt='%.4f')

fname = os.path.join(dataPath, 'make90.csv')
m90_out = open(fname, 'w')
datapath = os.path.join(basePath, 'DATA', 'dallas', 'csv')
m90, mids = sf.get_MAKE90_dallas(ids, f['meta'], datapath, m90_out, buffer=30, ct=2)

print('Ready for distance matrix calculation. Please run script \'calc_dms.py\'')
