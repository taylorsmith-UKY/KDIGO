import os
import h5py
import numpy as np
import pandas as pd
import kdigo_funcs as kf
import stat_funcs as sf
from cluster_funcs import assign_feature_vectors, dist_cut_cluster
import datetime

# ------------------------------- PARAMETERS ----------------------------------#
basePath = "../"
t_analyze = 'ICU'
xl_file = "KDIGO_full.xlsx"
timescale = 6  # in hours
id_ref = 'icu_valid_ids.csv'  # specify different file with subset of IDs if desired
incl_0 = False
h5_name = 'stats.h5'
folder_name = '/7days_071118/'
alpha = 1.0
transition_costs = [1.5,    # [0 - 1]
                    1.8,    # [1 - 2]
                    2.75,   # [2 - 3]
                    4.25]   # [3 - 4]

use_dic_dtw = False
use_dic_dist = False
# Dictionary explanation:
# cost(0, 1) = 1
# cost(1, 2) = 2 + 1 = 3
# cost(2, 3) = 6 + 3 = 9
# cost(3, 4) = 24 + 24 = 33
# order is reversed so the bray-curtis distance

# -----------------------------------------------------------------------------#


dtw_dic = {}
bc_dic = {}

s = 0
dtw_dic[0] = 0
for i in range(len(transition_costs)):
    s += transition_costs[i]
    dtw_dic[i + 1] = s

bc_dic[0] = s
for i in range(len(transition_costs) - 1):
    s -= transition_costs[i]
    bc_dic[i + 1] = s
bc_dic[len(transition_costs)] = 0


if use_dic_dtw:
    dm_tag = '_custcost'
else:
    dtw_dic = None
    dm_tag = '_norm'

if use_dic_dist:
    dm_tag += '_custcost'
else:
    bc_dic = None
    dm_tag += '_norm'

dm_tag += '_a%d' % alpha

sort_id = 'STUDY_PATIENT_ID'
sort_id_date = 'SCR_ENTERED'
dataPath = basePath + "DATA/"
outPath = dataPath + t_analyze.lower() + folder_name
resPath = basePath + 'RESULTS/' + t_analyze.lower() + folder_name
inFile = dataPath + xl_file
id_ref = outPath + id_ref
baseline_file = dataPath + 'baselines_1_7-365_mdrd.csv'
h5_name = resPath + h5_name
date_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')


def main():
    dem_m = None
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    if not os.path.exists(resPath):
        os.makedirs(resPath)

    # Try to load previously extracted data
    try:
        ids = np.loadtxt(id_ref, dtype=int)
        print('Loaded previous ids.')
        # try to load final KDIGO values
        try:
            kdigos = kf.load_csv(outPath + 'kdigo.csv', ids)
            print('Loaded previously extracted KDIGO vectors')
        # try to load extracted raw data
        except:
            scr = kf.load_csv(outPath + 'scr_raw.csv', ids)
            dates = kf.load_csv(outPath + 'dates.csv', ids, dt=str)
            masks = kf.load_csv(outPath + 'masks.csv', ids, dt=int)
            dmasks = kf.load_csv(outPath + 'dialysis.csv', ids, dt=int)
            baselines = kf.load_csv(outPath + 'baselines.csv', ids, sel=1)
            print('Loaded previously extracted raw data')

            try:
                post_interpo = kf.load_csv(outPath + 'scr_interp.csv', ids)
                dmasks_interp = kf.load_csv(outPath + 'dmasks_interp.csv', ids, dt=int)
                interp_masks = kf.load_csv(outPath + 'interp_masks.csv', ids, dt=int)
                days_interp = kf.load_csv(outPath + 'days_interp.csv', ids, dt=int)
                print('Loaded previously interpolated values')
            except:
                # Interpolate missing values
                print('Interpolating missing values')
                interpo_log = open(outPath + 'interpo_log.txt', 'w')
                post_interpo, dmasks_interp, days_interp, interp_masks = kf.linear_interpo(scr, ids, dates, masks,
                                                                                           dmasks, timescale,
                                                                                           interpo_log)
                kf.arr2csv(outPath + 'scr_interp.csv', post_interpo, ids)
                kf.arr2csv(outPath + 'dmasks_interp.csv', dmasks_interp, ids, fmt='%d')
                kf.arr2csv(outPath + 'days_interp.csv', days_interp, ids, fmt='%d')
                kf.arr2csv(outPath + 'interp_masks.csv', interp_masks, ids, fmt='%d')
                interpo_log.close()
            print('Converting to KDIGO')
            # Convert SCr to KDIGO
            kdigos = kf.scr2kdigo(post_interpo, baselines, dmasks_interp, days_interp, interp_masks)
            kf.arr2csv(outPath + 'kdigo.csv', kdigos, ids, fmt='%d')
    # If data loading unsuccesful start from scratch
    except:
        # Load raw data from individual CSV files
        ((date_m, hosp_locs, icu_locs, adisp_loc,
          surg_m, surg_des_loc,
          diag_m, diag_loc, diag_nb_loc,
          esrd_m, esrd_locs,
          dia_m, crrt_locs, hd_locs, pd_locs,
          scr_all_m, scr_date_loc, scr_val_loc, scr_desc_loc,
          dem_m, sex_loc, eth_loc,
          dob_m, birth_loc,
          mort_m, mdate_loc,
          io_m, charl_m, charl_loc, elix_m, elix_loc,
          blood_gas, pa_o2, pa_co2, p_h,
          clinical_oth, resp, fi_o2, g_c_s,
          clinical_vit, temp, m_a_p, cuff, h_r,
          labs, bili, pltlts, na, p_k, hemat, w_b_c,
          medications, med_name, med_date, med_dur,
          organ_sup, mech_vent_dates, mech_vent_days,
          scr_agg, s_c_r)) = kf.load_all_csv(dataPath, sort_id)

        # Determine relative admits
        if t_analyze == 'ICU':
            admit_info = kf.get_admits(date_m, icu_locs[0])
        elif t_analyze == 'HOSP':
            admit_info = kf.get_admits(date_m, hosp_locs[0])

        # Get mask inidicating which points are during dialysis
        dia_mask = kf.get_dialysis_mask(scr_all_m, scr_date_loc, dia_m, crrt_locs, hd_locs, pd_locs)

        # Get mask indicating whether each point was in hospital or ICU
        t_mask = kf.get_t_mask(scr_all_m, scr_date_loc, scr_val_loc, date_m, hosp_locs, icu_locs, admit_info)

        # Get mask for the desired data
        mask = np.zeros(len(scr_all_m))
        for i in range(len(scr_all_m)):
            if t_analyze == 'ICU':
                if t_mask[i] == 2:
                    if dia_mask[i]:
                        mask[i] = -1
                    else:
                        mask[i] = 1
            elif t_analyze == 'HOSP':
                if t_mask[i] >= 1:
                    if dia_mask[i]:
                        mask[i] = -1
                    else:
                        mask[i] = 1

        # Baselines

        print('Loading baselines...')
        try:
            bsln_m = pd.read_csv(baseline_file)
            bsln_scr_loc = bsln_m.columns.get_loc('bsln_val')
            admit_loc = bsln_m.columns.get_loc('admit_date')
            bsln_m = bsln_m.values

        except:
            kf.get_baselines(date_m, hosp_locs, scr_all_m, scr_val_loc, scr_date_loc, scr_desc_loc,
                             dem_m, sex_loc, eth_loc, dob_m, birth_loc, baseline_file)

            bsln_m = pd.read_csv(baseline_file)
            bsln_scr_loc = bsln_m.columns.get_loc('bsln_val')
            admit_loc = bsln_m.columns.get_loc('admit_date')
            bsln_m = bsln_m.values

        count_log = open(outPath + 'patient_summary.csv', 'w')
        exc_log = open(outPath + 'excluded_patients.csv', 'w')
        # Extract patients into separate list elements
        (ids, scr, dates, masks, dmasks, baselines,
         bsln_gfr, d_disp, t_range, ages) = kf.get_patients(scr_all_m, scr_val_loc, scr_date_loc, adisp_loc,
                                                            mask, dia_mask,
                                                            diag_m, diag_loc,
                                                            esrd_m, esrd_locs,
                                                            bsln_m, bsln_scr_loc, admit_loc,
                                                            date_m, icu_locs,
                                                            surg_m, surg_des_loc,
                                                            dem_m, sex_loc, eth_loc,
                                                            dob_m, birth_loc,
                                                            mort_m, mdate_loc,
                                                            count_log, exc_log)
        count_log.close()
        exc_log.close()
        kf.arr2csv(outPath + 'scr_raw.csv', scr, ids, fmt='%.3f')
        kf.arr2csv(outPath + 'dates.csv', dates, ids, fmt='%s')
        kf.arr2csv(outPath + 'masks.csv', masks, ids, fmt='%d')
        kf.arr2csv(outPath + 'dialysis.csv', dmasks, ids, fmt='%d')
        kf.arr2csv(outPath + 'baselines.csv', baselines, ids, fmt='%.3f')
        kf.arr2csv(outPath + 'baseline_gfr.csv', bsln_gfr, ids, fmt='%.3f')
        kf.arr2csv(outPath + 'time_ranges.csv', t_range, ids, fmt='%.3f')
        kf.arr2csv(outPath + 'ages.csv', ages, ids, fmt='%.2f')
        kf.arr2csv(outPath + 'disch_disp.csv', d_disp, ids, fmt='%s')
        np.savetxt(id_ref, ids, fmt='%d')

        # Interpolate missing values
        print('Interpolating missing values')
        interpo_log = open(outPath + 'interpo_log.txt', 'w')
        post_interpo, dmasks_interp, days_interp, interp_masks = kf.linear_interpo(scr, ids, dates, masks, dmasks,
                                                                                   timescale, interpo_log)
        kf.arr2csv(outPath + 'scr_interp.csv', post_interpo, ids)
        kf.arr2csv(outPath + 'days_interp.csv', days_interp, ids, fmt='%d')
        kf.arr2csv(outPath + 'interp_masks.csv', interp_masks, ids, fmt='%d')
        kf.arr2csv(outPath + 'dmasks_interp.csv', dmasks_interp, ids, fmt='%d')
        interpo_log.close()

        # Convert SCr to KDIGO
        print('Converting to KDIGO')
        kdigos = kf.scr2kdigo(post_interpo, baselines, dmasks_interp, days_interp, interp_masks)
        kf.arr2csv(outPath + 'kdigo.csv', kdigos, ids, fmt='%d')

    # Calculate clinical mortality prediction scores
    sofa = None
    if not os.path.exists(outPath + 'sofa.csv'):
        if dem_m is None:
            ((date_m, hosp_locs, icu_locs, adisp_loc,
              surg_m, surg_des_loc,
              diag_m, diag_loc, diag_nb_loc,
              esrd_m, esrd_locs,
              dia_m, crrt_locs, hd_locs, pd_locs,
              scr_all_m, scr_date_loc, scr_val_loc, scr_desc_loc,
              dem_m, sex_loc, eth_loc,
              dob_m, birth_loc,
              mort_m, mdate_loc,
              io_m, charl_m, charl_loc, elix_m, elix_loc,
              blood_gas, pa_o2, pa_co2, p_h,
              clinical_oth, resp, fi_o2, g_c_s,
              clinical_vit, temp, m_a_p, cuff, h_r,
              labs, bili, pltlts, na, p_k, hemat, w_b_c,
              medications, med_name, med_date, med_dur,
              organ_sup, mech_vent_dates, mech_vent_days,
              scr_agg, s_c_r)) = kf.load_all_csv(dataPath, sort_id)
        print('Getting SOFA scores')
        sofa = sf.get_sofa(ids,
                           date_m, icu_locs[0],
                           blood_gas, pa_o2,
                           clinical_oth, fi_o2, g_c_s,
                           clinical_vit, m_a_p, cuff,
                           labs, bili, pltlts,
                           medications, med_name, med_date, med_dur,
                           organ_sup, mech_vent_dates,
                           scr_agg, s_c_r,
                           out_name=outPath + 'sofa.csv')
    else:
        sofa = kf.load_csv(outPath + 'sofa.csv', ids, dt=int)
        sofa = np.array(sofa)

    apache = None
    if not os.path.exists(outPath + 'apache.csv'):
        if dem_m is None:
            ((date_m, hosp_locs, icu_locs, adisp_loc,
              surg_m, surg_des_loc,
              diag_m, diag_loc, diag_nb_loc,
              esrd_m, esrd_locs,
              dia_m, crrt_locs, hd_locs, pd_locs,
              scr_all_m, scr_date_loc, scr_val_loc, scr_desc_loc,
              dem_m, sex_loc, eth_loc,
              dob_m, birth_loc,
              mort_m, mdate_loc,
              io_m, charl_m, charl_loc, elix_m, elix_loc,
              blood_gas, pa_o2, pa_co2, p_h,
              clinical_oth, resp, fi_o2, g_c_s,
              clinical_vit, temp, m_a_p, cuff, h_r,
              labs, bili, pltlts, na, p_k, hemat, w_b_c,
              medications, med_name, med_date, med_dur,
              organ_sup, mech_vent_dates, mech_vent_days,
              scr_agg, s_c_r)) = kf.load_all_csv(dataPath, sort_id)
        print('Getting APACHE-II Scores')
        apache = sf.get_apache(ids, outPath,
                               clinical_vit, temp, m_a_p, cuff, h_r,
                               clinical_oth, resp, fi_o2, g_c_s,
                               blood_gas, pa_o2, pa_co2, p_h,
                               labs, na, p_k, hemat, w_b_c,
                               scr_agg, s_c_r,
                               out_name=outPath + 'apache.csv')

    else:
        apache = kf.load_csv(outPath + 'apache.csv', ids, dt=int)
        apache = np.array(apache)

    # Summarize patient stats
    try:
        f = h5py.File(h5_name, 'r+')
    except:
        f = h5py.File(h5_name, 'w')
    try:
        stats = f['meta']
        all_stats = f['meta_all']
        max_kdigo = all_stats['max_kdigo'][:]
        aki_idx = np.where(max_kdigo > 0)[0]
    except:
        if dem_m is None:
            # Load raw data from individual CSV files
            ((date_m, hosp_locs, icu_locs, adisp_loc,
              surg_m, surg_des_loc,
              diag_m, diag_loc, diag_nb_loc,
              esrd_m, esrd_locs,
              dia_m, crrt_locs, hd_locs, pd_locs,
              scr_all_m, scr_date_loc, scr_val_loc, scr_desc_loc,
              dem_m, sex_loc, eth_loc,
              dob_m, birth_loc,
              mort_m, mdate_loc,
              io_m, charl_m, charl_loc, elix_m, elix_loc,
              blood_gas, pa_o2, pa_co2, p_h,
              clinical_oth, resp, fi_o2, g_c_s,
              clinical_vit, temp, m_a_p, cuff, h_r,
              labs, bili, pltlts, na, p_k, hemat, w_b_c,
              medications, med_name, med_date, med_dur,
              organ_sup, mech_vent_dates, mech_vent_days,
              scr_agg, s_c_r)) = kf.load_all_csv(dataPath, sort_id)
        print('Summarizing stats')
        all_stats = sf.summarize_stats(ids, kdigos,
                                       dem_m, sex_loc, eth_loc,
                                       dob_m, birth_loc,
                                       diag_m, diag_loc, diag_nb_loc,
                                       charl_m, charl_loc, elix_m, elix_loc,
                                       organ_sup, mech_vent_days,
                                       date_m, hosp_locs, icu_locs,
                                       sofa, apache, io_m,
                                       h5_name, outPath, grp_name='meta_all')
        max_kdigo = all_stats['max_kdigo'][:]
        aki_idx = np.where(max_kdigo > 0)[0]
        stats = f.create_group('meta')
        print('Copying to KDIGO > 0')
        for i in range(len(list(all_stats))):
            name = list(all_stats)[i]
            try:
                stats.create_dataset(name, data=all_stats[name][:][aki_idx], dtype=all_stats[name].dtype)
            except:
                print(name + ' was not copied from meta_all to meta')
    f.close()

    aki_ids = np.array(ids)[aki_idx]
    aki_kdigos = []
    for i in range(len(kdigos)):
        if np.max(kdigos[i]) > 0:
            aki_kdigos.append(kdigos[i])

    # Calculate individual trajectory based features if not already done
    if not os.path.exists(resPath + 'features'):
        os.mkdir(resPath + 'features')

    if not os.path.exists(resPath + 'features/trajectory_individual/'):
        os.mkdir(resPath + 'features/trajectory_individual/')
        desc = kf.descriptive_trajectory_features(aki_kdigos, aki_ids,
                                                  filename=resPath + 'features/trajectory_individual/descriptive_features.csv')

        slope = kf.slope_trajectory_features(aki_kdigos, aki_ids,
                                             filename=resPath + 'features/trajectory_individual/slope_features.csv')
        slope_norm = kf.normalize_features(slope)
        kf.arr2csv(resPath + 'features/trajectory_individual/slope_norm.csv', slope_norm, aki_ids)

        temp = kf.template_trajectory_features(aki_kdigos, aki_ids,
                                               filename=resPath + 'features/trajectory_individual/template_features.csv')
        temp_norm = kf.normalize_features(temp)
        kf.arr2csv(resPath + 'features/trajectory_individual/template_norm.csv', temp_norm, aki_ids)

        all_traj_ind = np.hstack((desc, slope_norm, temp_norm))
        kf.arr2csv(resPath + 'features/trajectory_individual/all_trajectory.csv',
                   all_traj_ind, aki_ids, fmt='%.4f')
    else:
        desc = kf.load_csv(resPath + 'features/trajectory_individual/descriptive_features.csv', aki_ids, skip_header=True)
        temp_norm = kf.load_csv(resPath + 'features/trajectory_individual/template_norm.csv', aki_ids, skip_header=False)
        slope_norm = kf.load_csv(resPath + 'features/trajectory_individual/slope_norm.csv', aki_ids, skip_header=False)
        all_traj_ind = np.hstack((desc, slope_norm, temp_norm))

    # Calculate distance matrix
    dm = None
    if not os.path.exists(resPath + 'kdigo_dm' + dm_tag + '.csv'):
        dm = kf.pairwise_dtw_dist(aki_kdigos, aki_ids, resPath + 'kdigo_dm' + dm_tag + '.csv',
                                  resPath + 'kdigo_dtwlog' + dm_tag + '.csv',
                                  incl_0=False, alpha=alpha, dtw_dic=dtw_dic, bc_dic=bc_dic)
        np.save(resPath + 'kdigo_dm' + dm_tag, dm)

    # Load clusters or launch interactive clustering
    lbls = None
    if os.path.exists(resPath + 'clusters/%s/' % dm_tag[1:]):
        for (dirpath, dirnames, filenames) in os.walk(resPath + 'clusters/%s/composite/' % dm_tag[1:]):
            if lbls is None:
                for filename in filenames:
                    if filename == 'clusters.txt':
                        lbls = np.loadtxt(dirpath + '/' + filename, dtype=str)
    if lbls is None:
        if dm is None:
            dm = np.load(resPath + 'kdigo_dm' + dm_tag + '.pkl', dm)
        if not os.path.exists(resPath + 'clusters/%s/' % dm_tag[1:]):
            os.mkdir(resPath + 'clusters/%s/' % dm_tag[1:])
        lbls = dist_cut_cluster(h5_name, dm, aki_ids, path=resPath + 'clusters/%s/' % dm_tag[1:], interactive=True, save=True)

    n_clusters = len(np.unique(lbls))
    clust_str = '%dclusters-%s' % (n_clusters, date_str)
    if not os.path.exists(resPath + 'features/%s/' % dm_tag[1:]):
        os.mkdir(resPath + 'features/%s/' % dm_tag[1:])
    if not os.path.exists(resPath + 'features/%s/%s/' % (dm_tag[1:], clust_str)):
        os.mkdir(resPath + 'features/%s/%s/' % (dm_tag[1:], clust_str))

        desc_c, temp_c, slope_c = kf.cluster_feature_vectors(desc, temp_norm, slope_norm, lbls)
        all_desc_c = assign_feature_vectors(lbls, desc_c)
        all_temp_c = assign_feature_vectors(lbls, temp_c)
        all_slope_c = assign_feature_vectors(lbls, slope_c)
        kf.arr2csv(resPath + 'features/%s/%s/descriptive.csv' % (dm_tag[1:], clust_str),
                   all_desc_c, aki_ids, fmt='%.4f')
        kf.arr2csv(resPath + 'features/%s/%s/slope.csv' % (dm_tag[1:], clust_str),
                   all_slope_c, aki_ids, fmt='%.4f')
        kf.arr2csv(resPath + 'features/%s/%s/template.csv' % (dm_tag[1:], clust_str),
                   all_temp_c, aki_ids, fmt='%.4f')

        all_traj_c = np.hstack((all_desc_c, all_slope_c, all_temp_c))
        kf.arr2csv(resPath + 'features/%s/%s/all_trajectory.csv' % (dm_tag[1:], clust_str),
                   all_traj_c, aki_ids, fmt='%.4f')
    else:
        all_desc_c = kf.load_csv(resPath + 'features/%s/%s/descriptive.csv' % (dm_tag[1:], clust_str), aki_ids)
        all_slope_c = kf.load_csv(resPath + 'features/%s/%s/slope.csv' % (dm_tag[1:], clust_str), aki_ids)
        all_temp_c = kf.load_csv(resPath + 'features/%s/%s/template.csv' % (dm_tag[1:], clust_str), aki_ids)
        all_traj_c = np.hstack((all_desc_c, all_slope_c, all_temp_c))

    if sofa is None:
        sofa = kf.load_csv(outPath + 'sofa.csv', aki_ids, dt=int)
    else:
        if sofa.shape[0] == len(ids):
            sofa = sofa[aki_idx, :]
    sofa_norm = kf.normalize_features(sofa)

    if apache is None:
        apache = kf.load_csv(outPath + 'apache.csv', aki_ids, dt=int)
        apache = np.array(apache)
    else:
        if apache.shape[0] == len(ids):
            apache = np.array(apache[aki_idx, :])
    apache_norm = kf.normalize_features(apache)

    if not os.path.exists(resPath + 'features/clinical/'):
        os.mkdir(resPath + 'features/clinical/')
        kf.arr2csv(resPath + 'features/clinical/sofa.csv',
                   sofa, aki_ids, fmt='%d')
        kf.arr2csv(resPath + 'features/clinical/sofa_norm.csv',
                   sofa_norm, aki_ids, fmt='%.4f')
        kf.arr2csv(resPath + 'features/clinical/apache.csv',
                   apache, aki_ids, fmt='%d')
        kf.arr2csv(resPath + 'features/clinical/apache_norm.csv',
                   apache_norm, aki_ids, fmt='%.4f')
        all_clin = np.hstack((sofa_norm, apache_norm))
        kf.arr2csv(resPath + 'features/clinical/all_clinical.csv',
                   all_clin, aki_ids, fmt='%.4f')
    else:
        all_clin = kf.load_csv(resPath + 'features/clinical/all_clinical.csv', aki_ids)

    if not os.path.exists(resPath + 'features/trajectory_individual/everything.csv'):
        everything_ind = np.hstack((all_clin, all_traj_ind))
        kf.arr2csv(resPath + 'features/trajectory_individual/everything.csv',
                   everything_ind, aki_ids, fmt='%.4f')

    if not os.path.exists(resPath + 'features/%s/%s/everything.csv' % (dm_tag[1:], clust_str)):
        everything_clusters = np.hstack((all_clin, all_traj_c))
        kf.arr2csv(resPath + 'features/%s/%s/everything.csv' % (dm_tag[1:], clust_str),
                   everything_clusters, aki_ids, fmt='%.4f')

    print('Ready for classification. Please run script \'classify_features.py\'')


main()
