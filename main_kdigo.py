import os
import h5py
import numpy as np
import pandas as pd
import kdigo_funcs as kf
import stat_funcs as sf
from cluster_funcs import assign_feature_vectors
from kdigo_commands import post_dm_process

# ------------------------------- PARAMETERS ----------------------------------#
basePath = "../"
t_analyze = 'ICU'
xl_file = "KDIGO_full.xlsx"
timescale = 6  # in hours
id_ref = 'icu_valid_ids.csv'  # specify different file with subset of IDs if desired
incl_0 = False
h5_name = 'kdigo_dm.h5'
# -----------------------------------------------------------------------------#

sort_id = 'STUDY_PATIENT_ID'
sort_id_date = 'SCR_ENTERED'
dataPath = basePath + "DATA/"
outPath = dataPath + t_analyze.lower() + '/7days_061718/'
resPath = basePath + 'RESULTS/' + t_analyze.lower() + '/7days_061718/'
inFile = dataPath + xl_file
id_ref = outPath + id_ref
baseline_file = dataPath + 'baselines_1_7-365_mdrd.csv'
h5_name = resPath + h5_name


def main():
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
            _, kdigos = kf.load_csv(outPath + 'kdigo.csv', ids)
            print('Loaded previously extracted KDIGO vectors')
        # try to load extracted raw data
        except:
            _, scr = kf.load_csv(outPath + 'scr_raw.csv', ids)
            _, dates = kf.load_csv(outPath + 'dates.csv', ids, dt=str)
            _, masks = kf.load_csv(outPath + 'masks.csv', ids, dt=int)
            _, dmasks = kf.load_csv(outPath + 'dialysis.csv', ids, dt=int)
            _, baselines = kf.load_csv(outPath + 'baselines.csv', ids, sel=1)
            print('Loaded previously extracted raw data')

            try:
                _, post_interpo = kf.load_csv(outPath + 'scr_interp.csv', ids)
                _, dmasks_interp = kf.load_csv(outPath + 'dmasks_interp.csv', ids, dt=int)
                _, interp_masks = kf.load_csv(outPath + 'interp_masks.csv', ids, dt=int)
                _, days_interp = kf.load_csv(outPath + 'days_interp.csv', ids, dt=int)
                print('Loaded previously interpolated values')
            except:
                # Interpolate missing values
                print('Interpolating missing values')
                interpo_log = open(outPath + 'interpo_log.txt', 'w')
                post_interpo, dmasks_interp, days_interp, interp_masks = kf.linear_interpo(scr, ids, dates, masks, dmasks, timescale, interpo_log)
                kf.arr2csv(outPath + 'scr_interp.csv', post_interpo, ids)
                kf.arr2csv(outPath + 'dmasks_interp.csv', dmasks_interp, ids, fmt='%d')
                kf.arr2csv(outPath + 'days_interp.csv', days_interp, ids, fmt='%d')
                kf.arr2csv(outPath + 'interp_masks.csv', interp_masks, ids, fmt='%d')
                interpo_log.close()
            print('Converting to KDIGO')
            # Convert SCr to KDIGO
            kdigos = kf.scr2kdigo(post_interpo, baselines, dmasks_interp, days_interp, interp_masks)
            kf.arr2csv(outPath + 'kdigo.csv', kdigos, ids)
    # If data loading unsuccesful start from scratch
    except:
        print('Loading encounter info...')
        # Get IDs and find indices of all used metrics
        date_m = kf.get_mat(inFile, 'ADMISSION_INDX', [sort_id])
        hosp_locs = [date_m.columns.get_loc("HOSP_ADMIT_DATE"), date_m.columns.get_loc("HOSP_DISCHARGE_DATE")]
        icu_locs = [date_m.columns.get_loc("ICU_ADMIT_DATE"), date_m.columns.get_loc("ICU_DISCHARGE_DATE")]
        adisp_loc = date_m.columns.get_loc('DISCHARGE_DISPOSITION')
        date_m = date_m.as_matrix()

        ### GET SURGERY SHEET AND LOCATION
        print('Loading surgery information...')
        surg_m = kf.get_mat(inFile, 'SURGERY_INDX', [sort_id])
        surg_des_loc = surg_m.columns.get_loc("SURGERY_DESCRIPTION")
        surg_m = surg_m.as_matrix()

        ### GET DIAGNOSIS SHEET AND LOCATION
        print('Loading diagnosis information...')
        dx_m = kf.get_mat(inFile, 'DIAGNOSIS', [sort_id])
        dx_loc = dx_m.columns.get_loc("DIAGNOSIS_DESC")
        dx_m = dx_m.as_matrix()

        print('Loading ESRD status...')
        # ESRD status
        esrd_m = kf.get_mat(inFile, 'ESRD_STATUS', [sort_id])
        esrd_locs = [esrd_m.columns.get_loc("AT_ADMISSION_INDICATOR"),
                     esrd_m.columns.get_loc("DURING_INDEXED_INDICATOR"),
                     esrd_m.columns.get_loc("BEFORE_INDEXED_INDICATOR")]
        esrd_m = esrd_m.as_matrix()

        # Dialysis dates
        print('Loading dialysis dates...')
        dia_m = kf.get_mat(inFile, 'RENAL_REPLACE_THERAPY', [sort_id])
        crrt_locs = [dia_m.columns.get_loc('CRRT_START_DATE'), dia_m.columns.get_loc('CRRT_STOP_DATE')]
        hd_locs = [dia_m.columns.get_loc('HD_START_DATE'), dia_m.columns.get_loc('HD_STOP_DATE')]
        pd_locs = [dia_m.columns.get_loc('PD_START_DATE'), dia_m.columns.get_loc('PD_STOP_DATE')]
        dia_m = dia_m.as_matrix()

        # All SCR
        print('Loading SCr values (may take a while)...')
        scr_all_m = kf.get_mat(inFile, 'SCR_ALL_VALUES', [sort_id, sort_id_date])
        scr_date_loc = scr_all_m.columns.get_loc('SCR_ENTERED')
        scr_val_loc = scr_all_m.columns.get_loc('SCR_VALUE')
        scr_desc_loc = scr_all_m.columns.get_loc('SCR_ENCOUNTER_TYPE')
        scr_all_m = scr_all_m.as_matrix()

        # Demographics
        print('Loading demographics...')
        dem_m = kf.get_mat(inFile, 'DEMOGRAPHICS_INDX', [sort_id])
        sex_loc = dem_m.columns.get_loc('GENDER')
        eth_loc = dem_m.columns.get_loc('RACE')
        dem_m = dem_m.as_matrix()

        # DOB
        print('Loading birthdates...')
        dob_m = kf.get_mat(inFile, 'DOB', [sort_id])
        birth_loc = dob_m.columns.get_loc("DOB")
        dob_m = dob_m.as_matrix()

        # load death data
        mort_m = kf.get_mat(inFile, 'OUTCOMES', 'STUDY_PATIENT_ID')
        mdate_loc = mort_m.columns.get_loc("DECEASED_DATE")
        mort_m = mort_m.as_matrix()

        # Get mask inidicating which points are during dialysis
        dia_mask = kf.get_dialysis_mask(scr_all_m, scr_date_loc, dia_m, crrt_locs, hd_locs, pd_locs)

        # Get mask indicating whether each point was in hospital or ICU
        t_mask = kf.get_t_mask(scr_all_m, scr_date_loc, scr_val_loc, date_m, hosp_locs, icu_locs)

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
            bsln_m = bsln_m.as_matrix()

        except:
            kf.get_baselines(date_m, hosp_locs, scr_all_m, scr_val_loc, scr_date_loc, scr_desc_loc,
                             dem_m, sex_loc, eth_loc, dob_m, birth_loc, baseline_file)

            bsln_m = pd.read_csv(baseline_file)
            bsln_scr_loc = bsln_m.columns.get_loc('bsln_val')
            admit_loc = bsln_m.columns.get_loc('admit_date')
            bsln_m = bsln_m.as_matrix()

        count_log = open(outPath + 'patient_summary.csv', 'w')
        exc_log = open(outPath + 'excluded_patients.csv', 'w')
        # Extract patients into separate list elements
        (ids, scr, dates, masks, dmasks, baselines,
         bsln_gfr, d_disp, t_range, ages) = kf.get_patients(scr_all_m, scr_val_loc, scr_date_loc, adisp_loc,
                                                            mask, dia_mask,
                                                            dx_m, dx_loc,
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
        kf.arr2csv(outPath + 'scr_raw.csv', scr, ids)
        kf.arr2csv(outPath + 'dates.csv', dates, ids, fmt='%s')
        kf.arr2csv(outPath + 'masks.csv', masks, ids, fmt='%d')
        kf.arr2csv(outPath + 'dialysis.csv', dmasks, ids, fmt='%d')
        kf.arr2csv(outPath + 'baselines.csv', baselines, ids)
        kf.arr2csv(outPath + 'baseline_gfr.csv', bsln_gfr, ids)
        kf.arr2csv(outPath + 'time_ranges.csv', t_range, ids)
        kf.arr2csv(outPath + 'ages.csv', ages, ids)
        kf.arr2csv(outPath + 'disch_disp.csv', d_disp, ids, fmt='%s')
        np.savetxt(id_ref, ids, fmt='%d')

        # Interpolate missing values
        print('Interpolating missing values')
        interpo_log = open(outPath + 'interpo_log.txt', 'w')
        post_interpo, dmasks_interp, days_interp, interp_masks = kf.linear_interpo(scr, ids, dates, masks, dmasks, timescale, interpo_log)
        kf.arr2csv(outPath + 'scr_interp.csv', post_interpo, ids)
        kf.arr2csv(outPath + 'days_interp.csv', days_interp, ids, fmt='%d')
        kf.arr2csv(outPath + 'interp_masks.csv', interp_masks, ids, fmt='%d')
        kf.arr2csv(outPath + 'dmasks_interp.csv', dmasks_interp, ids, fmt='%d')
        interpo_log.close()

        # Convert SCr to KDIGO
        print('Converting to KDIGO')
        kdigos = kf.scr2kdigo(post_interpo, baselines, dmasks_interp, days_interp, interp_masks)
        kf.arr2csv(outPath + 'kdigo.csv', kdigos, ids, fmt='%d')

    # Get KDIGO Distance Matrix and summarize patient stats
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
        print('Summarizing stats')
        all_stats = sf.summarize_stats(outPath, h5_name, grp_name='meta_all')
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

    aki_ids = np.array(ids)[aki_idx]
    aki_kdigos = []
    for i in range(len(kdigos)):
        if np.max(kdigos[i]) > 0:
            aki_kdigos.append(kdigos[i])


    # Calculate clinical mortality prediction scores
    sofa = None
    if not os.path.exists(outPath + 'sofa.csv'):
        print('Getting SOFA scores')
        sofa = sf.get_sofa(id_ref, inFile, outPath + 'sofa.csv')

    apache = None
    if not os.path.exists(outPath + 'apache.csv'):
        print('Getting APACHE-II Scores')
        apache = sf.get_apache(id_ref, inFile, outPath + 'apache.csv')

    if 'sofa' not in list(stats):
        if sofa is None:
            _, sofa = kf.load_csv(outPath + 'sofa.csv', ids, dt=int)
            sofa = np.array(sofa)
        sofa = np.sum(sofa, axis=1)
        _ = all_stats.create_dataset('sofa', data=sofa, dtype=int)
        _ = stats.create_dataset('sofa', data=sofa[aki_idx], dtype=int)

    if 'apache' not in list(stats):
        if apache is None:
            _, apache = kf.load_csv(outPath + 'apache.csv', ids, dt=int)
            apache = np.array(apache)
        apache = np.sum(apache, axis=1)
        _ = all_stats.create_dataset('apache', data=apache, dtype=int)
        _ = stats.create_dataset('apache', data=apache[aki_idx], dtype=int)

    if 'dm' not in list(f):
        dm = kf.pairwise_dtw_dist(aki_kdigos, aki_ids, resPath + 'kdigo_dm.csv', resPath + 'kdigo_dtwlog.csv', incl_0=False)
        f = h5py.File(h5_name, 'w')
        f.create_dataset('dm', data=dm)

    # Calculate individual trajectory based statistics if not already done
    if 'features' in list(f):
        fg = f['features']
    else:
        fg = f.create_group('features')


    if 'descriptive_individual' in list(fg):
        desc = fg['descriptive_individual'][:]
    else:
        desc = kf.descriptive_trajectory_features(kdigos, ids, filename=outPath+'descriptive_features.csv')
        _ = fg.create_dataset('descriptive_individual', data=desc, dtype=int)

    if 'slope_individual' in list(fg):
        slope = fg['slope_individual'][:]
    else:
        slope = kf.slope_trajectory_features(kdigos, ids, filename=outPath + 'slope_features.csv')
        _ = fg.create_dataset('slope_individual', data=slope, dtype=int)

    if 'template_individual' in list(fg):
        temp = fg['template_individual'][:]
    else:
        temp = kf.template_trajectory_features(kdigos, ids, filename=outPath + 'template_features.csv')
        _ = fg.create_dataset('template_individual', data=temp, dtype=int)

    # Load clusters or launch interactive clustering
    try:
        lbls = np.loadtxt(resPath + 'clusters/composite/clusters.txt', dtype=str)
    except:
        if not os.path.exists(resPath + 'clusters/'):
            os.mkdir(resPath + 'clusters/')
        lbls = post_dm_process(h5_name, '', output_base_path=resPath + 'clusters/')

        if not os.path.exists(resPath + 'clusters/composite/'):
            os.mkdir(resPath + 'clusters/composite/')
        np.savetxt(resPath + 'clusters/composite/clusters.txt', lbls, fmt='%s')

    # Get corresponding cluster features
    try:
        _ = fg['descriptive_clusters'][:]
        _ = fg['template_clusters'][:]
        _ = fg['slope_clusters'][:]
    except:
        desc_c, temp_c, slope_c = kf.cluster_feature_vectors(desc, temp, slope, lbls)
        all_desc_c = assign_feature_vectors(lbls, desc_c)
        all_temp_c = assign_feature_vectors(lbls, temp_c)
        all_slope_c = assign_feature_vectors(lbls, slope_c)
        fg.create_dataset('descriptive_clusters', data=all_desc_c, dtype=int)
        fg.create_dataset('template_clusters', data=all_temp_c, dtype=float)
        fg.create_dataset('slope_clusters', data=all_slope_c, dtype=float)

    print('Ready for classification. Please run script \'classify_features.py\'')
    print('Available features:')
    for k in fg.keys():
        print('\t' + k)


main()
