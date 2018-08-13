from __future__ import division
import datetime
import math
import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, VarianceThreshold, RFECV
from sklearn.model_selection import StratifiedKFold
import h5py
from tqdm import tqdm


# %%
def get_dialysis_mask(scr_m, scr_date_loc, dia_m, crrt_locs, hd_locs, pd_locs, v=True):
    mask = np.zeros(len(scr_m))
    if v:
        print 'Getting mask for dialysis'
        print 'Number non-dialysis records, #CRRT, #HD, #PD'
    for i in range(len(mask)):
        this_id = scr_m[i, 0]
        date_str = str(scr_m[i, scr_date_loc]).lower()
        if date_str == 'nan' or date_str == 'nat':
            continue
        else:
            this_date = datetime.datetime.strptime(str(scr_m[i, scr_date_loc]), '%Y-%m-%d %H:%M:%S')
        rows = np.where(dia_m[:, 0] == this_id)[0]
        for row in rows:
            if dia_m[row, crrt_locs[0]]:
                if str(dia_m[row, crrt_locs[0]]) != 'nan' and \
                        str(dia_m[row, crrt_locs[1]]) != 'nan':
                    left = datetime.datetime.strptime(str(dia_m[row, crrt_locs[0]]), '%Y-%m-%d %H:%M:%S')
                    right = datetime.datetime.strptime(str(dia_m[row, crrt_locs[1]]), '%Y-%m-%d %H:%M:%S')
                    if left < this_date < right + datetime.timedelta(2):
                        mask[i] = 1
            if dia_m[row, hd_locs[0]]:
                if str(dia_m[row, hd_locs[0]]) != 'nan' and \
                        str(dia_m[row, hd_locs[1]]) != 'nan':
                    left = datetime.datetime.strptime(str(dia_m[row, hd_locs[0]]), '%Y-%m-%d %H:%M:%S')
                    right = datetime.datetime.strptime(str(dia_m[row, hd_locs[1]]), '%Y-%m-%d %H:%M:%S')
                    if left < this_date < right + datetime.timedelta(2):
                        mask[i] = 2
            if dia_m[row, pd_locs[0]]:
                if str(dia_m[row, pd_locs[0]]) != 'nan' and str(dia_m[row, pd_locs[1]]) != 'nan':
                    left = datetime.datetime.strptime(str(dia_m[row, pd_locs[0]]), '%Y-%m-%d %H:%M:%S')
                    right = datetime.datetime.strptime(str(dia_m[row, pd_locs[1]]), '%Y-%m-%d %H:%M:%S')
                    if left < this_date < right + datetime.timedelta(2):
                        mask[i] = 3
    if v:
        nwo = len(np.where(mask == 0)[0])
        ncrrt = len(np.where(mask == 1)[0])
        nhd = len(np.where(mask == 2)[0])
        npd = len(np.where(mask == 3)[0])
        print('%d, %d, %d, %d\n' % (nwo, ncrrt, nhd, npd))
    return mask


def get_dia_mask_dallas(scr_m, dia_locs):
    mask = np.zeros(len(scr_m), dtype=int)
    for col in dia_locs:
        mask[np.where(scr_m[:, col])] = 1
    return mask


# %%
def get_admits(date_m, admit_loc):
    temp_id = 0
    all_ids = np.unique(date_m[:, 0])
    admit_info = np.zeros((len(all_ids), 2), dtype='|S20')
    admit_info[:, 0] = all_ids
    for i in range(len(all_ids)):
        tid = all_ids[i]
        rows = np.where(date_m[:, 0] == tid)[0]
        admit = datetime.datetime.now()
        for row in rows:
            tdate = datetime.datetime.strptime(str(date_m[row, admit_loc]).split('.')[0], '%Y-%m-%d %H:%M:%S')
            if tdate < admit:
                admit = tdate
        admit_info[i, 1] = admit
    return admit_info


# %%
def get_t_mask(scr_m, scr_date_loc, scr_val_loc, date_m, hosp_locs, icu_locs, admits, t_lim=7, v=True):
    mask = np.zeros(len(scr_m))
    if v:
        print('Getting masks for icu and hospital admit-discharge')
    for i in range(len(mask)):
        this_id = scr_m[i, 0]
        date_str = str(scr_m[i, scr_date_loc]).lower()
        if date_str == 'nan' or date_str == 'nat' or date_str == '1900-01-00 00:00:00':
            continue
        this_date = datetime.datetime.strptime(str(scr_m[i, scr_date_loc]).split('.')[0], '%Y-%m-%d %H:%M:%S')
        this_val = scr_m[i, scr_val_loc]
        this_day = this_date.day
        if this_val == np.nan:
            continue
        try:
            admit_idx = np.where(admits[:, 0] == str(this_id))[0][0]
            admit = datetime.datetime.strptime(str(admits[admit_idx, 1]).split('.')[0], '%Y-%m-%d %H:%M:%S')
            admit_day = admit.day
        except:
            continue

        if t_lim is not None:
            if this_day - admit_day > t_lim:
                continue

        rows = np.where(date_m[:, 0] == this_id)[0]
        for row in rows:
            if date_m[row, icu_locs[0]] != np.nan:
                start = datetime.datetime.strptime(str(date_m[row, icu_locs[0]]).split('.')[0], '%Y-%m-%d %H:%M:%S')
                stop = datetime.datetime.strptime(str(date_m[row, icu_locs[1]]).split('.')[0], '%Y-%m-%d %H:%M:%S')
                if start < this_date < stop:
                    mask[i] = 2
                    continue
            elif date_m[row, hosp_locs[0]] != np.nan:
                start = datetime.datetime.strptime(str(date_m[row, hosp_locs[0]]).split('.')[0], '%Y-%m-%d %H:%M:%S')
                stop = datetime.datetime.strptime(str(date_m[row, hosp_locs[1]]).split('.')[0], '%Y-%m-%d %H:%M:%S')
                if start < this_date < stop:
                    mask[i] = 1
    if v:
        nop = len(np.where(mask == 0)[0])
        nhp = len(np.where(mask >= 1)[0])
        nicu = len(np.where(mask == 2)[0])
        print('Number records outside hospital: ' + str(nop))
        print('Number records in hospital: ' + str(nhp))
        print('Number records in ICU: ' + str(nicu))
    return mask


# %%
def get_patients(scr_all_m, scr_val_loc, scr_date_loc, d_disp_loc,
                 mask, dia_mask,
                 dx_m, dx_loc,
                 esrd_m, esrd_locs,
                 bsln_m, bsln_scr_loc, admit_loc,
                 date_m, icu_locs,
                 xplt_m, xplt_des_loc,
                 dem_m, sex_loc, eth_loc,
                 dob_m, birth_loc,
                 mort_m, mdate_loc,
                 log, exc_log, v=True):
    # Lists to store records for each patient
    scr = []
    tmasks = []  # time/date
    dmasks = []  # dialysis
    dates = []
    d_disp = []
    ids_out = []
    bslns = []
    bsln_gfr = []
    t_range = []
    ages = []
    days = []

    # Counters for total number of patients and how many removed for each exclusion criterium
    count = 0
    gfr_count = 0
    no_admit_info_count = 0
    # gap_icu_count = 0
    no_recs_count = 0
    no_bsln_count = 0
    kid_xplt_count = 0
    esrd_count = 0
    dem_count = 0
    ids = np.unique(scr_all_m[:, 0]).astype(int)
    ids.sort()
    if v:
        print('Getting patient vectors')
        print('Patient_ID\tAdmit_Date\tDischarge_Date\tBaseline_SCr\tMort_Date\tDays_To_Death')
        log.write('Patient_ID,Admit_Date,Discharge_Date,Baseline_SCr,Mort_Date,Days_To_Death\n')
    for idx in ids:
        skip = False  # required to skip patients where exclusion is found in interior loop
        all_rows = np.where(scr_all_m[:, 0] == idx)[0]
        sel = np.where(mask[all_rows] != 0)[0]
        keep = all_rows[sel]
        # Ensure this patient has values in time period of interest
        if len(sel) < 2:
            no_recs_count += 1
            if v:
                print(str(idx) + ', removed due to not enough values in the time period of interest')
                exc_log.write(str(idx) + ', removed due to not enough values in the time period of interest\n')
            continue

        # Get Baseline or remove if no admit dates provided
        bsln_idx = np.where(bsln_m[:, 0] == idx)[0]

        if bsln_idx.size == 0:
            no_admit_info_count += 1
            if v:
                print(str(idx) + ', removed due to missing admission info')
                exc_log.write(str(idx) + ', removed due to missing admission info\n')
            continue
        else:
            bsln_idx = bsln_idx[0]
        bsln = bsln_m[bsln_idx, bsln_scr_loc]
        if str(bsln).lower() == 'nan' or str(bsln).lower() == 'none' or str(bsln).lower() == 'nat':
            no_bsln_count += 1
            if v:
                print(str(idx) + ', removed due to missing baseline')
                exc_log.write(str(idx) + ', removed due to missing baseline\n')
            continue
        bsln = float(bsln)
        if bsln >= 4.0:
            no_bsln_count += 1
            if v:
                print(str(idx) + ', removed due to baseline SCr > 4.0')
                exc_log.write(str(idx) + ', removed due to baseline SCr > 4.0\n')
            continue
        admit = str(bsln_m[bsln_idx, admit_loc]).split('.')[0]
        admit = datetime.datetime.strptime(admit, '%Y-%m-%d %H:%M:%S')

        # get mortality date if available
        mort_idx = np.where(mort_m[:, 0] == idx)[0]
        mort_date = 'NA'
        if mort_idx.size > 0:
            for i in range(len(mort_idx)):
                tid = mort_idx[i]
                mdate = str(mort_m[tid, mdate_loc]).split('.')[0]
                try:
                    mort_date = datetime.datetime.strptime(mdate, '%Y-%m-%d %H:%M:%S')
                except:
                    mort_date = 'NA'
        if mort_date != 'NA':
            death_dur = (mort_date - admit).total_seconds() / (60 * 60 * 24)
        else:
            death_dur = np.nan
        #         if mdate - admit < datetime.timedelta(death_excl_dur):
        #             skip = True
        #             death_count += 1
        #             if v:
        #                 print(str(idx) + ', removed due to death in specified window')
        #                 log.write(str(idx) + ', removed due to death in specified window\n')
        #             continue
        # if skip:
        #     continue

        # get dob, sex, and race
        if idx not in dob_m[:, 0] or idx not in dem_m[:, 0]:
            # ids = np.delete(ids, count)
            dem_count += 1
            if v:
                print(str(idx) + ', removed due to missing DOB')
                exc_log.write(str(idx) + ', removed due to missing DOB\n')
            continue
        birth_idx = np.where(dob_m[:, 0] == idx)[0][0]
        dob = str(dob_m[birth_idx, birth_loc]).split('.')[0]
        dob = datetime.datetime.strptime(dob, '%Y-%m-%d %H:%M:%S')

        dem_idx = np.where(dem_m[:, 0] == idx)[0]
        if len(dem_idx) > 1:
            dem_idx = dem_idx[0]
        sex = dem_m[dem_idx, sex_loc]
        race = dem_m[dem_idx, eth_loc]
        age = admit - dob
        age = age.total_seconds() / (60 * 60 * 24 * 365)
        if age < 18:
            continue

        # remove if ESRD status
        esrd_idx = np.where(esrd_m[:, 0] == idx)[0]
        if len(esrd_idx) > 0:
            for loc in esrd_locs:
                if np.any(esrd_m[esrd_idx, loc] == 'Y'):
                    skip = True
                    # ids = np.delete(ids, count)
                    esrd_count += 1
                    if v:
                        print(str(idx) + ', removed due to ESRD status')
                        exc_log.write(str(idx) + ', removed due to ESRD status\n')
                    break
        if skip:
            continue

        # remove patients with required demographics missing
        if str(sex) == 'nan' or str(race) == 'nan':
            # ids = np.delete(ids, count)
            dem_count += 1
            if v:
                print(str(idx) + ', removed due to missing demographics')
                exc_log.write(str(idx) + ', removed due to missing demographics\n')
            continue

        # remove patients with baseline GFR < 15
        gfr = calc_gfr(bsln, sex, race, age)
        if gfr < 15:
            # ids = np.delete(ids, count)
            gfr_count += 1
            if v:
                print(str(idx) + ', removed due to initial GFR too low')
                exc_log.write(str(idx) + ', removed due to initial GFR too low\n')
            continue

        # remove patients with kidney transplant
        x_rows = np.where(xplt_m[:, 0] == idx)  # rows in surgery sheet
        for row in x_rows:
            str_des = str(xplt_m[row, xplt_des_loc]).upper()
            if 'KID' in str_des and 'TRANS' in str_des:
                skip = True
                kid_xplt_count += 1
                # ids = np.delete(ids, count)
                if v:
                    print(str(idx) + ', removed due to kidney transplant')
                    exc_log.write(str(idx) + ', removed due to kidney transplant\n')
                break
        if skip:
            continue

        d_rows = np.where(dx_m[:, 0] == idx)
        for row in d_rows:
            str_des = str(dx_m[row, dx_loc]).upper()
            if str_des == 'KIDNEY/PANCREAS FROM BATAVIA  ETA 1530':
                skip = True
                kid_xplt_count += 1
                if v:
                    print(str(idx) + ', removed due to kidney transplant')
                    exc_log.write(str(idx) + ', removed due to kidney transplant\n')
                break
            elif 'KID' in str_des and 'TRANS' in str_des:
                skip = True
                kid_xplt_count += 1
                if v:
                    print(str(idx) + ', removed due to kidney transplant')
                    exc_log.write(str(idx) + ', removed due to kidney transplant\n')
                break
        if skip:
            continue

        # get discharge date and check for multiple separate ICU stays
        # add code to make sure this discharge corresponds to the analyzed admission
        all_drows = np.where(date_m[:, 0] == idx)[0]
        delta = datetime.timedelta(0)
        discharge = datetime.datetime(1000, 1, 1, 1)
        for i in range(len(all_drows)):
            start = str(date_m[all_drows[i], icu_locs[1]]).split('.')[0]
            start = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
            if start > discharge:
                discharge = start
            td = datetime.timedelta(0)
            for j in range(len(all_drows)):
                tdate = str(date_m[all_drows[j], icu_locs[0]]).split('.')[0]
                tdate = datetime.datetime.strptime(tdate, '%Y-%m-%d %H:%M:%S')
                if tdate > start:
                    if td == datetime.timedelta(0):
                        td = tdate - start
                    elif (tdate - start) < td:
                        td = tdate - start
            if delta == datetime.timedelta(0):
                delta = td
            elif delta < td:
                delta = td
        # if delta > datetime.timedelta(3):
        #    #  np.delete(ids, count)
        #     gap_icu_count += 1
        #     if v:
        #         print(str(idx)+', removed due to different ICU stays > 3 days apart')
        #         exc_log.write(str(idx)+', removed due to different ICU stays > 3 days apart\n')
        #     continue

        # # remove patients who died <48 hrs after indexed admission
        disch_disp = date_m[all_drows[0], d_disp_loc]
        if type(disch_disp) == np.ndarray:
            disch_disp = disch_disp[0]
        disch_disp = str(disch_disp).upper()
        # if 'LESS THAN' in disch_disp:
        #    #  np.delete(ids, count)
        #     lt48_count += 1
        #     if v:
        #         print(str(idx)+', removed due to death within 48 hours of admission')
        #         exc_log.write(str(idx)+', removed due to  death within 48 hours of admission\n')
        #     continue

        # get duration vector
        tdates = scr_all_m[keep, scr_date_loc]
        tdates = [datetime.datetime.strptime(str(tdates[i]).split('.')[0], '%Y-%m-%d %H:%M:%S') for i in
                  range(len(tdates))]
        duration = [tdates[x] - admit for x in range(len(tdates))]
        duration = np.array(duration)

        tdays = np.array([(x - admit).days for x in tdates])

        if v:
            print('%d\t%s\t%s\t%.3f\t%s\t%.3f' % (idx, admit, discharge, bsln, mort_date, death_dur))
            log.write('%d,%s,%s,%.3f,%s,%.3f\n' % (idx, admit, discharge, bsln, mort_date, death_dur))
        d_disp.append(disch_disp)
        bslns.append(bsln)
        bsln_gfr.append(gfr)

        tmask = mask[keep]
        tmasks.append(tmask)
        dmask = dia_mask[keep]
        dmasks.append(dmask)
        scr.append(scr_all_m[keep, scr_val_loc])
        dates.append(tdates)
        ages.append(age)
        days.append(tdays)

        tmin = duration[0].total_seconds() / (60 * 60)
        tmax = duration[-1].total_seconds() / (60 * 60)
        ids_out.append(idx)
        t_range.append([tmin, tmax])
        count += 1
    bslns = np.array(bslns)
    if v:
        print('# Patients Kept: ' + str(count))
        print('# Patients removed for ESRD: ' + str(esrd_count))
        print('# Patients w/ GFR < 15: ' + str(gfr_count))
        print('# Patients w/ no admit info: ' + str(no_admit_info_count))
        print('# Patients w/ missing demographics: ' + str(dem_count))
        print('# Patients w/ < 2 ICU records: ' + str(no_recs_count))
        print('# Patients w/ no valid baseline: ' + str(no_bsln_count))
        print('# Patients w/ kidney transplant: ' + str(kid_xplt_count))
        exc_log.write('# Patients Kept: ' + str(count) + '\n')
        exc_log.write('# Patients removed for ESRD: ' + str(esrd_count) + '\n')
        exc_log.write('# Patients w/ GFR < 15: ' + str(gfr_count) + '\n')
        exc_log.write('# Patients w/ no admit info: ' + str(no_admit_info_count) + '\n')
        exc_log.write('# Patients w/ missing demographics: ' + str(dem_count) + '\n')
        exc_log.write('# Patients w/ < 2 ICU records: ' + str(no_recs_count) + '\n')
        exc_log.write('# Patients w/ no valid baseline: ' + str(no_bsln_count) + '\n')
        exc_log.write('# Patients w/ kidney transplant: ' + str(kid_xplt_count) + '\n')
    del scr_all_m
    del bsln_m
    del dx_m
    del date_m
    del xplt_m
    del dem_m
    del dob_m
    return ids_out, scr, dates, days, tmasks, dmasks, bslns, bsln_gfr, d_disp, t_range, ages


# %%
def get_patients_dallas(scr_all_m, scr_val_loc, scr_date_loc,
                        mask, dia_mask,
                        esrd_m, esrd_locs,
                        bsln_m, bsln_scr_loc, admit_loc, disch_loc,
                        date_m,
                        dem_m, sex_loc, eth_loc, dob_loc, dod_locs,
                        log, exc_log, v=True):
    # Lists to store records for each patient
    scr = []
    tmasks = []  # time/date
    dmasks = []  # dialysis
    dates = []
    # d_disp = []
    ids_out = []
    bslns = []
    bsln_gfr = []
    t_range = []
    ages = []
    days = []

    # Counters for total number of patients and how many removed for each exclusion criterium
    count = 0
    gfr_count = 0
    no_admit_info_count = 0
    # gap_icu_count = 0
    no_recs_count = 0
    no_bsln_count = 0
    esrd_count = 0
    dem_count = 0
    ids = np.unique(scr_all_m[:, 0]).astype(int)
    ids.sort()
    if v:
        print('Getting patient vectors')
        print('Patient_ID\tAdmit_Date\tDischarge_Date\tBaseline_SCr\tMort_Date\tDays_To_Death')
        log.write('Patient_ID,Admit_Date,Discharge_Date,Baseline_SCr,Mort_Date,Days_To_Death\n')
    for idx in ids:
        skip = False  # required to skip patients where exclusion is found in interior loop
        all_rows = np.where(scr_all_m[:, 0] == idx)[0]
        sel = np.where(mask[all_rows] != 0)[0]
        keep = all_rows[sel]
        # Ensure this patient has values in time period of interest
        if len(sel) < 2:
            no_recs_count += 1
            if v:
                print(str(idx) + ', removed due to not enough values in the time period of interest')
                exc_log.write(str(idx) + ', removed due to not enough values in the time period of interest\n')
            continue

        # Get Baseline or remove if no admit dates provided
        bsln_idx = np.where(bsln_m[:, 0] == idx)[0]

        if bsln_idx.size == 0:
            no_admit_info_count += 1
            if v:
                print(str(idx) + ', removed due to missing admission info')
                exc_log.write(str(idx) + ', removed due to missing admission info\n')
            continue
        else:
            bsln_idx = bsln_idx[0]
        bsln = bsln_m[bsln_idx, bsln_scr_loc]
        if str(bsln).lower() == 'nan' or str(bsln).lower() == 'none' or str(bsln).lower() == 'nat':
            no_bsln_count += 1
            if v:
                print(str(idx) + ', removed due to missing baseline')
                exc_log.write(str(idx) + ', removed due to missing baseline\n')
            continue
        bsln = float(bsln)
        if bsln >= 4.0:
            no_bsln_count += 1
            if v:
                print(str(idx) + ', removed due to baseline SCr > 4.0')
                exc_log.write(str(idx) + ', removed due to baseline SCr > 4.0\n')
            continue

        admit = str(bsln_m[bsln_idx, admit_loc]).split('.')[0]
        if admit == 'nat':
            no_admit_info_count += 1
            if v:
                print(str(idx) + ', removed due to missing admission info')
                exc_log.write(str(idx) + ', removed due to missing admission info\n')
            continue
        admit = datetime.datetime.strptime(admit, '%Y-%m-%d %H:%M:%S')

        discharge = str(bsln_m[bsln_idx, disch_loc]).split('.')[0]
        discharge = datetime.datetime.strptime(discharge, '%Y-%m-%d %H:%M:%S')

        # get mortality date if available
        dem_idx = np.where(dem_m[:, 0] == idx)[0]
        if dem_idx.size == 0:
            dem_count += 1
            if v:
                print(str(idx) + ', removed due to missing demographics')
                exc_log.write(str(idx) + ', removed due to missing demographics\n')
            continue
        else:
            dem_idx = dem_idx[0]
        mort_date = 'NA'
        for i in range(len(dod_locs)):
            try:
                mdate = dem_m[dem_idx, dod_locs[i]]
                mort_date = datetime.datetime.strptime(mdate, '%Y-%m-%d %H:%M:%S')
            except:
                a = 1
        if mort_date != 'NA':
            death_dur = (mort_date - admit).total_seconds() / (60 * 60 * 24)
        else:
            death_dur = np.nan
        #         if mdate - admit < datetime.timedelta(death_excl_dur):
        #             skip = True
        #             death_count += 1
        #             if v:
        #                 print(str(idx) + ', removed due to death in specified window')
        #                 log.write(str(idx) + ', removed due to death in specified window\n')
        #             continue
        # if skip:
        #     continue

        # get dob, sex, and race

        dob = str(dem_m[dem_idx, dob_loc]).split('.')[0]
        dob = datetime.datetime.strptime(dob, '%Y-%m-%d %H:%M:%S')
        sex = dem_m[dem_idx, sex_loc]
        race = dem_m[dem_idx, eth_loc]
        age = admit - dob
        age = age.total_seconds() / (60 * 60 * 24 * 365)
        if age < 18:
            print(str(idx) + ', removed because age < 18 yrs')
            continue

        # remove if ESRD status
        esrd_idx = np.where(esrd_m[:, 0] == idx)[0]
        if esrd_idx.size > 0:
            esrd_idx = esrd_idx[0]
            for loc in esrd_locs:
                if np.any(str(esrd_m[esrd_idx, loc]).lower() != 'nan'):
                    skip = True
                    # ids = np.delete(ids, count)
                    esrd_count += 1
                    if v:
                        print(str(idx) + ', removed due to ESRD status')
                        exc_log.write(str(idx) + ', removed due to ESRD status\n')
                    break
        if skip:
            continue

        # remove patients with required demographics missing
        if str(sex) == 'nan' or str(race) == 'nan':
            # ids = np.delete(ids, count)
            dem_count += 1
            if v:
                print(str(idx) + ', removed due to missing demographics')
                exc_log.write(str(idx) + ', removed due to missing demographics\n')
            continue

        # remove patients with baseline GFR < 15
        gfr = calc_gfr(bsln, sex, race, age)
        if gfr < 15:
            # ids = np.delete(ids, count)
            gfr_count += 1
            if v:
                print(str(idx) + ', removed due to initial GFR too low')
                exc_log.write(str(idx) + ', removed due to initial GFR too low\n')
            continue

        # # get discharge date and check for multiple separate ICU stays
        # # add code to make sure this discharge corresponds to the analyzed admission
        # all_drows = np.where(date_m[:, 0] == idx)[0]
        # delta = datetime.timedelta(0)
        # discharge = datetime.datetime(1000, 1, 1, 1)
        # for i in range(len(all_drows)):
        #     start = str(date_m[all_drows[i], icu_locs[1]]).split('.')[0]
        #     start = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
        #     if start > discharge:
        #         discharge = start
        #     td = datetime.timedelta(0)
        #     for j in range(len(all_drows)):
        #         tdate = str(date_m[all_drows[j], icu_locs[0]]).split('.')[0]
        #         tdate = datetime.datetime.strptime(tdate, '%Y-%m-%d %H:%M:%S')
        #         if tdate > start:
        #             if td == datetime.timedelta(0):
        #                 td = tdate - start
        #             elif (tdate - start) < td:
        #                 td = tdate - start
        #     if delta == datetime.timedelta(0):
        #         delta = td
        #     elif delta < td:
        #         delta = td
        # # if delta > datetime.timedelta(3):
        # #    #  np.delete(ids, count)
        # #     gap_icu_count += 1
        # #     if v:
        # #         print(str(idx)+', removed due to different ICU stays > 3 days apart')
        # #         exc_log.write(str(idx)+', removed due to different ICU stays > 3 days apart\n')
        # #     continue
        #
        # # # remove patients who died <48 hrs after indexed admission
        # disch_disp = date_m[all_drows[0], d_disp_loc]
        # if type(disch_disp) == np.ndarray:
        #     disch_disp = disch_disp[0]
        # disch_disp = str(disch_disp).upper()
        # if 'LESS THAN' in disch_disp:
        #    #  np.delete(ids, count)
        #     lt48_count += 1
        #     if v:
        #         print(str(idx)+', removed due to death within 48 hours of admission')
        #         exc_log.write(str(idx)+', removed due to  death within 48 hours of admission\n')
        #     continue

        # get duration vector
        tdates = scr_all_m[keep, scr_date_loc]
        tdates = [datetime.datetime.strptime(str(tdates[i]).split('.')[0], '%Y-%m-%d %H:%M:%S') for i in
                  range(len(tdates))]
        duration = [tdates[x] - admit for x in range(len(tdates))]
        duration = np.array(duration)

        tdays = np.array([(x - admit).days for x in tdates])

        if v:
            print('%d\t%s\t%s\t%.3f\t%s\t%.3f' % (idx, admit, discharge, bsln, mort_date, death_dur))
            log.write('%d,%s,%s,%.3f,%s,%.3f\n' % (idx, admit, discharge, bsln, mort_date, death_dur))
        # d_disp.append(disch_disp)
        bslns.append(bsln)
        bsln_gfr.append(gfr)

        tmask = mask[keep]
        tmasks.append(tmask)
        dmask = dia_mask[keep]
        dmasks.append(dmask)
        scr.append(scr_all_m[keep, scr_val_loc])
        dates.append(tdates)
        ages.append(age)
        days.append(tdays)

        tmin = duration[0].total_seconds() / (60 * 60)
        tmax = duration[-1].total_seconds() / (60 * 60)
        ids_out.append(idx)
        t_range.append([tmin, tmax])
        count += 1
    bslns = np.array(bslns)
    if v:
        print('# Patients Kept: ' + str(count))
        print('# Patients removed for ESRD: ' + str(esrd_count))
        print('# Patients w/ GFR < 15: ' + str(gfr_count))
        print('# Patients w/ no admit info: ' + str(no_admit_info_count))
        print('# Patients w/ missing demographics: ' + str(dem_count))
        print('# Patients w/ < 2 ICU records: ' + str(no_recs_count))
        print('# Patients w/ no valid baseline: ' + str(no_bsln_count))
        exc_log.write('# Patients Kept: ' + str(count) + '\n')
        exc_log.write('# Patients removed for ESRD: ' + str(esrd_count) + '\n')
        exc_log.write('# Patients w/ GFR < 15: ' + str(gfr_count) + '\n')
        exc_log.write('# Patients w/ no admit info: ' + str(no_admit_info_count) + '\n')
        exc_log.write('# Patients w/ missing demographics: ' + str(dem_count) + '\n')
        exc_log.write('# Patients w/ < 2 ICU records: ' + str(no_recs_count) + '\n')
        exc_log.write('# Patients w/ no valid baseline: ' + str(no_bsln_count) + '\n')
    del scr_all_m
    del bsln_m
    del date_m
    del dem_m
    return ids_out, scr, dates, days, tmasks, dmasks, bslns, bsln_gfr, t_range, ages


# %%
def linear_interpo(scr, ids, dates, masks, dmasks, scale, log, v=True):
    post_interpo = []
    dmasks_interp = []
    days_interps = []
    interp_masks = []
    count = 0
    if v:
        log.write('Raw SCr\n')
        log.write('Stretched SCr\n')
        log.write('Interpolated\n')
        log.write('Original Dialysis\n')
        log.write('Interpolated Dialysis\n')
    print('Interpolating missing values')
    for i in range(len(scr)):
        print('Patient #' + str(ids[i]))
        mask = masks[i]
        dmask = dmasks[i]
        print(mask)
        tmin = datetime.datetime.strptime(str(dates[i][0]).split('.')[0], '%Y-%m-%d %H:%M:%S')
        tmax = datetime.datetime.strptime(str(dates[i][-1]).split('.')[0], '%Y-%m-%d %H:%M:%S')
        tstart = tmin.hour
        n = nbins(tmin, tmax, scale)
        thisp = np.repeat(-1., n)
        interp_mask = np.zeros(n, dtype=int)
        this_start = datetime.datetime.strptime(str(dates[i][0]).split('.')[0], '%Y-%m-%d %H:%M:%S')
        thisp[0] = scr[i][0]
        dmask_i = np.repeat(-1, len(thisp))
        dmask_i[0] = dmask[0]
        for j in range(1, len(scr[i])):
            tdate = datetime.datetime.strptime(str(dates[i][j]).split('.')[0], '%Y-%m-%d %H:%M:%S')
            dt = (tdate - this_start).total_seconds()
            idx = int(math.floor(dt / (60 * 60 * scale)))
            if mask[j] != -1:
                thisp[idx] = scr[i][j]
                interp_mask[idx] = 1
            dmask_i[idx] = dmask[j]
        for j in range(len(dmask_i)):
            if dmask_i[j] != -1:
                k = j + 1
                while k < len(dmask_i) and dmask_i[k] == -1:
                    dmask_i[k] = dmask_i[j]
                    k += 1
        print(str(thisp))
        if v:
            log.write('%d\n' % (ids[i]))
            log.write(arr2str(scr[i]) + '\n')
            log.write(arr2str(thisp) + '\n')
        print(dmask_i)
        dmasks_interp.append(dmask_i)
        interp_masks.append(interp_mask)
        j = 0
        while j < len(thisp):
            if thisp[j] == -1:
                pre_id = j - 1
                pre_val = thisp[pre_id]
                while thisp[j] == -1 and j < len(thisp) - 1:
                    j += 1
                post_id = j
                post_val = thisp[post_id]
                if post_val == -1:
                    post_val = pre_val
                step = (post_val - pre_val) / (post_id - pre_id)
                for k in range(pre_id + 1, post_id + 1):
                    thisp[k] = thisp[k - 1] + step
            j += 1
        if v:
            log.write(arr2str(thisp) + '\n')
            log.write(arr2str(dmask) + '\n')
            log.write(arr2str(dmask_i) + '\n')
            log.write('\n')
        print(str(thisp))
        post_interpo.append(thisp)
        interp_len = len(thisp)

        if tstart >= 18:
            n_zeros = 1
        elif tstart >= 12:
            n_zeros = 2
        elif tstart >= 6:
            n_zeros = 3
        else:
            n_zeros = 4

        days_interp = np.zeros(interp_len, dtype=int)
        tday = 1
        ct = 0
        for i in range(n_zeros, interp_len):
            days_interp[i] = tday
            if ct == 3:
                ct = 0
                tday += 1
            else:
                ct += 1
        days_interps.append(days_interp)

        count += 1
    return post_interpo, dmasks_interp, days_interps, interp_masks


# %%
def nbins(start, stop, scale):
    dt = (stop - start).total_seconds()
    div = scale * 60 * 60  # hrs * minutes * seconds
    bins, _ = divmod(dt, div)
    return int(bins + 1)


# %%
def get_baselines(date_m, hosp_locs, scr_all_m, scr_val_loc, scr_date_loc, scr_desc_loc,
                  dem_m, sex_loc, eth_loc, dob_m, dob_loc, fname, outp_rng=(1, 365), inp_rng=(7, 365)):
    log = open(fname, 'w')
    log.write('ID,bsln_val,bsln_type,bsln_date,admit_date,time_delta\n')
    cur_id = None

    out_lim = (datetime.timedelta(outp_rng[0]), datetime.timedelta(outp_rng[1]))
    inp_lim = (datetime.timedelta(inp_rng[0]), datetime.timedelta(inp_rng[1]))

    for i in range(len(date_m)):
        idx = date_m[i, 0]
        if cur_id != idx:
            cur_id = idx
            log.write(str(idx) + ',')
            # determine earliest admission date
            admit = datetime.datetime.now()
            didx = np.where(date_m[:, 0] == idx)[0]
            for did in didx:
                if date_m[did, hosp_locs[0]] < admit:
                    admit = date_m[did, hosp_locs[0]]
            if type(admit) == np.ndarray:
                admit = admit[0]

            # find indices of all SCr values for this patient
            all_rows = np.where(scr_all_m[:, 0] == idx)[0]

            # extract record types
            # i.e. indexed admission, before indexed, after indexed
            scr_desc = scr_all_m[all_rows, scr_desc_loc]
            for j in range(len(scr_desc)):
                scr_desc[j] = scr_desc[j].split()[0].upper()

            scr_tp = scr_all_m[all_rows, scr_desc_loc]
            for j in range(len(scr_tp)):
                scr_tp[j] = scr_tp[j].split()[-1].upper()

            # find indexed admission rows for this patient
            idx_rows = np.where(scr_desc == 'INDEXED')[0]
            idx_rows = all_rows[idx_rows]

            pre_admit_rows = np.where(scr_desc == 'BEFORE')[0]
            inp_rows = np.where(scr_tp == 'INPATIENT')[0]
            out_rows = np.where(scr_tp == 'OUTPATIENT')[0]

            pre_inp_rows = np.intersect1d(pre_admit_rows, inp_rows)
            pre_inp_rows = list(all_rows[pre_inp_rows])

            pre_out_rows = np.intersect1d(pre_admit_rows, out_rows)
            pre_out_rows = list(all_rows[pre_out_rows])

            # default values
            bsln_val = None
            bsln_date = None
            bsln_type = None
            bsln_delta = None

            # find the baseline
            if len(idx_rows) == 0:  # no indexed tpts, so disregard entirely
                bsln_type = 'No_indexed_values'
            elif np.all([np.isnan(scr_all_m[x, scr_val_loc]) for x in idx_rows]):
                bsln_type = 'No_indexed_values'
            # BASELINE CRITERIUM A
            # first look for valid outpatient values before admission
            if len(pre_out_rows) != 0 and bsln_type is None:
                row = pre_out_rows.pop(-1)
                this_date = scr_all_m[row, scr_date_loc]
                delta = admit - this_date
                # find latest point that is > 24 hrs prior to admission
                while delta < out_lim[0] and len(pre_out_rows) > 0:
                    row = pre_out_rows.pop(-1)
                    this_date = scr_all_m[row, scr_date_loc]
                    delta = admit - this_date
                # if valid point found save it
                if out_lim[0] < delta < out_lim[1]:
                    bsln_date = str(this_date).split('.')[0]
                    bsln_val = scr_all_m[row, scr_val_loc]
                    bsln_type = 'OUTPATIENT'
                    bsln_delta = delta.total_seconds() / (60 * 60 * 24)
            # BASLINE CRITERIUM B
            # no valid outpatient, so look for valid inpatient
            if len(pre_inp_rows) != 0 and bsln_type is None:
                row = pre_inp_rows.pop(-1)
                this_date = scr_all_m[row, scr_date_loc]
                delta = admit - this_date
                # find latest point that is > 24 hrs prior to admission
                while delta < inp_lim[0] and len(pre_inp_rows) > 0:
                    row = pre_inp_rows.pop(-1)
                    this_date = scr_all_m[row, scr_date_loc]
                    delta = admit - this_date
                # if valid point found save it
                if inp_lim[0] < delta < inp_lim[1]:
                    bsln_date = str(this_date).split('.')[0]
                    bsln_val = scr_all_m[row, scr_val_loc]
                    bsln_type = 'INPATIENT'
                    bsln_delta = delta.total_seconds() / (60 * 60 * 24)
            # BASELINE CRITERIUM C
            # no values prior to admission, calculate MDRD derived
            if bsln_type is None:
                dem_idx = np.where(dem_m[:, 0] == cur_id)[0][0]
                dob_idx = np.where(dob_m[:, 0] == cur_id)[0]
                if dob_idx.size == 0:
                    bsln_type = 'no_dob'
                else:
                    dob_idx = dob_idx[0]
                    sex = dem_m[dem_idx, sex_loc]
                    eth = dem_m[dem_idx, eth_loc]
                    dob = dob_m[dob_idx, dob_loc]
                    age = float((admit - dob).total_seconds()) / (60 * 60 * 24 * 365)
                    if age > 0:
                        bsln_val = baseline_est_gfr_mdrd(75, sex, eth, age)
                        bsln_date = str(admit).split('.')[0]
                        bsln_type = 'mdrd'
                        bsln_delta = 'na'
            admit = str(admit).split('.')[0]
            log.write(str(bsln_val) + ',' + str(bsln_type) + ',' + str(bsln_date) + ',' +
                      str(admit) + ',' + str(bsln_delta) + '\n')
    log.close()


# %%
def get_baselines_dallas(date_m, icu_locs, bsln_loc, bsln_date_loc, bsln_typ_loc, fname):
    log = open(fname, 'w')
    log.write('ID,bsln_val,bsln_type,bsln_date,admit_date,disch_date,time_delta\n')

    for i in range(len(date_m)):
        idx = date_m[i, 0]
        log.write(str(idx) + ',')
        # determine earliest admission date
        tstr = str(date_m[i, icu_locs[0]]).lower()
        if tstr != 'nat' and tstr != 'nan':
            admit = datetime.datetime.strptime(tstr, "%Y-%m-%d %H:%M:%S")
        else:
            admit = 'nat'
        tstr = str(date_m[i, icu_locs[1]]).lower()
        if tstr != 'nat' and tstr != 'nan':
            disch = datetime.datetime.strptime(tstr, "%Y-%m-%d %H:%M:%S")
        else:
            disch = 'nat'
        bsln_val = date_m[i, bsln_loc]
        bsln_type = date_m[i, bsln_typ_loc]
        tstr = str(date_m[i, bsln_date_loc]).lower()
        if tstr != 'nat' and tstr != 'nan':
            bsln_date = datetime.datetime.strptime(tstr, "%Y-%m-%d %H:%M:%S")
        else:
            bsln_date = 'nat'
        if admit != 'nat' and bsln_date != 'nat':
            bsln_delta = (admit - bsln_date).total_seconds() / (60 * 60 * 24)
        else:
            bsln_delta = 'nan'

        log.write(str(bsln_val) + ',' + str(bsln_type) + ',' + str(bsln_date) + ',' +
                  str(admit) + ',' + str(disch) + ',' + str(bsln_delta) + '\n')
    log.close()


# %%
# def calc_gfr(bsln, date_base, sex, race, dob):
#     if type(date_base) == str or type(date_base) == unicode:
#         date_base = datetime.datetime.strptime(date_base.split('.')[0], '%Y-%m-%d %H:%M:%S')
#     if type(dob) == str or type(dob) == unicode:
#         dob = datetime.datetime.strptime(dob.split('.')[0], '%Y-%m-%d %H:%M:%S')
#     time_diff = date_base - dob
#     year_val = time_diff.years
#     mon_val = time_diff.months
#     age = year_val+mon_val/12
#     min_value = 1
#     max_value = 1
#     race_value = 1
#     if sex == 'M':
#         k_value = 0.9
#         a_value = -0.411
#         f_value = 1
#     else:  # female
#         k_value = 0.7
#         a_value = -0.329
#         f_value = 1.018
#     if race == "BLACK/AFR AMERI":
#         race_value = 1.159
#     if bsln/k_value < 1:
#         min_value = bsln/k_value
#     else:
#         max_value = bsln/k_value
#     min_power = math.pow(min_value, a_value)
#     max_power = math.pow(max_value, -1.209)
#     age_power = math.pow(0.993, age)
#     gfr = 141 * min_power * max_power * age_power * f_value * race_value
#     return gfr
def calc_gfr(scr, sex, race, age):
    if sex == 'M':
        sex_val = 1.0
    else:  # female
        sex_val = 0.742
    if race == "BLACK/AFR AMERI":
        race_val = 1.212
    else:
        race_val = 1.0
    gfr = 174 * math.pow(scr, -1.154) * math.pow(age, -0.203) * sex_val * race_val
    return gfr


# %%
def baseline_est_gfr_epi(gfr, sex, race, age):
    race_value = 1
    sel = 0
    if sex:
        k_value = 0.9
        a_value = -0.411
        f_value = 1
    else:  # female
        k_value = 0.7
        a_value = -0.329
        f_value = 1.018
    if sex and not race and age >= 109:
        sel = 1
    if sex and race and age >= 88:
        sel = 1
    if not sex and not race and age >= 90:
        sel = 1
    if not sex and race and age >= 111:
        sel = 1

    if race:
        race_value = 1.159
    if sel == 0:
        a_value = -1.209
    numerator = gfr * math.pow(k_value, a_value)
    denominator = 141 * math.pow(0.993, age) * f_value * race_value
    scr = math.pow((numerator / denominator), 1. / a_value)

    return scr


def load_bsln_dates(bsln_file):
    bsln_dates = np.loadtxt(bsln_file, delimiter=',', usecols=3, dtype=str)
    b_dates = []
    for i in range(len(bsln_dates)):
        try:
            b_dates.append(datetime.datetime.strptime(bsln_dates[i].split('.')[0], '%Y-%m-%d %H:%M:%S'))
        except:
            b_dates.append(datetime.datetime(1, 1, 1))
    b_dates = np.array(b_dates)
    return b_dates


def baseline_est_gfr_mdrd(gfr, sex, race, age):
    race_value = 1
    if sex:
        f_value = 1
    else:  # female
        f_value = 0.742
    if race:
        race_value = 1.212
    numerator = gfr
    denominator = 175 * age ** (-0.203) * f_value * race_value
    scr = (numerator / denominator) ** (1. / -1.154)
    return scr


# %%
def scr2kdigo(scr, base, masks, days, valid):
    kdigos = []
    for i in range(len(scr)):
        kdigo = np.zeros(len(scr[i]), dtype=int)
        for j in range(len(scr[i])):
            if masks[i][j] > 0:
                kdigo[j] = 4
                continue
            elif scr[i][j] <= (1.5 * base[i]):
                if j > 7:
                    window = np.where(days[i] >= days[i][j] - 2)[0]
                    window = window[np.where(window < j)[0]]
                    window = np.intersect1d(window, np.where(valid[i])[0])
                    if window.size > 0:
                        if scr[i][j] >= np.min(scr[i][window]) + 0.3:
                            kdigo[j] = 1
                        else:
                            kdigo[j] = 0
                    else:
                        kdigo[j] = 0
                else:
                    kdigo[j] = 0
            elif scr[i][j] < (2 * base[i]):
                kdigo[j] = 1
            elif scr[i][j] < (3 * base[i]):
                kdigo[j] = 2
            elif (scr[i][j] >= (3 * base[i])) or (scr[i][j] >= 4.0):
                kdigo[j] = 3
            elif scr[i][j] >= 4.0:
                kdigo[j] = 3
        kdigos.append(kdigo)
    return kdigos


# %%
def pairwise_dtw_dist(patients, days, ids, dm_fname, dtw_name, v=True,
                      mismatch=lambda y, yy: abs(y-yy),
                      extension=lambda y: 0,
                      dist=distance.braycurtis,
                      alpha=1.0, t_lim=7,
                      desc='DTW and Distance Calculation'):
    df = open(dm_fname, 'w')
    dis = []
    pdic = {}
    if v and dtw_name is not None:
        log = open(dtw_name, 'w')
    for i in tqdm(range(len(patients)), desc=desc):
        # if v:
            # print('#' + str(i + 1) + ' vs #' + str(i + 2) + ' to ' + str(len(patients)))
        sel = np.where(days[i] <= t_lim)[0]
        patient1 = np.array(patients[i])[sel]
        if tuple(patient1) in list(pdic):
            tlist = pdic[tuple(patient1)]
            start = len(tlist) - (len(patients) - i + 1)
            ct = 0
            for j in tqdm(range(i + 1, len(patients)), desc='Patient %d' % ids[i]):
                dis.append(tlist[start + ct])
        else:
            dlist = []
            for j in range(i + 1, len(patients)):
                df.write('%d,%d,' % (ids[i], ids[j]))
                sel = np.where(days[j] < t_lim)[0]
                patient2 = np.array(patients[j])[sel]
                if np.all(patients[i] == patients[j]):
                    df.write('%f\n' % 0)
                    dis.append(0)
                    dlist.append(0)
                else:
                    if len(patients[i]) > 1 and len(patients[j]) > 1:
                        dist, _, _, path = dtw_p(patient1, patient2, mismatch, extension, alpha)
                        p1_path = path[0]
                        p2_path = path[1]
                        p1 = [patient1[p1_path[x]] for x in range(len(p1_path))]
                        p2 = [patient2[p2_path[x]] for x in range(len(p2_path))]
                    elif len(patients[i]) == 1:
                        p1 = np.repeat(patient1[0], len(patient2))
                        p2 = patient2
                    elif len(patients[j]) == 1:
                        p1 = patient1
                        p2 = np.repeat(patient2[0], len(patient1))
                    if np.all(p1 == p2):
                        df.write('%f\n' % 0)
                        dis.append(0)
                        dlist.append(0)
                    else:
                        d = dist(p1, p2)
                        df.write('%f\n' % d)
                        dis.append(d)
                        dlist.append(d)
                if v and dtw_name is not None:
                    log.write(arr2str(p1, fmt='%d') + '\n')
                    log.write(arr2str(p2, fmt='%d') + '\n\n')
            pdic[tuple(patient1)] = dlist
    if v and dtw_name is not None:
        log.close()
    return dis


def dtw_p(x, y, mismatch=lambda y, yy: abs(y-yy),
                extension=lambda y: 0,
                alpha=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences with weighted penalty exponentiation.
    Designed for sequences of distinct integer values in the set [0, 1, 2, 3, 4]

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure

    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the warp path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = np.zeros((r + 1, c + 1))  # distance matrix
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = mismatch(x[i], y[j])
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += np.min((D0[i, j], D0[i, j + 1], D0[i + 1, j]))
    if len(x) == 1:
        path = np.zeros(len(y))
    elif len(y) == 1:
        path = np.zeros(len(x))
    else:
        path = _traceback(D0, x, y, extension, alpha)
    return D1[-1, -1] / sum(D1.shape), C, D1, path


def _traceback(D, x, y, extension=lambda y: 0,
                        alpha=1.0):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    px = extension(x[i])
    py = extension(y[j])
    while i > 0 or j > 0:
        tb = np.argmin((D[i, j], D[i, j + 1] + (alpha * px), D[i + 1, j] + (alpha * py)))
        if tb == 0:
            i -= 1
            j -= 1
            px = extension(x[i])
            py = extension(y[j])
        elif tb == 1:
            i -= 1
            px = extension(x[i])
            py += extension(y[j])
        else:  # (tb == 2):
            j -= 1
            px += extension(x[i])
            py = extension(y[j])
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)


# %%
def count_bsln_types(bsln_file, outfname):
    pre_outp = 0
    pre_inp = 0
    ind_inp = 0
    no_ind = 0
    all_rrt = 0

    f = open(bsln_file, 'r')
    _ = f.readline()
    for line in f:
        l = line.rstrip().split(',')
        t = l[2]
        if t == 'OUTPATIENT':
            pre_outp += 1
        elif t == 'INPATIENT':
            pre_inp += 1
        elif t.split()[0] == 'INDEXED':
            ind_inp += 1
        elif t == 'No_indexed_values':
            no_ind += 1
        elif t == 'none':
            all_rrt += 1

    lbls = ['Pre-admit Outpatient', 'Pre-admit Inpatient', 'Indexed Inpatient', 'All RRT', 'No Indexed Recs']
    counts = [pre_outp, pre_inp, ind_inp, all_rrt, no_ind]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar([0, 1, 2, 3, 4], [pre_outp, pre_inp, ind_inp, all_rrt, no_ind], tick_label=lbls)
    ax.set_xticklabels(lbls, rotation=45)
    plt.ylabel('Number of Patients')
    plt.title('Distribution of Baseline Types')
    plt.savefig('../DATA/bsln_dist.png')

    outf = open(outfname, 'w')
    for i in range(len(lbls)):
        outf.write(lbls[i] + ' - ' + str(counts[i]) + '\n')
        print lbls[i] + ' - ' + str(counts[i])
    f.close()
    outf.close()
    return zip([lbls, counts])


# %%
def rel_scr(scr_fname, bsln_fname):
    scr_f = open(scr_fname, 'r')
    bsln_f = open(bsln_fname, 'r')
    ids = []
    pct_chg = []
    abs_chg = []
    for scr in scr_f:
        idx = int(scr.split(',')[0])
        scr = np.array(scr.split(',')[1:], dtype=float)
        bsln = bsln_f.readline()
        bsln = float(bsln.split(',')[1])
        ab = scr - bsln
        pct = scr / bsln

        ids.append(idx)
        abs_chg.append(ab)
        pct_chg.append(pct)

    arr2csv('scr_pct_chg.csv', pct_chg, ids)
    arr2csv('scr_abs_chg.csv', abs_chg, ids)


# %%
def arr2csv(fname, inds, ids=None, fmt='%f', header=False):
    outFile = open(fname, 'w')
    if ids is None:
        ids = np.arange(len(inds))
    if header:
        outFile.write(header)
        outFile.write('\n')
    for i in range(len(inds)):
        outFile.write('%d' % (ids[i]))
        if np.size(inds[i]) > 1:
            for j in range(len(inds[i])):
                outFile.write(',' + fmt % (inds[i][j]))
        else:
            outFile.write(',' + fmt % (inds[i]))
        outFile.write('\n')
    outFile.close()


# %%
def str2csv(fname, inds, ids):
    outFile = open(fname, 'w')
    if len(np.shape(inds)) > 1:
        for i in range(len(inds)):
            outFile.write('%d' % (ids[i]))
            for j in range(len(inds[i])):
                outFile.write(',%s' % (inds[i][j]))
            outFile.write('\n')
    else:
        for i in range(len(inds)):
            outFile.write('%d,%s\n' % (ids[i], inds[i]))
    outFile.close()


# %%
def arr2str(arr, fmt='%f'):
    s = fmt % (arr[0])
    for i in range(1, len(arr)):
        s = s + ',' + fmt % (arr[i])
    return s


# %%
def get_xl_subset(file_name, sheet_names, ids):
    for sheet in sheet_names:
        ds = get_mat(file_name, sheet, 'STUDY_PATIENT_ID')
        mask = [ds['STUDY_PATIENT_ID'][x] in ids for x in range(len(ds['STUDY_PATIENT_ID']))]
        ds[mask].to_excel(sheet + '.xlsx')


# %%
def get_mat(fname, page_name, sort_id):
    return pd.read_excel(fname, page_name).sort_values(sort_id)


# %%
def load_csv(fname, ids, dt=float, skip_header=False, sel=None):
    res = []
    rid = []
    f = open(fname, 'r')
    if skip_header:
        if skip_header == 'keep':
            hdr = f.readline().rstrip().split(',')
        else:
            _ = f.readline()
    for line in f:
        l = line.rstrip()
        if ids is None or int(l.split(',')[0]) in ids:
            if sel is None:
                res.append(np.array(l.split(',')[1:], dtype=dt))
            else:
                res.append(np.array(l.split(',')[sel], dtype=dt))
            rid.append(int(l.split(',')[0]))
    if len(rid) != len(ids):
        print('Missing ids in file: ' + fname)
        return
    else:
        rid = np.array(rid)
        ids = np.array(ids)
        if len(rid) == len(ids):
            if not np.all(rid == ids):
                temp = res
                res = []
                for i in range(len(ids)):
                    idx = ids[i]
                    sel = np.where(rid == idx)[0][0]
                    res.append(temp[sel])
        try:
            if np.all([len(res[x]) == len(res[0]) for x in range(len(res))]):
                res = np.array(res)
                if res.ndim > 1:
                    if res.shape[1] == 1:
                        res = np.squeeze(res)
        except:
            res = res
        if skip_header == 'keep':
            return hdr, res
        else:
            return res


def load_dm(fname, ids):
    f = open(fname, 'r')
    dm = []
    for line in f:
        (id1, id2, d) = line.rstrip().split(',')
        id1 = int(id1)
        id2 = int(id2)
        if id1 in ids and id2 in ids:
            dm.append(float(d))
    return np.array(dm)


def descriptive_trajectory_features(kdigos, ids, days=None, t_lim=None, filename='descriptive_features.csv'):
    npts = len(kdigos)
    features = np.zeros((npts, 26))
    header = 'id,no_AKI,peak_at_KDIGO1,peak_at_KDIGO2,peak_at_KDIGO3,peak_at_KDIGO3D,KDIGO1_at_admit,KDIGO2_at_admit,' + \
             'KDIGO3_at_admit,KDIGO3D_at_admit,KDIGO1_at_disch,KDIGO2_at_disch,KDIGO3_at_disch,KDIGO3D_at_disch,' + \
             'onset_lt_3days,onset_gte_3days,complete_recovery_lt_3days,multiple_hits,KDIGO1_gt_24hrs,KDIGO2_gt_24hrs,' + \
             'KDIGO3_gt_24hrs,KDIGO4_gt_24hrs,flat,strictly_increase,strictly_decrease,slope_posTOneg,slope_negTOpos'
    for i in range(len(kdigos)):
        kdigo = kdigos[i]
        if days is not None:
            tdays = days[i]
            sel = np.where(tdays <= t_lim)[0]
            kdigo = kdigo[sel]
        kdigo1 = np.where(kdigo == 1)[0]
        kdigo2 = np.where(kdigo == 2)[0]
        kdigo3 = np.where(kdigo == 3)[0]
        kdigo4 = np.where(kdigo == 4)[0]
        # No AKI
        if np.all(kdigo == 0):
            features[i, 0] = 1
        # KDIGO 1 Peak
        temp = 0  # previous score
        direction = 0  # 1: Increased     -1: Decreased
        for j in range(len(kdigo)):
            if kdigo[j] < temp:
                if direction == 1:
                    features[i, temp] = 1
                direction = -1

            elif kdigo[j] > temp:
                direction = 1
            temp = kdigo[j]

        if direction == 1:
            features[i, temp] = 1
        '''
        if kdigo[0] > 0:
            temp = kdigo[0]
            i = 1
            while i < len(kdigo):
                if kdigo[i] > temp:
                    break
                elif kdigo[i] < temp:
                    if temp == 1:
                        features[i, 1] = 1
                    elif temp == 2:
                        features[i, 2] = 1
                    elif temp == 3:
                        features[i, 3] = 1
                    elif temp == 4:
                        features[i, 4] = 1
                    break
                i += 1
        '''
        # KDIGO @ admit
        if kdigo[0] == 1:
            features[i, 5] = 1
        elif kdigo[0] == 2:
            features[i, 6] = 1
        elif kdigo[0] == 3:
            features[i, 7] = 1
        elif kdigo[0] == 4:
            features[i, 8] = 1

        # KDIGO @ discharge
        if kdigo[-1] == 1:
            features[i, 9] = 1
        elif kdigo[-1] == 2:
            features[i, 10] = 1
        elif kdigo[-1] == 3:
            features[i, 11] = 1
        elif kdigo[-1] == 4:
            features[i, 12] = 1

        # AKI onset <3 days after admit
        if kdigo[0] == 0 and np.any(kdigo[:12] > 0):
            features[i, 13] = 1

        # AKI onset >=3 days after admit
        if np.all(kdigo[:12] == 0) and np.any(kdigo[12:] > 0):
            features[i, 14] = 1

        # Complete recovery within 3 days
        if len(kdigo) >= 12:
            if np.any(kdigo > 0) and np.all(kdigo[12:] == 0):
                features[i, 15] = 1
        else:
            if kdigo[-1] == 0:
                features[i, 15] = 1

        # Multiple hits separated by >= 24 hrs
        temp = 0
        count = 0
        for j in range(len(kdigo)):
            if kdigo[j] > 0:
                if temp == 0:
                    if count >= 4:
                        features[i, 16] = 1
                temp = 1
                count = 0
            else:
                if temp == 1:
                    count = 1
                temp = 0
                if count > 0:
                    count += 1

        # >=24 hrs at KDIGO 1
        if len(kdigo1) >= 4:
            features[i, 17] = 1
        # >=24 hrs at KDIGO 2
        if len(kdigo2) >= 4:
            features[i, 18] = 1
        # >=24 hrs at KDIGO 3
        if len(kdigo3) >= 4:
            features[i, 19] = 1
        # >=24 hrs at KDIGO 3D
        if len(kdigo4) >= 4:
            features[i, 20] = 1
        # Flat trajectory
        if np.all(kdigo == kdigo[0]):
            features[i, 21] = 1

        # KDIGO strictly increases
        diff = kdigo[1:] - kdigo[:-1]
        if np.any(diff > 0):
            if np.all(diff >= 0):
                features[i, 22] = 1
        '''
        j = 1
        while j < len(kdigo):
            if kdigo[j] < kdigo[j-1]:
                break
            if j == len(kdigo) - 1:
                if kdigo[j] >= kdigo[0]:
                    features[i, 22] = 1
            j += 1
        '''

        # KDIGO strictly decreases
        if np.any(diff < 0):
            if np.all(diff <= 0):
                features[i, 23] = 1
        '''
        j = 1
        while j < len(kdigo):
            if kdigo[j] > kdigo[j - 1]:
                break
            if j == len(kdigo) - 1:
                if kdigo[j] <= kdigo[0]:
                    features[i, 23] = 1
            j += 1
        '''

        # Slope changes sign
        direction = 0
        temp = kdigo[0]
        for j in range(len(kdigo)):
            if kdigo[j] < temp:
                # Pos to neg
                if direction == 1:
                    features[i, 24] = 1
                direction = -1
            elif kdigo[j] > temp:
                # Neg to pos
                if direction == -1:
                    features[i, 25] = 1
                direction = 1
            temp = kdigo[j]
    arr2csv(filename, features, ids, fmt='%d', header=header)
    return features


def template_trajectory_features(kdigos, ids, days=None,t_lim=None, filename='template_trajectory_features.csv',
                                 scores=np.array([0, 1, 2, 3, 4], dtype=int), npoints=3):
    combination = scores
    for i in range(npoints - 1):
        combination = np.vstack((combination, scores))
    npts = len(kdigos)
    templates = cartesian(combination)
    header = 'id'
    for i in range(len(templates)):
        header += ',' + str(templates[i])
    features = np.zeros((npts, len(templates)), dtype=int)
    for i in range(npts):
        kdigo = kdigos[i]
        if days is not None:
            tdays = days[i]
            sel = np.where(tdays <= t_lim)[0]
            kdigo = kdigo[sel]
        nwin = len(kdigo) - npoints + 1
        for j in range(nwin):
            tk = kdigo[j:j + npoints]
            sel = np.where(templates[:, 0] == tk[0])[0]
            loc = [x for x in sel if np.all(templates[x, :] == tk)]
            features[i, loc] += 1
    arr2csv(filename, features, ids, fmt='%d', header=header)
    return features


def slope_trajectory_features(kdigos, ids, days=None,t_lim=None, scores=np.array([0, 1, 2, 3, 4]), filename='slope_features.csv'):
    slopes = []
    header = 'ids'
    for i in range(len(scores)):
        slopes.append(scores[i] - scores[0])
        header += ',%d' % (scores[i] - scores[0])
    for i in range(len(slopes)):
        if slopes[i] > 0:
            slopes.append(-slopes[i])
            header += ',%d' % (-slopes[i])
    slopes = np.array(slopes)
    npts = len(kdigos)
    features = np.zeros((npts, len(slopes)), dtype=int)
    for i in range(npts):
        kdigo = kdigos[i]
        if days is not None:
            tdays = days[i]
            sel = np.where(tdays <= t_lim)[0]
            kdigo = kdigo[sel]
        nwin = len(kdigo) - 1
        for j in range(nwin):
            ts = kdigo[j + 1] - kdigo[j]
            loc = np.where(slopes == ts)[0][0]
            features[i, loc] += 1
    arr2csv(filename, features, ids, fmt='%d', header=header)
    return features


def normalize_features(ds):
    norm = mms()
    nds = norm.fit_transform(ds)
    return nds


def feature_voting(ds):
    features = np.zeros(ds.shape[1])
    flags = np.zeros(ds.shape[1])
    for i in range(ds.shape[1]):
        avg = np.mean(ds[:, i], axis=0)
        std = np.std(ds[:, i], axis=0)
        if avg > 0.5:
            features[i] = 1
        if avg < 2 * std:
            flags[i] = 1
    print('Features with High Variability:')
    sel = np.where(flags)[0]
    for idx in sel:
        print('Feature #' + str(idx))
    return features


def perf_measure(y_actual, y_hat):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            tp += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            fp += 1
        if y_actual[i] == y_hat[i] == 0:
            tn += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            fn += 1
    prec = precision_score(y_actual, y_hat)
    rec = recall_score(y_actual, y_hat)
    f1 = f1_score(y_actual, y_hat)
    return np.array((prec, rec, f1, tp, fp, tn, fn))


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype
    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)
    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def feature_selection(data, lbls, method='univariate', params=[]):
    '''

    :param data: Matrix of the form n_samples x n_features
    :param lbls: n_samples binary features
    :param method: {'variance', 'univariate', 'recursive', 'FromModel'}

    :param params: variance     - [float pct]
                   univariate   - [int n_feats, function scoring]
                   recursive    - [obj estimator, int cv]
                   FromModel
                        linear  - [float c]
                        tree    - []

    :return:
    '''

    if method == 'variance':
        assert len(params) == 1
        pct = params[0]
        sel = VarianceThreshold(threshold=(pct * (1 - pct)))
        sf = sel.fit_transform(data)
        return sel, sf

    elif method == 'univariate':
        assert len(params) == 2
        n_feats = params[0]
        scoring = params[1]
        sel = SelectKBest(scoring, k=n_feats)
        sf = sel.fit_transform(data, lbls)
        return sel, sf

    elif method == 'recursive':
        assert len(params) == 2
        estimator = params[0]
        cv = params[1]
        rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(cv),
                      scoring='accuracy')
        rfecv.fit(data, lbls)
        sel = rfecv.support_
        sf = data[:, sel]
        return rfecv, sf

    elif method == 'linear':
        assert len(params) == 1
        c = params[0]
        lsvc = LinearSVC(C=c, penalty='l1', dual=False).fit(data, lbls)
        model = SelectFromModel(lsvc, prefit=True)
        sf = model.transform(data)
        return model, sf

    elif method == 'tree':
        assert params == []
        clf = RandomForestClassifier()
        clf.fit(data, lbls)
        model = SelectFromModel(clf, prefit=True)
        sf = model.transform(data)
        return model, sf

    else:
        print("Please select one of the following methods:")
        print("[variance, univariate, recursive, linear, tree")


def excel_to_h5(file_name, h5_name, keep_dic, append=False):
    xl = pd.read_excel(file_name, sheetname=None, header=0)
    if append:
        f = h5py.File(h5_name, 'r+')
    else:
        f = h5py.File(h5_name, 'w')

    for sheet in keep_dic.keys():
        grp = f.create_group(sheet)
        sheet_len = len(xl[sheet])
        for col in keep_dic[sheet]:
            if sheet == 'DIAGNOSIS' and col[1] == 'transplant':
                vals = xl[sheet][col[0]]
                ds = grp.create_dataset(col[1], data=np.zeros(sheet_len, ), dtype=int)
                for i in range(len(vals)):
                    if type(vals[i]) == unicode:
                        if 'KID' in vals[i] and 'TRANS' in vals[i]:
                            ds[i] = 1
            elif type(xl[sheet][col[0]][0]) == pd.tslib.Timestamp:
                ds = grp.create_dataset(col[1], shape=(sheet_len,), dtype='|S19')
                vals = xl[sheet][col[0]].values
                if vals.dtype == 'O':
                    val = val.astype(str)
                for i in range(len(vals)):
                    ds[i] = str(vals[i]).split('.')[0]
            else:
                try:
                    grp.create_dataset(col[1], data=xl[sheet][col[0]].values)
                except:
                    grp.create_dataset(col[1], data=xl[sheet][col[0]].values.astype(str))


def combine_labels(lbls):
    '''
    :param lbls: -   Tuple of cluster labels
                     lbls[0] contains the list of cluster labels for the initial clustering
                     lbls[1:] contain two entries:
                        lbls[n][0] = list of parent clusters
                                     i.e. if this corresponds to breaking down cluster 2, followed by the corresponding
                                     cluster 5, this would be (2, 5)
                        lbls[n][1] = corresponding labels for further stratification
    :return:
    '''
    out = np.array(lbls[0], dtype=int, copy=True)
    lbl_sets = {'root': np.array(out, copy=True)}
    for i in range(1, len(lbls)):
        shift = np.max(out)
        p = lbls[i][0]
        nlbls = lbls[i][1]
        lbl_key = '%d' % p[0]
        for j in range(1, len(p)):
            lbl_key += (',%d' % p[j])
        lbl_sets[lbl_key] = np.array(nlbls, copy=True)
        sel = np.where(lbls[0] == p[0])[0]
        if len(p) == 1:
            nlbls += shift
            out[sel] = nlbls
        else:
            lbl_key_temp = '%d' % p[0]
            for j in range(1, len(p)):
                sel = sel[np.where(lbl_sets[lbl_key_temp] == p[j])]
                lbl_key_temp += (',%d' % p[j])
            nlbls += shift
            out[sel] = nlbls
    all_lbls = np.unique(out)
    glob_shift = np.min(out[np.where(out >= 0)]) - 1
    out[np.where(out >= 0)] -= glob_shift
    for i in range(len(all_lbls) - 1):
        if all_lbls[i] < 0:
            continue
        if all_lbls[i + 1] - all_lbls[i] > 1:
            gap = all_lbls[i + 1] - all_lbls[i] - 1
            out[np.where(out > all_lbls[i])] -= gap
            all_lbls = np.unique(out)
    return out


def cross_val_classify(features, labels, n_splits=10, clf_type='svm', clf_params=[5, 'entropy', 'sqrt']):
    X = features[:]
    y = labels[:]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

    perf = np.array((n_splits, 7))
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train = data[train_idx]
        y_train = labels[train_idx]
        X_val = data[val_idx]
        y_val = labels[val_idx]
        if clf_type == 'svm':
            if len(clf_params) != 3:
                print('SVM requires 3 parameter arguments:')
                print('Kernel (\'rbf\', \'linear\', \'poly\', \'sigmoid\')')
                print('Gamma (float)')
                print('C (float)')
            clf = SVC(kernel=clf_params[0], gamma=clf_params[1], C=clf_params[2])
        elif clf_type == 'rf':
            if len(clf_params) != 3:
                print('RF requires 3 parameter arguments:')
                print('Num_Estimators (int)')
                print('Criterion (\'gini\', \'entropy\')')
                print('Max_Features (\'auto\', \'sqrt\',\'log2\')')
            clf = RandomForestClassifier(n_estimators=clf_params[0], criterion=clf_params[1],
                                         max_features=clf_params[2])
        else:
            print('Currently only supports SVM and RF')

        clf.fit(X_train, y_train)
        predicted = clf.predict(X_val)
        perf[i, :] = perf_measure(y_val, predicted)

    return perf


def daily_max_kdigo(scr, dates, bsln, admit_date, dmask, tlim=7):
    mk = []
    temp = dates[0].day
    tmax = 0
    # Prior 2 days minimum (IRRESPECTIVE OF TIME!!!)
    #   i.e. any record on day 2 is compared to the minimum value during days 0-1, even if the minimum is at the
    #        beginning of day 0 and the current point is at the end of day 2
    for i in range(len(scr)):
        date = dates[i]
        day = date.day
        if day != temp:
            mk.append(tmax)
            tmax = 0
            temp = day
        if dmask[i] > 0:
            tmax = 4
        elif scr[i] <= (1.5 * bsln):
            if date - admit_date >= datetime.timedelta(2):
                minim = scr[i]
                for j in range(i)[::-1]:
                    delta = date - dates[j]
                    if delta < datetime.timedelta(2):
                        break
                    if scr[i] >= minim + 0.3:
                        if tmax < 1:
                            tmax = 1
                        break
                    else:
                        if scr[i] < minim:
                            minim = scr[i]
        elif scr[i] < (2 * bsln):
            if tmax < 1:
                tmax = 1
        elif scr[i] < (3 * bsln):
            if tmax < 2:
                tmax = 2
        elif (scr[i] >= (3 * bsln)) or (scr[i] >= 4.0):
            if tmax < 3:
                tmax = 3
    mk.append(tmax)
    return mk


def cluster_feature_vectors(desc, temp, slope, lbls):
    clbls = np.unique(lbls)
    n_clust = len(clbls)
    n_desc = desc.shape[1]
    n_temp = temp.shape[1]
    n_slope = slope.shape[1]
    desc_c = np.zeros((n_clust, n_desc))
    temp_c = np.zeros((n_clust, n_temp))
    slope_c = np.zeros((n_clust, n_slope))
    for i in range(n_clust):
        idx = np.where(lbls == clbls[i])[0]
        tdesc = desc[idx, :]
        ttemp = temp[idx, :]
        tslope = slope[idx, :]

        tdesc = np.mean(tdesc, axis=0)
        tdesc[np.where(tdesc >= 0.5)] = 1
        tdesc[np.where(tdesc < 1)] = 0
        ttemp = np.mean(ttemp, axis=0)
        tslope = np.mean(tslope, axis=0)

        desc_c[i, :] = tdesc
        temp_c[i, :] = ttemp
        slope_c[i, :] = tslope
    return desc_c, temp_c, slope_c


def mismatch_penalty_func(*tcosts):
    '''
    Returns distance function for the mismatch penalty between any two KDIGO
     scores in the range [0, len(tcosts) - 1]
    :param tcosts: List of float values corresponding to transitions between
                   consecutive KDIGO scores. E.g.
                    tcosts[i] = cost(i, i + 1)
    :return:
    '''
    cost_dic = {}
    for i in range(len(tcosts)):
        cost_dic[tuple(set((i, i+1)))] = tcosts[i]
    for i in range(len(tcosts)):
        for j in range(i + 2, len(tcosts) + 1):
            cost_dic[tuple(set((i, j)))] = cost_dic[tuple(set((i, j-1)))] + cost_dic[tuple(set((j-1, j)))]

    def penalty(x, y):
        if x == y:
            return 0
        elif x < y:
            return cost_dic[tuple(set((x, y)))]
        else:
            return cost_dic[tuple(set((y, x)))]

    return penalty


def extension_penalty_func(*tcosts):
    costs = np.zeros(len(tcosts) + 1)
    for i in range(len(tcosts)):
        costs[i + 1] = tcosts[i]

    def penalty(x):
        return costs[x]

    return penalty


def get_custom_braycurtis(*tcosts, **kwargs):
    coordinates = np.zeros(len(tcosts) + 1)
    shift = 0
    if 'shift' in list(kwargs):
        shift = kwargs['shift']

    coordinates[0] = np.sum(tcosts) + shift
    for i in range(len(tcosts)):
        coordinates[i + 1] = coordinates[i] - tcosts[i]

    def dist(x, y):
        return distance.braycurtis(coordinates[x], coordinates[y])

    return dist


def get_pop_dist(*tcosts, **kwargs):
    coordinates = np.zeros(len(tcosts) + 1)
    for i in range(len(tcosts)):
        coordinates[i + 1] = coordinates[i] + tcosts[i]

    def dist(x, y):
        return np.sum(np.abs(coordinates[x] - coordinates[y]))

    return dist


def count_transitions(ids, kdigos, out_name):
    k_lbls = range(5)
    transitions = cartesian((k_lbls, k_lbls))
    t_counts = np.zeros((len(kdigos), len(transitions)))
    for i in range(len(kdigos)):
        for j in range(len(kdigos[i]) - 1):
            idx = np.intersect1d(np.where(transitions[:, 0] == kdigos[i][j])[0],
                                 np.where(transitions[:, 1] == kdigos[i][j + 1])[0])
            t_counts[i, idx] += 1

    header = 'id'
    for i in range(len(transitions)):
        header += ',' + str(transitions[i])

    rows = np.array(ids)[:, np.newaxis]
    with open(out_name, 'w') as f:
        f.write(header + '\n')
        np.savetxt(f, np.hstack((rows, t_counts)), delimiter=',', fmt='%d')
    return t_counts


def load_all_csv(datapath, sort_id='STUDY_PATIENT_ID'):
    print('Loading encounter info...')
    # Get IDs and find indices of all used metrics
    date_m = pd.read_csv(datapath + 'all_sheets/ADMISSION_INDX.csv')
    date_m.sort_values(by=[sort_id, 'HOSP_ADMIT_DATE'], inplace=True)
    hosp_locs = [date_m.columns.get_loc("HOSP_ADMIT_DATE"), date_m.columns.get_loc("HOSP_DISCHARGE_DATE")]
    icu_locs = [date_m.columns.get_loc("ICU_ADMIT_DATE"), date_m.columns.get_loc("ICU_DISCHARGE_DATE")]
    adisp_loc = date_m.columns.get_loc('DISCHARGE_DISPOSITION')
    date_m = date_m.values

    ### GET SURGERY SHEET AND LOCATION
    print('Loading surgery information...')
    surg_m = pd.read_csv(datapath + 'all_sheets/SURGERY_INDX.csv')
    surg_m.sort_values(by=sort_id, inplace=True)
    surg_des_loc = surg_m.columns.get_loc("SURGERY_DESCRIPTION")
    surg_m = surg_m.values

    ### GET DIAGNOSIS SHEET AND LOCATION
    print('Loading diagnosis information...')
    diag_m = pd.read_csv(datapath + 'all_sheets/DIAGNOSIS_new.csv')
    diag_m.sort_values(by=sort_id, inplace=True)
    diag_loc = diag_m.columns.get_loc("DIAGNOSIS_DESC")
    diag_nb_loc = diag_m.columns.get_loc("DIAGNOSIS_SEQ_NB")
    diag_m = diag_m.values

    print('Loading ESRD status...')
    # ESRD status
    esrd_m = pd.read_csv(datapath + 'all_sheets/ESRD_STATUS.csv')
    esrd_m.sort_values(by=sort_id, inplace=True)
    esrd_locs = [esrd_m.columns.get_loc("AT_ADMISSION_INDICATOR"),
                 esrd_m.columns.get_loc("DURING_INDEXED_INDICATOR"),
                 esrd_m.columns.get_loc("BEFORE_INDEXED_INDICATOR")]
    esrd_m = esrd_m.values

    # Dialysis dates
    print('Loading dialysis dates...')
    dia_m = pd.read_csv(datapath + 'all_sheets/RENAL_REPLACE_THERAPY.csv')
    dia_m.sort_values(by=sort_id, inplace=True)
    crrt_locs = [dia_m.columns.get_loc('CRRT_START_DATE'), dia_m.columns.get_loc('CRRT_STOP_DATE')]
    hd_locs = [dia_m.columns.get_loc('HD_START_DATE'), dia_m.columns.get_loc('HD_STOP_DATE')]
    pd_locs = [dia_m.columns.get_loc('PD_START_DATE'), dia_m.columns.get_loc('PD_STOP_DATE')]
    dia_m = dia_m.values

    # All SCR
    print('Loading SCr values (may take a while)...')
    scr_all_m = pd.read_csv(datapath + 'all_sheets/SCR_ALL_VALUES.csv')
    scr_all_m.sort_values(by=[sort_id, 'SCR_ENTERED'], inplace=True)
    scr_date_loc = scr_all_m.columns.get_loc('SCR_ENTERED')
    scr_val_loc = scr_all_m.columns.get_loc('SCR_VALUE')
    scr_desc_loc = scr_all_m.columns.get_loc('SCR_ENCOUNTER_TYPE')
    scr_all_m = scr_all_m.values

    # Demographics
    print('Loading demographics...')
    dem_m = pd.read_csv(datapath + 'all_sheets/DEMOGRAPHICS_INDX.csv')
    dem_m.sort_values(by=sort_id, inplace=True)
    sex_loc = dem_m.columns.get_loc('GENDER')
    eth_loc = dem_m.columns.get_loc('RACE')
    dem_m = dem_m.values

    # DOB
    print('Loading birthdates...')
    dob_m = pd.read_csv(datapath + 'all_sheets/DOB.csv')
    dob_m.sort_values(by=sort_id, inplace=True)
    birth_loc = dob_m.columns.get_loc("DOB")
    dob_m = dob_m.values

    # load death data
    print('Loading dates of death...')
    mort_m = pd.read_csv(datapath + 'all_sheets/OUTCOMES.csv')
    mort_m.sort_values(by=sort_id, inplace=True)
    mdate_loc = mort_m.columns.get_loc("DECEASED_DATE")
    mort_m = mort_m.values

    # load fluid input/output info
    print('Loading fluid I/O totals...')
    io_m = pd.read_csv(datapath + 'all_sheets/IO_TOTALS.csv')
    io_m.sort_values(by=sort_id, inplace=True)
    io_m = io_m.values

    # load dates of birth
    print('Loading charlson scores...')
    charl_m = pd.read_csv(datapath + 'all_sheets/CHARLSON_SCORE.csv')
    charl_m.sort_values(by=sort_id, inplace=True)
    charl_loc = charl_m.columns.get_loc("CHARLSON_INDEX")
    charl_m = charl_m.values

    # load dates of birth
    print('Loading elixhauser scores...')
    elix_m = pd.read_csv(datapath + 'all_sheets/ELIXHAUSER_SCORE.csv')
    elix_m.sort_values(by=sort_id, inplace=True)
    elix_loc = elix_m.columns.get_loc("ELIXHAUSER_INDEX")
    elix_m = elix_m.values

    print('Loading blood gas labs...')
    blood_gas = pd.read_csv(datapath + 'all_sheets/BLOOD_GAS.csv')
    blood_gas.sort_values(by=sort_id, inplace=True)
    pa_o2 = [blood_gas.columns.get_loc('PO2_D1_LOW_VALUE'),
             blood_gas.columns.get_loc('PO2_D1_HIGH_VALUE')]
    pa_co2 = [blood_gas.columns.get_loc('PCO2_D1_LOW_VALUE'),
              blood_gas.columns.get_loc('PCO2_D1_HIGH_VALUE')]
    p_h = [blood_gas.columns.get_loc('PH_D1_LOW_VALUE'),
           blood_gas.columns.get_loc('PH_D1_HIGH_VALUE')]
    blood_gas = blood_gas.values

    print('Loading clinical others...')
    clinical_oth = pd.read_csv(datapath + 'all_sheets/CLINICAL_OTHERS.csv')
    clinical_oth.sort_values(by=sort_id, inplace=True)
    resp = [clinical_oth.columns.get_loc('RESP_RATE_D1_LOW_VALUE'),
            clinical_oth.columns.get_loc('RESP_RATE_D1_HIGH_VALUE')]
    fi_o2 = [clinical_oth.columns.get_loc('FI02_D1_LOW_VALUE'),
             clinical_oth.columns.get_loc('FI02_D1_HIGH_VALUE')]
    g_c_s = clinical_oth.columns.get_loc('GLASGOW_SCORE_D1_LOW_VALUE')
    clinical_oth = clinical_oth.values

    print('Loading clinical vitals...')
    clinical_vit = pd.read_csv(datapath + 'all_sheets/CLINICAL_VITALS.csv')
    clinical_vit.sort_values(by=sort_id, inplace=True)
    temp = [clinical_vit.columns.get_loc('TEMPERATURE_D1_LOW_VALUE'),
            clinical_vit.columns.get_loc('TEMPERATURE_D1_HIGH_VALUE')]
    m_a_p = [clinical_vit.columns.get_loc('ART_MEAN_D1_LOW_VALUE'),
             clinical_vit.columns.get_loc('ART_MEAN_D1_HIGH_VALUE')]
    cuff = [clinical_vit.columns.get_loc('CUFF_MEAN_D1_LOW_VALUE'),
            clinical_vit.columns.get_loc('CUFF_MEAN_D1_HIGH_VALUE')]
    h_r = [clinical_vit.columns.get_loc('HEART_RATE_D1_LOW_VALUE'),
           clinical_vit.columns.get_loc('HEART_RATE_D1_HIGH_VALUE')]
    clinical_vit = clinical_vit.values

    print('Loading standard labs...')
    labs = pd.read_csv(datapath + 'all_sheets/LABS_SET1.csv')
    labs.sort_values(by=sort_id, inplace=True)
    bili = labs.columns.get_loc('BILIRUBIN_D1_HIGH_VALUE')
    pltlts = labs.columns.get_loc('PLATELETS_D1_LOW_VALUE')
    na = [labs.columns.get_loc('SODIUM_D1_LOW_VALUE'),
          labs.columns.get_loc('SODIUM_D1_HIGH_VALUE')]
    p_k = [labs.columns.get_loc('POTASSIUM_D1_LOW_VALUE'),
           labs.columns.get_loc('POTASSIUM_D1_HIGH_VALUE')]
    hemat = [labs.columns.get_loc('HEMATOCRIT_D1_LOW_VALUE'),
             labs.columns.get_loc('HEMATOCRIT_D1_HIGH_VALUE')]
    w_b_c = [labs.columns.get_loc('WBC_D1_LOW_VALUE'),
             labs.columns.get_loc('WBC_D1_HIGH_VALUE')]
    labs = labs.values

    print('Loading medications...')
    medications = pd.read_csv(datapath + 'all_sheets/MEDICATIONS_INDX.csv')
    medications.sort_values(by=sort_id, inplace=True)
    med_name = medications.columns.get_loc('MEDICATION_TYPE')
    med_date = medications.columns.get_loc('ORDER_ENTERED_DATE')
    med_dur = medications.columns.get_loc('DAYS_ON_MEDICATION')
    medications = medications.values

    print('Loading mechanical ventilation support data...')
    organ_sup = pd.read_csv(datapath + 'all_sheets/ORGANSUPP_VENT.csv')
    organ_sup.sort_values(by=sort_id, inplace=True)
    mech_vent_dates = [organ_sup.columns.get_loc('VENT_START_DATE'), organ_sup.columns.get_loc('VENT_STOP_DATE')]
    mech_vent_days = organ_sup.columns.get_loc('TOTAL_DAYS')
    organ_sup = organ_sup.values

    print('Loading aggregate SCr data...')
    scr_agg = pd.read_csv(datapath + 'all_sheets/SCR_INDX_AGG.csv')
    scr_agg.sort_values(by=sort_id, inplace=True)
    s_c_r = scr_agg.columns.get_loc('DAY1_MAX_VALUE')
    scr_agg = scr_agg.values

    return ((date_m, hosp_locs, icu_locs, adisp_loc,
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
             scr_agg, s_c_r))


def load_all_csv_dallas(datapath, sort_id='PATIENT_NUM'):
    print('Loading encounter info...')
    # Get IDs and find indices of all used metrics
    date_m = pd.read_csv(datapath + 'csv/tIndexedIcuAdmission.csv')
    date_m.sort_values(by=[sort_id, 'HSP_ADMSN_TIME'], inplace=True)
    hosp_locs = [date_m.columns.get_loc("HSP_ADMSN_TIME"), date_m.columns.get_loc("HSP_DISCH_TIME")]
    icu_locs = [date_m.columns.get_loc("ICU_ADMSN_TIME"), date_m.columns.get_loc("ICU_DISCH_TIME")]
    bsln_loc = date_m.columns.get_loc("BASELINE_SCR")
    bsln_date_loc = date_m.columns.get_loc("DATE_BASELINE_SCR")
    bsln_typ_loc = date_m.columns.get_loc("BASELINE_SCR_LOC")
    # adisp_loc = date_m.columns.get_loc('DISCHARGE_DISPOSITION')
    date_m = date_m.values

    print('Loading ESRD status...')
    # ESRD status
    esrd_m = pd.read_csv(datapath + 'csv/tHospitalization.csv')
    esrd_m.sort_values(by=sort_id, inplace=True)
    esrd_locs = [esrd_m.columns.get_loc("ESRD_BEFORE_INDEXED_ADT"),
                 esrd_m.columns.get_loc("ESRD_DURING_INDEXED_ADT")]
    esrd_m = esrd_m.values

    # All SCR
    print('Loading SCr values (may take a while)...')
    scr_all_m = pd.read_csv(datapath + 'csv/all_scr_data.csv')
    scr_all_m.sort_values(by=[sort_id, 'SPECIMN_TEN_TIME'], inplace=True)
    scr_date_loc = scr_all_m.columns.get_loc('SPECIMN_TEN_TIME')
    scr_val_loc = scr_all_m.columns.get_loc('ORD_VALUE')
    dia_locs = [scr_all_m.columns.get_loc('CRRT_0_1'),
                scr_all_m.columns.get_loc('HD_0_1'),
                scr_all_m.columns.get_loc('PD_0_1')]
    scr_all_m = scr_all_m.values

    # Demographics
    print('Loading demographics...')
    dem_m = pd.read_csv(datapath + 'csv/tPatients.csv')
    dem_m.sort_values(by=sort_id, inplace=True)
    sex_loc = dem_m.columns.get_loc('SEX_ID')
    eth_loc = dem_m.columns.get_loc('RACE_BLACK')
    dob_loc = dem_m.columns.get_loc('DOB')
    dod_locs = [dem_m.columns.get_loc('DOD_Epic'),
                dem_m.columns.get_loc('DOD_NDRI'),
                dem_m.columns.get_loc('DMF_DEATH_DATE')]
    dem_m = dem_m.values

    # load lab values
    print('Loading laboratory values...')
    lab_m = pd.read_csv(datapath + 'csv/icu_lab_data.csv')
    lab_col = lab_m.columns.get_loc('TERM_GRP_NAME')
    lab_min = lab_m.columns.get_loc('D_MIN_VAL')
    lab_max = lab_m.columns.get_loc('D_MAX_VAL')
    lab_m = lab_m.values

    # load blood gas/flow values
    print('Loading laboratory values...')
    flw_m = pd.read_csv(datapath + 'csv/icu_flw_data.csv')
    flw_col = flw_m.columns.get_loc('TERM_GRP_NAME')
    flw_day = flw_m.columns.get_loc('DAY_NO')
    flw_min = flw_m.columns.get_loc('D_MIN_VAL')
    flw_max = flw_m.columns.get_loc('D_MAX_VAL')
    flw_m = flw_m.values

    print('Loading medications...')
    medications = pd.read_csv(datapath + 'csv/tMedications.csv')
    medications.sort_values(by=sort_id, inplace=True)
    press_loc = medications.columns.get_loc('PRESSOR_INOTROPE')
    medications = medications.values

    print('Loading mechanical ventilation support data...')
    organ_sup = pd.read_csv(datapath + 'csv/tAOS.csv')
    organ_sup.sort_values(by=sort_id, inplace=True)
    mech_vent_dates = [organ_sup.columns.get_loc('MV_START_DATE'), organ_sup.columns.get_loc('MV_END_DATE')]
    mech_vent_days = organ_sup.columns.get_loc('DAYS_ON_MV')
    organ_sup = organ_sup.values

    return ((date_m, hosp_locs, icu_locs, bsln_loc, bsln_date_loc, bsln_typ_loc,
             esrd_m, esrd_locs,
             scr_all_m, scr_date_loc, scr_val_loc, dia_locs,
             dem_m, sex_loc, eth_loc, dob_loc, dod_locs,
             lab_m, lab_col, lab_min, lab_max,
             flw_m, flw_col, flw_day, flw_min, flw_max,
             medications, press_loc,
             organ_sup, mech_vent_dates, mech_vent_days))


