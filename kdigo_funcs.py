from __future__ import division
import datetime
import math
import numpy as np
import pandas as pd
from scipy.spatial import distance
import dtw
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, VarianceThreshold, RFECV
from sklearn.model_selection import StratifiedKFold
import h5py


# %%
def get_dialysis_mask(scr_m, scr_date_loc, dia_m, crrt_locs, hd_locs, pd_locs, v=True):
    mask = np.zeros(len(scr_m))
    if v:
        print 'Getting mask for dialysis'
        print 'Number non-dialysis records, #CRRT, #HD, #PD'
    for i in range(len(mask)):
        this_id = scr_m[i, 0]
        this_date = scr_m[i, scr_date_loc]
        this_date.resolution = datetime.timedelta(0, 1)
        rows = np.where(dia_m[:, 0] == this_id)[0]
        for row in rows:
            if dia_m[row, crrt_locs[0]]:
                if str(dia_m[row, crrt_locs[0]]) != 'nan' and \
                        str(dia_m[row, crrt_locs[1]]) != 'nan':
                    if dia_m[row, crrt_locs[0]] < this_date < dia_m[row, crrt_locs[1]] + datetime.timedelta(2):
                        mask[i] = 1
            if dia_m[row, hd_locs[0]]:
                if str(dia_m[row, hd_locs[0]]) != 'nan' and \
                        str(dia_m[row, hd_locs[1]]) != 'nan':
                    if dia_m[row, hd_locs[0]] < this_date < dia_m[row, hd_locs[1]] + datetime.timedelta(2):
                        mask[i] = 2
            if dia_m[row, pd_locs[0]]:
                if str(dia_m[row, pd_locs[0]]) != 'nan' and str(dia_m[row, pd_locs[1]]) != 'nan':
                    if dia_m[row, pd_locs[0]] < this_date < dia_m[row, pd_locs[1]] + datetime.timedelta(2):
                        mask[i] = 3
    if v:
        nwo = len(np.where(mask == 0)[0])
        ncrrt = len(np.where(mask == 1)[0])
        nhd = len(np.where(mask == 2)[0])
        npd = len(np.where(mask == 3)[0])
        print('%d, %d, %d, %d\n' % (nwo, ncrrt, nhd, npd))
    return mask


# %%
def get_t_mask(scr_m, scr_date_loc, scr_val_loc, date_m, hosp_locs, icu_locs, v=True):
    mask = np.zeros(len(scr_m))
    if v:
        print('Getting masks for icu and hospital admit-discharge')
    for i in range(len(mask)):
        this_id = scr_m[i, 0]
        this_date = scr_m[i, scr_date_loc]
        this_val = scr_m[i, scr_val_loc]
        if this_val == np.nan:
            continue
        rows = np.where(date_m[:, 0] == this_id)[0]
        for row in rows:
            if date_m[row, icu_locs[0]] != np.nan:
                if date_m[row, icu_locs[0]] < this_date < date_m[row, icu_locs[1]]:
                    mask[i] = 2
                    break
            elif date_m[row, hosp_locs[0]] != np.nan:
                if date_m[row, hosp_locs[0]] < this_date < date_m[row, hosp_locs[1]]:
                    mask[i] = 1
    if v:
        nop = len(np.where(mask == 0)[0])
        nhp = len(np.where(mask >= 1)[0])
        nicu = len(np.where(mask == 2)[0])
        print('Number records outside hospital: '+str(nop))
        print('Number records in hospital: '+str(nhp))
        print('Number records in ICU: '+str(nicu))
    return mask


# %%
def get_patients(scr_all_m, scr_val_loc, scr_date_loc, d_disp_loc,
                 mask, dia_mask,
                 dx_m, dx_loc,
                 esrd_m, esrd_locs,
                 bsln_m, bsln_scr_loc, admit_loc,
<<<<<<< HEAD
=======
                 mort_m, mdate_loc,
>>>>>>> cc45cee6634fdc61b6a6c3046a10c45ae20b6d23
                 date_m, id_loc,
                 xplt_m, xplt_des_loc,
                 dem_m, sex_loc, eth_loc,
                 dob_m, birth_loc,
                 log, v=True):
    # Lists to store records for each patient
    scr = []
    tmasks = []     # time/date
    dmasks = []     # dialysis
    dates = []
    d_disp = []
    ids_out = []
    bslns = []
    bsln_gfr = []
    t_range = []
    ages = []

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
    ids = np.unique(scr_all_m[:, 0])
    ids.sort()
    if v:
        print('Getting patient vectors')
        print('Patient ID,\tBaseline,\tTotal no. records,\tno. selected records')
        log.write('Patient ID,\tBaseline,\tTotal no. records,\tno. selected records\n')
    for idx in ids:
        skip = False            # required to skip patients where exclusion is found in interior loop
        all_rows = np.where(scr_all_m[:, 0] == idx)[0]
        sel = np.where(mask[all_rows] != 0)[0]
        keep = all_rows[sel]
        # Ensure this patient has values in time period of interest
        if len(sel) < 2:
            ids = np.delete(ids, count)
            no_recs_count += 1
            if v:
                print(str(idx)+', removed due to not enough values in the time period of interest')
                log.write(str(idx)+', removed due to not enough values in the time period of interest\n')
            continue

        # Get Baseline or remove if no admit dates provided
        bsln_idx = np.where(bsln_m[:, 0] == idx)[0]
        if bsln_idx.size == 0:
            ids = np.delete(ids, count)
            no_admit_info_count += 1
            if v:
                print(str(idx)+', removed due to missing admission info')
                log.write(str(idx)+', removed due to missing admission info\n')
            continue
        else:
            bsln_idx = bsln_idx[0]
        bsln = bsln_m[bsln_idx, bsln_scr_loc]
        if str(bsln).lower() == 'nan' or str(bsln).lower() == 'none' or str(bsln).lower() == 'nat':
            ids = np.delete(ids, count)
            no_bsln_count += 1
            if v:
                print(str(idx)+', removed due to missing baseline')
                log.write(str(idx)+', removed due to missing baseline\n')
            continue
        bsln = float(bsln)
        if bsln >= 4.0:
            ids = np.delete(ids, count)
            no_bsln_count += 1
            if v:
                print(str(idx)+', removed due to baseline SCr > 4.0')
                log.write(str(idx)+', removed due to baseline SCr > 4.0\n')
            continue
        admit = str(bsln_m[bsln_idx, admit_loc]).split('.')[0]
        admit = datetime.datetime.strptime(admit, '%Y-%m-%d %H:%M:%S')

        # get mortality and exclude if in window
        # mort_idx = np.where(mort_m[:, 0] == idx)[0]
        # if mort_idx.size > 0:
        #     for i in range(len(mort_idx)):
        #         tid = mort_idx[i]
        #         mdate = mort_m[tid, mdate_loc]
        #         if mdate - admit < datetime.timedelta(death_excl_dur):
        #             skip = True
        #             ids = np.delete(ids, count)
        #             death_count += 1
        #             if v:
        #                 print(str(idx) + ', removed due to death in specified window')
        #                 log.write(str(idx) + ', removed due to death in specified window\n')
        #             continue
        # if skip:
        #     continue

        # get dob, sex, and race
        if idx not in dob_m[:, 0] or idx not in dem_m[:, 0]:
            ids = np.delete(ids, count)
            dem_count += 1
            if v:
                print(str(idx)+', removed due to missing DOB')
                log.write(str(idx)+', removed due to missing DOB\n')
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

        # remove if ESRD status
        esrd_idx = np.where(esrd_m[:, 0] == idx)[0]
        if len(esrd_idx) > 0:
            for loc in esrd_locs:
                if np.any(esrd_m[esrd_idx, loc] == 'Y'):
                    skip = True
                    ids = np.delete(ids, count)
                    esrd_count += 1
                    if v:
                        print(str(idx)+', removed due to ESRD status')
                        log.write(str(idx)+', removed due to ESRD status\n')
                    break
        if skip:
            continue

        # remove patients with required demographics missing
        if str(sex) == 'nan' or str(race) == 'nan':
            ids = np.delete(ids, count)
            dem_count += 1
            if v:
                print(str(idx)+', removed due to missing demographics')
                log.write(str(idx)+', removed due to missing demographics\n')
            continue

        # remove patients with baseline GFR < 15
        gfr = calc_gfr(bsln, sex, race, age)
        if gfr < 15:
            ids = np.delete(ids, count)
            gfr_count += 1
            if v:
                print(str(idx)+', removed due to initial GFR too low')
                log.write(str(idx)+', removed due to initial GFR too low\n')
            continue

        # remove patients with kidney transplant
        x_rows = np.where(xplt_m[:, 0] == idx)         # rows in surgery sheet
        for row in x_rows:
            str_des = str(xplt_m[row, xplt_des_loc]).upper()
            if 'KID' in str_des and 'TRANS' in str_des:
                skip = True
                kid_xplt_count += 1
                np.delete(ids, count)
                if v:
                    print(str(idx)+', removed due to kidney transplant')
                    log.write(str(idx)+', removed due to kidney transplant\n')
                break
        if skip:
            continue

        d_rows = np.where(dx_m[:, 0] == idx)
        for row in d_rows:
            str_des = str(dx_m[row, dx_loc]).upper()
            if str_des == 'KIDNEY/PANCREAS FROM BATAVIA  ETA 1530':
                skip = True
                kid_xplt_count += 1
                np.delete(ids, count)
                if v:
                    print(str(idx)+', removed due to kidney transplant')
                    log.write(str(idx)+', removed due to kidney transplant\n')
                break
            elif 'KID' in str_des and 'TRANS' in str_des:
                skip = True
                np.delete(ids, count)
                kid_xplt_count += 1
                if v:
                    print(str(idx)+', removed due to kidney transplant')
                    log.write(str(idx)+', removed due to kidney transplant\n')
                break
        if skip:
            continue

        all_drows = np.where(date_m[:, id_loc] == idx)[0]
        '''
        delta = datetime.timedelta(0)
        for i in range(len(all_drows)):
            start = str(date_m[all_drows[i], icu_locs[1]]).split('.')[0]
            start = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
            td = datetime.timedelta(0)
            for j in range(len(all_drows)):
                tdate = str(date_m[all_drows[j], icu_locs[0]]).split('.')[0]
                tdate = datetime.datetime.strptime(tdate, '%Y-%m-%d %H:%M:%S')
                if tdate > start:
                    if td == datetime.timedelta(0):
                        td = tdate-start
                    elif (date_m[all_drows[j], icu_locs[0]]-start) < td:
                        td = tdate-start
            if delta == datetime.timedelta(0):
                delta = td
            elif delta < td:
                delta = td
        if delta > datetime.timedelta(3):
            np.delete(ids, count)
            gap_icu_count += 1
            if v:
                print(str(idx)+', removed due to different ICU stays > 3 days apart')
                log.write(str(idx)+', removed due to different ICU stays > 3 days apart\n')
            continue
        '''

        # # remove patients who died <48 hrs after indexed admission
        disch_disp = date_m[all_drows[0], d_disp_loc]
        if type(disch_disp) == np.ndarray:
            disch_disp = disch_disp[0]
        disch_disp = str(disch_disp).upper()
        # if 'LESS THAN' in disch_disp:
        #     np.delete(ids, count)
        #     lt48_count += 1
        #     if v:
        #         print(str(idx)+', removed due to death within 48 hours of admission')
        #         log.write(str(idx)+', removed due to  death within 48 hours of admission\n')
        #     continue

        # get duration vector
        tdates = scr_all_m[keep, scr_date_loc]
        duration = [tdates[x] - admit for x in range(len(tdates))]
        duration = np.array(duration)
        dkeep = np.where(duration < datetime.timedelta(7))[0]
        duration = duration[dkeep]
        keep = keep[dkeep]

        if len(dkeep) < 2:
            np.delete(ids, count)
            no_recs_count += 1
            if v:
                print(str(idx)+', removed due to not enough values in the time period of interest')
                log.write(str(idx)+', removed due to not enough values in the time period of interest\n')
            continue

        # points to keep = where duration < 7 days
        # i.e. set any points in 'keep' corresponding to points > 7 days
        # from start to 0
        d_disp.append(disch_disp)
        bslns.append(bsln)
        bsln_gfr.append(gfr)
        if v:
            print('%d,\t\t%f,\t\t%d,\t\t%d' % (idx, bsln, len(all_rows), len(sel)))
            log.write('%d,\t\t%f,\t\t%d,\t\t%d\n' % (idx, bsln, len(all_rows), len(sel)))
        tmask = mask[keep]
        tmasks.append(tmask)
        dmask = dia_mask[keep]
        dmasks.append(dmask)
        scr.append(scr_all_m[keep, scr_val_loc])
        dates.append(scr_all_m[keep, scr_date_loc])
        ages.append(age)

        tmin = duration[0].total_seconds() / (60 * 60)
        tmax = duration[-1].total_seconds() / (60 * 60)
        ids_out.append(idx)
        t_range.append([tmin, tmax])
        count += 1
    bslns = np.array(bslns)
    if v:
        print('# Patients Kept: '+str(count))
        print('# Patients removed for ESRD: '+str(esrd_count))
        print('# Patients w/ GFR < 15: '+str(gfr_count))
        print('# Patients w/ no admit info: '+str(no_admit_info_count))
        print('# Patients w/ missing demographics: '+str(dem_count))
        print('# Patients w/ < 2 ICU records: '+str(no_recs_count))
        print('# Patients w/ no valid baseline: '+str(no_bsln_count))
        print('# Patients w/ kidney transplant: '+str(kid_xplt_count))
        log.write('# Patients Kept: '+str(count)+'\n')
        log.write('# Patients removed for ESRD: '+str(esrd_count)+'\n')
        log.write('# Patients w/ GFR < 15: '+str(gfr_count)+'\n')
        log.write('# Patients w/ no admit info: '+str(no_admit_info_count)+'\n')
        log.write('# Patients w/ missing demographics: '+str(dem_count)+'\n')
        log.write('# Patients w/ < 2 ICU records: '+str(no_recs_count)+'\n')
        log.write('# Patients w/ no valid baseline: '+str(no_bsln_count)+'\n')
        log.write('# Patients w/ kidney transplant: '+str(kid_xplt_count)+'\n')
<<<<<<< HEAD
=======
        log.write('# Patients who died < 48 hrs after admission: '+str(death_count)+'\n')
>>>>>>> cc45cee6634fdc61b6a6c3046a10c45ae20b6d23
    del scr_all_m
    del bsln_m
    del dx_m
    del date_m
    del xplt_m
    del dem_m
    del dob_m
    return ids_out, scr, dates, tmasks, dmasks, bslns, bsln_gfr, d_disp, t_range, ages


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
        print('Patient #'+str(ids[i]))
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
            dt = (tdate-this_start).total_seconds()
            idx = int(math.floor(dt/(60*60*scale)))
            if mask[j] != -1:
                thisp[idx] = scr[i][j]
                interp_mask[idx] = 1
            dmask_i[idx] = dmask[j]
        for j in range(len(dmask_i)):
            if dmask_i[j] != -1:
                k = j+1
                while k < len(dmask_i) and dmask_i[k] == -1:
                    dmask_i[k] = dmask_i[j]
                    k += 1
        print(str(thisp))
        if v:
            log.write('%d\n' % (ids[i]))
            log.write(arr2str(scr[i])+'\n')
            log.write(arr2str(thisp)+'\n')
        print(dmask_i)
        dmasks_interp.append(dmask_i)
        interp_masks.append(interp_mask)
        j = 0
        while j < len(thisp):
            if thisp[j] == -1:
                pre_id = j-1
                pre_val = thisp[pre_id]
                while thisp[j] == -1 and j < len(thisp)-1:
                    j += 1
                post_id = j
                post_val = thisp[post_id]
                if post_val == -1:
                    post_val = pre_val
                step = (post_val-pre_val)/(post_id-pre_id)
                for k in range(pre_id+1, post_id+1):
                    thisp[k] = thisp[k-1]+step
            j += 1
        if v:
            log.write(arr2str(thisp)+'\n')
            log.write(arr2str(dmask)+'\n')
            log.write(arr2str(dmask_i)+'\n')
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
    dt = (stop-start).total_seconds()
    div = scale*60*60       # hrs * minutes * seconds
    bins, _ = divmod(dt, div)
    return int(bins+1)


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
                delta = admit-this_date
                # find latest point that is > 24 hrs prior to admission
                while delta < out_lim[0] and len(pre_out_rows) > 0:
                    row = pre_out_rows.pop(-1)
                    this_date = scr_all_m[row, scr_date_loc]
                    delta = admit-this_date
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
                delta = admit-this_date
                # find latest point that is > 24 hrs prior to admission
                while delta < inp_lim[0] and len(pre_inp_rows) > 0:
                    row = pre_inp_rows.pop(-1)
                    this_date = scr_all_m[row, scr_date_loc]
                    delta = admit-this_date
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
            log.write(str(bsln_val)+',' + str(bsln_type) + ',' + str(bsln_date) + ',' +
                      str(admit) + ',' + str(bsln_delta) + '\n')
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
    scr = math.pow((numerator / denominator), 1./a_value)

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
    denominator = 175 * age**(-0.203) * f_value * race_value
    scr = (numerator / denominator)**(1./-1.154)
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
                if days[i][j] > 1:
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
def pairwise_dtw_dist(patients, ids, dm_fname, dtw_name, incl_0=True, v=True, alpha=1.0):
    df = open(dm_fname, 'w')
    dis = []
    if v and dtw_name is not None:
        log = open(dtw_name, 'w')
    for i in range(len(patients)):
        if incl_0 == False and np.all(patients[i] == 0):
            continue
        if v:
            print('#'+str(i+1)+' vs #'+str(i+2)+' to '+str(len(patients)))
        for j in range(i+1,len(patients)):
            if not incl_0 and np.all(patients[j] == 0):
                    continue
            df.write('%d,%d,' % (ids[i],ids[j]))
            if np.all(patients[i] == 0) and np.all(patients[j] == 0):
                df.write('%f\n' % 0)
                dis.append(0)
            else:
                if len(patients[i]) > 1 and len(patients[j]) > 1:
                    dist, _, _, path = dtw_p(patients[i], patients[j], lambda y, yy: np.abs(y-yy), alpha=alpha)
                    p1_path = path[0]
                    p2_path = path[1]
                    p1 = [patients[i][p1_path[x]] for x in range(len(p1_path))]
                    p2 = [patients[j][p2_path[x]] for x in range(len(p2_path))]
                elif len(patients[i]) == 1:
                    p1 = np.repeat(patients[i][0],len(patients[j]))
                    p2 = patients[j]
                elif len(patients[j]) == 1:
                    p1 = patients[i]
                    p2 = np.repeat(patients[j][0],len(patients[i]))
                if np.all(p1 == p2):
                    df.write('%f\n' % 0)
                    dis.append(0)
                else:
                    df.write('%f\n' % (distance.braycurtis(p1,p2)))
                    dis.append(distance.braycurtis(p1,p2))
            if v and dtw_name is not None:
                log.write(arr2str(p1, fmt='%d')+'\n')
                log.write(arr2str(p2, fmt='%d')+'\n\n')
    if v and dtw_name is not None:
        log.close()
    return dis


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
        outf.write(lbls[i] + ' - ' + str(counts[i])+'\n')
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
        pct = scr/bsln

        ids.append(idx)
        abs_chg.append(ab)
        pct_chg.append(pct)

    arr2csv('scr_pct_chg.csv', pct_chg, ids)
    arr2csv('scr_abs_chg.csv', abs_chg, ids)


# %%
def arr2csv(fname, inds, ids, fmt='%f', header=False):
    outFile = open(fname, 'w')
    if header:
        outFile.write(header)
        outFile.write('\n')
    try:
        for i in range(len(inds)):
            outFile.write('%d' % (ids[i]))
            for j in range(len(inds[i])):
                outFile.write(','+fmt % (inds[i][j]))
            outFile.write('\n')
        outFile.close()
    except:
        outFile.write(',' + fmt % (inds[0]) + '\n')
        for i in range(1, len(inds)):
            outFile.write('%d' % (ids[i]))
            outFile.write(','+fmt % (inds[i])+'\n')
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
        ds[mask].to_excel(sheet+'.xlsx')


# %%
def get_mat(fname, page_name, sort_id):
    return pd.read_excel(fname, page_name).sort_values(sort_id)


# %%
def load_csv(fname, ids, dt=float, skip_header=False, sel=None):
    res = []
    rid = []
    f = open(fname, 'r')
    if skip_header:
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
        return rid, res


def descriptive_trajectory_features(kdigos, ids, filename='descriptive_features.csv'):
    npts = len(kdigos)
    features = np.zeros((npts, 26))
    header = 'id,no_AKI,peak_at_KDIGO1,peak_at_KDIGO2,peak_at_KDIGO3,peak_at_KDIGO3D,KDIGO1_at_admit,KDIGO2_at_admit,' +\
             'KDIGO3_at_admit,KDIGO3D_at_admit,KDIGO1_at_disch,KDIGO2_at_disch,KDIGO3_at_disch,KDIGO3D_at_disch,' +\
             'onset_lt_3days,onset_gte_3days,complete_recovery_lt_3days,multiple_hits,KDIGO1_gt_24hrs,KDIGO2_gt_24hrs,' +\
             'KDIGO3_gt_24hrs,KDIGO4_gt_24hrs,flat,strictly_increase,strictly_decrease,slope_posTOneg,slope_negTOpos'
    for i in range(len(kdigos)):
        kdigo = np.array(kdigos[i])
        kdigo1 = np.where(kdigo == 1)[0]
        kdigo2 = np.where(kdigo == 2)[0]
        kdigo3 = np.where(kdigo == 3)[0]
        kdigo4 = np.where(kdigo == 4)[0]
        # No AKI
        if np.all(kdigo == 0):
            features[i, 0] = 1
        # KDIGO 1 Peak
        temp = 0    # previous score
        direction = 0   # 1: Increased     -1: Decreased
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


def template_trajectory_features(kdigos, ids, filename='template_trajectory_features.csv', scores=np.array([0, 1, 2, 3, 4], dtype=int), npoints=3):
    combination = scores
    for i in range(npoints - 1):
        combination = np.vstack((combination, scores))
    npts = len(kdigos)
    templates = cartesian(combination)
    header = 'id'
    for i in range(len(templates)):
        header += ','+str(templates[i])
    features = np.zeros((npts, len(templates)), dtype=int)
    for i in range(npts):
        kdigo = np.array(kdigos[i])
        nwin = len(kdigo) - npoints + 1
        for j in range(nwin):
            tk = kdigo[j:j + npoints]
            sel = np.where(templates[:, 0] == tk[0])[0]
            loc = [x for x in sel if np.all(templates[x, :] == tk)]
            features[i, loc] += 1
    arr2csv(filename, features, ids, fmt='%d', header=header)
    return features


def slope_trajectory_features(kdigos, ids, scores=np.array([0, 1, 2, 3, 4]), filename='slope_features.csv'):
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
        kdigo = np.array(kdigos[i])
        nwin = len(kdigo) - 1
        for j in range(nwin):
            ts = kdigo[j+1] - kdigo[j]
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
        if avg < 2*std:
            flags[i] = 1
    print('Features with High Variability:')
    sel = np.where(flags)[0]
    for idx in sel:
        print('Feature #'+str(idx))
    return features


def perf_measure(y_actual, y_hat):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           tp += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           fp += 1
        if y_actual[i]==y_hat[i]==0:
           tn += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
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
        return sf

    elif method == 'univariate':
        assert len(params) == 2
        n_feats = params[0]
        scoring = params[1]
        sf = SelectKBest(scoring, k=n_feats).fit_transform(data, lbls)
        return sf

    elif method == 'recursive':
        assert len(params) == 2
        estimator = params[0]
        cv = params[1]
        rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(cv),
                      scoring='accuracy')
        rfecv.fit(data, lbls)
        sel = rfecv.support_
        sf = data[:, sel]
        return sf

    elif method == 'linear':
        assert len(params) == 1
        c = params[0]
        lsvc = LinearSVC(C=c, penalty='l1', dual=False).fit(data, lbls)
        model = SelectFromModel(lsvc, prefit=True)
        sf = model.transform(data)
        return sf

    elif method == 'tree':
        assert params == []
        clf = ExtraTreesClassifier()
        clf.fit(data, lbls)
        model = SelectFromModel(clf, prefit=True)
        sf = model.transform(data)
        return sf

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
                ds = grp.create_dataset(col[1], data=np.zeros(sheet_len,), dtype=int)
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
    for i in range(len(all_lbls)-1):
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
            clf = RandomForestClassifier(n_estimators=clf_params[0], criterion=clf_params[1], max_features=clf_params[2])
        else:
            print('Currently only supports SVM and RF')

        clf.fit(X_train, y_train)
        predicted = clf.predict(X_val)
        perf[i, :] = perf_measure(y_val, predicted)

    return perf


def daily_max_kdigo(scr, dates, bsln, admit_date, dmask):
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


def dtw_p(x, y, dist, alpha=1.0, agg='mult'):
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
            D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += np.min((D0[i, j], D0[i, j+1], D0[i+1, j]))
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0, x, y, alpha, agg)
    return D1[-1, -1] / sum(D1.shape), C, D1, path


def _traceback(D, x, y, alpha=1.0, agg='mult'):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    px = 0
    py = 0
    while i > 0 or j > 0:
        tb = np.argmin((D[i, j], D[i, j+1] + (alpha * px) + x[i], D[i+1, j] + (alpha * py) + y[j]))
        if tb == 0:
            i -= 1
            j -= 1
            px = 0
            py = 0
        elif tb == 1:
            i -= 1
            if px == 0:
                px = x[i]
            else:
                if agg == 'mult':
                    px *= x[i]
                elif agg == 'add':
                    px += x[i]
            py = 0
        else:  # (tb == 2):
            j -= 1
            if py == 0:
                py = y[j]
            else:
                if agg == 'mult':
                    py *= y[j]
                elif agg == 'add':
                    py += y[j]
            px = 0
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)


def count_transitions(ids, kdigos, out_name):
    k_lbls = range(5)
    transitions = cartesian((k_lbls, k_lbls))
    t_counts = np.zeros((len(kdigos), len(transitions)))
    for i in range(len(kdigos)):
        for j in range(len(kdigos[i]) - 1):
            idx = np.intersect1d(np.where(transitions[:, 0] == kdigos[i][j])[0],
                                 np.where(transitions[:, 1] == kdigos[i][j+1])[0])
            t_counts[i, idx] += 1

    header = 'id'
    for i in range(len(transitions)):
        header += ',' + str(transitions[i])

    rows = np.array(ids)[:, np.newaxis]
    with open(out_name, 'w') as f:
        f.write(header + '\n')
        np.savetxt(f, np.hstack((rows, t_counts)), delimiter=',', fmt='%d')
    return t_counts


