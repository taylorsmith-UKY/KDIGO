from __future__ import division
import datetime
import math
import numpy as np
import pandas as pd
from scipy.spatial import distance
import dtw
from dateutil import relativedelta as rdelta
from numpy.random import permutation as permute
import re
import matplotlib.pyplot as plt


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
                    if this_date > dia_m[row, crrt_locs[0]] and this_date < dia_m[row, crrt_locs[1]] + datetime.timedelta(2):
                        mask[i] = 1
            if dia_m[row, hd_locs[0]]:
                if str(dia_m[row, hd_locs[0]]) != 'nan' and \
                        str(dia_m[row, hd_locs[1]]) != 'nan':
                    if this_date > dia_m[row, hd_locs[0]] and this_date < dia_m[row, hd_locs[1]] + datetime.timedelta(2):
                        mask[i] = 2
            if dia_m[row,pd_locs[0]]:
                if str(dia_m[row,pd_locs[0]]) != 'nan' and str(dia_m[row,pd_locs[1]]) != 'nan':
                    if this_date > dia_m[row, pd_locs[0]] and this_date < dia_m[row, pd_locs[1]] + datetime.timedelta(2):
                        mask[i] = 3
    if v:
        nwo = len(np.where(mask == 0)[0])
        ncrrt = len(np.where(mask == 1)[0])
        nhd = len(np.where(mask == 2)[0])
        npd = len(np.where(mask == 3)[0])
        print('%d, %d, %d, %d\n' % (nwo,ncrrt,nhd,npd))
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
        rows = np.where(date_m[:,0]==this_id)[0]
        for row in rows:
            if date_m[row,icu_locs[0]] != np.nan:
                if this_date > date_m[row,icu_locs[0]] and this_date < date_m[row,icu_locs[1]]:
                    mask[i] = 2
                    break
            elif date_m[row,hosp_locs[0]] != np.nan:
                if this_date > date_m[row,hosp_locs[0]] and this_date < date_m[row,hosp_locs[1]]:
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
                 bsln_m, bsln_scr_loc, bsln_date_loc,
                 date_m, id_loc, icu_locs,
                 xplt_m, xplt_loc, xplt_des_loc,
                 dem_m, sex_loc, eth_loc,
                 dob_m, birth_loc,
                 log, v=True):
    scr = []
    tmasks = []     # time/date
    dmasks = []     # dialysis
    dates = []
    d_disp = []
    ids = np.unique(scr_all_m[:,0])
    ids.sort()
    ids_out = []
    bslns = []
    bsln_gfr = []
    count = 0
    gfr_count = 0
    no_admit_info_count = 0
    no_recs_count = 0
    no_bsln_count = 0
    kid_xplt_count = 0
    esrd_count = 0
    dem_count = 0
    lt48_count = 0
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
            np.delete(ids,count)
            no_recs_count += 1
            if v:
                print('Patient '+str(idx)+' removed due to not enough values in the time period of interest')
                log.write('Patient '+str(idx)+' removed due to not enough values in the time period of interest\n')
            continue

        # Get Baseline or remove if no admit dates provided
        bsln_idx = np.where(bsln_m[:,0] == idx)[0][0]
        if bsln_idx.size == 0:
            np.delete(ids, bsln_idx)
            no_admit_info_count += 1
            if v:
                print('Patient '+str(idx)+' removed due to missing admission info')
                log.write('Patient '+str(idx)+' removed due to missing admission info\n')
            continue
        bsln = bsln_m[bsln_idx, bsln_scr_loc]
        if str(bsln).lower() == 'nan' or str(bsln).lower() == 'none' or str(bsln).lower() == 'nat':
            np.delete(ids,bsln_idx)
            no_bsln_count += 1
            if v:
                print('Patient '+str(idx)+' removed due to missing baseline')
                log.write('Patient '+str(idx)+' removed due to missing baseline\n')
            continue
        bsln = float(bsln)
        bsln_date = bsln_m[bsln_idx,bsln_date_loc].split('.')[0]
        bsln_date = datetime.datetime.strptime(bsln_date,'%Y-%m-%d %H:%M:%S')

        # remove if ESRD status
        esrd_idx = np.where(esrd_m[:,0]==idx)[0]
        if len(esrd_idx) > 0:
            for loc in esrd_locs:
                if np.any(esrd_m[esrd_idx,loc] == 'Y'):
                    skip = True
                    np.delete(ids,bsln_idx)
                    esrd_count+=1
                    if v:
                        print('Patient '+str(idx)+' removed due to ESRD status')
                        log.write('Patient '+str(idx)+' removed due to ESRD status\n')
                    break
        if skip:
            continue

        # get dob, sex, and race
        if idx not in dob_m[:,0] or idx not in dem_m[:,0]:
            np.delete(ids,bsln_idx)
            dem_count += 1
            if v:
                print('Patient '+str(idx)+' removed due to missing DOB')
                log.write('Patient '+str(idx)+' removed due to missing DOB\n')
            continue
        birth_idx = np.where(dob_m[:, 0] == idx)[0]
        dob = dob_m[birth_idx, birth_loc][0].to_pydatetime()

        dem_idx = np.where(dem_m[:,0] == idx)[0]
        if len(dem_idx) > 1:
            dem_idx = dem_idx[0]
        sex = dem_m[dem_idx, sex_loc]
        race = dem_m[dem_idx, eth_loc]

        # remove patients with required demographics missing
        if str(sex) == 'nan' or str(race) == 'nan':
            np.delete(ids,bsln_idx)
            dem_count += 1
            if v:
                print('Patient '+str(idx)+' removed due to missing demographics')
                log.write('Patient '+str(idx)+' removed due to missing demographics\n')
            continue

        # remove patients with baseline GFR < 15
        gfr = calc_gfr(bsln,bsln_date,sex,race,dob)
        if gfr < 15:
            np.delete(ids,bsln_idx)
            gfr_count += 1
            if v:
                print('Patient '+str(idx)+' removed due to initial GFR too low')
                log.write('Patient '+str(idx)+' removed due to initial GFR too low\n')
            continue

        # remove patients with kidney transplant
        x_rows = np.where(xplt_m[:,0] == idx)         # rows in surgery sheet
        for row in x_rows:
            str_des = str(xplt_m[row, xplt_des_loc]).upper()
            if 'KID' in str_des and 'TRANS' in str_des:
                skip = True
                kid_xplt_count += 1
                np.delete(ids, count)
                if v:
                    print('Patient '+str(idx)+' removed due to kidney transplant')
                    log.write('Patient '+str(idx)+' removed due to kidney transplant\n')
                break
        if skip:
            continue

        d_rows = np.where(dx_m[:, 0] == idx)
        for row in d_rows:
            str_des = str(dx_m[row,dx_loc]).upper()
            if str_des == 'KIDNEY/PANCREAS FROM BATAVIA  ETA 1530':
                skip = True
                kid_xplt_count += 1
                np.delete(ids,count)
                if v:
                    print('Patient '+str(idx)+' removed due to kidney transplant')
                    log.write('Patient '+str(idx)+' removed due to kidney transplant\n')
                break
            elif 'KID' in str_des and 'TRANS' in str_des:
                skip = True
                np.delete(ids, count)
                kid_xplt_count += 1
                if v:
                    print('Patient '+str(idx)+' removed due to kidney transplant')
                    log.write('Patient '+str(idx)+' removed due to kidney transplant\n')
                break
        if skip:
            continue


        all_drows = np.where(date_m[:,id_loc] == idx)[0]
        '''
        delta = datetime.timedelta(0)
        for i in range(len(all_drows)):
            start = date_m[all_drows[i],icu_locs[1]]
            td = datetime.timedelta(0)
            for j in range(len(all_drows)):
                if date_m[all_drows[j],icu_locs[0]] > start:
                    if td == datetime.timedelta(0):
                        td = date_m[all_drows[j],icu_locs[0]]-start
                    elif (date_m[all_drows[j],icu_locs[0]]-start) < td:
                        td = date_m[all_drows[j],icu_locs[0]]-start
            if delta == datetime.timedelta(0):
                delta = td
            elif delta < td:
                delta = td
        if delta > datetime.timedelta(3):
            np.delete(ids,count)
            gap_icu_count+=1
            if v:
                print('Patient '+str(idx)+' removed due to different ICU stays > 3 days apart')
                log.write('Patient '+str(idx)+' removed due to different ICU stays > 3 days apart\n')
            continue
        '''
        # remove patients who died <48 hrs after indexed admission
        disch_disp = str(date_m[all_drows[0], d_disp_loc]).upper()
        if 'LESS THAN' in disch_disp:
            np.delete(ids, count)
            lt48_count += 1
            if v:
                print('Patient '+str(idx)+' removed due to death within 48 hours of admission')
                log.write('Patient '+str(idx)+' removed due to  death within 48 hours of admission\n')
            continue

        # get duration vector
        tdates = scr_all_m[keep,scr_date_loc]
        duration = [datetime.timedelta(0)] + [tdates[x] - tdates[0] for x in range(1, len(tdates))]
        duration = np.array(duration)
        dkeep = np.where(duration < datetime.timedelta(7))[0]
        keep = keep[dkeep]

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
        scr.append(scr_all_m[keep,scr_val_loc])
        dates.append(scr_all_m[keep,scr_date_loc])
        ids_out.append(idx)
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
        print('# Patients who died < 48 hrs after admission: '+str(lt48_count))
        log.write('# Patients Kept: '+str(count)+'\n')
        log.write('# Patients removed for ESRD: '+str(esrd_count)+'\n')
        log.write('# Patients w/ GFR < 15: '+str(gfr_count)+'\n')
        log.write('# Patients w/ no admit info: '+str(no_admit_info_count)+'\n')
        log.write('# Patients w/ missing demographics: '+str(dem_count)+'\n')
        log.write('# Patients w/ < 2 ICU records: '+str(no_recs_count)+'\n')
        log.write('# Patients w/ no valid baseline: '+str(no_bsln_count)+'\n')
        log.write('# Patients w/ kidney transplant: '+str(kid_xplt_count)+'\n')
        log.write('# Patients who died < 48 hrs after admission: '+str(lt48_count)+'\n')
    del scr_all_m
    del bsln_m
    del dx_m
    del date_m
    del xplt_m
    del dem_m
    del dob_m
    return ids_out, scr, dates, tmasks, dmasks, bslns, bsln_gfr, d_disp


# %%
def linear_interpo(scr, ids, dates, masks, dmasks, scale, log, v=True):
    post_interpo = []
    dmasks_interp = []
    count=0
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
        tmin = dates[i][0]
        tmax = dates[i][-1]
        n = nbins(tmin,tmax,scale)
        thisp = np.repeat(-1.,n)
        this_start = dates[i][0]
        thisp[0] = scr[i][0]
        dmask_i = np.repeat(-1,len(thisp))
        dmask_i[0] = dmask[0]
        for j in range(1,len(scr[i])):
            dt = (dates[i][j]-this_start).total_seconds()
            idx = int(math.floor(dt/(60*60*scale)))
            if mask[j] != -1:
                thisp[idx] = scr[i][j]
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
                for k in range(pre_id+1,post_id+1):
                    thisp[k] = thisp[k-1]+step
            j+=1
        if v:
            log.write(arr2str(thisp)+'\n')
            log.write(arr2str(dmask)+'\n')
            log.write(arr2str(dmask_i)+'\n')
            log.write('\n')
        print(str(thisp))
        post_interpo.append(thisp)
        count += 1
    return post_interpo, dmasks_interp


# %%
def nbins(start, stop, scale):
    dt = (stop-start).total_seconds()
    div = scale*60*60       # hrs * minutes * seconds
    bins, _ = divmod(dt,div)
    return bins+1


# %%
def get_baselines(date_m, hosp_locs, scr_all_m, scr_val_loc, scr_date_loc,scr_desc_loc,
                    dia_m, crrt_locs, hd_locs, pd_locs, fname):

    log = open(fname, 'w')
    log.write('ID,bsln_val,bsln_type,bsln_date\n')
    cur_id = None
    for i in range(len(date_m)):
        idx = date_m[i, 0]
        if cur_id != idx:
            cur_id = idx
            log.write(str(idx) + ',')
            # determine earliest admission date
            admit = datetime.datetime.now()
            didx = np.where(date_m[:, 0] == idx)[0]
            for did in didx:
                if date_m[did,hosp_locs[0]] < admit:
                    admit = date_m[did,hosp_locs[0]]

            # find indices of all SCr values for this patient
            all_rows = np.where(scr_all_m[:,0] == idx)[0]

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

            pre_inp_rows = np.intersect1d(pre_admit_rows,inp_rows)
            pre_inp_rows = list(all_rows[pre_inp_rows])

            pre_out_rows = np.intersect1d(pre_admit_rows,out_rows)
            pre_out_rows = list(all_rows[pre_out_rows])

            d_lim = datetime.timedelta(1)
            y_lim = datetime.timedelta(365)
            dia_lim = datetime.timedelta(2)

            # default values
            bsln_val = None
            bsln_date = None
            bsln_type = None
            # find the baseline
            if len(idx_rows) == 0:  # no indexed tpts, so disregard entirely
                bsln_type = 'No_indexed_values'
            elif np.all([np.isnan(scr_all_m[x, scr_val_loc]) for x in idx_rows]):
                bsln_type = 'No_indexed_values'
            # first look for valid outpatient values before admission
            if len(pre_out_rows) != 0 and bsln_type == None:
                row = pre_out_rows.pop(-1)
                this_date = scr_all_m[row,scr_date_loc]
                delta = admit-this_date
                # find latest point that is > 24 hrs prior to admission
                while delta < d_lim and len(pre_out_rows) > 0:
                    row = pre_out_rows.pop(-1)
                    this_date = scr_all_m[row,scr_date_loc]
                    delta = admit-this_date
                # if valid point found save it
                if y_lim > delta > d_lim:
                    bsln_date = this_date
                    bsln_val = scr_all_m[row, scr_val_loc]
                    bsln_type = 'OUTPATIENT'
            # no valid outpatient, so look for valid inpatient
            if len(pre_inp_rows) != 0 and bsln_type is None:
                row = pre_inp_rows.pop(-1)
                this_date = scr_all_m[row,scr_date_loc]
                delta = admit-this_date
                # find latest point that is > 48 hrs prior to admission
                while delta < d_lim and len(pre_inp_rows) > 0:
                    row = pre_inp_rows.pop(-1)
                    this_date = scr_all_m[row,scr_date_loc]
                    delta = admit-this_date
                    # if valid point found save it
                if delta < y_lim and delta > d_lim:
                    bsln_date = this_date
                    bsln_val = scr_all_m[row,scr_val_loc]
                    bsln_type = 'INPATIENT'
                    # no values prior to admission, find minimum during
            if bsln_type is None:
                bsln_val = np.min(scr_all_m[idx_rows, scr_val_loc])
                row = np.argmin(scr_all_m[idx_rows, scr_val_loc])
                row = idx_rows[row]
                bsln_date = scr_all_m[row, scr_date_loc]
                bdt = bsln_date.to_pydatetime()
                c = 1
                if str(bsln_val) != 'nan':
                    drows = np.where(dia_m[:,0] == idx)[0]
                    if len(drows) > 0:
                        bad = True
                        while bad and c <= len(idx_rows):
                            bad = False
                            for drow in drows:
                                # crrt
                                if str(dia_m[drow, crrt_locs[0]]) != 'NaT':
                                    if bdt > dia_m[drow, crrt_locs[0]].to_pydatetime():
                                        if bdt < (dia_m[drow, crrt_locs[1]].to_pydatetime() + dia_lim):
                                            bad = True
                                # hd
                                if str(dia_m[drow, hd_locs[0]]) != 'NaT':
                                    if bdt > dia_m[drow, hd_locs[0]].to_pydatetime():
                                        if bdt < (dia_m[drow, hd_locs[1]].to_pydatetime() + dia_lim):
                                            bad = True
                                # pd
                                if str(dia_m[drow, pd_locs[0]]) != 'NaT':
                                    if bdt > dia_m[drow, pd_locs[0]].to_pydatetime():
                                        if bdt < (dia_m[drow, pd_locs[1]].to_pydatetime() + dia_lim):
                                            bad = True
                            if bad and c < len(idx_rows):
                                bsln_val = np.sort(scr_all_m[idx_rows, scr_val_loc])[c]
                                row = np.where(scr_all_m[idx_rows, scr_val_loc] == bsln_val)[0]
                                row = idx_rows[row]
                            c += 1
                        if bad:
                            bsln_val = 'nan'
                            bsln_type = 'none'
                            bsln_date = 'all_RRT'
                        else:
                            bsln_type = scr_all_m[row,scr_desc_loc].upper()
                    else:
                        bsln_type = scr_all_m[row,scr_desc_loc].upper()
            log.write(str(bsln_val)+',' + str(bsln_type) + ',' + str(bsln_date) + '\n')
    log.close()


# %%
def calc_gfr(bsln, date_base, sex, race, dob):
    # date_base = datetime.datetime.strptime(date_base.split('.')[0], '%Y-%m-%d %H:%M:%S')
    time_diff = rdelta.relativedelta(date_base,dob)
    year_val = time_diff.years
    mon_val = time_diff.months
    age = year_val+mon_val/12
    min_value = 1
    max_value = 1
    race_value = 1
    if sex == 'M':
        k_value = 0.9
        a_value = -0.411
        f_value = 1
    else:  # female
        k_value = 0.7
        a_value = -0.329
        f_value = 1.018
    if race == "BLACK/AFR AMERI":
        race_value = 1.159
    if bsln/k_value < 1:
        min_value = bsln/k_value
    else:
        max_value = bsln/k_value
    min_power = math.pow(min_value, a_value)
    max_power = math.pow(max_value, -1.209)
    age_power = math.pow(0.993, age)
    GFR = 141 * min_power * max_power * age_power * f_value * race_value
    return GFR


# %%
def scr2kdigo(scr, base, masks):
    kdigos = []
    for i in range(len(scr)):
        kdigo = np.zeros(len(scr[i]),dtype=int)
        for j in range(len(scr[i])):
            if masks[i][j] > 0:
                kdigo[j] = 4
                continue
            elif scr[i][j] <= (1.5 * base[i]):
                if scr[i][j] >= base[i] + 0.3:
                    kdigo[j] = 1
                else:
                    kdigo[j] = 0
            elif scr[i][j] < (2*base[i]):
                kdigo[j] = 1
            elif scr[i][j] < (3 * base[i]):
                kdigo[j] = 2
            elif scr[i][j] >= (3 * base[i]):
                kdigo[j] = 3
            elif scr[i][j] >= 4.0:
                kdigo[j] = 3
        kdigos.append(kdigo)
    return kdigos


# %%
def pairwise_dtw_dist(patients, ids, dm_fname, dtw_name, incl_0=True, v=True):
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
                df.write('%f\n' % (0))
                dis.append(0)
            else:
                if len(patients[i]) > 1 and len(patients[j]) > 1:
                    dist, _, _, path=dtw.dtw(patients[i], patients[j], lambda y, yy: np.abs(y-yy))
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
                    df.write('%f\n' % (0))
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
def sparsify(fname, outname, thresh):
    inf = open(fname, 'r')
    out = open(outname, 'w')
    for l in inf:
        dist = float(l.rstrip().split(',')[-1])
        if dist < thresh:
            out.write(l)
    return


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


#%%
def mort_subset_ids(n_pts=300, path='../DATA/icu/',set_name='subset2'):
    n_alive = 0
    n_lt = 0
    n_gt = 0
    n_xfer = 0
    n_ama = 0
    n_unk = 0
    alive_ids = []
    dead_ids = []
    f = open(path+'disch_disp.csv', 'r')
    for line in f:
        l = line.strip()
        idx = int(l.split(',')[0])
        str_disp = l.split(',')[1].upper()
        if re.search('EXP', str_disp):
            dead_ids.append(idx)
            if re.search('LESS', str_disp):
                n_lt += 1
            elif re.search('MORE', str_disp):
                n_gt += 1
        elif re.search('ALIVE', str_disp):
            alive_ids.append(idx)
            n_alive += 1
        elif re.search('XFER', str_disp) or re.search('TRANS', str_disp):
            n_xfer += 1
        elif re.search('AMA', str_disp):
            n_ama += 1
        else:
            n_unk += 1
    fname = path + set_name + '_ids.csv'
    np.savetxt(path+'inp_died_ids.csv',dead_ids, fmt='%d')
    np.savetxt(path+'alive_ids.csv',alive_ids, fmt='%d')
    if len(alive_ids)*2 < n_pts or len(dead_ids)*2 < n_pts:
        print('Change number of desired patients')
        print('Current: %d' % n_pts)
        print('Number died: %d' % len(dead_ids))
        print('Number alive: %d' % len(alive_ids))
    else:
        alive = permute(alive_ids)[:(n_pts/2)]
        dead = permute(dead_ids)[:(n_pts/2)]
        ids = permute(np.concatenate((alive, dead)))
        np.savetxt(fname, ids, fmt='%d')


# %%
def arr2csv(fname, inds, ids, fmt='%f', header=False):
    outFile = open(fname, 'w')
    if header:
        outFile.write('id')
        for idx in ids:
            outFile.write(',%d' % idx)
        outFile.write('\n')
    try:
        for i in range(len(inds)):
            outFile.write('%d' % (ids[i]))
            for j in range(len(inds[i])):
                outFile.write(','+fmt % (inds[i][j]))
            outFile.write('\n')
        outFile.close()
    except:
        for i in range(len(inds)):
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
def load_csv(fname, ids, dt=float):
    res = []
    rid = []
    f = open(fname, 'r')
    for line in f:
        l = line.rstrip()
        if int(l.split(',')[0]) in ids or ids is None:
            res.append(np.array(l.split(',')[1:], dtype=dt))
            rid.append(int(l.split(',')[0]))
    return rid, res
