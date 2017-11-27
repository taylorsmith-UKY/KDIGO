from __future__ import division
import datetime
import math
import numpy as np
import pandas as pd
from scipy.spatial import distance
import dtw
from dateutil import relativedelta as rdelta
import re


def get_dialysis_mask(scr_m, scr_date_loc, dia_m, crrt_locs, hd_locs,
                      pd_locs, v=True):
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
                    if this_date > dia_m[row, crrt_locs[0]] and this_date < \
                            dia_m[row, crrt_locs[1]] + datetime.timedelta(2):
                        mask[i] = 1
            if dia_m[row, hd_locs[0]]:
                if str(dia_m[row, hd_locs[0]]) != 'nan' and \
                        str(dia_m[row, hd_locs[1]]) != 'nan':
                    if this_date > dia_m[row, hd_locs[0]] and this_date < \
                            dia_m[row, hd_locs[1]] + datetime.timedelta(2):
                        mask[i] = 2
            if dia_m[row,pd_locs[0]]:
                if str(dia_m[row,pd_locs[0]]) != 'nan' and str(dia_m[row,pd_locs[1]]) != 'nan':
                    if this_date > dia_m[row,pd_locs[0]] and this_date < dia_m[row,pd_locs[1]] + datetime.timedelta(2):
                        mask[i] = 3
    if v:
        nwo = len(np.where(mask == 0)[0])
        ncrrt = len(np.where(mask == 1)[0])
        nhd = len(np.where(mask == 2)[0])
        npd = len(np.where(mask == 3)[0])
        print('%d, %d, %d, %d\n' % (nwo,ncrrt,nhd,npd))
    return mask


def get_t_mask(scr_m,scr_date_loc,date_m,hosp_locs,icu_locs,v=True):
    mask = np.zeros(len(scr_m))
    if v:
        print('Getting masks for icu and hospital admit-discharge')
    for i in range(len(mask)):
        this_id = scr_m[i,0]
        this_date = scr_m[i,scr_date_loc]
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


def get_esrd_mask(scr_m,id_loc,esrd_m,esrd_locs,v=True):
    '''
    purpose: find out the patient id who have esrd data "Y"
    input: encountmatr---excel sheet data,id_loc---encounter id index;
            esrd_loc---esrd_indicator index
    output:yesesrd
    '''
    mask = np.zeros(len(scr_m))
    if v:
        print('Getting mask for dialysis')
    for i in range(len(mask)):
        this_id = scr_m[i][id_loc]
        rows = np.where(esrd_m[0,:]==this_id)[0]
        for loc in esrd_locs:
            if np.any(esrd_m[rows,loc] == 'Y'):
                mask[i] = 1
    if v:
        nw = len(np.where(mask == 1)[0])
        nwo = len(mask)-nw
        print('Number records for patients with ESRD: '+str(nw))
        print('Number records for patients without ESRD: '+str(nwo))
    return mask

def get_patients(scr_all_m,scr_val_loc,scr_date_loc,\
                 mask,dia_mask,\
                 dx_m,dx_loc,\
                 bsln_m,bsln_scr_loc,bsln_date_loc,\
                 date_m,id_loc,icu_locs,\
                 xplt_m,xplt_loc,xplt_des_loc,\
                 dem_m,sex_loc,eth_loc,\
                 dob_m,birth_loc,v=True):
    scr = []
    tmasks = []     #time/date
    dmasks = []     #dialysis
    dates = []
    ids = np.unique(scr_all_m[:,0])
    ids.sort()
    ids_out = []
    bslns = []
    count=0
    gfr_count=0
    no_bsln_count=0
    no_recs_count=0
    gap_icu_count=0
    kid_xplt_count=0
    log = open('result/record_counts.csv','w')
    if v:
        print('Getting patient vectors')
        print('Patient ID,\tBaseline,\tTotal no. records,\tno. selected records')
        log.write('Patient ID,\tBaseline,\tTotal no. records,\tno. selected records\n')
    for idx in ids:
        ### Get Baseline
        bsln_idx = np.where(bsln_m[:,0] == idx)[0]
        bsln = bsln_m[bsln_idx,bsln_scr_loc][0]
        bsln_date = bsln_m[bsln_idx,bsln_date_loc][0]
        #remove if no baseline
        if str(bsln) == 'nan':
            np.delete(ids,bsln_idx)
            no_bsln_count+=1
            if v:
                print('Patient '+str(idx)+' removed due to missing baseline')
                log.write('Patient '+str(idx)+' removed due to missing baseline\n')
            continue
        #get dob, sex, and race
        birth_idx = np.where(dob_m[:,0] == idx)[0]
        dob = dob_m[birth_idx,birth_loc][0]

        dem_idx = np.where(dem_m[:,0] == idx)[0]
        sex = dem_m[dem_idx,sex_loc]
        race = dem_m[dem_idx,eth_loc]

        gfr = calc_gfr(bsln,bsln_date,sex,race,dob)
        if gfr < 15:
            np.delete(ids,bsln_idx)
            gfr_count+=1
            if v:
                print('Patient '+str(idx)+' removed due to initial GFR too low')
                log.write('Patient '+str(idx)+' removed due to initial GFR too lown')
            continue
        all_rows = np.where(scr_all_m[:,0] == idx)[0]
        sel = np.where(mask[all_rows] != 0)[0]
        if len(sel) == 0:
            np.delete(ids,count)
            no_recs_count+=1
            if v:
                print('Patient '+str(idx)+' removed due to no values in the time period of interest')
                log.write('Patient '+str(idx)+' removed due to no values in the time period of interest\n')
            continue

        #test for kidney transplant
        x_rows=np.where(xplt_m[:,0] == idx)         #rows in surgery sheet
        for row in x_rows:
            str_des = str(xplt_m[row,xplt_des_loc]).upper()
            if 'KID' in str_des and 'TRANS' in str_des:
                kid_xplt_count+=1
                np.delete(ids,count)
                if v:
                    print('Patient '+str(idx)+' removed due to kidney transplant')
                    log.write('Patient '+str(idx)+' removed due to kidney transplant\n')
                continue

        d_rows = np.where(dx_m[:,0] == idx)
        for row in d_rows:
            str_des = str(dx_m[row,dx_loc]).upper()
            if str_des == 'KIDNEY/PANCREAS FROM BATAVIA  ETA 1530':
                kid_xplt_count+=1
                np.delete(ids,count)
                if v:
                    print('Patient '+str(idx)+' removed due to kidney transplant')
                    log.write('Patient '+str(idx)+' removed due to kidney transplant\n')
                continue
            elif 'KID' in str_des and 'TRANS' in str_des:
                np.delete(ids,count)
                kid_xplt_count+=1
                if v:
                    print('Patient '+str(idx)+' removed due to kidney transplant')
                    log.write('Patient '+str(idx)+' removed due to kidney transplant\n')
                continue

        keep = all_rows[sel]
        all_drows = np.where(date_m[:,id_loc] == idx)[0]
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

        bslns.append(bsln_m[bsln_idx,bsln_scr_loc][0])
        if v:
            print('%d,\t\t%f,\t\t%d,\t\t%d' % (idx,bsln,len(all_rows),len(sel)))
            log.write('%d,\t\t%f,\t\t%d,\t\t%d\n' % (idx,bsln,len(all_rows),len(sel)))
        tmask = mask[keep]
        tmasks.append(tmask)
        dmask = dia_mask[keep]
        dmasks.append(dmask)
        scr.append(scr_all_m[keep,scr_val_loc])
        dates.append(scr_all_m[keep,scr_date_loc])
        ids_out.append(idx)
        count+=1
    bslns = np.array(bslns)
    if v:
        print('# Patients Kept: '+str(count))
        print('# Patients w/ GFR < 15: '+str(gfr_count))
        print('# Patients w/ no baseline: '+str(no_bsln_count))
        print('# Patients w/ no ICU records: '+str(no_recs_count))
        print('# Patients w/ gap in ICU > 3 days: '+str(gap_icu_count))
        print('# Patients w/ kidney transplant: '+str(kid_xplt_count))
        log.write('# Patients Kept: '+str(count)+'\n')
        log.write('# Patients w/ GFR < 15: '+str(gfr_count)+'\n')
        log.write('# Patients w/ no baseline: '+str(no_bsln_count)+'\n')
        log.write('# Patients w/ no ICU records: '+str(no_recs_count)+'\n')
        log.write('# Patients w/ gap in ICU > 3 days: '+str(gap_icu_count)+'\n')
        log.write('# Patients w/ kidney transplant: '+str(kid_xplt_count)+'\n')
        log.close()
    del scr_all_m
    del bsln_m
    del dx_m
    del date_m
    del xplt_m
    del dem_m
    del dob_m
    return ids_out, scr, dates, tmasks, dmasks, bslns


def linear_interpo(scr,ids,dates,masks,dmasks,scale,v=True):
    post_interpo = []
    dmasks_interp = []
    count=0
    if v:
        log = open('result/interpo_log.txt','w')
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
                    dmask_i[k]=dmask_i[j]
                    k+=1
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
                    j+=1
                post_id = j
                post_val = thisp[post_id]
                if post_val == -1:
                    post_val = pre_val
                step = (post_val-pre_val)/(post_id-pre_id)
                for k in range(pre_id+1,post_id+1):
                    thisp[k]=thisp[k-1]+step
            j+=1
        if v:
            log.write(arr2str(thisp)+'\n')
            log.write(arr2str(dmask)+'\n')
            log.write(arr2str(dmask_i)+'\n')
            log.write('\n')
        print(str(thisp))
        post_interpo.append(thisp)
        count+=1
    if v:
        log.close()
    return post_interpo, dmasks_interp

def nbins(start,stop,scale):
    dt = (stop-start).total_seconds()
    div = scale*60*60       #hrs * minutes * seconds
    bins, _ = divmod(dt,div)
    return bins+1

def pairwise_dtw_dist(patients,dm_fname,dtw_name,v=True):
    df = open(dm_fname,'w')
    if v and dtw_name is not None:
        log = open(dtw_name,'w')
    for i in range(len(patients)):
        if v:
            print('#'+str(i+1)+' vs #'+str(i+2)+' to '+str(len(patients)))
        for j in range(i+1,len(patients)):
            if np.all(patients[i] == 0) and np.all(patients[j] == 0):
                df.write('%f\n' % (0))
            else:
                if len(patients[i]) > 1 and len(patients[j]) > 1:
                    dist,_,_,path=dtw.dtw(patients[i],patients[j],lambda x,y: np.abs(x-y))
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
                else:
                    df.write('%f\n' % (distance.braycurtis(p1,p2)))
            if v and dtw_name is not None:
                log.write(arr2str(p1)+'\n')
                log.write(arr2str(p2)+'\n\n')
    if v and dtw_name is not None:
        log.close()
    return df

#===========================================================================
def scr2kdigo(scr,base,masks):
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
            elif scr[i][j] >= (3* base[i]):
                kdigo[j] = 3
            elif scr[i][j] >= 4.0:
                kdigo[j] = 3
        kdigos.append(kdigo)
    return kdigos


### Helper functions for descriptive statistics

def get_disch_date(date_m,hosp_locs,idx):
    rows = np.where(date_m[0] == idx)
    dd = datetime.timedelta(0)
    for row in rows:
        if date_m[row,hosp_locs[1]] > dd:
            dd = date_m[row,hosp_locs[1]]
    #dd.resolution=datetime.timedelta(1)
    return dd

def get_dod(date_m,outcome_m,dod_loc,idx):
    rows = np.where(date_m[0] == idx)
    if rows == None:
        return rows
    dd = datetime.timedelta(0)
    for row in rows:
        if outcome_m[row,dod_loc] > dd:
            dd = outcome_m[row,dod_loc]
    if dd == datetime.timedelta(0):
        return None
    return dd


def get_baselines(date_m, hosp_locs, bsln_m, bsln_scr_loc,bsln_type_loc,
                   scr_all_m, scr_val_loc, scr_date_loc, scr_desc_loc):
    #%%
    log = open('baseline_calc.csv', 'w')
    log.write('ID,provided_bsln,calc_bsln,provided_type,calc_type,calc_date\n')
    for i in range(len(bsln_m)):
        idx = bsln_m[i,0]
        log.write(str(idx))
        bsln = bsln_m[i,bsln_scr_loc]
        bsln_type = bsln_m[i,bsln_type_loc]
        #skip if no baseline
#        if str(bsln) == 'nan':
#            log.write(',no baseline\n')
#            ref_base = None
        log.write(','+str(bsln)+',')
        #determine earliest admission date
        admit = datetime.datetime.now()
        didx = np.where(date_m[:,0] == idx)[0]
        for did in didx:
            if date_m[did,hosp_locs[0]] < admit:
                admit = date_m[did,hosp_locs[0]]

        #find indices of all SCr values for this patient
        all_rows = np.where(scr_all_m[:,0] == idx)[0]

        #extract record types
        #i.e. indexed admission, before indexed, after indexed
        scr_desc = scr_all_m[all_rows,scr_desc_loc]
        for j in range(len(scr_desc)):
            scr_desc[j]=scr_desc[j].split()[0].upper()

        scr_tp = scr_all_m[all_rows,scr_desc_loc]
        for j in range(len(scr_tp)):
            scr_tp[j]=scr_tp[j].split()[-1].upper()

        #find location of first indexed, inpatient record
        idx_rows = np.where(scr_desc == 'INDEXED')[0]
        idx_rows = all_rows[idx_rows]

        #find locations of before indexed inpatient and outpatient
        bo_rows = np.intersect1d(np.where(scr_desc == 'BEFORE')[0],np.where(scr_tp == 'OUTPATIENT')[0])
        bo_rows = all_rows[bo_rows]
        bi_rows = np.intersect1d(np.where(scr_desc == 'BEFORE')[0],np.where(scr_tp == 'INPATIENT')[0])
        bi_rows = all_rows[bi_rows]
        true_date = 'missing'
        true_type = 'nan'
        #find the baseline
        if idx_rows == []: # no indexed tpts, so disregard
            log.write(',no indexed values\n')
            true_base = None
            true_date = None
            true_type = None
            log.write('\n')
        #no values prior to admission
        elif idx_rows[0] == all_rows[0]:
            true_base = np.min(scr_all_m[idx_rows,scr_val_loc])
            row = np.argmin(scr_all_m[idx_rows,scr_val_loc])
            row = idx_rows[row]
            true_date = scr_all_m[row,scr_date_loc]
            if str(true_base) != 'nan':
                true_type = scr_all_m[row,scr_desc_loc].upper()
        else: # start at first indexed tpt and move back until valid point found
            row = idx_rows[0]
            end_date = scr_all_m[row,scr_date_loc]
            row-=1
            prev_date = scr_all_m[row,scr_date_loc]
            prev_type = scr_all_m[row,scr_desc_loc].upper().split()[-1]
            while prev_type == 'EMERGENCY' and row > all_rows[0]:
                row-=1
                prev_date = scr_all_m[row,scr_date_loc]
                prev_type = scr_all_m[row,scr_desc_loc].upper().split()[-1]
            if prev_type != 'EMERGENCY' and rdelta.relativedelta(end_date,prev_date).years < 1:
                true_base = scr_all_m[row,scr_val_loc]
                true_date = prev_date
                if str(true_base) != 'nan':
                    true_type = scr_all_m[row,scr_desc_loc].upper()
            else:
                true_base = np.min(scr_all_m[idx_rows,scr_val_loc])
                true_date_idx = np.argmin(scr_all_m[idx_rows,scr_val_loc])
                row = idx_rows[true_date_idx]
                true_date = scr_all_m[row,scr_date_loc]
                if str(true_base) != 'nan':
                    true_type = scr_all_m[row,scr_desc_loc].upper()
        log.write(str(true_base) + ','+str(bsln_type)+',' + str(true_type) + ',' + str(true_date) + '\n')
    log.close()


'''
            row = idx_rows[0]
            end_date = scr_all_m[row,scr_date_loc]
            row-=1
            true_base = 'None'
            true_date = 'None'
            #see if there are any outpatients before index and use the latest
            if len(bi_rows) > 0 and len(bo_rows) > 0:
                tr = bo_rows[-1]
                tdi = scr_all_m[tr,scr_date_loc]

            elif len(bo_rows) > 0:
                tr = bo_rows[-1]
                td = scr_all_m[tr,scr_date_loc]
                if rdelta.relativedelta(end_date,td).years < 1:
                    true_base = scr_all_m[tr,scr_val_loc]
                    true_date = scr_all_m[tr,scr_date_loc]
                    true_type = 'INPATIENT - latest <1 yr prior to admission'
            #see if any inpatients
            elif len(bi_rows) > 0:
                #only count inpatient records < 1 yr from admission
                tr = bi_rows[-1]
                td = scr_all_m[tr,scr_date_loc]
                if rdelta.relativedelta(end_date,td).years < 1:
                    true_base = scr_all_m[tr,scr_val_loc]
                    true_date = scr_all_m[tr,scr_date_loc]
                    true_type = 'INPATIENT - latest <1 yr prior to admission'
'''

#%%

def arr2csv(fname,inds,ids):
    outFile=open(fname,'w')
    try:
        for i in range(len(inds)):
            outFile.write('%d' % (ids[i]))
            for j in range(len(inds[i])):
                outFile.write(', %f' % (inds[i][j]))
            outFile.write('\n')
        outFile.close()
    except:
        for i in range(len(inds)):
            outFile.write('%d' % (ids[i]))
            outFile.write(', %f\n' % (inds[i]))
        outFile.close()

def str2csv(fname,inds,ids):
    outFile=open(fname,'w')
    for i in range(len(inds)):
        outFile.write('%d' % (ids[i]))
        for j in range(len(inds[i])):
            outFile.write(', %s' % (inds[i][j]))
        outFile.write('\n')
    outFile.close()

def arr2str(arr,fmt='%f'):
    s = fmt % (arr[0])
    for i in range(1,len(arr)):
        s = s + ', ' + fmt % (arr[i])
    return s

### finish GFR calculation function
def calc_gfr(bsln,date_base,sex,race,dob):
    date_base = datetime.datetime.strptime(date_base.split('.')[0], '%Y-%m-%d %H:%M:%S')
    time_diff=rdelta.relativedelta(date_base,dob)
    year_val = time_diff.years
    mon_val = time_diff.months
    age = year_val+mon_val/12
    min_value=1
    max_value=1
    race_value=1
    if sex == 'M':
        k_value = 0.9
        a_value = -0.411
        f_value =1
    else:
        k_value=0.7#female
        a_value=-0.329#female
        f_value=1.018#female
    if race == "BLACK/AFR AMERI":
        race_value = 1.159
    if bsln/k_value < 1:
        min_value = bsln/k_value
    else:
        max_value = bsln/k_value
    min_power=math.pow(min_value,a_value)
    max_power= math.pow(max_value,-1.209)
    age_power= math.pow(0.993,age)
    GFR = 141 * min_power * max_power * age_power * f_value * race_value
    return GFR

def get_mat(fname,page_name,sort_id):
    return pd.read_excel(fname,page_name).sort_values(sort_id)
