#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:15:39 2017

@author: taylorsmith
"""
from __future__ import division
from kdigo_funcs import load_csv, get_mat
import datetime
import numpy as np
from collections import Counter
import re

#%%
def get_cstats(clustFile,idFile,dataPath,outPath):
    #get IDs and Clusters in order from cluster file
    ids = np.loadtxt(idFile,dtype=int,delimiter=',')
    clusts = np.loadtxt(clustFile,dtype=int,delimiter=',')
    '''
    f = open(clustFile,'r')
    _ = f.readline() #first line is header
    for l in f:
        ids.append(int(l.split(',')[0][1:-1]))
        clusts.append(int(l.rstrip().split(',')[-1]))
    f.close()
    '''
    n_ids = len(ids)

    n_clusts = len(set(clusts))

    #load KDIGO and discharge dispositions
    k_ids,kdigos = load_csv(dataPath + 'kdigo.csv',ids,dt=int)
    d_ids,d_disp = load_csv(dataPath + 'disch_disp.csv',ids,dt='|S')
    k_ids = np.array(k_ids,dtype=int)
    d_ids = np.array(d_ids,dtype=int)

    #load outcome data
    date_m = get_mat('../DATA/KDIGO_full.xlsx','ADMISSION_INDX','STUDY_PATIENT_ID')
    hosp_locs=[date_m.columns.get_loc("HOSP_ADMIT_DATE"),date_m.columns.get_loc("HOSP_DISCHARGE_DATE")]
    icu_locs=[date_m.columns.get_loc("ICU_ADMIT_DATE"),date_m.columns.get_loc("ICU_DISCHARGE_DATE")]
    date_m = date_m.as_matrix()
    date_ids=date_m[:,0]
    date_all_ids=[]
    for i in range(len(date_ids)):
        date_all_ids.append(date_ids[i])
    all_apache = np.loadtxt('../DATA/icu/7days_final/apache.csv',delimiter=',',dtype=int)
    all_sofa = np.loadtxt('../DATA/icu/7days_final/sofa.csv',delimiter=',',dtype=int)

    #arrays to store individual results
    disch_disp = np.zeros(n_ids,dtype=int)
    kdigo_max = np.zeros(n_ids,dtype=int)
    kdigo_counts = np.zeros([n_ids,5],dtype=int)
    kdigo_pcts = np.zeros([n_ids,5],dtype=float)
    los = np.zeros([n_ids,2],dtype=float)    #   [:,0]=hospital   [:,1]=ICU
    n_eps = np.zeros(n_ids,dtype=int)
    sofa = np.zeros(n_ids,dtype=int)
    apache = np.zeros(n_ids,dtype=int)
#    surv_t = np.zeros(n_ids,dtype=datetime.datetime)

    #arrays to store cluster results (each is in the form [mean,std])
    c_header = 'cluster_id,count,mort_avg,mort_std,kdigo_max_avg,kdig_max_std,'+\
                'n_eps_mean,n_eps_std,hosp_los_mean,hosp_los_std,icu_los_mean,'+\
                'icu_los_std,sofa_mean,sofa_std,apache_mean,apache_std'
    c_stats = np.zeros([n_clusts,16],dtype=float)
    c_stats[:,0] = [x for x in range(1,n_clusts+1)]


    mk = [] #max kdigo
    di = [] #death indicator
    n = [] #n_eps
    hl = [] #hosp los
    il = [] #icu los
    so = [] #sofa
    ap = [] #apache
    cnum = 0
    this_clust = clusts[0]
    for i in range(len(ids)):
        if this_clust != clusts[i]:
            c_stats[cnum,0] = this_clust
            c_stats[cnum,2] = np.mean(di)
            c_stats[cnum,3] = np.std(di)
            c_stats[cnum,4] = np.mean(mk)
            c_stats[cnum,5] = np.std(mk)
            c_stats[cnum,6] = np.mean(n)
            c_stats[cnum,7] = np.std(n)
            c_stats[cnum,8] = np.mean(hl)
            c_stats[cnum,9] = np.std(hl)
            c_stats[cnum,10] = np.mean(il)
            c_stats[cnum,11] = np.std(il)
            c_stats[cnum,12] = np.mean(so)
            c_stats[cnum,13] = np.std(so)
            c_stats[cnum,14] = np.mean(ap)
            c_stats[cnum,15] = np.std(ap)
            mk = []
            di = []
            hl = [] #los
            il = []
            so = [] #sofa
            ap = [] #apache
            n = []
            cnum += 1
            this_clust = clusts[i]
        idx = ids[i]
        c_stats[cnum,1] += 1
        #assign death group
        drow = np.where(d_ids==idx)[0][0]
        str_disp = d_disp[drow][0].upper()
        died = 0
        if re.search('EXP',str_disp):
            died = 1
            if re.search('LESS',str_disp):
                disch_disp[i] = 1
            elif re.search('MORE',str_disp):
                disch_disp[i] = 2
        elif re.search('ALIVE',str_disp):
            disch_disp[i] = 0
        elif re.search('XFER',str_disp) or re.search('TRANS',str_disp):
            disch_disp[i] = 3
        elif re.search('AMA',str_disp):
            disch_disp[i] = 4
        else:
            disch_disp[i] = -1
        di.append(died)


        #get max and avg kdigo, as well as percent time at each stage
        krow = np.where(k_ids == idx)[0][0]
        kdigo = np.array(kdigos[krow],dtype=int)
        for j in range(5):
            kdigo_counts[i,j] = len(np.where(kdigo == j)[0])
            kdigo_pcts[i,j] = kdigo_counts[i,j] / len(kdigo)
        kdigo_max[i] = np.max(kdigo)
        mk.append(kdigo_max[i])

        los[i,:] = get_los(idx,date_m,hosp_locs,icu_locs)
        hl.append(los[i,0])
        il.append(los[i,1])

        #Add function to calculate number of episodes
        n_eps[i] = count_eps(kdigo)
        n.append(n_eps[i])

        sofa_idx = np.where(all_sofa[:,0] == idx)[0]
        so.append(np.sum(all_sofa[sofa_idx,1:]))
        sofa[i] = np.sum(all_sofa[sofa_idx,1:])

        apache_idx = np.where(all_apache[:,0] == idx)[0]
        ap.append(np.sum(all_apache[apache_idx,1:]))
        apache[i] = np.sum(all_apache[apache_idx,1:])

        #surv_t = sf.get_survival_time(PID,date_m,hosp_locs,outcome_m,death_loc)


    np.savetxt(outPath+'/disch_codes.csv',disch_disp,fmt='%d',delimiter=',')
    np.savetxt(outPath+'/kdigo_max.csv',kdigo_max,fmt='%d',delimiter=',')
    np.savetxt(outPath+'/kdigo_pct.csv',kdigo_pcts,fmt='%f',delimiter=',')
    np.savetxt(outPath+'/n_episodes.csv',n_eps,fmt='%d',delimiter=',')
    np.savetxt(outPath+'/all_los.csv',los,header='hospital_LOS,ICU_LOS',fmt='%f',delimiter=',')
    np.savetxt(outPath+'/sofa.csv',sofa,fmt='%d',delimiter=',')
    np.savetxt(outPath+'/apache.csv',apache,fmt='%d',delimiter=',')

    c_stats[cnum,0] = this_clust
    c_stats[cnum,2] = np.mean(di)
    c_stats[cnum,3] = np.std(di)
    c_stats[cnum,4] = np.mean(mk)
    c_stats[cnum,5] = np.std(mk)
    c_stats[cnum,6] = np.mean(n)
    c_stats[cnum,7] = np.std(n)
    c_stats[cnum,8] = np.mean(hl)
    c_stats[cnum,9] = np.std(hl)
    c_stats[cnum,10] = np.mean(il)
    c_stats[cnum,11] = np.std(il)
    c_stats[cnum,12] = np.mean(so)
    c_stats[cnum,13] = np.std(so)
    c_stats[cnum,14] = np.mean(ap)
    c_stats[cnum,15] = np.std(ap)
    np.savetxt(outPath+'/cluster_stats.csv',c_stats,header=c_header,fmt='%f',delimiter=',')
    #np.savetxt(res_path+'/survival_duration.csv',surv_t,fmt=datetime.datetime)



#%%
def get_disch_date(idx,date_m,hosp_locs):
    rows = np.where(date_m[0] == idx)
    dd = datetime.timedelta(0)
    for row in rows:
        if date_m[row,hosp_locs[1]] > dd:
            dd = date_m[row,hosp_locs[1]]
    #dd.resolution=datetime.timedelta(1)
    return dd

#%%
def get_dod(idx,date_m,outcome_m,dod_loc):
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


#%%
def count_eps(kdigo,t_gap=48,timescale=6):
    aki = np.where(kdigo)[0]
    if len(aki) > 0:
        count = 1
    else:
        return 0
    gap_ct = t_gap / timescale
    for i in range(1,len(aki)):
        if (aki[i] - aki[i-1]) >= gap_ct:
            count += 1
    return count

def get_los(PID,date_m,hosp_locs,icu_locs):
    date_rows = np.where(date_m[:,0]==PID)[0]
    hosp = []
    icu = []
    for i in range(len(date_rows)):
        idx = date_rows[i]
        add_hosp = True
        for j in range(len(hosp)):
            #new start is before saved start
            if date_m[idx,hosp_locs[0]] < hosp[j][0]:
                #new stop is after saved stop - replace interior window
                if date_m[idx,hosp_locs[1]] > hosp[j][1]:
                    hosp[j][0] = date_m[idx,hosp_locs[0]]
                    hosp[j][1] = date_m[idx,hosp_locs[1]]
                    add_hosp = False
                #new stop is after saved start - replace saved start
                elif date_m[idx,hosp_locs[1]] > hosp[j][0]:
                    hosp[j][0] = date_m[idx,hosp_locs[0]]
                    add_hosp = False
            #new start is before saved stop
            elif date_m[idx,hosp_locs[0]] < hosp[j][1]:
                add_hosp = False
                #new stop is after saved stop - replace saved stop
                if date_m[idx,hosp_locs[1]] < hosp[j][1]:
                    hosp[j][1] = date_m[idx,hosp_locs[1]]
        if add_hosp:
            hosp.append([date_m[idx,hosp_locs[0]],date_m[idx,hosp_locs[1]]])

        add_icu = True
        for j in range(len(icu)):
            #new start is before saved start
            if date_m[idx,icu_locs[0]] < icu[j][0]:
                #new stop is after saved stop - replace interior window
                if date_m[idx,icu_locs[1]] > icu[j][1]:
                    icu[j][0] = date_m[idx,icu_locs[0]]
                    icu[j][1] = date_m[idx,icu_locs[1]]
                    add_icu = False
                #new stop is after saved start - replace saved start
                elif date_m[idx,icu_locs[1]] > icu[j][0]:
                    icu[j][0] = date_m[idx,icu_locs[0]]
                    add_icu = False
            #new start is before saved stop
            elif date_m[idx,icu_locs[0]] < icu[j][1]:
                add_icu = False
                #new stop is after saved stop - replace saved stop
                if date_m[idx,icu_locs[1]] < icu[j][1]:
                    icu[j][1] = date_m[idx,icu_locs[1]]
        if add_icu:
            icu.append([date_m[idx,icu_locs[0]],date_m[idx,icu_locs[1]]])

    h_dur = datetime.timedelta(0)
    for i in range(len(hosp)):
        h_dur += hosp[i][1].to_pydatetime() - hosp[i][0].to_pydatetime()

    icu_dur = datetime.timedelta(0)
    for i in range(len(icu)):
        icu_dur += icu[i][1].to_pydatetime() - icu[i][0].to_pydatetime()

    h_dur = h_dur.days + float(h_dur.seconds)/86400
    icu_dur = icu_dur.days + float(icu_dur.seconds)/86400

    return h_dur, icu_dur

#%% Summarize discharge dispositions for a file
def get_disch_summary(idFile,statFile,ids=None):
    '''
    Code:
        0 - Dead in hospital
        1 - Dead after 48 hrs
        2 - Alive
        3 - Transfered
        4 - AMA
    '''

    #get IDs in order
    f = open(idFile,'r')
    sf = open(statFile,'w')

    n_alive = 0
    n_lt = 0
    n_gt = 0
    n_xfer = 0
    n_ama = 0
    n_unk = 0
    n_oth = 0
    str_disp = 'empty'
    for l in f:
        line=l.strip()
        this_id = line.split(',')[0]
        if ids is not None and this_id not in ids:
            continue
        str_disp = line.split(',')[-1].upper()
        if re.search('EXP',str_disp):
            if re.search('LESS',str_disp):
                n_lt += 1
            elif re.search('MORE',str_disp):
                n_gt += 1
        elif re.search('ALIVE',str_disp):
            n_alive += 1
        elif re.search('XFER',str_disp) or re.search('TRANS',str_disp):
            n_xfer += 1
        elif re.search('AMA',str_disp):
            n_ama += 1
        elif str_disp == '':
            n_unk += 1
        else:
            n_oth += 1
    f.close()

    sf.write('All Alive: %d\n' % (n_alive+n_xfer+n_ama))
    sf.write('Alive - Routine: %d\n' % (n_alive))
    sf.write('Transferred: %d\n' % (n_xfer))
    sf.write('Against Medical Advice: %d\n' % (n_ama))
    sf.write('Died: %d\n' % (n_lt+n_gt))
    sf.write('   <48 hrs: %d\n' % (n_lt))
    sf.write('   >48 hrs: %d\n\n' % (n_gt))
    sf.write('Other: %d\n' % (n_oth))
    sf.write('Unknown (not provided):\t%d\n' % (n_unk))

    sf.close()

def get_SOFA(id_file,in_name,out_name):
    ids = np.loadtxt(id_file,dtype=int)

    blood_gas = get_mat(in_name,'BLOOD_GAS','STUDY_PATIENT_ID')
    PaO2 = blood_gas.columns.get_loc('PO2_D1_HIGH_VALUE')
    blood_gas = blood_gas.as_matrix()

    organ_sup = get_mat(in_name,'ORGANSUPP_VENT','STUDY_PATIENT_ID')
    Mech_vent = [organ_sup.columns.get_loc('VENT_START_DATE'),organ_sup.columns.get_loc('VENT_STOP_DATE')]
    organ_sup = organ_sup.as_matrix()


    clinical_oth = get_mat(in_name,'CLINICAL_OTHERS','STUDY_PATIENT_ID')
    FiO2 = clinical_oth.columns.get_loc('FI02_D1_HIGH_VALUE')
    GCS = clinical_oth.columns.get_loc('GLASGOW_SCORE_D1_LOW_VALUE')
    clinical_oth = clinical_oth.as_matrix()


    clinical_vit = get_mat(in_name,'CLINICAL_VITALS','STUDY_PATIENT_ID')
    MAP = clinical_vit.columns.get_loc('ART_MEAN_D1_LOW_VALUE')
    clinical_vit = clinical_vit.as_matrix()



    labs = get_mat(in_name,'LABS_SET1','STUDY_PATIENT_ID')
    Bili = labs.columns.get_loc('BILIRUBIN_D1_HIGH_VALUE')
    Pltlts = labs.columns.get_loc('PLATELETS_D1_LOW_VALUE')
    labs = labs.as_matrix()


    medications = get_mat(in_name,'MEDICATIONS_INDX','STUDY_PATIENT_ID')
    Med_name = medications.columns.get_loc('MEDICATION_TYPE')
    medications = medications.as_matrix()


    scr_agg = get_mat(in_name,'SCR_INDX_AGG','STUDY_PATIENT_ID')
    SCR = scr_agg.columns.get_loc('DAY1_MAX_VALUE')
    scr_agg = scr_agg.as_matrix()

    out = open(out_name,'w')
    for idx in ids:
        out.write('%d' % (idx))
        bg_rows = np.where(blood_gas[:,0]==idx)
        co_rows = np.where(clinical_oth[:,0]==idx)
        cv_rows = np.where(clinical_vit[:,0]==idx)
        lab_rows = np.where(labs[:,0]==idx)
        med_rows = np.where(medications[:,0]==idx)
        scr_rows = np.where(scr_agg[:,0]==idx)
        mv_rows = np.where(organ_sup[:,0]==idx)

        s1_pa = blood_gas[bg_rows,PaO2]
        s1_fi = clinical_oth[co_rows,FiO2]
        #s1_vent = organ_sup[mv_rows,Mech_vent]

        s2_gcs = clinical_oth[co_rows,GCS]

        s3_map = clinical_vit[cv_rows,MAP]
        s3_med = medications[med_rows,Med_name]

        s4_bili = labs[lab_rows,Bili]

        s5_plt = labs[lab_rows,Pltlts]

        s6_scr = scr_agg[scr_rows,SCR]

        score = np.zeros(6,dtype=int)

        if len(mv_rows) > 0:
            vent = 1
        else:
            vent = 0
        try:
            s1 = s1_fi
            if s1_pa <= s1_fi:
                s1 = s1_pa
            if s1_pa < 400:
                score[0] += 1
                if s1_pa < 300:
                    score[0] += 1
                    if vent:
                        if s1_pa < 200:
                            score[0] += 1
                            if s1_pa < 100:
                                score[0] += 1
        except:
            score[0] = np.nan

        s2 = s2_gcs
        try:
            if s2 < 15:
                score[1] += 1
                if s2 < 13:
                    score[1] += 1
                    if s2 < 10:
                        score[1] += 1
                        if s2 < 6:
                            score[1] += 1
        except:
            score[1] = 0

        dopa = 0
        epi = 0
        for med_typ in s3_med:
            try:
                dm = med_typ[0][0].lower()
                if 'dopamine' in dm or 'dobutamine' in dm:
                    dopa = 1
                elif 'epinephrine' in dm:
                    epi = 1
            except:
                continue
        s3 = s3_map
        if epi:
            score[2] = 3
        elif dopa:
            score[2] = 2
        elif s3 < 70:
            score[2] = 1

        s4 = s4_bili
        try:
            if s4 > 1.2:
                score[3] += 1
                if s4 > 2.0:
                    score[3] += 1
                    if s4 > 6.0:
                        score[3] += 1
                        if s4 > 12.0:
                            score[3] += 1
        except:
            score[3] = 0

        s5 = s5_plt
        try:
            if s5 < 150:
                score[4] += 1
                if s5 < 100:
                    score[4] += 1
                    if s5 < 50:
                        score[4] += 1
                        if s5  < 20:
                            score[4] += 1
        except:
            score[4] = 0

        s6 = s6_scr
        try:
            if s6 > 1.2:
                score[5] += 1
                if s6 > 2.0:
                    score[5] += 1
                    if s6 > 3.5:
                        score[5] += 1
                        if s6 > 5.0:
                            score[5] += 1
        except:
            score[5] = 0

        out.write(',%d,%d,%d,%d,%d,%d\n' % (int(score[0]),int(score[1]),int(score[2]),
                                           int(score[3]),int(score[4]),int(score[5])))
        print(np.sum(score))



def get_APACHE(id_file,in_name,out_name):
    ids = np.loadtxt(id_file,dtype=int)

    clinical_vit = get_mat(in_name,'CLINICAL_VITALS','STUDY_PATIENT_ID')
    Temp = [clinical_vit.columns.get_loc('TEMPERATURE_D1_LOW_VALUE'),clinical_vit.columns.get_loc('TEMPERATURE_D1_HIGH_VALUE')]
    MAP = [clinical_vit.columns.get_loc('ART_MEAN_D1_LOW_VALUE'),clinical_vit.columns.get_loc('ART_MEAN_D1_HIGH_VALUE')]
    HR = [clinical_vit.columns.get_loc('HEART_RATE_D1_LOW_VALUE'),clinical_vit.columns.get_loc('HEART_RATE_D1_HIGH_VALUE')]
    clinical_vit = clinical_vit.as_matrix()



    clinical_oth = get_mat(in_name,'CLINICAL_OTHERS','STUDY_PATIENT_ID')
    Resp = [clinical_oth.columns.get_loc('RESP_RATE_D1_LOW_VALUE'),clinical_oth.columns.get_loc('RESP_RATE_D1_HIGH_VALUE')]
    FiO2 = [clinical_oth.columns.get_loc('FI02_D1_LOW_VALUE'),clinical_oth.columns.get_loc('FI02_D1_HIGH_VALUE')]
    GCS = clinical_oth.columns.get_loc('GLASGOW_SCORE_D1_LOW_VALUE')
    clinical_oth = clinical_oth.as_matrix()




    blood_gas = get_mat(in_name,'BLOOD_GAS','STUDY_PATIENT_ID')
    PaO2 = [blood_gas.columns.get_loc('PO2_D1_LOW_VALUE'),blood_gas.columns.get_loc('PO2_D1_HIGH_VALUE')]
    PaCO2 = [blood_gas.columns.get_loc('PCO2_D1_LOW_VALUE'),blood_gas.columns.get_loc('PCO2_D1_HIGH_VALUE')]
    pH = [blood_gas.columns.get_loc('PH_D1_LOW_VALUE'),blood_gas.columns.get_loc('PH_D1_HIGH_VALUE')]
    blood_gas = blood_gas.as_matrix()


    labs = get_mat(in_name,'LABS_SET1','STUDY_PATIENT_ID')
    Na = [labs.columns.get_loc('SODIUM_D1_LOW_VALUE'),labs.columns.get_loc('SODIUM_D1_HIGH_VALUE')]
    pK = [labs.columns.get_loc('POTASSIUM_D1_LOW_VALUE'),labs.columns.get_loc('POTASSIUM_D1_HIGH_VALUE')]
    Hemat = [labs.columns.get_loc('HEMATOCRIT_D1_LOW_VALUE'),labs.columns.get_loc('HEMATOCRIT_D1_HIGH_VALUE')]
    WBC = [labs.columns.get_loc('WBC_D1_LOW_VALUE'),labs.columns.get_loc('WBC_D1_HIGH_VALUE')]
    labs = labs.as_matrix()



    dob = get_mat(in_name,'DOB','STUDY_PATIENT_ID')
    dob_idx = dob.columns.get_loc('DOB')
    dob = dob.as_matrix()

    bsln = get_mat(in_name,'BASELINE_SCR','STUDY_PATIENT_ID')
    bsln_date = bsln.columns.get_loc('BASELINE_DATE')
    bsln = bsln.as_matrix()

    scr_agg = get_mat(in_name,'SCR_INDX_AGG','STUDY_PATIENT_ID')
    SCR = scr_agg.columns.get_loc('DAY1_MAX_VALUE')
    scr_agg = scr_agg.as_matrix()

    out = open(out_name,'w')
    for idx in ids:
        out.write('%d' % (idx))

        bg_rows = np.where(blood_gas[:,0]==idx)
        co_rows = np.where(clinical_oth[:,0]==idx)
        cv_rows = np.where(clinical_vit[:,0]==idx)
        lab_rows = np.where(labs[:,0]==idx)
        scr_rows = np.where(scr_agg[:,0]==idx)
        dob_rows = np.where(dob[:,0]==idx)
        bsln_rows = np.where(bsln[:,0]==idx)

        s1_low = clinical_vit[cv_rows,Temp[0]]
        s1_high = clinical_vit[cv_rows,Temp[1]]

        s2_low = clinical_vit[cv_rows,MAP[0]]
        s2_high = clinical_vit[cv_rows,MAP[1]]

        s3_low = clinical_vit[cv_rows,HR[0]]
        s3_high = clinical_vit[cv_rows,HR[1]]

        s4_low = clinical_oth[co_rows,Resp[0]]
        s4_high = clinical_oth[co_rows,Resp[1]]

        try:
            s5_po = float(blood_gas[bg_rows,PaO2[1]]) / 100
        except:
            s5_po = np.nan
        try:
            s5_pco = float(blood_gas[bg_rows,PaCO2[1]]) / 100
        except:
            s5_pco = np.nan
        try:
            s5_f = float(blood_gas[bg_rows,FiO2[1]]) / 100
        except:
            s5_f = np.nan

        s6_low = labs[lab_rows,pH[0]]
        s6_high = labs[lab_rows,pH[1]]

        s7_low = labs[lab_rows,Na[0]]
        s7_high = labs[lab_rows,Na[1]]

        s8_low = labs[lab_rows,pK[0]]
        s8_high = labs[lab_rows,pK[1]]

        s9 = scr_agg[scr_rows,SCR]

        s10_low = labs[lab_rows,Hemat[0]]
        s10_high = labs[lab_rows,Hemat[1]]

        s11_low = labs[lab_rows,WBC[0]]
        s11_high = labs[lab_rows,WBC[1]]

        s12 = clinical_oth[co_rows,GCS]

        s13_dob = dob[dob_rows,dob_idx]
        s13_admit = bsln[bsln_rows,bsln_date]

        score = np.zeros(13)

        if s1_low < 30 or s1_high > 40:
            score[0] = 4
        elif s1_low < 32 or s1_high > 39:
            score[0] = 3
        elif s1_low < 34:
            score[0] = 2
        elif s1_low < 36 or s1_high > 38.5:
            score[0] = 1

        if s2_low < 49 or s2_high > 160:
            score[1] = 4
        elif s2_high > 130:
            score[1] = 3
        elif s2_low < 70 or s2_high > 110:
            score[1] = 2


        if s3_low < 49 or s3_high > 160:
            score[2] = 4
        elif s3_high > 130:
            score[2] = 3
        elif s3_low < 70 or s3_high > 110:
            score[2] = 2

        if s4_low <= 5 or s4_high >= 50:
            score[3] = 4
        elif s4_high >= 35:
            score[3] = 3
        elif s4_low <= 10:
            score[3] = 2
        elif s4_low < 12 or s4_high >= 25:
            score[3] = 1


        if s5_f >= 0.5:
            aado2 = s5_f*713 - (s5_pco/0.8) - s5_po
            if aado2 > 4:
                score[4] = 4
            elif aado2 > 3.5:
                score[4] = 3
            elif aado2 > 2:
                score[4] = 2
        else:
            if s5_po < 0.55:
                score[4] = 4
            elif s5_po < 0.60:
                score[4] = 3
            elif s5_po < 0.70:
                score[4] = 1

        if s6_low <= 7.15 or s6_high >= 7.7:
            score[5] = 4
        elif s6_low < 7.25 or s6_high >= 7.6:
            score[5] = 3
        elif s6_low < 7.33:
            score[5] = 2
        elif s6_high >= 7.5:
            score[5] = 1

        if s7_low <= 110 or s7_high >= 180:
            score[6] = 4
        elif s7_low < 120 or s7_high >= 160:
            score[6] = 3
        elif s7_low < 130 or s7_high >= 155:
            score[6] = 2
        elif s7_high >= 150:
            score[6] = 1

        if s8_low < 2.5 or s8_high >= 7:
            score[7] = 4
        elif s8_high <= 6:
            score[7] = 3
        elif s8_low < 3:
            score[7] = 2
        elif s8_low < 3.5 or s7_high >= 5.5:
            score[7] = 1

        if s9 >= 3.5:
            score[8] = 4
        elif s9 >= 2:
            score[8] = 3
        elif s9 >= 1.5 or s9 < 0.6:
            score[8] = 2

        if s10_low < 20 or s10_high >= 60:
            score[9] = 4
        elif s10_low < 30 or s10_high >= 50:
            score[9] = 2
        elif s10_high >= 46:
            score[9] = 1

        if s11_low < 1 or s11_high >= 40:
            score[10] = 4
        elif s11_low < 3 or s11_high >= 20:
            score[10] = 2
        elif s11_high >= 15:
            score[10] = 1

        try:
            gcs = int(s12.split('-')[-1])
            score[11] = 15 - gcs
        except:
            score[11] = 0

        age = s13_admit[0][0]-s13_dob[0][0]
        if age >= datetime.timedelta(75*365):
            score[12] = 6
        elif age >= datetime.timedelta(65*365) and age < datetime.timedelta(75*365):
            score[12] = 5
        elif age >= datetime.timedelta(55*365) and age < datetime.timedelta(65*365) :
            score[12] = 3
        elif age >= datetime.timedelta(45*365) and age < datetime.timedelta(55*365):
            score[12] = 2

        for i in range(len(score)):
            out.write(',%d' % (score[i]))
        out.write('\n')
        print(np.sum(score))


