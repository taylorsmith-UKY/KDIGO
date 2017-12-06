#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:15:39 2017

@author: taylorsmith
"""
from __future__ import division
from kdigo_funcs import load_csv
import datetime
import numpy as np
from dateutil import relativedelta as rdelta
import re

#%%
def get_cstats(clustFile,dataPath,outPath):
    #get IDs and Clusters in order from cluster file
    ids = []
    clusts = []
    f = open(clustFile,'r')
    _ = f.readline() #first line is header
    for l in f:
        ids.append(int(l.split(',')[0][1:-1]))
        clusts.append(int(l.rstrip().split(',')[-1]))
    f.close()
    n_ids = len(ids)

    n_clusts = max(clusts)

    #load KDIGO and discharge dispositions
    k_ids,kdigos = load_csv(dataPath + 'kdigo.csv',ids,dt=int)
    d_ids,d_disp = load_csv(dataPath + 'disch_disp.csv',ids,dt='|S')
    k_ids = np.array(k_ids,dtype=int)
    d_ids = np.array(d_ids,dtype=int)

    #load outcome data
    date_ids=date_m[:,0]
    date_all_ids=[]
    for i in range(len(date_ids)):
        date_all_ids.append(date_ids[i])
    outcome_ids = outcome_m[:,0]
    outcome_allid=[]
    for i in range(len(outcome_ids)):
        outcome_allid.append(outcome_ids[i])

    #arrays to store individual results
    disch_disp = np.zeros(n_ids,dtype=int)
    kdigo_max = np.zeros(n_ids,dtype=int)
    kdigo_counts = np.zeros([n_ids,5],dtype=int)
    kdigo_pcts = np.zeros([n_ids,5],dtype=float)
#    los = np.zeros([n_ids,2],dtype=datetime.datetime)    #   [:,0]=hospital   [:,1]=ICU
#    surv_t = np.zeros(n_ids,dtype=datetime.datetime)

    #arrays to store cluster results (each is in the form [mean,std])
    c_header = 'cluster_id,count,mort_avg,mort_std,kdigo_max_avg,kdig_max_std'
    c_stats = np.zeros([n_clusts,6],dtype=float)
    c_stats[:,0] = [x for x in range(1,n_clusts+1)]


    mk = [] #max kdigo
    di = [] #death indicator
    cnum = 0
    for i in range(len(ids)):
        if (cnum+1) != clusts[i]:
            c_stats[cnum,0] = cnum+1
            c_stats[cnum,2] = np.mean(di)
            c_stats[cnum,3] = np.std(di)
            c_stats[cnum,4] = np.mean(mk)
            c_stats[cnum,5] = np.std(mk)
            mk = []
            di = []
            cnum += 1
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

        #los = sf.get_los(PID,date_m,hosp_locs,icu_locs)

        #surv_t = sf.get_survival_time(PID,date_m,hosp_locs,outcome_m,death_loc)


        #get max and avg kdigo, as well as percent time at each stage
        krow = np.where(k_ids == idx)[0][0]
        kdigo = np.array(kdigos[krow],dtype=int)
        for j in range(5):
            kdigo_counts[i,j] = len(np.where(kdigo == j)[0])
            kdigo_pcts[i,j] = kdigo_counts[i,j] / len(kdigo)
        kdigo_max[i] = np.max(kdigo)
        mk.append(kdigo_max[i])

        #Add function to calculate number of episodes
        #n_eps = count_eps(kdigo,timescale,gap)
    c_stats[cnum,0] = cnum
    c_stats[cnum,2] = np.mean(di)
    c_stats[cnum,3] = np.std(di)
    c_stats[cnum,4] = np.mean(mk)
    c_stats[cnum,5] = np.std(mk)

    np.savetxt(outPath+'/disch_codes.csv',disch_disp,fmt='%d')
    np.savetxt(outPath+'/kdigo_max.csv',kdigo_max,fmt='%d')
    np.savetxt(outPath+'/kdigo_pct.csv',kdigo_pcts,fmt='%f')
    np.savetxt(outPath+'/cluster_stats.csv',c_stats,header=c_header,fmt='%f')
    #np.savetxt(res_path+'/n_episodes.csv',n_eps,fmt='%d')
    #np.savetxt(res_path+'/all_los.csv',los,fmt=datetime.datetime)
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
def count_eps(kdigo,timescale,gap):
    count = 0#count how many episode
    mult_0 = 0 #count how many 0s between two non-zero values
    num_t = gap/timescale#how many time point need to skip
    total_0=-1
    #all zero in this list return 0
    if all(i == 0 for i in kdigo):
        return count
    else:
        count_val=Counter(kdigo)
        for i,k in count_val.items():
             if i == 0:
                total_0 = k
                break
        #the list have less 0 (gap/timescale) means no episode count return 1
        if total_0 < num_t:
            count = count+1
        else:
            for j in range(len(kdigo)):
                if kdigo[j] !=0:#non zero number appear
                    if j > num_t+1:#if we find two non-zero have 8 zero in between, the non zero is at non-zero at list have 10 values,so index should be 9
                        for w in range(j-1,0,-1):#loop back to find the number of 0s
                            if kdigo[w] !=0:
                                break
                            else:
                                mult_0=mult_0+1
                        if mult_0 >= num_t:
                            count+=1
                            mult_0=0
        return count
#%%
def get_los(PID,date_m,hosp_locs,icu_locs,date_all_ids):
    #this function will get the first PID index
    index = date_all_ids.index(PID)
    hosp_times=[]#store all hospital  times into list with same patient
    icu_times=[]#store all icu times into a list with same patient
    temp_hosp=[]
    temp_icu=[]
    for i in range(index,len(date_all_ids)):
        if date_all_ids[i] == index:
            temp_hosp.append(date_m[i,hosp_locs[0]])
            temp_hosp.append(date_m[i,hosp_locs[1]])
            temp_icu.append(date_m[i,icu_locs[0]])
            temp_icu.append(date_m[i,icu_locs[1]])
            hosp_times.append(temp_hosp)
            icu_times.append(temp_icu)
            temp_hosp=[]
            temp_icu=[]
        else:
            break
    lst_set_hosp=[]
    lst_set_icu=[]
    los_hosp=0
    los_icu=0
    union_hosp=[]
    union_icu=[]
    for i in range(len(hosp_times)):
        hosp=[]
        icu=[]
        date_hosp1=hosp_times[i]
        date_hosp2=hosp_times[i+1]
        date_icu1=icu_times[i]
        date_icu2=icu_times[i+1]
        delta_hosp=date_hosp2-date_hosp1
        delta_icu = date_icu2-date_icu1
        for j in range(delta_hosp.days+1):
            hosp.append(hosp_times[i]+timedelta(days=j))
        for j in range(delta_icu.days+1):
            icu.append(date_icu1+timedelta(days=j))
        lst_set_hosp.append(hosp)
        lst_set_icu.append(icu)
        i=i+1
    #find the union of two list
    for i in range(len(lst_set_hosp)):
        union_hosp=sorted(list(set(union_hosp) | set(lst_set_hosp[i])))
        union_icu = sorted(list(set(union_icu) | set(lst_set_icu[i])))
    los_hosp=len(union_hosp)
    los_icu = len(union_icu)
    return los_hosp,los_icu

#%%
def get_survival_time(PID,date_m,hosp_locs,outcome_m,death_loc,date_all_ids,outcome_allid):
    #if duration = -1 means no death date provide for this id
    index_hosp=date_all_ids.index(PID)
    index_death = outcome_allid.index(PID)
    max_date=date_m[index_hosp][hosp_locs[1]]
    duration=-1
    if len(str(outcome_m[index_death][death_loc])) >5:
        #death date provide
        death_date = outcome_m[index_death][death_loc]
        for i in range(index_hosp,len(date_all_ids)):
            if date_m[index_hosp][0] == PID:
                if date_m[index_hosp][hosp_locs[1]] > max_date:
                    max_date = date_m[index_hosp][hosp_loc[1]]
            else:
                break
        diff = death_date - max_date
        duration = diff.days
    return duration

#%% Summarize discharge dispositions for a file
def get_disch_summary(inFile,statFile,ids=None):
    '''
    Code:
        0 - Dead in hospital
        1 - Dead after 48 hrs
        2 - Alive
        3 - Transfered
        4 - AMA
    '''

    #get IDs in order from cluster file
    f = open(inFile,'r')
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