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

#def count_eps(kdigo,timescale,gap):

#def get_los(PID,date_m,hosp_locs,icu_locs):

#def get_survival_time(PID,date_m,hosp_locs,outcome_m,death_loc):
    #disch = get_disch_date(idx,date_m,hosp_locs)
    #dod = get_dod(idx,date_m,outcome_m,dod_loc)
    #return rdelta.relativedelta(dod,disch)

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