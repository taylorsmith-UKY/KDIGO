from __future__ import division
import numpy as np
import re
from kdigo_funcs import load_csv
import stat_funcs as sf
from kdigo_funcs import get_mat
import datetime

#------------------------------- PARAMETERS ----------------------------------#
#root directory for study
base_path = '/Users/taylorsmith/Google Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/DATA/shared/'
sep = 'icu/'
set_name = 'subset2'
sort_id = 'STUDY_PATIENT_ID'
#-----------------------------------------------------------------------------#
#generate paths and filenames
data_path = base_path+'DATA/'
inFile = base_path + "KDIGO_full.xlsx"

res_path = base_path+'RESULTS/'+sep+'/'+ set_name
idFile = res_path + '/clusters_matorder.csv'

'''
Code:
    0 - Alive
    1 - Died < 48 hrs after admission
    2 - Died > 48 hrs after admission
    3 - Transfered
    4 - AMA
    -1 - Unknown
'''
##Get hospital and ICU admit/discharge dates
#date_m = get_mat(inFile,'ADMISSION_INDX',[sort_id])
#id_loc=date_m.columns.get_loc("STUDY_PATIENT_ID")
#hosp_locs=[date_m.columns.get_loc("HOSP_ADMIT_DATE"),date_m.columns.get_loc("HOSP_DISCHARGE_DATE")]
#icu_locs=[date_m.columns.get_loc("ICU_ADMIT_DATE"),date_m.columns.get_loc("ICU_DISCHARGE_DATE")]
#date_m=date_m.as_matrix()
#

##Outcomes
#outcome_m = get_mat(inFile,'OUTCOMES',sort_id)
#death_loc=outcome_m.columns.get_loc("STUDY_PATIENT_ID")
#outcome_m=outcome_m.as_matrix()
#-----------start-------------------------------
date_ids=date_m[:,0]
date_all_ids=[]
for i in range(len(date_ids)):
    date_all_ids.append(date_ids[i])
outcome_ids = outcome_m[:,0]
outcome_allid=[]
for i in range(len(outcome_ids)):
    outcome_allid.append(outcome_ids[i])
#------------end-------------------------------
#get IDs in order from cluster file
ids = []
f = open(idFile,'r')
_ = f.readline()
for l in f:
    ids.append(int(l.split(',')[0][1:-1]))
f.close()
n_ids = len(ids)

#load KDIGO and discharge dispositions
k_ids,kdigos = load_csv(data_path + sep + 'kdigo.csv',ids,dt=int)
d_ids,d_disp = load_csv(data_path + sep + 'disch_disp.csv',ids,dt='|S')
k_ids = np.array(k_ids,dtype=int)
d_ids = np.array(d_ids,dtype=int)

#arrays to store results
disch_disp = np.zeros(n_ids,dtype=int)
kdigo_max = np.zeros(n_ids,dtype=int)
kdigo_counts = np.zeros([n_ids,5],dtype=int)
kdigo_pcts = np.zeros([n_ids,5],dtype=float)
los = np.zeros([n_ids,2],dtype=datetime.datetime)    #   [:,0]=hospital   [:,1]=ICU
surv_t = np.zeros(n_ids,dtype=datetime.datetime)

for i in range(len(ids)):
    idx = ids[i]
    #assign death group
    drow = np.where(d_ids==idx)[0][0]
    str_disp = d_disp[drow][0].upper()
    if re.search('EXP',str_disp):
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

    #los = sf.get_los(PID,date_m,hosp_locs,icu_locs)

    #surv_t = sf.get_survival_time(PID,date_m,hosp_locs,outcome_m,death_loc)




    #get max and avg kdigo, as well as percent time at each stage
    krow = np.where(k_ids == idx)[0][0]
    kdigo = np.array(kdigos[krow],dtype=int)
    for j in range(5):
        kdigo_counts[i,j] = len(np.where(kdigo == j)[0])
        kdigo_pcts[i,j] = kdigo_counts[i,j] / len(kdigo)
    kdigo_max[i] = np.max(kdigo)

    #Add function to calculate number of episodes
    #n_eps = count_eps(kdigo,timescale,gap)


np.savetxt(res_path+'/disch_codes.csv',disch_disp,fmt='%d')
np.savetxt(res_path+'/kdigo_max.csv',kdigo_max,fmt='%d')
np.savetxt(res_path+'/kdigo_pct.csv',kdigo_pcts,fmt='%f')
#np.savetxt(res_path+'/n_episodes.csv',n_eps,fmt='%d')
#np.savetxt(res_path+'/all_los.csv',los,fmt=datetime.datetime)
#np.savetxt(res_path+'/survival_duration.csv',surv_t,fmt=datetime.datetime)
#================================================================
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
#======================================================================================
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
#================================================================================
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
