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