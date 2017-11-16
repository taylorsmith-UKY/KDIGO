from __future__ import division
import pandas as pd
import numpy as np
import datetime
import re
from kdigo_funcs import *

sort_id = 'STUDY_PATIENT_ID'
inFile = "../DATA/KDIGO_full.xlsx"

'''
Code:
    0 - Dead in hospital
    1 - Dead after 48 hrs
    2 - Alive
    3 - Transfered
    4 - AMA
'''

date_m = get_mat(inFile,'ADMISSION_INDX',[sort_id])
id_loc=date_m.columns.get_loc("STUDY_PATIENT_ID")
hosp_locs=[date_m.columns.get_loc("HOSP_ADMIT_DATE"),date_m.columns.get_loc("HOSP_DISCHARGE_DATE")]
icu_locs=[date_m.columns.get_loc("ICU_ADMIT_DATE"),date_m.columns.get_loc("ICU_DISCHARGE_DATE")]
adisp_loc = date_m.columns.get_loc('DISCHARGE_DISPOSITION')
date_m = date_m.as_matrix()

outcome_m = get_mat(inFile,'OUTCOMES',[sort_id])
disp_loc = outcome_m.columns.get_loc('INDX_DISCHARGE_DISPOSITION')
dod_loc = outcome_m.columns.get_loc('DECEASED_DATE')
outcome_m = outcome_m.as_matrix()

dob_m = get_mat(inFile,'DOB',[sort_id])
dob_loc = dob_m.columns.get_loc('DOB')
dob_m = dob_m.as_matrix()

f = open('result/kdigo_sub_no0.csv','r')
rids = []
kdigos = []
for l in f:
    rids.append(int(l.split(', ')[0]))
    kdigos.append([int(float(x)) for x in l.split(',')[1:]])
    

f = open('result/kdigo_sub_no0_6clusters_matorder.csv','r')
clusters = []
ids = []
f.readline()
for l in f:
    ids.append(int(l.rstrip().split(',')[0][1:-1]))
    clusters.append(int(l.rstrip().split(',')[1]))
f.close()
ids = np.array(ids)
n_ids = len(ids)

dead_inp = np.zeros(n_ids,dtype=bool)
death_dur = np.repeat(datetime.timedelta(10000),n_ids)

kdigo_max = np.zeros(n_ids,dtype=int)
kdigo_counts = np.zeros([n_ids,5],dtype=int)
kdigo_pcts = np.zeros([n_ids,5],dtype=int)

for i in range(len(ids)):
    idx = ids[i]
    #assign death group
    rows = np.where(date_m[:,0]==idx)[0]
    for row in rows:
        str_disp = str(date_m[row,adisp_loc]).upper()
        if re.search('LESS THAN',str_disp):
            dead_inp[i]=1
    
    #get duration of death
    #disch = get_disch_date(date_m,hosp_locs,idx)
    #dod = get_dod(date_m,outcome_m,dod_loc,idx)
    #if dod is not None:
    #    death_dur[i] = dod - disch
    
    #get max and avg kdigo, as well as percent time at each stage
    kidx = np.where(rids == idx)[0][0]
    kdigo = kdigos[kidx]
    for j in range(5):
        kdigo_counts[i,j] = len(np.where(kdigo == i)[0])
    kdigo_max[i] = np.max(kdigo)
    kdigo_pcts = kdigo_counts[i,:] / np.sum(kdigo_counts[i,:])
    
#row_lbls = np.array(ids, dtype='|S5')[:, np.newaxis]

#ds = np.hstack((row_lbls,clusters,dead_inp,kdigo_max,kdigo_pcts))

np.savetxt('result/death_ind.csv',dead_inp)
np.savetxt('result/kdigo_max.csv',kdigo_max)
np.savetxt('result/kdigo_pct.csv',kdigo_pcts)

