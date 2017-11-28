from __future__ import division
import numpy as np
import re
from kdigo_funcs import get_mat

sort_id = 'STUDY_PATIENT_ID'
inFile = "/Users/taylorsmith/Google Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/DATA/KDIGO_full.xlsx"

'''
Code:
    0 - Dead in hospital
    1 - Dead after 48 hrs
    2 - Alive
    3 - Transfered
    4 - AMA
'''

date_m = get_mat(inFile,'ADMISSION_INDX',[sort_id])
adisp_loc = date_m.columns.get_loc('DISCHARGE_DISPOSITION')
date_m = date_m.as_matrix()


ids = np.loadtxt('result/6clusters_matorder.csv')[:,0]

f = open('kdigo.csv','r')
kdigos = []
k_ids = []
for l in f:
    if l.split(',')[0] in ids:
        k_ids.append(l.split(',')[0])
        kdigos.append([int(float(x)) for x in l.split(',')[1:]])
n_ids = len(ids)
dead_inp = np.zeros(n_ids,dtype=int)
kdigo_max = np.zeros(n_ids,dtype=int)
kdigo_counts = np.zeros([n_ids,5],dtype=int)
kdigo_pcts = np.zeros([n_ids,5],dtype=int)
n_alive = 0
n_lt = 0
n_gt = 0
n_xfer = 0
n_ama = 0
n_unk = 0

for i in range(len(ids)):
    idx = ids[i]
    #assign death group
    row = np.where(date_m[:,0]==idx)[0][0]
    str_disp = str(date_m[row,adisp_loc]).upper()
    if re.search('MORE THAN',str_disp):
        dead_inp[i] = 1
        n_gt += 1
    elif re.search('LESS THAN',str_disp):
        dead_inp[i] = 1
        n_lt += 1
    elif re.search('ALIVE',str_disp):
        n_alive += 1
    elif re.search('XFER',str_disp) or re.search('TRANS',str_disp):
        n_xfer += 1
    elif re.search('AMA',str_disp):
        n_ama += 1
    else:
        n_unk += 1

    #get duration of death
    #disch = get_disch_date(date_m,hosp_locs,idx)
    #dod = get_dod(date_m,outcome_m,dod_loc,idx)
    #if dod is not None:
    #    death_dur[i] = dod - disch

    #get max and avg kdigo, as well as percent time at each stage
    kidx = np.where(k_ids == idx)[0][0]
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

