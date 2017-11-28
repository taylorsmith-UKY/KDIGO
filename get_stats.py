from __future__ import division
import numpy as np
import re
from kdigo_funcs import load_csv

#------------------------------- PARAMETERS ----------------------------------#
#root directory for study
base_path = '/Users/taylorsmith/Google Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/'
sep = 'icu/'
id_fname = '6clusters_matorder.csv'

#-----------------------------------------------------------------------------#
#generate paths and filenames
data_path = base_path+'DATA/'+sep

res_path = base_path+'RESULTS/'+sep

idFile = res_path + id_fname

'''
Code:
    0 - Dead in hospital
    1 - Dead after 48 hrs
    2 - Alive
    3 - Transfered
    4 - AMA
'''

#get IDs in order from cluster file
ids = []
f = open(idFile,'r')
_ = f.readline()
for l in f:
    ids.append(int(l.split(',')[0][1:-1]))
f.close()
n_ids = len(ids)

k_ids,kdigos = load_csv(data_path + 'kdigo.csv',ids,dt=int)
d_ids,d_disp = load_csv(data_path + 'disch_disp.csv',ids,dt='|S')

dead_inp = np.zeros(n_ids,dtype=int)
kdigo_max = np.zeros(n_ids,dtype=int)
kdigo_counts = np.zeros([n_ids,5],dtype=int)
kdigo_pcts = np.zeros([n_ids,5],dtype=float)
n_alive = 0
n_died = 0
n_lt = 0
n_gt = 0
n_xfer = 0
n_ama = 0
n_unk = 0

for i in range(len(ids)):
    idx = ids[i]
    #assign death group
    drow = np.where(d_ids==idx)[0][0]
    str_disp = d_disp[drow].upper()
    if re.search('EXP',str_disp):
        dead_inp[i] = 1
        n_died += 1
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
    else:
        n_unk += 1

    #get duration of death
    #disch = get_disch_date(date_m,hosp_locs,idx)
    #dod = get_dod(date_m,outcome_m,dod_loc,idx)
    #if dod is not None:
    #    death_dur[i] = dod - disch

    #get max and avg kdigo, as well as percent time at each stage
    krow = np.where(k_ids == idx)[0][0]
    for j in range(5):
        kdigo_counts[i,j] = len(np.where(kdigos[krow] == i)[0])
        kdigo_pcts = kdigo_counts[i,j] / len(kdigos[krow])
    kdigo_max[i] = np.max(kdigos[krow])


np.savetxt('result2/death_ind.csv',dead_inp)
np.savetxt('result2/kdigo_max.csv',kdigo_max)
np.savetxt('result2/kdigo_pct.csv',kdigo_pcts)

