import numpy as np
from scipy.spatial.distance import squareform
from kdigo_funcs import pairwise_dtw_dist as ppd
from kdigo_funcs import arr2csv
import os

#------------------------------- PARAMETERS ----------------------------------#
#root directory for study
base_path = '/Users/taylorsmith/Google Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/DATA/shared/'
sep = 'icu/'

#base for output filenames
set_name = 'subset2'
#-----------------------------------------------------------------------------#
#generate paths and filenames
data_path = base_path+'DATA/'+sep
id_file = data_path + set_name + '_ids.csv'

res_path = base_path+'RESULTS/'+sep+set_name
if not os.path.exists(res_path):
    os.makedirs(res_path)

dmname = res_path+'/dm.csv'
dtwname = res_path+'/dtw.csv'

dmsname = res_path+'/dm_square.csv'

#get desired ids
keep_ids = np.loadtxt(id_file,dtype=int)

#load kdigo vector for each patient
kd_fname = data_path + 'kdigo.csv'
f = open(kd_fname,'r')
ids = []
kdigos = []
for l in f:
    if int(l.split(',')[0]) in keep_ids:
        ids.append(l.split(',')[0])
        kdigos.append([int(float(x)) for x in l.split(',')[1:]])
ids = np.array(ids,dtype=int)
f.close()

#perform pairwise DTW + distance calculation
cdm = ppd(kdigos, ids, dmname, dtwname)


#condensed matrix -> square
ds = squareform(cdm)

arr2csv(dmsname,ds,ids,fmt='%f',header=True)
