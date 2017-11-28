import numpy as np
from scipy.spatial.distance import squareform

ids = np.loadtxt('ex_keep_ids.txt',dtype=int)

cdm = np.loadtxt('result/kdigo_ex_dm.csv')

ds = squareform(cdm)

f=open('result/kdigo_ex_dm_square.csv','w')
f.write('id')
for idx in ids:
    f.write(',%d' % (idx))
f.write('\n')

for i in range(ds.shape[0]):
    f.write('%d' % (ids[i]))
    for j in range(ds.shape[1]):
        f.write(',%f' % (ds[i][j]))
    f.write('\n')
f.close()