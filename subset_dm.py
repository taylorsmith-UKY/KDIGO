import numpy as np
from kdigo_funcs import *

outPath = 'result/'

# id_f = open(outPath+'sub_ids.csv')
# ids = []
# for l in id_f:
# 	ids.append(int(l.split(',')[0]))
# id_f.close()

kdigo_f = open(outPath+'kdigo.csv')
nf = open(outPath+'kdigo_all_no0.csv','w')
kdigo = []
for l in kdigo_f:
	#if int(l.split(',')[0]) in ids:
	k = np.array([int(float(x)) for x in l.split(', ')[1:]])
	if not np.all(k == 0):
		kdigo.append(k)
		nf.write(l)
kdigo_f.close()
nf.close()

kdigo_dm = pairwise_dtw_dist(kdigo,outPath+'kdigo_all_no0_dm.csv',None)
