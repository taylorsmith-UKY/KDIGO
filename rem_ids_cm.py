#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:41:20 2018

@author: taylorsmith
"""
import numpy as np

bad_id_file = '../DATA/icu/7days/all0_ids.txt'
all_data_file = '../RESULTS/icu/7days/kdigo_dm_final.csv'
cleaned_file = '../RESULTS/icu/7days/kdigo_dm_final_no0.csv'

bad_ids = np.loadtxt(bad_id_file,dtype=int)

inf = open(all_data_file,'r')
out = open(cleaned_file,'w')

for l in inf:
    id1,id2,d = l.rstrip().split(',')
    id1 = int(id1)
    id2 = int(id2)
    d = float(d)
    if id1 in bad_ids or id2 in bad_ids:
        continue
    else:
        out.write('%d,%d,%.3f\n' % (id1,id2,d))