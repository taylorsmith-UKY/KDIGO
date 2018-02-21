#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:41:20 2018

@author: taylorsmith
"""
import numpy as np

bsln_file = '../DATA/baselines_final.csv'
count_file = '../DATA/icu/7days/record_counts.csv'
bad_id_file = '../DATA/bad_ids.txt'

rrt_ids = []
f = open(bsln_file,'r')
_ = f.readline()
for l in f:
    idx,bsln_val,bsln_type,bsln_date = l.rstrip().split(',')
    if bsln_date == 'all_RRT':
        rrt_ids.append(int(idx))
f.close()

low_count_ids = []
f = open(count_file,'r')
_ = f.readline()
for l in f:
    if len(l.rstrip().split(',')) < 4:
        continue
    idx,bsln,tot,sel = l.rstrip().split(',')
    if int(sel) < 2:
        low_count_ids.append(int(idx))
f.close()

rrt_ids = np.sort(rrt_ids)
low_count_ids = np.sort(low_count_ids)

f = open(bad_id_file,'w')
f.write('id,reason\n')

dbl = False
rsn = ''
while len(rrt_ids) > 0 and len(low_count_ids) > 0:
    if rrt_ids[0] < low_count_ids[0]:
        idx = rrt_ids[0]
        rrt_ids = np.delete(rrt_ids,0)
        rsn += 'all_rrt'
        f.write('%d,%s\n' % (idx,rsn))
        rsn = ''
    elif rrt_ids[0] > low_count_ids[0]:
        idx = low_count_ids[0]
        low_count_ids = np.delete(low_count_ids,0)
        rsn += '<2_records'
        f.write('%d,%s\n' % (idx,rsn))
        rsn = ''
    else:
        rsn = 'all_rrt-'
        rrt_ids = np.delete(rrt_ids,0)

if len(rrt_ids) > 0:
    for i in range(len(rrt_ids)):
        idx = rrt_ids[0]
        rrt_ids = np.delete(rrt_ids,0)
        rsn = 'all_rrt'
        f.write('%d,%s\n' % (idx,rsn))

if len(low_count_ids) > 0:
    for i in range(len(low_count_ids)):
        idx = low_count_ids[0]
        low_count_ids = np.delete(low_count_ids,0)
        rsn = '<2_records'
        f.write('%d,%s\n' % (idx,rsn))