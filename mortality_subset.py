#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:08:43 2017

This file is used to extract `n_pts` ids from the full list of patients such
that there is an equal number of patients who died prior to discharge as were
alive at discharge. Also prints general statistics regarding discharge disposition.

@author: taylorsmith
"""

import numpy as np
from numpy.random import permutation as permute
import re

#------------------------------- PARAMETERS ----------------------------------#
base_path = '/Users/taylorsmith/Google Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/DATA/shared/'
sep = 'icu/'
set_name = 'subset2'
n_pts = 200
#-----------------------------------------------------------------------------#
data_path = base_path + 'DATA/'
path = data_path + sep
disp_file = path + 'disch_disp.csv'
fname = path + set_name + '_ids.csv'

def main():
    n_alive = 0
    n_lt = 0
    n_gt = 0
    n_xfer = 0
    n_ama = 0
    n_unk = 0
    alive_ids = []
    dead_ids = []
    f = open(disp_file,'r')
    for l in f:
        idx = int(l.split(',')[0])
        str_disp = l.split(',')[1].upper()
        if re.search('EXP',str_disp):
            dead_ids.append(idx)
            if re.search('LESS',str_disp):
                n_lt += 1
            elif re.search('MORE',str_disp):
                n_gt += 1
        elif re.search('ALIVE',str_disp):
            alive_ids.append(idx)
            n_alive += 1
        elif re.search('XFER',str_disp) or re.search('TRANS',str_disp):
            n_xfer += 1
        elif re.search('AMA',str_disp):
            n_ama += 1
        else:
            n_unk += 1
    print('Number died less than 48 hrs: ' + str(n_lt))
    print('Number died after 48 hrs: ' + str(n_gt))
    print('Number alive, routine: ' + str(n_alive))
    print('Number transfered: ' + str(n_xfer))
    print('Number left against medical advice: ' + str(n_ama))
    print('Number unknown disposition: ' + str(n_unk) + '\n')

    np.savetxt(path+'inp_died_ids.csv',dead_ids,fmt='%d')
    np.savetxt(path+'alive_ids.csv',alive_ids,fmt='%d')
    if len(alive_ids)*2 < n_pts or len(dead_ids)*2 < n_pts:
        print('Change number of desired patients')
        print('Current: %d' % (n_pts))
        print('Number died: %d' % (len(dead_ids)))
        print('Number alive: %d' % (len(alive_ids)))
    else:
        alive = permute(alive_ids)[:(n_pts/2)]
        dead = permute(dead_ids)[:(n_pts/2)]
        ids = permute(np.concatenate((alive,dead)))
        np.savetxt(fname,ids,fmt='%d')

main()