#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:08:43 2017

@author: taylorsmith
"""

import pandas as pd
import numpy as np
import re
from kdigo_funcs import get_mat

sort_id = 'STUDY_PATIENT_ID'
inFile = "../DATA/KDIGO_full.xlsx"
outPath = "result/"

'''
Code:
    0 - Died < 48 hrs
    1 - Dead > 48 hrs
    2 - Alive
    3 - Transfered
    4 - AMA
'''

def main():
    date_m = get_mat(inFile,'ADMISSION_INDX',[sort_id])
    hosp_locs = [date_m.columns.get_loc("HOSP_ADMIT_DATE"),date_m.columns.get_loc("HOSP_DISCHARGE_DATE")]
    icu_locs = [date_m.columns.get_loc("ICU_ADMIT_DATE"),date_m.columns.get_loc("ICU_DISCHARGE_DATE")]
    adisp_loc = date_m.columns.get_loc('DISCHARGE_DISPOSITION')
    date_m = date_m.as_matrix()

    outcome_m = get_mat(inFile,'OUTCOMES',[sort_id])
    disp_loc = outcome_m.columns.get_loc('INDX_DISCHARGE_DISPOSITION')
    dd_loc = outcome_m.columns.get_loc('DECEASED_DATE')
    outcome_m = outcome_m.as_matrix()

    ids = date_m[:,0]
    ids = np.unique(ids)

    mlog = open(outPath + 'inpt_mortality.csv','w') #main log
    alog = open(outPath + 'alive_ids.csv','w')      #alive log
    dlog = open(outPath + 'died_inp_ids.csv','w')   #dead log
    n_alive = 0
    n_lt = 0
    n_gt = 0
    n_xfer = 0
    n_ama = 0
    n_unk = 0
    for i in range(len(ids)):
        idx = ids[i]
#        row = np.where(outcome_m[:,0]==idx)[0]
#        if len(row) > 0:
#            disp_str = str(outcome_m[row[0],disp_loc]).upper()
#            if str(outcome_m[row[0],dd_loc]) != 'nan':
#                dd = outcome_m[row[0],dd_loc]
#                disch = datetime.timedelta(0)
#                rows = np.where(date_m[:,0]==idx)[0]
#                for r in rows:
#                    if date_m[r,hosp_locs[1]] > disch:
#                        disch = date_m[r,hosp_locs[1]]
#                if disch - dd < datetime.timedelta(1):
#                    mlog.write('%d, date, 1\n' % (idx))
#                else:
#                    mlog.write('%d, date, 0, %s\n' % (idx,str(disch - dd)))
#            else:
#                if re.search('DIE',disp_str) or re.search('EXPIRE',disp_str) or re.search('MORG',disp_str) or re.search('DEATH',disp_str) or re.search('DECEASE',disp_str) or re.search('FUNERAL',disp_str):
#                    mlog.write('%d, disp, 1, %s\n' % (idx,disp_str))
#                else:
#                    mlog.write('%d, disp, 0, %s\n' % (idx,disp_str))
#        else:
        row = np.where(date_m[:,0]==idx)[0][0]
        str_disp = str(date_m[row,adisp_loc]).upper()
        if re.search('MORE THAN',str_disp):
            mlog.write('%d, 1, %s\n' % (idx,str_disp))
            dlog.write('%d\n' % (idx))
            n_gt += 1
        elif re.search('LESS THAN',str_disp):
            mlog.write('%d, 0, %s\n' % (idx,str_disp))
            dlog.write('%d\n' % (idx))
            n_lt += 1
        elif re.search('ALIVE',str_disp):
            mlog.write('%d, 2, %s\n' % (idx,str_disp))
            alog.write('%d\n' % (idx))
            n_alive += 1
        elif re.search('XFER',str_disp) or re.search('TRANS',str_disp):
            mlog.write('%d, 3, %s\n' % (idx,str_disp))
            n_xfer += 1
        elif re.search('AMA',str_disp):
            mlog.write('%d, 4, %s\n' % (idx,str_disp))
            n_ama += 1
        else:
            mlog.write('%d, 5, disposition empty\n' % (idx))
            n_unk += 1
    print('Number died less than 48 hrs: ' + str(n_lt))
    print('Number died after 48 hrs: ' + str(n_gt))
    print('Number alive, routine: ' + str(n_alive))
    print('Number transfered: ' + str(n_xfer))
    print('Number left against medical advice: ' + str(n_ama))
    print('Number unknown disposition: ' + str(n_unk) + '\n')
    mlog.write('Number died less than 48 hrs: ' + str(n_lt) + '\n')
    mlog.write('Number died after 48 hrs: ' + str(n_gt) + '\n')
    mlog.write('Number alive, routine: ' + str(n_alive) + '\n')
    mlog.write('Number transfered: ' + str(n_xfer) + '\n')
    mlog.write('Number left against medical advice: ' + str(n_ama) + '\n')
    mlog.write('Number unknown disposition: ' + str(n_unk) + '\n')
    mlog.close()

main()