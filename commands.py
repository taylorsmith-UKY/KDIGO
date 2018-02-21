#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 12:06:33 2017

@author: taylorsmith
"""
from kdigo_funcs import get_mat, get_baselines, load_csv
from vis_funcs import hist
import numpy as np
from scipy.spatial.distance import squareform
from kdigo_funcs import pairwise_dtw_dist as ppd
from kdigo_funcs import arr2csv
import os

#%% Calculate Baselines from Raw Data in Excel and store in CSV
def calc_baselines():
    inFile = "/Users/taylorsmith/Google Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/DATA/KDIGO_full.xlsx"
    sort_id = 'STUDY_PATIENT_ID'
    sort_id_date = 'SCR_ENTERED'

    date_m = get_mat(inFile,'ADMISSION_INDX',[sort_id])
    hosp_locs=[date_m.columns.get_loc("HOSP_ADMIT_DATE"),date_m.columns.get_loc("HOSP_DISCHARGE_DATE")]
    date_m=date_m.as_matrix()

    scr_all_m = get_mat(inFile,'SCR_ALL_VALUES',[sort_id,sort_id_date])
    scr_date_loc = scr_all_m.columns.get_loc('SCR_ENTERED')
    scr_val_loc = scr_all_m.columns.get_loc('SCR_VALUE')
    scr_desc_loc = scr_all_m.columns.get_loc('SCR_ENCOUNTER_TYPE')
    scr_all_m = scr_all_m.as_matrix()

    dia_m = get_mat(inFile,'RENAL_REPLACE_THERAPY',[sort_id])
    crrt_locs = [dia_m.columns.get_loc('CRRT_START_DATE'),dia_m.columns.get_loc('CRRT_STOP_DATE')]
    hd_locs = [dia_m.columns.get_loc('HD_START_DATE'),dia_m.columns.get_loc('HD_STOP_DATE')]
    pd_locs = [dia_m.columns.get_loc('PD_START_DATE'),dia_m.columns.get_loc('PD_STOP_DATE')]
    dia_m = dia_m.as_matrix()

    get_baselines(date_m,hosp_locs,scr_all_m,scr_val_loc,scr_date_loc,scr_desc_loc,dia_m,crrt_locs,hd_locs,pd_locs)


#%% Get histograms for distributions of SCr values and # records
def get_hists():
    f = open('../DATA/icu/masks.csv','r')
    fname = '../DATA/icu/raw_scr_hist.pdf'
    hist(f,fname,'Raw SCr Values in ICU',x_lbl='SCr Value',y_lbl='# Measurements',x_rng=(0,10))
    f.close()
    f = open('../DATA/icu/scr_raw.csv','r')
    fname = '../DATA/icu/record_count_hist.pdf'
    hist(f,fname,'Distribution of Number of Measurements in ICU',x_lbl='Number of Measurements',y_lbl='# Patients',x_rng=(0,50))


#%%Generate subset DM from previously extracted patient data
def subset_dm():
    #------------------------------- PARAMETERS ----------------------------------#
    #root directory for study
    data_path = '../DATA/icu/'
    #base for output filenames
    set_name = 'subset2'
    res_path = '../RESULTS/icu/'+set_name+'/'
    #-----------------------------------------------------------------------------#
    #generate paths and filenames
    id_file = data_path + set_name + '_ids.csv'

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    dmname = res_path+'dm.csv'
    dtwname = res_path+'dtw.csv'

    dmsname = res_path+'dm_square.csv'

    #get desired ids
    keep_ids = np.loadtxt(id_file,dtype=int)

    #load kdigo vector for each patient
    kd_fname = data_path + 'kdigo.csv'
    f = open(kd_fname,'r')
    ids,kdigos = load_csv(kd_fname,keep_ids,dt=int)
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
