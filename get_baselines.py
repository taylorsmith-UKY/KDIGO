#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:11:19 2017

@author: taylorsmith
"""

from kdigo_funcs import get_mat
from kdigo_funcs import get_baselines


#------------------------------- PARAMETERS ----------------------------------#
inFile = "/Users/taylorsmith/Google Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/DATA/KDIGO_full.xlsx"
sort_id = 'STUDY_PATIENT_ID'
sort_id_date = 'SCR_ENTERED'

date_m = get_mat(inFile,'ADMISSION_INDX',[sort_id])
id_loc=date_m.columns.get_loc("STUDY_PATIENT_ID")
hosp_locs=[date_m.columns.get_loc("HOSP_ADMIT_DATE"),date_m.columns.get_loc("HOSP_DISCHARGE_DATE")]
icu_locs=[date_m.columns.get_loc("ICU_ADMIT_DATE"),date_m.columns.get_loc("ICU_DISCHARGE_DATE")]
date_m=date_m.as_matrix()

scr_all_m = get_mat(inFile,'SCR_ALL_VALUES',[sort_id,sort_id_date])
scr_date_loc = scr_all_m.columns.get_loc('SCR_ENTERED')
scr_val_loc = scr_all_m.columns.get_loc('SCR_VALUE')
scr_desc_loc = scr_all_m.columns.get_loc('SCR_ENCOUNTER_TYPE')
scr_all_m = scr_all_m.as_matrix()

bsln_m = get_mat(inFile,'BASELINE_SCR',[sort_id])
bsln_scr_loc = bsln_m.columns.get_loc('BASELINE_VALUE')
bsln_type_loc = bsln_m.columns.get_loc('BASELINE_TYPE')
bsln_date_loc = bsln_m.columns.get_loc('BASELINE_DATE')
bsln_m = bsln_m.as_matrix()

get_baselines(date_m,hosp_locs,bsln_m,bsln_scr_loc,bsln_type_loc,\
                   scr_all_m,scr_val_loc,scr_date_loc,scr_desc_loc)
