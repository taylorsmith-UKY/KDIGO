#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:15:39 2017

@author: taylorsmith
"""
import numpy as np
import datetime
from dateutil import relativedelta as rdelta

def get_disch_date(idx,date_m,hosp_locs):
    rows = np.where(date_m[0] == idx)
    dd = datetime.timedelta(0)
    for row in rows:
        if date_m[row,hosp_locs[1]] > dd:
            dd = date_m[row,hosp_locs[1]]
    #dd.resolution=datetime.timedelta(1)
    return dd

#%%
def get_dod(idx,date_m,outcome_m,dod_loc):
    rows = np.where(date_m[0] == idx)
    if rows == None:
        return rows
    dd = datetime.timedelta(0)
    for row in rows:
        if outcome_m[row,dod_loc] > dd:
            dd = outcome_m[row,dod_loc]
    if dd == datetime.timedelta(0):
        return None
    return dd

#def count_eps(kdigo,timescale,gap):

#def get_los(PID,date_m,hosp_locs,icu_locs):

#def get_survival_time(PID,date_m,hosp_locs,outcome_m,death_loc):
    #disch = get_disch_date(idx,date_m,hosp_locs)
    #dod = get_dod(idx,date_m,outcome_m,dod_loc)
    #return rdelta.relativedelta(dod,disch)