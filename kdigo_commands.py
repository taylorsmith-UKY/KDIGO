#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Created on Wed Dec  6 12:06:33 2017

@author: taylorsmith
"""
from kdigo_funcs import get_mat, get_baselines, load_csv
from vis_funcs import hist
import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import normaltest
from kdigo_funcs import pairwise_dtw_dist as ppd
from kdigo_funcs import arr2csv, daily_max_kdigo, calc_gfr
from cluster_funcs import inter_intra_dist
from stat_funcs import get_cstats
from fastcluster import ward
from scipy.cluster.hierarchy import to_tree, fcluster
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import datetime
import os
import h5py


#%% Calculate Baselines from Raw Data in Excel and store in CSV
def calc_baselines(min_diff, max_diff, fname, data=None):
    inFile = "../DATA/KDIGO_full.xlsx"
    sort_id = 'STUDY_PATIENT_ID'
    sort_id_date = 'SCR_ENTERED'
    if data == None:
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
    else:
        date_m, hosp_locs, scr_all_m, scr_date_loc, scr_val_loc, scr_desc_loc, dia_m, crrt_locs, hd_locs, pd_locs = data

    get_baselines(date_m, hosp_locs, scr_all_m, scr_val_loc, scr_date_loc, scr_desc_loc,
                  dia_m, crrt_locs, hd_locs, pd_locs, fname, min_diff=min_diff, max_diff=max_diff)
    return (date_m, hosp_locs, scr_all_m, scr_date_loc, scr_val_loc, scr_desc_loc, dia_m, crrt_locs, hd_locs, pd_locs)


def count_bsln_types(bsln_file, fname, fig_fname):
    pre_outp = 0
    pre_inp = 0
    ind_inp = 0
    no_ind = 0
    all_rrt = 0

    f = open(bsln_file,'r')
    _ = f.readline()
    for line in f:
        l = line.rstrip().split(',')
        t = l[2]
        if t == 'OUTPATIENT':
            pre_outp += 1
        elif t == 'INPATIENT':
            pre_inp += 1
        elif t.split()[0] == 'INDEXED':
            ind_inp += 1
        elif t == 'No_indexed_values':
            no_ind += 1
        elif t == 'none':
            all_rrt += 1

    lbls = ['Pre-admit Outpatient', 'Pre-admit Inpatient', 'Indexed Inpatient', 'All RRT', 'No Indexed Recs']
    counts = [pre_outp, pre_inp, ind_inp, all_rrt, no_ind]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar([0, 1, 2, 3, 4], [pre_outp, pre_inp, ind_inp, all_rrt, no_ind], tick_label=lbls)
    ax.set_xticklabels(lbls, rotation=45)
    plt.ylabel('Number of Patients')
    plt.title('Distribution of Baseline Types')
    plt.savefig(fig_fname)

    outf = open(fname, 'w')
    for i in range(len(lbls)):
        outf.write(lbls[i] + ' - ' + str(counts[i])+'\n')
        print(lbls[i] + ' - ' + str(counts[i]))
    f.close()
    outf.close()
    return zip([lbls, counts])


def summarize_baselines(exp_baselines, b_types, epi_baselines, mdrd_baselines,
                        mk_exp, mk_epi, mk_mdrd, bsln_lim = None):
    optA_idx = np.where(b_types == 'OUTPATIENT')[0]
    optB_idx = np.where(b_types == 'INPATIENT')[0]
    optC_idx = np.where(b_types == 'INDEXED - INPATIENT')[0]
    bad_mdrd = np.where(mdrd_baselines[optC_idx] == np.inf)
    optC_idx_mdrd = np.setdiff1d(optC_idx, optC_idx[bad_mdrd])
    summaries = []

    if bsln_lim is None:
        summaries.append(baseline_string_summarize(exp_baselines, optA_idx, mk_exp))
        summaries.append(baseline_string_summarize(exp_baselines, optB_idx, mk_exp))
        summaries.append(baseline_string_summarize(exp_baselines, optC_idx, mk_exp))
        summaries.append(baseline_string_summarize(epi_baselines, optC_idx, mk_epi))
        summaries.append(baseline_string_summarize(mdrd_baselines, optC_idx_mdrd, mk_mdrd))
    else:
        excl_A = np.where(exp_baselines[optA_idx] >= bsln_lim)[0]
        excl_B = np.where(exp_baselines[optB_idx] >= bsln_lim)[0]
        excl_C = np.where(exp_baselines[optC_idx] >= bsln_lim)[0]
        excl_C_mdrd = np.where(exp_baselines[optC_idx_mdrd] >= bsln_lim)[0]
        sel_A = np.setdiff1d(optA_idx, optA_idx[excl_A])
        sel_B = np.setdiff1d(optB_idx, optB_idx[excl_B])
        sel_C = np.setdiff1d(optC_idx, optC_idx[excl_C])
        sel_C_mdrd = np.setdiff1d(optC_idx_mdrd, optC_idx_mdrd[excl_C_mdrd])

        summaries.append(baseline_string_summarize(exp_baselines, sel_A, mk_exp))
        summaries.append(baseline_string_summarize(exp_baselines, sel_B, mk_exp))
        summaries.append(baseline_string_summarize(exp_baselines, sel_C, mk_exp))
        summaries.append(baseline_string_summarize(epi_baselines, sel_C, mk_epi))
        summaries.append(baseline_string_summarize(mdrd_baselines, sel_C_mdrd, mk_mdrd))
    return summaries


def summarize_baseline_deltas(baselines, b_types, deltas, bsln_lim = None):
    optA_idx = np.where(b_types == 'OUTPATIENT')[0]
    optB_idx = np.where(b_types == 'INPATIENT')[0]
    summaries = []

    if bsln_lim is None:
        summaries.append(baseline_delta_string_summarize(deltas, optA_idx))
        summaries.append(baseline_delta_string_summarize(deltas, optB_idx))
    else:
        excl_A = np.where(baselines[optA_idx] >= bsln_lim)[0]
        excl_B = np.where(baselines[optB_idx] >= bsln_lim)[0]
        sel_A = np.setdiff1d(optA_idx, optA_idx[excl_A])
        sel_B = np.setdiff1d(optB_idx, optB_idx[excl_B])

        summaries.append(baseline_delta_string_summarize(deltas, sel_A))
        summaries.append(baseline_delta_string_summarize(deltas, sel_B))
    return summaries


def baseline_string_summarize(baselines, selection, max_kdigo):
    # Inpatient Lowest - Explicit Option C
    count = len(selection)
    mean = np.nanmean(baselines[selection])
    std = np.nanstd(baselines[selection])
    med = np.nanmedian(baselines[selection])
    p25 = np.nanpercentile(baselines[selection], 25)
    p75 = np.nanpercentile(baselines[selection], 75)
    minv = np.nanmin(baselines[selection])
    maxv = np.nanmax(baselines[selection])
    k0 = len(np.where(max_kdigo[selection] == 0)[0])
    k1 = len(np.where(max_kdigo[selection] == 1)[0])
    k2 = len(np.where(max_kdigo[selection] == 2)[0])
    k3 = len(np.where(max_kdigo[selection] == 3)[0])
    k4 = len(np.where(max_kdigo[selection] == 4)[0])
    return '%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d,%d' % \
           (count, mean, std, med, p25, p75, minv, maxv, k0, k1, k2, k3, k4)


def baseline_delta_string_summarize(deltas, selection):
    # Inpatient Lowest - Explicit Option C
    count = len(selection)
    mean = np.nanmean(deltas[selection])
    std = np.nanstd(deltas[selection])
    med = np.nanmedian(deltas[selection])
    p25 = np.nanpercentile(deltas[selection], 25)
    p75 = np.nanpercentile(deltas[selection], 75)
    minv = np.nanmin(deltas[selection])
    maxv = np.nanmax(deltas[selection])
    return '%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f' % (count, mean, std, med, p25, p75, minv, maxv)
