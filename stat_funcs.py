#!/usr/bin/env python2
# -*- coding: utf-8 -*-`
"""
Created on Wed Nov 29 13:15:39 2017

@author: taylorsmith
"""
from __future__ import division
from kdigo_funcs import load_csv, calc_gfr, get_date, daily_max_kdigo_interp, daily_max_kdigo_aligned, dtw_p,\
    mismatch_penalty_func, extension_penalty_func
import datetime
import numpy as np
import re
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import sem, t, ttest_ind, kruskal, normaltest
from sklearn.utils import resample


def summarize_stats(ids, kdigos, days, scrs, icu_windows, hosp_windows,
                    dem_m, sex_loc, eth_loc,
                    dob_m, birth_loc,
                    diag_m, diag_loc,
                    charl_m, charl_loc, elix_m, elix_loc,
                    organ_sup_mv, mech_vent_dates,
                    organ_sup_ecmo, ecmo_dates,
                    organ_sup_iabp, iabp_dates,
                    organ_sup_vad, vad_dates,
                    io_m,
                    mort_m, mdate_loc,
                    clinical_vit, map_locs, cuff_locs, temp_locs, hr_locs,
                    clinical_oth, height_loc, weight_loc, fio2_locs, gcs_loc, resp_locs,
                    dia_m, crrt_locs, hd_locs, hd_trt_loc,
                    med_m, med_type, med_date, med_name, med_dur,
                    labs1_m, hemat_locs, hemo_locs, bili_loc, bun_loc, pltlt_loc, na_locs, pk_locs, wbc_locs,
                    labs2_m, alb_loc, lac_loc,
                    blood_m, ph_locs, pao2_locs, paco2_locs,
                    urine_m, urine_locs,
                    smoke_m, former_smoke, current_smoke,
                    out_name, data_path, grp_name='meta', tlim=7):
    dd = np.array(load_csv(data_path + 'disch_disp.csv', ids, str, idxs=[1, ]))
    if type(icu_windows) == list:
        icu_windows = np.array(icu_windows)
    if type(hosp_windows) == list:
        hosp_windows = np.array(hosp_windows)

    n_eps = []
    for i in range(len(kdigos)):
        n_eps.append(count_eps(kdigos[i]))

    bsln_scrs = load_csv(data_path + 'baselines.csv', ids, float)
    bsln_gfrs = load_csv(data_path + 'baseline_gfr.csv', ids, float)
    bsln_types = load_csv(data_path + 'baseline_types.csv', ids, str)

    print('Getting patient demographics...')
    genders, eths, ages = get_uky_demographics(ids, hosp_windows[:, 0], dem_m, sex_loc, eth_loc, dob_m, birth_loc)
    print('Getting patient vitals...')
    temps, maps, hrs = get_uky_vitals(ids, clinical_vit, map_locs, cuff_locs, temp_locs, hr_locs)
    print('Getting patient comorbidities...')
    c_scores, e_scores, smokers = get_uky_comorbidities(ids, charl_m, charl_loc, elix_m, elix_loc, smoke_m, former_smoke, current_smoke,)
    print('Getting clinical others...')
    bmis, weights, heights, fio2s, resps, gcss = get_uky_clinical_others(ids, clinical_oth, height_loc, weight_loc,
                                                                         fio2_locs, resp_locs, gcs_loc)
    print('Getting diagnoses...')
    sepsiss, diabetics, hypertensives = get_uky_diagnoses(ids, diag_m, diag_loc)
    print('Getting fluid IO...')
    net_fluids, gross_fluids, fos = get_uky_fluids(ids, weights, io_m)
    print('Getting mortalities...')
    dieds, dtds, mdates = get_uky_mortality(ids, dd, hosp_windows, icu_windows, mort_m, mdate_loc)

    print('Getting dialysis...')
    rrt_flags, hd_dayss, crrt_dayss,\
    hd_frees_7d, hd_frees_28d, hd_trtmts,\
    crrt_frees_7d, crrt_frees_28d = get_uky_rrt(ids, dieds, icu_windows[:, 0], dia_m, crrt_locs, hd_locs, hd_trt_loc)
    print('Getting other organ support...')
    mv_flags, mv_days, mv_frees, ecmos, vads, iabps = get_uky_organsupp(ids, dieds, icu_windows[:, 0],
                                                                        organ_sup_mv, mech_vent_dates,
                                                                        organ_sup_ecmo, ecmo_dates,
                                                                        organ_sup_vad, vad_dates,
                                                                        organ_sup_iabp, iabp_dates)
    print('Getting medications...')
    nephrotox_cts, vasopress_cts, dopas, epis = get_uky_medications(ids, icu_windows[:, 0], med_m,
                                                                    med_type, med_date, med_name, med_dur)
    print('Getting labs...')
    anemics, bilis, buns,\
    hemats, hemos, pltlts,\
    nas, pks, albs, lacs, phs, \
    pao2s, paco2s, wbcs = get_uky_labs(ids, genders, labs1_m, bili_loc, bun_loc, hemat_locs, hemo_locs, pltlt_loc,
                                       na_locs, pk_locs, labs2_m, alb_loc, lac_loc, blood_m, ph_locs, pao2_locs,
                                       paco2_locs, wbc_locs)
    print('Getting urine flow...')
    urine_outs, urine_flows = get_uky_urine(ids, weights, urine_m, urine_locs)

    mks_7d = []
    mks_w = []
    nepss = []
    hosp_days = []
    hosp_frees = []
    icu_days = []
    icu_frees = []

    admit_scrs = []
    peak_scrs = []

    for i in range(len(ids)):
        td = days[i]
        mask = np.where(td <= tlim)[0]
        mk7d = np.max(kdigos[i][mask])
        mk = np.max(kdigos[i])
        
        (hosp_admit, hosp_disch) = hosp_windows[i]
        (icu_admit, icu_disch) = icu_windows[i]

        eps = n_eps[i]

        admit_scr = scrs[i][0]
        peak_scr = np.max(scrs[i])

        hlos = (hosp_disch - hosp_admit).total_seconds() / (60 * 60 * 24)
        ilos = (icu_disch - icu_admit).total_seconds() / (60 * 60 * 24)

        hfree = 28 - hlos
        ifree = 28 - ilos
        if hfree < 0:
            hfree = 0
        if ifree < 0:
            ifree = 0

        mks_7d.append(mk7d)
        mks_w.append(mk)
        nepss.append(eps)
        hosp_days.append(hlos)
        hosp_frees.append(hfree)
        icu_days.append(ilos)
        icu_frees.append(ifree)
        admit_scrs.append(admit_scr)
        peak_scrs.append(peak_scr)

    icu_windows_save = []
    for i in range(len(icu_windows)):
        icu_windows_save.append([str(icu_windows[i][0]), str(icu_windows[i][1])])
    icu_windows = np.array(icu_windows_save, dtype='|S20')
    hosp_windows_save = []
    for i in range(len(hosp_windows)):
        hosp_windows_save.append([str(hosp_windows[i][0]), str(hosp_windows[i][1])])
    hosp_windows = np.array(hosp_windows_save, dtype='|S20')
    n_eps = np.array(n_eps, dtype=int)
    bsln_scrs = np.array(bsln_scrs, dtype=float)
    bsln_gfrs = np.array(bsln_gfrs, dtype=float)
    bsln_types = np.array(bsln_types, dtype='|S8')
    genders = np.array(genders, dtype=bool)
    eths = np.array(eths, dtype=bool)
    ages = np.array(ages, dtype=float)
    temps = np.array(temps, dtype=float)
    maps = np.array(maps, dtype=float)
    hrs = np.array(hrs, dtype=float)
    c_scores = np.array(c_scores, dtype=float)  # Float bc possible NaNs
    e_scores = np.array(e_scores, dtype=float)  # Float bc possible NaNs
    smokers = np.array(smokers, dtype=bool)
    bmis = np.array(bmis, dtype=float)
    weights = np.array(weights, dtype=float)
    fio2s = np.array(fio2s, dtype=float)
    resps = np.array(resps, dtype=float)
    gcss = np.array(gcss, dtype=float)
    sepsiss = np.array(sepsiss, dtype=bool)
    diabetics = np.array(diabetics, dtype=bool)
    hypertensives = np.array(hypertensives, bool)
    dieds = np.array(dieds, dtype=int)
    dtds = np.array(dtds, dtype=float)
    net_fluids = np.array(net_fluids, dtype=float)
    gross_fluids = np.array(gross_fluids, dtype=float)
    fos = np.array(fos, dtype=float)
    rrt_flags = np.array(rrt_flags, dtype=bool)
    hd_dayss = np.array(hd_dayss, dtype=int)
    crrt_dayss = np.array(crrt_dayss, dtype=int)
    hd_frees_7d = np.array(hd_frees_7d, dtype=int)
    hd_frees_28d = np.array(hd_frees_28d, dtype=int)
    crrt_frees_7d = np.array(crrt_frees_7d, dtype=int)
    hd_trtmts = np.array(hd_trtmts, dtype=int)
    hd_frees_28d = np.array(hd_frees_28d, dtype=int)
    mv_flags = np.array(mv_flags, dtype=bool)
    mv_days = np.array(mv_days, dtype=float)
    mv_frees = np.array(mv_frees, dtype=float)
    ecmos = np.array(ecmos, dtype=int)
    iabps = np.array(iabps, dtype=int)
    vads = np.array(vads, dtype=int)
    nephrotox_cts = np.array(nephrotox_cts, dtype=int)
    vasopress_cts = np.array(vasopress_cts, dtype=int)
    dopas = np.array(dopas)
    epis = np.array(epis)
    anemics = np.array(anemics, dtype=bool)
    bilis = np.array(bilis, dtype=float)
    buns = np.array(buns, dtype=float)
    hemats = np.array(hemats, dtype=float)
    hemos = np.array(hemos, dtype=float)
    pltlts = np.array(pltlts, dtype=float)
    nas = np.array(nas, dtype=float)
    pks = np.array(pks, dtype=float)
    albs = np.array(albs, dtype=float)
    lacs = np.array(lacs, dtype=float)
    phs = np.array(phs, dtype=float)
    pao2s = np.array(pao2s, dtype=float)
    paco2s = np.array(paco2s, dtype=float)
    wbcs = np.array(wbcs, dtype=float)
    urine_outs = np.array(urine_outs, dtype=float)
    urine_flows = np.array(urine_flows, dtype=float)
    mks_7d = np.array(mks_7d, dtype=int)
    mks_w = np.array(mks_w, dtype=int)
    hosp_days = np.array(hosp_days, dtype=float)
    hosp_frees = np.array(hosp_frees, dtype=float)
    icu_days = np.array(icu_days, dtype=float)
    icu_frees = np.array(icu_frees, dtype=float)
    admit_scrs = np.array(admit_scrs, dtype=float)
    peak_scrs = np.array(peak_scrs, dtype=float)

    try:
        f = h5py.File(out_name, 'r+')
    except:
        f = h5py.File(out_name, 'w')

    try:
        meta = f[grp_name]
    except:
        meta = f.create_group(grp_name)

    meta.create_dataset('ids', data=ids, dtype=int)
    meta.create_dataset('age', data=ages, dtype=float)
    meta.create_dataset('gender', data=genders, dtype=int)
    meta.create_dataset('race', data=eths, dtype=int)
    meta.create_dataset('bmi', data=bmis, dtype=float)
    meta.create_dataset('weight', data=weights, dtype=float)
    meta.create_dataset('charlson', data=c_scores, dtype=int)
    meta.create_dataset('elixhauser', data=e_scores, dtype=int)
    meta.create_dataset('diabetic', data=diabetics, dtype=int)
    meta.create_dataset('hypertensive', data=hypertensives, dtype=int)
    meta.create_dataset('hosp_los', data=hosp_days, dtype=float)
    meta.create_dataset('hosp_free_days', data=hosp_frees, dtype=float)
    meta.create_dataset('icu_los', data=icu_days, dtype=float)
    meta.create_dataset('icu_free_days', data=icu_frees, dtype=float)
    meta.create_dataset('mv_flag', data=mv_flags, dtype=int)
    meta.create_dataset('mv_days', data=mv_days, dtype=float)
    meta.create_dataset('mv_free_days', data=mv_frees, dtype=float)
    meta.create_dataset('ecmo', data=ecmos, dtype=int)
    meta.create_dataset('iabp', data=iabps, dtype=int)
    meta.create_dataset('vad', data=vads, dtype=int)
    meta.create_dataset('sepsis', data=sepsiss, dtype=int)
    meta.create_dataset('died_inp', data=dieds, dtype=int)
    meta.create_dataset('baseline_scr', data=bsln_scrs, dtype=float)
    meta.create_dataset('baseline_gfr', data=bsln_gfrs, dtype=float)
    meta.create_dataset('baseline_type', data=bsln_types, dtype='|S8')
    meta.create_dataset('admit_scr', data=admit_scrs, dtype=float)
    meta.create_dataset('peak_scr', data=peak_scrs, dtype=float)
    meta.create_dataset('net_fluid', data=net_fluids, dtype=int)
    meta.create_dataset('gross_fluid', data=gross_fluids, dtype=int)
    meta.create_dataset('fluid_overload', data=fos, dtype=float)
    meta.create_dataset('hd_days', data=hd_dayss, dtype=int)
    meta.create_dataset('crrt_days', data=crrt_dayss, dtype=int)
    meta.create_dataset('hd_free_7d', data=hd_frees_7d, dtype=int)
    meta.create_dataset('crrt_free_7d', data=crrt_frees_7d, dtype=int)
    meta.create_dataset('hd_free_28d', data=hd_frees_28d, dtype=int)
    meta.create_dataset('crrt_free_28d', data=crrt_frees_28d, dtype=int)
    meta.create_dataset('hd_treatments', data=hd_trtmts, dtype=int)
    meta.create_dataset('icu_dates', data=icu_windows, dtype='|S20')
    meta.create_dataset('hosp_dates', data=hosp_windows, dtype='|S20')

    meta.create_dataset('max_kdigo_7d', data=mks_7d, dtype=int)
    meta.create_dataset('max_kdigo', data=mks_w, dtype=int)
    meta.create_dataset('n_episodes', data=nepss, dtype=int)
    meta.create_dataset('days_to_death', data=dtds, dtype=float)

    meta.create_dataset('nephrotox_ct', data=nephrotox_cts, dtype=int)
    meta.create_dataset('vasopress_ct', data=vasopress_cts, dtype=int)
    meta.create_dataset('anemia', data=anemics, dtype=bool)
    meta.create_dataset('urine_out', data=urine_outs, dtype=float)
    meta.create_dataset('urine_flow', data=urine_flows, dtype=float)
    meta.create_dataset('smoker', data=smokers, dtype=bool)

    meta.create_dataset('ph', data=phs, dtype=float)
    meta.create_dataset('map', data=maps, dtype=float)
    meta.create_dataset('albumin', data=albs, dtype=float)
    meta.create_dataset('lactate', data=lacs, dtype=float)
    meta.create_dataset('bilirubin', data=bilis, dtype=float)
    meta.create_dataset('bun', data=buns, dtype=float)

    meta.create_dataset('num_episodes', data=n_eps, dtype=int)
    meta.create_dataset('temperature', data=temps, dtype=float)
    meta.create_dataset('heart_rate', data=hrs, dtype=float)
    meta.create_dataset('fio2', data=fio2s, dtype=float)
    meta.create_dataset('respiration', data=resps, dtype=float)
    meta.create_dataset('glasgow', data=gcss, dtype=float)
    meta.create_dataset('rrt_flag', data=rrt_flags, dtype=bool)
    meta.create_dataset('dopa', data=dopas, dtype=float)
    meta.create_dataset('epinephrine', data=epis, dtype=float)
    meta.create_dataset('hematocrit', data=hemats, dtype=float)
    meta.create_dataset('hemoglobin', data=hemos, dtype=float)
    meta.create_dataset('platelets', data=pltlts, dtype=float)
    meta.create_dataset('sodium', data=nas, dtype=float)
    meta.create_dataset('potassium', data=pks, dtype=float)
    meta.create_dataset('pao2', data=pao2s, dtype=float)
    meta.create_dataset('paco2', data=paco2s, dtype=float)
    meta.create_dataset('wbc', data=wbcs, dtype=float)
    meta.create_dataset('hosp_window', data=hosp_windows_save, dtype='|S20')
    meta.create_dataset('icu_window', data=icu_windows_save, dtype='|S20')
    return f


# def summarize_stats(ids, kdigos, days, scrs,
#                     dem_m, sex_loc, eth_loc,
#                     dob_m, birth_loc,
#                     diag_m, diag_loc,
#                     charl_m, charl_loc, elix_m, elix_loc,
#                     organ_sup_mv, mech_vent_dates,
#                     organ_sup_ecmo, ecmo_dates,
#                     organ_sup_iabp, iabp_dates,
#                     organ_sup_vad, vad_dates,
#                     date_m, hosp_locs, icu_locs,
#                     sofas, apaches, io_m,
#                     mort_m, mdate_loc,
#                     clinical_vit, map_loc,
#                     clinical_oth, height_loc, weight_loc,
#                     dia_m, crrt_locs, hd_locs,
#                     med_m, med_type, med_date,
#                     labs1_m, hemat_loc, hemo_loc, bili_loc, bun_loc,
#                     labs2_m, alb_loc, lac_loc,
#                     blood_m, ph_loc,
#                     urine_m, urine_locs,
#                     smoke_m, former_smoke, current_smoke,
#                     out_name, data_path, grp_name='meta', tlim=7):
#
#     dd = np.array(load_csv(data_path + 'disch_disp.csv', ids, str, sel=1))
#
#     n_eps = []
#     for i in range(len(kdigos)):
#         n_eps.append(count_eps(kdigos[i]))
#
#     bsln_scrs = load_csv(data_path + 'baselines.csv', ids, float)
#     bsln_gfrs = load_csv(data_path + 'baseline_gfr.csv', ids, float)
#     bsln_types = load_csv(data_path + 'baseline_types.csv', ids, str)
#
#     ages = []
#     genders = []
#     mks_7d = []
#     mks = []
#     dieds = []
#     nepss = []
#     hosp_days = []
#     hosp_frees = []
#     icu_days = []
#     icu_frees = []
#     sepsiss = []
#     net_fluids = []
#     gross_fluids = []
#     fos = []
#     c_scores = []
#     e_scores = []
#     mv_flags = []
#     mv_days = []
#     mv_frees = []
#     eths = []
#     dtds = []
#
#     weights = []
#     bmis = []
#     diabetics = []
#     hypertensives = []
#     ecmos = []
#     iabps = []
#     vads = []
#     smokers = []
#
#     admit_scrs = []
#     peak_scrs = []
#     hd_dayss = []
#     crrt_dayss = []
#     hd_frees_7d = []
#     hd_frees_28d = []
#     crrt_frees_7d = []
#     crrt_frees_28d = []
#
#     nephrotox_cts = []
#     vasopress_cts = []
#     anemics = []
#     urine_outs = []
#     urine_flows = []
#
#     phs = []
#     maps = []
#     albs = []
#     lacs = []
#     bilis = []
#     buns = []
#
#     for i in range(len(ids)):
#         idx = ids[i]
#         td = days[i]
#         mask = np.where(td <= tlim)[0]
#         mk7d = np.max(kdigos[i][mask])
#         mk = np.max(kdigos[i])
#         died = 0
#         if 'EXPIRED' in dd[i]:
#             died += 1
#             if 'LESS' in dd[i]:
#                 died += 1
#
#         eps = n_eps[i]
#
#         sepsis = 0
#         diabetic = 0
#         hypertensive = 0
#         diag_ids = np.where(diag_m[:, 0] == idx)[0]
#         for j in range(len(diag_ids)):
#             tid = diag_ids[j]
#             if 'sep' in str(diag_m[tid, diag_loc]).lower():
#                 # if int(diag_m[tid, diag_nb_loc]) == 1:
#                 sepsis = 1
#             if 'diabe' in str(diag_m[tid, diag_loc]).lower():
#                 diabetic = 1
#             if 'hypert' in str(diag_m[tid, diag_loc]).lower():
#                 hypertensive = 1
#
#         admit_scr = scrs[i][0]
#         peak_scr = np.max(scrs[i])
#
#         date_idx = np.where(date_m[:, 0] == idx)[0][0]
#         admit = get_date(date_m[date_idx, icu_locs[0]])
#         hd_days = 0
#         crrt_days = 0
#         hd_free_7d = np.ones(7)
#         hd_free_28d = np.ones(28)
#         crrt_free_7d = np.ones(7)
#         crrt_free_28d = np.ones(28)
#         dia_ids = np.where(dia_m[:, 0] == idx)[0]
#         if np.size(dia_ids) > 0:
#             for row in dia_ids:
#                 # CRRT
#                 start = get_date(dia_m[row, crrt_locs[0]])
#                 stop = get_date(dia_m[row, crrt_locs[1]])
#                 if start != 'nan' and stop != 'nan':
#                     crrt_days += (stop - start).days
#                     if admit < start <= admit + datetime.timedelta(7):
#                         sidx = (start - admit).days
#                         send = min(7, sidx + (stop - start).days)
#                         crrt_free_7d[sidx:send] = 0
#                     if admit < start <= admit + datetime.timedelta(28):
#                         sidx = (start - admit).days
#                         send = min(28, sidx + (stop - start).days)
#                         crrt_free_28d[sidx:send] = 0
#                     if admit < stop <= admit + datetime.timedelta(7):
#                         sidx = (stop - admit).days
#                         send = min(7, sidx)
#                         crrt_free_7d[:send] = 0
#                     if admit < stop <= admit + datetime.timedelta(28):
#                         sidx = (start - admit).days
#                         send = min(28, sidx + (stop - start).days)
#                         crrt_free_28d[:send] = 0
#                     if start < admit and stop > admit + datetime.timedelta(7):
#                         crrt_free_7d[:] = 0
#
#                 start = get_date(dia_m[row, hd_locs[0]])
#                 stop = get_date(dia_m[row, hd_locs[1]])
#                 if start != 'nan' and stop != 'nan':
#                     hd_days += (stop - start).days
#                     if admit < start <= admit + datetime.timedelta(7):
#                         sidx = (start - admit).days
#                         send = min(7, sidx + (stop - start).days)
#                         hd_free_7d[sidx:send] = 0
#                     if admit < start <= admit + datetime.timedelta(28):
#                         sidx = (start - admit).days
#                         send = min(28, sidx + (stop - start).days)
#                         hd_free_28d[sidx:send] = 0
#                     if admit < stop <= admit + datetime.timedelta(7):
#                         sidx = (stop - admit).days
#                         send = min(7, sidx)
#                         hd_free_7d[:send] = 0
#                     if admit < stop <= admit + datetime.timedelta(28):
#                         sidx = (start - admit).days
#                         send = min(28, sidx + (stop - start).days)
#                         hd_free_28d[:send] = 0
#                     if start < admit and stop > admit + datetime.timedelta(7):
#                         hd_free_7d[:] = 0
#
#         hd_free_7d = np.sum(hd_free_7d)
#         hd_free_28d = np.sum(hd_free_28d)
#         crrt_free_7d = np.sum(crrt_free_7d)
#         crrt_free_28d = np.sum(crrt_free_28d)
#
#         male = 0
#         dem_idx = np.where(dem_m[:, 0] == idx)[0][0]
#         if str(dem_m[dem_idx, sex_loc]).upper()[0] == 'M':
#             male = 1
#
#         dob_idx = np.where(dob_m[:, 0] == idx)[0][0]
#         dob = get_date(dob_m[dob_idx, birth_loc])
#         date_idx = np.where(date_m[:, 0] == idx)[0][0]
#         admit = get_date(date_m[date_idx, hosp_locs[0]])
#         tage = admit - dob
#         age = tage.total_seconds() / (60 * 60 * 24 * 365)
#
#         race = str(dem_m[dem_idx, eth_loc]).upper()
#         if "BLACK" in race:
#             eth = 1
#         else:
#             eth = 0
#
#         bmi = np.nan
#         weight = np.nan
#         co_rows = np.where(clinical_oth[:, 0] == idx)[0]
#         if co_rows.size > 0:
#             row = co_rows[0]
#             h = clinical_oth[row, height_loc] / 100
#             w = clinical_oth[row, weight_loc]
#             if h > 0.2 and w != 0:
#                 bmi = w / (h * h)
#                 weight = w
#
#         hlos, ilos = get_los(idx, date_m, hosp_locs, icu_locs)
#         hfree = 28 - hlos
#         ifree = 28 - ilos
#         if hfree < 0:
#             hfree = 0
#         if ifree < 0:
#             ifree = 0
#
#         mech_flag = 0
#         mech_day = 0
#         mech = 28
#         mech_idx = np.where(organ_sup_mv[:, 0] == idx)[0]
#         if mech_idx.size > 0:
#             mech_idx = mech_idx[0]
#             try:
#                 start_str = organ_sup_mv[mech_idx, mech_vent_dates[0]]
#                 mech_start = get_date(start_str)
#                 stop_str = str(organ_sup_mv[mech_idx, mech_vent_dates[1]])
#                 mech_stop = get_date(stop_str)
#                 if mech_stop < admit:
#                     pass
#                 elif (mech_start - admit).days > 28:
#                     pass
#                 else:
#                     if mech_start < admit:
#                         mech_day = (mech_stop - admit).days
#                     else:
#                         mech_day = (mech_stop - mech_start).days
#
#                     mech = 28 - mech_day
#                     mech_flag = 1
#             except ValueError:
#                 pass
#         if mech < 0:
#             mech = 0
#
#         ecmo = 0
#         ecmo_idx = np.where(organ_sup_ecmo[:, 0] == idx)[0]
#         if ecmo_idx.size > 0:
#             ecmo_idx = ecmo_idx[0]
#             try:
#                 start_str = str(organ_sup_ecmo[ecmo_idx, ecmo_dates[0]])
#                 ecmo_start = get_date(start_str)
#                 stop_str = str(organ_sup_ecmo[ecmo_idx, ecmo_dates[1]])
#                 ecmo_stop = get_date(stop_str)
#                 if ecmo_stop < admit:
#                     pass
#                 elif (ecmo_start - admit).days > 28:
#                     pass
#                 else:
#                     ecmo = 1
#             except ValueError:
#                 pass
#
#         iabp = 0
#         iabp_idx = np.where(organ_sup_iabp[:, 0] == idx)[0]
#         if iabp_idx.size > 0:
#             iabp_idx = iabp_idx[0]
#             try:
#                 start_str = str(organ_sup_iabp[iabp_idx, iabp_dates[0]])
#                 iabp_start = get_date(start_str)
#                 stop_str = str(organ_sup_iabp[iabp_idx, iabp_dates[1]])
#                 iabp_stop = get_date(stop_str)
#                 if iabp_stop < admit:
#                     pass
#                 elif (iabp_start - admit).days > 28:
#                     pass
#                 else:
#                     iabp = 1
#             except ValueError:
#                 pass
#
#         vad = 0
#         vad_idx = np.where(organ_sup_vad[:, 0] == idx)[0]
#         if vad_idx.size > 0:
#             vad_idx = vad_idx[0]
#             try:
#                 start_str = str(organ_sup_vad[vad_idx, vad_dates[0]])
#                 vad_start = get_date(start_str)
#                 stop_str = str(organ_sup_vad[vad_idx, vad_dates[1]])
#                 vad_stop = get_date(stop_str)
#                 if vad_stop < admit:
#                     pass
#                 elif (vad_start - admit).days > 28:
#                     pass
#                 else:
#                     vad = 1
#             except ValueError:
#                 pass
#
#         if died:
#             if mech < 28:
#                 mech = 0
#             if hfree < 28:
#                 hfree = 0
#             if ifree < 28:
#                 ifree = 0
#
#         io_idx = np.where(io_m[:, 0] == idx)[0]
#         if io_idx.size > 0:
#             net = 0
#             tot = 0
#             for tid in io_idx:
#                 if not np.isnan(io_m[tid, 1]) and not np.isnan(io_m[tid, 2]):
#                     net += (io_m[tid, 1] - io_m[tid, 2])
#                     tot += (io_m[tid, 1] + io_m[tid, 2])
#
#                 if not np.isnan(io_m[tid, 3]) and not np.isnan(io_m[tid, 4]):
#                     net += (io_m[tid, 3] - io_m[tid, 4])
#                     tot += (io_m[tid, 3] + io_m[tid, 4])
#
#                 if not np.isnan(io_m[tid, 5]) and not np.isnan(io_m[tid, 6]):
#                     net += (io_m[tid, 5] - io_m[tid, 6])
#                     tot += (io_m[tid, 5] + io_m[tid, 6])
#         else:
#             net = np.nan
#             tot = np.nan
#
#         if tot < 0:
#             net = np.nan
#             tot = np.nan
#
#         if net != np.nan and weight_loc != np.nan:
#             fo = net / weight_loc
#         else:
#             fo = np.nan
#
#         charl_idx = np.where(charl_m[:, 0] == idx)[0]
#         charl = np.nan
#         if charl_idx.size > 0 and charl_m[charl_idx, charl_loc] > 0:
#             charl = charl_m[charl_idx, charl_loc]
#
#         elix_idx = np.where(elix_m[:, 0] == idx)[0]
#         elix = np.nan
#         if elix_idx.size > 0 and elix_m[elix_idx, elix_loc] > 0:
#             elix = elix_m[elix_idx, elix_loc]
#
#         mort_idx = np.where(mort_m[:, 0] == idx)[0]
#         dtd = np.nan
#         if mort_idx.size > 0:
#             tstr = str(mort_m[mort_idx[0], mdate_loc]).split('.')[0]
#             mdate = get_date(tstr)
#             if mdate != 'nan':
#                 dtd = (mdate - admit).total_seconds() / (60 * 60 * 24)
#
#         med_idx = np.where(med_m[:, 0] == idx)[0]
#         neph_ct = 0
#         vaso_ct = 0
#         if med_idx.size > 0:
#             for tid in med_idx:
#                 tstr = str(med_m[tid, med_date]).split('.')[0]
#                 tdate = get_date(tstr)
#                 if tdate != 'nan':
#                     if admit.day <= tdate.day <= admit.day + 1:
#                         # Nephrotoxins
#                         if np.any([med_m[tid, med_type].lower() in x for x in ['acei', 'ace inhibitors',
#                                                                                'angiotensin receptor blockers', 'arb',
#                                                                                'aminoglycosides', 'nsaids']]):
#                             neph_ct += 1
#
#                         # Vasoactive Drugs
#                         if np.any([med_m[tid, med_type].lower() in x for x in ['pressor', 'inotrope',
#                                                                                'pressor or inotrope']]):
#                             vaso_ct += 1
#
#         # Anemia, bilirubin, and BUN (labs set 1)
#         anemic = np.zeros(3)
#         bili = np.nan
#         bun = np.nan
#         lab_idx = np.where(labs1_m[:, 0] == idx)[0]
#         if lab_idx.size > 0:
#             lab_idx = lab_idx[0]
#             bili = labs1_m[lab_idx, bili_loc]
#             bun = labs1_m[lab_idx, bun_loc]
#             hemat = labs1_m[lab_idx, hemat_loc[0]]
#             hemo = labs1_m[lab_idx, hemo_loc[0]]
#             # definition A
#             if male:
#                 if hemat < 39 or hemo < 18:
#                     anemic[0] = 1
#             else:
#                 if hemat < 36 or hemo < 12:
#                     anemic[0] = 1
#             # definition B
#             if hemat < 30 or hemo < 10:
#                 anemic[1] = 1
#             # definition C
#             if hemat < 27 or hemo < 9:
#                 anemic[2] = 1
#
#         # Urine output
#         urine_idx = np.where(urine_m[:, 0] == idx)[0]
#         urine_out = np.zeros(int(len(urine_locs) / 2))
#         urine_flow = np.zeros(int(len(urine_locs) / 2))
#         if urine_idx.size > 0:
#             urine_idx = urine_idx[0]
#             d = []
#             for j in range(len(urine_out)):
#                 tidx = 2*j
#                 urine_out[j] = np.nansum((urine_m[urine_idx, urine_locs[tidx]], urine_m[urine_idx, urine_locs[tidx + 1]]))
#             urine_flow = urine_out / weight / 24
#
#         # Smoking status
#         smoke_idx = np.where(smoke_m[:,0] == idx)[0]
#         smoker = 0
#         if smoke_idx.size > 0:
#             smoke_idx = smoke_idx[0]
#             if smoke_m[smoke_idx, former_smoke] == 'FORMER SMOKER':
#                 smoker = 1
#             if smoke_m[smoke_idx, current_smoke] == 'YES':
#                 smoker = 1
#
#         # Blood gas
#         blood_idx = np.where(blood_m[:, 0] == idx)[0]
#         ph = np.nan
#         if blood_idx.size > 0:
#             blood_idx = blood_idx[0]
#             ph = blood_m[blood_idx, ph_loc[0]]
#
#         # Labs set 2
#         lab_idx = np.where(labs2_m[:, 0] == idx)[0]
#         alb = np.nan
#         lac = np.nan
#         if lab_idx.size > 0:
#             lab_idx = lab_idx[0]
#             alb = labs2_m[lab_idx, alb_loc]
#             lac = labs2_m[lab_idx, lac_loc]
#
#         # Clinical vitals
#         vit_idx = np.where(clinical_vit[:, 0] == idx)[0]
#         tmap = np.nan
#         if vit_idx.size > 0:
#             vit_idx = vit_idx[0]
#             tmap = clinical_vit[vit_idx, map_loc[0]]
#
#         ages.append(age)
#         genders.append(male)
#         mks_7d.append(mk7d)
#         mks.append(mk)
#         dieds.append(died)
#         nepss.append(eps)
#         hosp_days.append(hlos)
#         hosp_frees.append(hfree)
#         icu_days.append(ilos)
#         icu_frees.append(ifree)
#         sepsiss.append(sepsis)
#         diabetics.append(diabetic)
#         hypertensives.append(hypertensive)
#         admit_scrs.append(admit_scr)
#         peak_scrs.append(peak_scr)
#
#         bmis.append(bmi)
#         weights.append(weight)
#         net_fluids.append(net)
#         gross_fluids.append(tot)
#         fos.append(fo)
#         c_scores.append(charl)
#         e_scores.append(elix)
#         mv_flags.append(mech_flag)
#         mv_days.append(mech_day)
#         mv_frees.append(mech)
#         ecmos.append(ecmo)
#         iabps.append(iabp)
#         vads.append(vad)
#         eths.append(eth)
#         dtds.append(dtd)
#         hd_dayss.append(hd_days)
#         crrt_dayss.append(crrt_days)
#         hd_frees_7d.append(hd_free_7d)
#         hd_frees_28d.append(hd_free_28d)
#         crrt_frees_7d.append(crrt_free_7d)
#         crrt_frees_28d.append(crrt_free_28d)
#
#         nephrotox_cts.append(neph_ct)
#         vasopress_cts.append(vaso_ct)
#         anemics.append(anemic)
#         urine_outs.append(urine_out)
#         urine_flows.append(urine_flow)
#         smokers.append(smoker)
#
#         phs.append(ph)
#         maps.append(tmap)
#         albs.append(alb)
#         lacs.append(lac)
#         bilis.append(bili)
#         buns.append(bun)
#
#     ages = np.array(ages, dtype=float)
#     genders = np.array(genders, dtype=bool)
#     eths = np.array(eths, dtype=bool)
#     bmis = np.array(bmis, dtype=float)
#     weights = np.array(weights, dtype=float)
#     c_scores = np.array(c_scores, dtype=float)  # Float bc possible NaNs
#     e_scores = np.array(e_scores, dtype=float)  # Float bc possible NaNs
#     diabetics = np.array(diabetics, dtype=bool)
#     hypertensives = np.array(hypertensives, bool)
#     if sofas.ndim == 2:
#         sofas = np.sum(sofas, axis=1)
#     if apaches.ndim == 2:
#         apaches = np.sum(apaches, axis=1)
#     hosp_days = np.array(hosp_days, dtype=float)
#     hosp_frees = np.array(hosp_frees, dtype=float)
#     icu_days = np.array(icu_days, dtype=float)
#     icu_frees = np.array(icu_frees, dtype=float)
#     mv_flags = np.array(mv_flags, dtype=bool)
#     mv_days = np.array(mv_days, dtype=float)
#     mv_frees = np.array(mv_frees, dtype=float)
#     ecmos = np.array(ecmos, dtype=int)
#     iabps = np.array(iabps, dtype=int)
#     vads = np.array(vads, dtype=int)
#     sepsiss = np.array(sepsiss, dtype=bool)
#     dieds = np.array(dieds, dtype=int)
#
#     bsln_scrs = np.array(bsln_scrs, dtype=float)
#     bsln_gfrs = np.array(bsln_gfrs, dtype=float)
#     admit_scrs = np.array(admit_scrs, dtype=float)
#     peak_scrs = np.array(peak_scrs, dtype=float)
#     mks_7d = np.array(mks_7d, dtype=int)
#     mks = np.array(mks, dtype=int)
#     net_fluids = np.array(net_fluids, dtype=float)
#     gross_fluids = np.array(gross_fluids, dtype=float)
#     fos = np.array(fos, dtype=float)
#     nepss = np.array(nepss, dtype=int)
#     hd_dayss = np.array(hd_dayss, dtype=int)
#     crrt_dayss = np.array(crrt_dayss, dtype=int)
#     hd_frees_7d = np.array(hd_frees_7d, dtype=int)
#     hd_frees_28d = np.array(hd_frees_28d, dtype=int)
#     crrt_frees_7d = np.array(crrt_frees_7d, dtype=int)
#     crrt_frees_28d = np.array(crrt_frees_28d, dtype=int)
#
#     dtds = np.array(dtds, dtype=float)
#
#     nephrotox_cts = np.array(nephrotox_cts, dtype=int)
#     vasopress_cts = np.array(vasopress_cts, dtype=int)
#     anemics = np.array(anemics, dtype=bool)
#     urine_outs = np.array(urine_outs, dtype=float)
#     urine_flows = np.array(urine_flows, dtype=float)
#     smokers = np.array(smokers, dtype=bool)
#
#     phs = np.array(phs, dtype=float)
#     maps = np.array(maps, dtype=float)
#     albs = np.array(albs, dtype=float)
#     lacs = np.array(lacs, dtype=float)
#     bilis = np.array(bilis, dtype=float)
#     buns = np.array(buns, dtype=float)
#
#     try:
#         f = h5py.File(out_name, 'r+')
#     except:
#         f = h5py.File(out_name, 'w')
#
#     try:
#         meta = f[grp_name]
#     except:
#         meta = f.create_group(grp_name)
#
#     meta.create_dataset('ids', data=ids, dtype=int)
#     meta.create_dataset('age', data=ages, dtype=float)
#     meta.create_dataset('gender', data=genders, dtype=int)
#     meta.create_dataset('race', data=eths, dtype=int)
#     meta.create_dataset('bmi', data=bmis, dtype=float)
#     meta.create_dataset('weight', data=weights, dtype=float)
#     meta.create_dataset('charlson', data=c_scores, dtype=int)
#     meta.create_dataset('elixhauser', data=e_scores, dtype=int)
#     meta.create_dataset('diabetic', data=diabetics, dtype=int)
#     meta.create_dataset('hypertensive', data=hypertensives, dtype=int)
#     meta.create_dataset('sofa', data=sofas, dtype=int)
#     meta.create_dataset('apache', data=apaches, dtype=int)
#     meta.create_dataset('hosp_los', data=hosp_days, dtype=float)
#     meta.create_dataset('hosp_free_days', data=hosp_frees, dtype=float)
#     meta.create_dataset('icu_los', data=icu_days, dtype=float)
#     meta.create_dataset('icu_free_days', data=icu_frees, dtype=float)
#     meta.create_dataset('mv_flag', data=mv_flags, dtype=int)
#     meta.create_dataset('mv_days', data=mv_days, dtype=float)
#     meta.create_dataset('mv_free_days', data=mv_frees, dtype=float)
#     meta.create_dataset('ecmo', data=ecmos, dtype=int)
#     meta.create_dataset('iabp', data=iabps, dtype=int)
#     meta.create_dataset('vad', data=vads, dtype=int)
#     meta.create_dataset('sepsis', data=sepsiss, dtype=int)
#     meta.create_dataset('died_inp', data=dieds, dtype=int)
#     meta.create_dataset('baseline_scr', data=bsln_scrs, dtype=float)
#     meta.create_dataset('baseline_gfr', data=bsln_gfrs, dtype=float)
#     meta.create_dataset('baseline_type', data=bsln_types, dtype='|S8')
#     meta.create_dataset('admit_scr', data=admit_scrs, dtype=float)
#     meta.create_dataset('peak_scr', data=peak_scrs, dtype=float)
#     meta.create_dataset('net_fluid', data=net_fluids, dtype=int)
#     meta.create_dataset('gross_fluid', data=gross_fluids, dtype=int)
#     meta.create_dataset('fluid_overload', data=fos, dtype=float)
#     meta.create_dataset('hd_days', data=hd_dayss, dtype=int)
#     meta.create_dataset('crrt_days', data=crrt_dayss, dtype=int)
#     meta.create_dataset('hd_free_7d', data=hd_frees_7d, dtype=int)
#     meta.create_dataset('crrt_free_7d', data=crrt_frees_7d, dtype=int)
#     meta.create_dataset('hd_free_28d', data=hd_frees_28d, dtype=int)
#     meta.create_dataset('crrt_free_28d', data=crrt_frees_28d, dtype=int)
#
#     meta.create_dataset('max_kdigo_7d', data=mks_7d, dtype=int)
#     meta.create_dataset('max_kdigo', data=mks, dtype=int)
#     meta.create_dataset('n_episodes', data=nepss, dtype=int)
#     meta.create_dataset('days_to_death', data=dtds, dtype=float)
#
#     meta.create_dataset('nephrotox_ct', data=nephrotox_cts, dtype=int)
#     meta.create_dataset('vasopress_ct', data=vasopress_cts, dtype=int)
#     meta.create_dataset('anemia', data=anemics, dtype=bool)
#     meta.create_dataset('urine_out', data=urine_outs, dtype=float)
#     meta.create_dataset('urine_flow', data=urine_outs, dtype=float)
#     meta.create_dataset('smoker', data=smokers, dtype=bool)
#
#     meta.create_dataset('ph', data=phs, dtype=float)
#     meta.create_dataset('map', data=maps, dtype=float)
#     meta.create_dataset('albumin', data=albs, dtype=float)
#     meta.create_dataset('lactate', data=lacs, dtype=float)
#     meta.create_dataset('bilirubin', data=bilis, dtype=float)
#     meta.create_dataset('bun', data=buns, dtype=float)
#
#     return f


def get_uky_vitals(ids, clinical_vit, map_locs, cuff_locs, temp_locs, hr_locs):
    temps = np.zeros((len(ids), 2))
    maps = np.zeros((len(ids), 2))
    hrs = np.zeros((len(ids), 2))
    for i in range(len(ids)):
        tid = ids[i]
        # Clinical vitals
        vit_idx = np.where(clinical_vit[:, 0] == tid)[0]
        tmap = np.array((np.nan, np.nan))
        temp = np.array((np.nan, np.nan))
        hr = np.array((np.nan, np.nan))
        if vit_idx.size > 0:
            vit_idx = vit_idx[0]
            tmap = clinical_vit[vit_idx, map_locs]
            if np.isnan(tmap[0]):
                tmap = clinical_vit[vit_idx, cuff_locs]
            temp = clinical_vit[vit_idx, temp_locs]
            hr = clinical_vit[vit_idx, hr_locs]
        temps[i] = temp
        maps[i] = tmap
        hrs[i] = hr
    return temps, maps, hrs


def get_uky_urine(ids, weights, urine_m, urine_locs):
    urine_outs = np.zeros((len(ids), 3))
    urine_flows = np.zeros((len(ids), 3))
    for i in range(len(ids)):
        tid = ids[i]
        weight = weights[i]
        urine_idx = np.where(urine_m[:, 0] == tid)[0]
        urine_out = np.zeros(3)
        urine_flow = np.zeros(3)
        if urine_idx.size > 0:
            urine_idx = urine_idx[0]
            for j in range(3):
                tidx = 2 * j
                urine_out[j] = np.nansum(
                    (urine_m[urine_idx, urine_locs[tidx]], urine_m[urine_idx, urine_locs[tidx + 1]]))
            urine_flow = urine_out / weight / 24
        urine_outs[i] = urine_out
        urine_flows[i] = urine_flow
    return urine_outs, urine_flows


def get_uky_medications(ids, admits, med_m, med_type, med_date, med_name, med_dur):
    neph_cts = np.zeros(len(ids))
    vaso_cts = np.zeros(len(ids))
    dopas = np.zeros(len(ids))
    epis = np.zeros(len(ids))
    for i in range(len(ids)):
        tid = ids[i]
        admit = admits[i]
        if type(admit) == str:
            admit = get_date(admit)
        med_idx = np.where(med_m[:, 0] == tid)[0]
        neph_ct = 0
        vaso_ct = 0
        dopa = 0
        epi = 0
        if med_idx.size > 0:
            for tid in med_idx:
                tstr = str(med_m[tid, med_date]).split('.')[0]
                tdate = get_date(tstr)
                tname = med_m[tid, med_name].lower()
                start = tdate
                dur = float(med_m[tid, med_dur])
                if not np.isnan(dur):
                    stop = start + datetime.timedelta(dur)
                    if tdate != 'nan':
                        if start <= admit:
                            if admit <= stop:
                                if 'dopamine' in tname or 'dobutamine' in tname:
                                    dopa = 1
                                elif 'epinephrine' in tname:
                                    epi = 1
                        if admit.day <= tdate.day <= admit.day + 1:
                            # Nephrotoxins
                            if np.any([med_m[tid, med_type].lower() in x for x in ['acei', 'ace inhibitors',
                                                                                   'angiotensin receptor blockers', 'arb',
                                                                                   'aminoglycosides', 'nsaids']]):
                                neph_ct += 1

                            # Vasoactive Drugs
                            if np.any([med_m[tid, med_type].lower() in x for x in ['pressor', 'inotrope',
                                                                                   'pressor or inotrope']]):
                                vaso_ct += 1
        neph_cts[i] = neph_ct
        vaso_cts[i] = vaso_ct
        dopas[i] = dopa
        epis[i] = epi
    return neph_cts, vaso_cts, dopas, epis


def get_uky_mortality(ids, disch_disps, hosp_windows, icu_windows, mort_m, mdate_loc):
    dieds = np.zeros(len(ids))
    dtds = np.zeros(len(ids))
    dods = np.zeros(len(ids), dtype='|S20')
    dods[:] = 'nan'
    for i in range(len(ids)):
        tid = ids[i]
        died = 0
        if 'EXPIRED' in disch_disps[i]:
            died += 1

        mort_idx = np.where(mort_m[:, 0] == tid)[0]
        dtd = np.nan
        admit = icu_windows[i][0]
        disch = hosp_windows[i][1]
        if type(admit) == str:
            admit = get_date(admit)
            disch = get_date(disch)
        if mort_idx.size > 0:
            tstr = str(mort_m[mort_idx[0], mdate_loc]).split('.')[0]
            mdate = get_date(tstr)
            if mdate != 'nan':
                dods[i] = tstr
                dtd = (mdate - admit).total_seconds() / (60 * 60 * 24)
                if disch == 'nan' or admit < mdate < disch:
                    died = 1
        dieds[i] = died
        dtds[i] = dtd
    return dieds, dtds, dods


def get_uky_comorbidities(ids, charl_m, charl_loc, elix_m, elix_loc, smoke_m, former_smoke, current_smoke,):
    charls = np.zeros(len(ids))
    elixs = np.zeros(len(ids))
    smokers = np.zeros(len(ids))
    for i in range(len(ids)):
        tid = ids[i]
        charl_idx = np.where(charl_m[:, 0] == tid)[0]
        charl = np.nan
        if charl_idx.size > 0 and charl_m[charl_idx, charl_loc] > 0:
            charl = charl_m[charl_idx, charl_loc]

        elix_idx = np.where(elix_m[:, 0] == tid)[0]
        elix = np.nan
        if elix_idx.size > 0 and elix_m[elix_idx, elix_loc] > 0:
            elix = elix_m[elix_idx, elix_loc]
        charls[i] = charl
        elixs[i] = elix

        smoke_idx = np.where(smoke_m[:, 0] == tid)[0]
        smoker = 0
        if smoke_idx.size > 0:
            smoke_idx = smoke_idx[0]
            if smoke_m[smoke_idx, former_smoke] == 'FORMER SMOKER':
                smoker = 1
            if smoke_m[smoke_idx, current_smoke] == 'YES':
                smoker = 1
        smokers[i] = smoker
    return charls, elixs, smokers


def get_uky_fluids(ids, weights, io_m):
    nets = np.zeros(len(ids))
    tots = np.zeros(len(ids))
    fos = np.zeros(len(ids))
    for i in range(len(ids)):
        tid = ids[i]
        weight = weights[i]
        io_idx = np.where(io_m[:, 0] == tid)[0]
        if io_idx.size > 0:
            net = 0
            tot = 0
            for tid in io_idx:
                if not np.isnan(io_m[tid, 1]) and not np.isnan(io_m[tid, 2]):
                    net += (io_m[tid, 1] - io_m[tid, 2])
                    tot += (io_m[tid, 1] + io_m[tid, 2])

                if not np.isnan(io_m[tid, 3]) and not np.isnan(io_m[tid, 4]):
                    net += (io_m[tid, 3] - io_m[tid, 4])
                    tot += (io_m[tid, 3] + io_m[tid, 4])

                if not np.isnan(io_m[tid, 5]) and not np.isnan(io_m[tid, 6]):
                    net += (io_m[tid, 5] - io_m[tid, 6])
                    tot += (io_m[tid, 5] + io_m[tid, 6])
        else:
            net = np.nan
            tot = np.nan

        if tot < 0:
            net = np.nan
            tot = np.nan

        if net != np.nan and weight != np.nan:
            fo = net / weight
        else:
            fo = np.nan

        nets[i] = net
        tots[i] = tot
        fos[i] = fo
    return nets, tots, fos


def get_uky_organsupp(ids, dieds, admits, organ_sup_mv, mech_vent_dates, organ_sup_ecmo, ecmo_dates,
                      organ_sup_vad, vad_dates, organ_sup_iabp, iabp_dates):
    mech_flags = np.zeros(len(ids))
    mech_days = np.zeros(len(ids))
    mech_frees = np.zeros(len(ids))
    ecmos = np.zeros(len(ids))
    vads = np.zeros(len(ids))
    iabps = np.zeros(len(ids))
    for i in range(len(ids)):
        tid = ids[i]
        died = dieds[i]
        admit = admits[i]
        if type(admit) == 'str':
            admit = get_date(admit)
        mech_flag = 0
        mech_day = 0
        mech = 28
        mech_idx = np.where(organ_sup_mv[:, 0] == tid)[0]
        if mech_idx.size > 0:
            mech_idx = mech_idx[0]
            try:
                start_str = organ_sup_mv[mech_idx, mech_vent_dates[0]]
                mech_start = get_date(start_str)
                stop_str = str(organ_sup_mv[mech_idx, mech_vent_dates[1]])
                mech_stop = get_date(stop_str)
                if mech_stop < admit:
                    pass
                elif (mech_start - admit).days > 28:
                    pass
                else:
                    if mech_start < admit:
                        mech_day = (mech_stop - admit).days
                    else:
                        mech_day = (mech_stop - mech_start).days

                    mech = 28 - mech_day
                    mech_flag = 1
            except ValueError:
                pass
        if mech < 0:
            mech = 0

        ecmo = 0
        ecmo_idx = np.where(organ_sup_ecmo[:, 0] == tid)[0]
        if ecmo_idx.size > 0:
            ecmo_idx = ecmo_idx[0]
            try:
                start_str = str(organ_sup_ecmo[ecmo_idx, ecmo_dates[0]])
                ecmo_start = get_date(start_str)
                stop_str = str(organ_sup_ecmo[ecmo_idx, ecmo_dates[1]])
                ecmo_stop = get_date(stop_str)
                if ecmo_stop < admit:
                    pass
                elif (ecmo_start - admit).days > 28:
                    pass
                else:
                    ecmo = 1
            except ValueError:
                pass

        iabp = 0
        iabp_idx = np.where(organ_sup_iabp[:, 0] == tid)[0]
        if iabp_idx.size > 0:
            iabp_idx = iabp_idx[0]
            try:
                start_str = str(organ_sup_iabp[iabp_idx, iabp_dates[0]])
                iabp_start = get_date(start_str)
                stop_str = str(organ_sup_iabp[iabp_idx, iabp_dates[1]])
                iabp_stop = get_date(stop_str)
                if iabp_stop < admit:
                    pass
                elif (iabp_start - admit).days > 28:
                    pass
                else:
                    iabp = 1
            except ValueError:
                pass

        vad = 0
        vad_idx = np.where(organ_sup_vad[:, 0] == tid)[0]
        if vad_idx.size > 0:
            vad_idx = vad_idx[0]
            try:
                start_str = str(organ_sup_vad[vad_idx, vad_dates[0]])
                vad_start = get_date(start_str)
                stop_str = str(organ_sup_vad[vad_idx, vad_dates[1]])
                vad_stop = get_date(stop_str)
                if vad_stop < admit:
                    pass
                elif (vad_start - admit).days > 28:
                    pass
                else:
                    vad = 1
            except ValueError:
                pass

        if mech_day > 0:
            if died or mech < 28:
                mech = 0

        mech_flags[i] = mech_flag
        mech_days[i] = mech_day
        mech_frees[i] = mech
        ecmos[i] = ecmo
        vads[i] = vad
        iabps[i] = iabp
    return mech_flags, mech_days, mech_frees, ecmos, vads, iabps


def get_uky_clinical_others(ids, clinical_oth, height_loc, weight_loc, fio2_locs, resp_locs, gcs_loc):
    bmis = np.zeros(len(ids))
    weights = np.zeros(len(ids))
    heights = np.zeros(len(ids))
    fio2s = np.zeros((len(ids), 2))
    gcss = np.zeros((len(ids), 2))
    resps = np.zeros((len(ids), 2))
    for i in range(len(ids)):
        tid = ids[i]

        bmi = np.nan
        weight = np.nan
        fio2 = np.array((np.nan, np.nan))
        gcs = np.array((np.nan, np.nan))
        resp = np.array((np.nan, np.nan))
        co_rows = np.where(clinical_oth[:, 0] == tid)[0]
        if co_rows.size > 0:
            row = co_rows[0]
            height = clinical_oth[row, height_loc] / 100
            weight = clinical_oth[row, weight_loc]
            if height > 0.2 and weight != 0:
                bmi = weight / (height * height)
                weight = weight
            fio2 = clinical_oth[row, fio2_locs]
            resp = clinical_oth[row, resp_locs]
            try:
                gcs_str = np.array(str(clinical_oth[row, gcs_loc]).split('-'))
                if len(gcs_str) == 2:
                    gcs[0] = float(gcs_str[0])
                    gcs[1] = float(gcs_str[1])
            except ValueError:
                pass
        bmis[i] = bmi
        weights[i] = weight
        heights[i] = height
        fio2s[i] = fio2
        resps[i] = resp
        gcss[i] = gcs
    return bmis, weights, heights, fio2s, resps, gcss


def get_uky_demographics(ids, admits, dem_m, sex_loc, eth_loc,
                         dob_m, birth_loc):
    males = np.zeros(len(ids))
    blacks = np.zeros(len(ids))
    ages = np.zeros(len(ids))
    for i in range(len(ids)):
        tid = ids[i]
        male = 'nan'
        race = 'nan'
        dem_idx = np.where(dem_m[:, 0] == tid)[0]
        if dem_idx.size > 0:
            male = 0
            if str(dem_m[dem_idx, sex_loc]).upper()[0] == 'M':
                male = 1
            if type(admits) != dict:
                admit = admits[i]
            else:
                admit = admits[tid][0]
            if type(admit) == str:
                admit = get_date(admit)
            race = str(dem_m[dem_idx, eth_loc]).upper()
            if "BLACK" in race:
                black = 1
            else:
                black = 0
        age = 'nan'
        dob_idx = np.where(dob_m[:, 0] == tid)[0]
        if dob_idx.size > 0:
            dob = get_date(dob_m[dob_idx, birth_loc])
            tage = admit - dob
            age = tage.total_seconds() / (60 * 60 * 24 * 365)

        males[i] = male
        blacks[i] = black
        ages[i] = age
    return males, blacks, ages


def get_uky_rrt(ids, dieds, admits, dia_m, crrt_locs, hd_locs, hd_trt_loc):
    hd_dayss = np.zeros(len(ids))
    crrt_dayss = np.zeros(len(ids))
    hd_frees_7d = np.zeros(len(ids))
    hd_frees_28d = np.zeros(len(ids))
    crrt_frees_7d = np.zeros(len(ids))
    crrt_frees_28d = np.zeros(len(ids))
    rrt_flags = np.zeros(len(ids))
    hd_trtmts = np.zeros(len(ids))
    for i in range(len(ids)):
        tid = ids[i]
        admit = admits[i]
        died = dieds[i]
        if type(admit) == str:
            admit = get_date(admit)
        hd_days = datetime.timedelta(0)
        crrt_days = datetime.timedelta(0)
        rrt_flag = 0
        hd_trtmt = 0
        hd_free_7d = np.ones(7)
        hd_free_28d = np.ones(28)
        crrt_free_7d = np.ones(7)
        crrt_free_28d = np.ones(28)
        dia_ids = np.where(dia_m[:, 0] == tid)[0]
        if np.size(dia_ids) > 0:
            for row in dia_ids:
                # CRRT
                start = get_date(dia_m[row, crrt_locs[0]])
                stop = get_date(dia_m[row, crrt_locs[1]])
                if start != 'nan' and stop != 'nan':
                    crrt_days += (stop - start)
                    if admit < start <= admit + datetime.timedelta(7):
                        sidx = (start - admit).days
                        send = min(7, sidx + (stop - start).days)
                        crrt_free_7d[sidx:send] = 0
                    if admit < start <= admit + datetime.timedelta(28):
                        sidx = (start - admit).days
                        send = min(28, sidx + (stop - start).days)
                        crrt_free_28d[sidx:send] = 0
                    if admit < stop <= admit + datetime.timedelta(7):
                        sidx = (stop - admit).days
                        send = min(7, sidx)
                        crrt_free_7d[:send] = 0
                    if admit < stop <= admit + datetime.timedelta(28):
                        sidx = (start - admit).days
                        send = min(28, sidx + (stop - start).days)
                        crrt_free_28d[:send] = 0
                    if start < admit and stop > admit + datetime.timedelta(7):
                        crrt_free_7d[:] = 0
    
                start = get_date(dia_m[row, hd_locs[0]])
                stop = get_date(dia_m[row, hd_locs[1]])
                if start != 'nan' and stop != 'nan':
                    if stop > admit:
                        hd_days += (stop - start)
                        hd_trtmt += dia_m[row, hd_trt_loc]
                        if admit < start <= admit + datetime.timedelta(7):
                            sidx = (start - admit).days
                            send = min(7, sidx + (stop - start).days)
                            hd_free_7d[sidx:send] = 0
                        if admit < start <= admit + datetime.timedelta(28):
                            sidx = (start - admit).days
                            send = min(28, sidx + (stop - start).days)
                            hd_free_28d[sidx:send] = 0
                        if admit < stop <= admit + datetime.timedelta(7):
                            sidx = (stop - admit).days
                            send = min(7, sidx)
                            hd_free_7d[:send] = 0
                        if admit < stop <= admit + datetime.timedelta(28):
                            sidx = (start - admit).days
                            send = min(28, sidx + (stop - start).days)
                            hd_free_28d[:send] = 0
                        if start < admit and stop > admit + datetime.timedelta(7):
                            hd_free_7d[:] = 0
        # crrt_days = crrt_days.total_seconds() / (60 * 60 * 24)
        # hd_days = hd_days.total_seconds() / (60 * 60 * 24)
    
        hd_free_7d = np.sum(hd_free_7d)
        hd_free_28d = np.sum(hd_free_28d)
        crrt_free_7d = np.sum(crrt_free_7d)
        crrt_free_28d = np.sum(crrt_free_28d)

        crrt_days = 28 - crrt_free_28d
        hd_days = 28 - hd_free_28d
        
        if hd_days or crrt_days:
            rrt_flag = 1
            if died:
                hd_free_7d = 0
                hd_free_28d = 0
                crrt_free_7d = 0
                crrt_free_28d = 0
        
        hd_dayss[i] = hd_days
        crrt_dayss[i] = crrt_days
        hd_frees_7d[i] = hd_free_7d
        hd_frees_28d[i] = hd_free_28d
        crrt_frees_7d[i] = crrt_free_7d
        crrt_frees_28d[i] = crrt_free_28d
        rrt_flags[i] = rrt_flag
        hd_trtmts[i] = hd_trtmt
    return rrt_flags, hd_dayss, crrt_dayss, hd_frees_7d, hd_frees_28d, hd_trtmts, crrt_frees_7d, crrt_frees_28d


def get_uky_diagnoses(ids, diag_m, diag_loc):
    sepsiss = np.zeros(len(ids))
    diabetics = np.zeros(len(ids))
    hypertensives = np.zeros(len(ids))
    for i in range(len(ids)):
        tid = ids[i]
        sepsis = 0
        diabetic = 0
        hypertensive = 0
        diag_ids = np.where(diag_m[:, 0] == tid)[0]
        for j in range(len(diag_ids)):
            tid = diag_ids[j]
            if 'sep' in str(diag_m[tid, diag_loc]).lower():
                # if int(diag_m[tid, diag_nb_loc]) == 1:
                sepsis = 1
            if 'diabe' in str(diag_m[tid, diag_loc]).lower():
                diabetic = 1
            if 'hypert' in str(diag_m[tid, diag_loc]).lower():
                hypertensive = 1
        sepsiss[i] = sepsis
        diabetics[i] = diabetic
        hypertensives[i] = hypertensive
    return sepsiss, diabetics, hypertensives


def get_uky_labs(ids, males, labs1_m, bili_loc, bun_loc, hemat_locs, hemo_locs, pltlt_loc, na_locs, pk_locs,
                 labs2_m, alb_loc, lac_loc, blood_m, ph_locs, pao2_locs, paco2_locs, wbc_locs):
    anemics = np.zeros((len(ids), 3))
    bilis = np.zeros(len(ids))
    buns = np.zeros(len(ids))
    hemats = np.zeros((len(ids), 2))
    hemos = np.zeros((len(ids), 2))
    pltlts = np.zeros(len(ids))
    nas = np.zeros((len(ids), 2))
    pks = np.zeros((len(ids), 2))
    albs = np.zeros(len(ids))
    lacs = np.zeros(len(ids))
    phs = np.zeros((len(ids), 2))
    pao2s = np.zeros((len(ids), 2))
    paco2s = np.zeros((len(ids), 2))
    wbcs = np.zeros((len(ids), 2))
    for i in range(len(ids)):
        tid = ids[i]
        male = males[i]
        # Anemia, bilirubin, and BUN (labs set 1)
        anemic = np.zeros(3)
        bili = np.nan
        bun = np.nan
        hemat = np.array((np.nan, np.nan))
        hemo = np.array((np.nan, np.nan))
        pltlt = np.nan
        na = np.array((np.nan, np.nan))
        pk = np.array((np.nan, np.nan))
        wbc = np.array((np.nan, np.nan))
        lab_idx = np.where(labs1_m[:, 0] == tid)[0]
        if lab_idx.size > 0:
            lab_idx = lab_idx[0]
            bili = labs1_m[lab_idx, bili_loc]
            bun = labs1_m[lab_idx, bun_loc]
            hemat = labs1_m[lab_idx, hemat_locs]
            hemo = labs1_m[lab_idx, hemo_locs]
            pltlt = labs1_m[lab_idx, pltlt_loc]
            na = labs1_m[lab_idx, na_locs]
            pk = labs1_m[lab_idx, pk_locs]
            wbc = labs1_m[lab_idx, wbc_locs]
            # definition A
            if male:
                if hemat[0] < 39 or hemo[0] < 18:
                    anemic[0] = 1
            else:
                if hemat[0] < 36 or hemo[0] < 12:
                    anemic[0] = 1
            # definition B
            if hemat[0] < 30 or hemo[0] < 10:
                anemic[1] = 1
            # definition C
            if hemat[0] < 27 or hemo[0] < 9:
                anemic[2] = 1

        # Labs set 2
        lab_idx = np.where(labs2_m[:, 0] == tid)[0]
        alb = np.nan
        lac = np.nan
        if lab_idx.size > 0:
            lab_idx = lab_idx[0]
            alb = labs2_m[lab_idx, alb_loc]
            lac = labs2_m[lab_idx, lac_loc]

        anemics[i] = anemic
        bilis[i] = bili
        buns[i] = bun
        hemats[i] = hemat
        hemos[i] = hemo
        pltlts[i] = pltlt
        nas[i] = na
        pks[i] = pk
        albs[i] = alb
        lacs[i] = lac
        wbcs[i] = wbc

        # ph_loc, pao2_loc, paco2_loc,
        # Blood gas
        blood_idx = np.where(blood_m[:, 0] == tid)[0]
        ph = np.array((np.nan, np.nan))
        pao2 = np.array((np.nan, np.nan))
        paco2 = np.array((np.nan, np.nan))
        if blood_idx.size > 0:
            blood_idx = blood_idx[0]
            ph = blood_m[blood_idx, ph_locs]
            pao2 = blood_m[blood_idx, pao2_locs]
            paco2 = blood_m[blood_idx, paco2_locs]
        phs[i] = ph
        pao2s[i] = pao2
        paco2s[i] = paco2
    return anemics, bilis, buns, hemats, hemos, pltlts, nas, pks, albs, lacs, phs, pao2s, paco2s, wbcs


def summarize_stats_dallas(ids, kdigos, days, scrs, icu_windows, hosp_windows,
                           dem_m, sex_loc, eth_loc, dob_loc, dod_loc,
                           diag_m, diag_loc,
                           dia_m, dia_start_loc, dia_stop_loc, dia_type_loc,
                           organ_sup, mech_vent_dates, iabp_dates, ecmo_dates, vad_dates,
                           flw_m, flw_col, flw_day, flw_min, flw_max,
                           med_m, amino_loc, nsaid_loc, acei_loc, arb_loc, press_loc,
                           lab_m, lab_col, lab_day, lab_min, lab_max,
                           date_m, hosp_locs, icu_locs,
                           # sofas, apaches,
                           out_name, data_path, grp_name='meta', tlim=7, v=False):
    # dd = np.array(load_csv(data_path + 'disch_disp.csv', ids, str, sel=1))

    n_eps = []
    for i in range(len(kdigos)):
        n_eps.append(count_eps(kdigos[i]))

    bsln_scrs = load_csv(data_path + 'baselines.csv', ids, float)
    bsln_gfrs = load_csv(data_path + 'baseline_gfr.csv', ids, float)
    bsln_types = load_csv(data_path + 'baseline_types.csv', ids, str)

    admits = []
    discharges = []
    ages = []
    genders = []
    mks_7d = []
    mks = []
    dieds = []
    nepss = []
    hosp_days = []
    hosp_frees = []
    icu_days = []
    icu_frees = []
    sepsiss = []
    net_fluids = []
    gross_fluids = []
    fos = []
    c_scores = []
    e_scores = []
    mv_flags = []
    mv_days = []
    mv_frees = []
    eths = []
    dtds = []

    diabetics = []
    hypertensives = []
    ecmos = []
    iabps = []
    vads = []
    smokers = []

    admit_scrs = []
    peak_scrs = []
    hd_dayss = []
    pd_dayss = []
    crrt_dayss = []
    hd_frees_7d = []
    hd_frees_28d = []
    crrt_frees_7d = []
    crrt_frees_28d = []
    pd_frees_7d = []
    pd_frees_28d = []

    nephrotox_cts = []
    vasopress_cts = []

    # icu flw
    phs = []
    maps = []
    temps = []
    hrs = []
    urine_outs = []
    urine_flows = []
    weights = []
    bmis = []
    glasgows = []
    scr_aggs = []
    oxs = []

    # icu labs
    albs = []
    lacs = []
    bilis = []
    buns = []
    chlorides = []
    hemats = []
    hemos = []
    pltlts = []
    sods = []
    wbcs = []
    fio2s = []
    resps = []
    potass = []
    coos = []
    anemics = []

    for i in range(len(ids)):
        idx = ids[i]
        if v:
            print('Summarizing Patient #%d (%d/%d)' % (idx, i+1, len(ids)))
        td = days[i]
        mask = np.where(td <= tlim)[0]
        mk7d = int(np.max(kdigos[i][mask]))
        mk = int(np.max(kdigos[i]))
        # died = 0
        # if 'died' in dd[i]:
        #     died += 1

        eps = n_eps[i]

        # tHospitalAdmissionDiagnosis
        sepsis = 0
        diabetic = 0
        hypertensive = 0
        diag_ids = np.where(diag_m[:, 0] == idx)[0]
        for j in range(len(diag_ids)):
            tid = diag_ids[j]
            if 'sep' in str(diag_m[tid, diag_loc]).lower():
                # if int(diag_m[tid, diag_nb_loc]) == 1:
                sepsis = 1
            if 'diabe' in str(diag_m[tid, diag_loc]).lower():
                diabetic = 1
            if 'hypert' in str(diag_m[tid, diag_loc]).lower():
                hypertensive = 1

        admit_scr = scrs[i][0]
        peak_scr = np.max(scrs[i])

        icu_admit = get_date(icu_windows[i][0])
        icu_discharge = get_date(icu_windows[i][1])

        hosp_admit = get_date(hosp_windows[i][0])
        hosp_disch = get_date(hosp_windows[i][1])

        # tDialysis
        hd_days = 0
        crrt_days = 0
        pd_days = 0
        hd_free_7d = np.ones(7)
        hd_free_28d = np.ones(28)
        crrt_free_7d = np.ones(7)
        crrt_free_28d = np.ones(28)
        pd_free_7d = np.ones(7)
        pd_free_28d = np.ones(28)

        dia_ids = np.where(dia_m[:, 0] == idx)[0]
        if np.size(dia_ids) > 0:
            for row in dia_ids:
                start = get_date(dia_m[row, dia_start_loc])
                stop = get_date(dia_m[row, dia_stop_loc])
                rrt_type = dia_m[row, dia_type_loc]
                if start != 'nan' and stop != 'nan':
                    if rrt_type == 'CRRT':
                        crrt_days += (stop - start).days
                        if icu_admit < start <= icu_admit + datetime.timedelta(7):
                            sidx = (start - icu_admit).days
                            send = min(7, sidx + (stop - start).days)
                            crrt_free_7d[sidx:send] = 0
                        if icu_admit < start <= icu_admit + datetime.timedelta(28):
                            sidx = (start - icu_admit).days
                            send = min(28, sidx + (stop - start).days)
                            crrt_free_28d[sidx:send] = 0
                        if icu_admit < stop <= icu_admit + datetime.timedelta(7):
                            sidx = (stop - icu_admit).days
                            send = min(7, sidx)
                            crrt_free_7d[:send] = 0
                        if icu_admit < stop <= icu_admit + datetime.timedelta(28):
                            sidx = (start - icu_admit).days
                            send = min(28, sidx + (stop - start).days)
                            crrt_free_28d[:send] = 0
                        if start < icu_admit and stop > icu_admit + datetime.timedelta(7):
                            crrt_free_7d[:] = 0
                    elif rrt_type == 'HD':
                        hd_days += (stop - start).days
                        if icu_admit < start <= icu_admit + datetime.timedelta(7):
                            sidx = (start - icu_admit).days
                            send = min(7, sidx + (stop - start).days)
                            hd_free_7d[sidx:send] = 0
                        if icu_admit < start <= icu_admit + datetime.timedelta(28):
                            sidx = (start - icu_admit).days
                            send = min(28, sidx + (stop - start).days)
                            hd_free_28d[sidx:send] = 0
                        if icu_admit < stop <= icu_admit + datetime.timedelta(7):
                            sidx = (stop - icu_admit).days
                            send = min(7, sidx)
                            hd_free_7d[:send] = 0
                        if icu_admit < stop <= icu_admit + datetime.timedelta(28):
                            sidx = (start - icu_admit).days
                            send = min(28, sidx + (stop - start).days)
                            hd_free_28d[:send] = 0
                        if start < icu_admit and stop > icu_admit + datetime.timedelta(7):
                            hd_free_7d[:] = 0
                    elif rrt_type == 'PD':
                        pd_days += (stop - start).days
                        if icu_admit < start <= icu_admit + datetime.timedelta(7):
                            sidx = (start - icu_admit).days
                            send = min(7, sidx + (stop - start).days)
                            pd_free_7d[sidx:send] = 0
                        if icu_admit < start <= icu_admit + datetime.timedelta(28):
                            sidx = (start - icu_admit).days
                            send = min(28, sidx + (stop - start).days)
                            pd_free_28d[sidx:send] = 0
                        if icu_admit < stop <= icu_admit + datetime.timedelta(7):
                            sidx = (stop - icu_admit).days
                            send = min(7, sidx)
                            pd_free_7d[:send] = 0
                        if icu_admit < stop <= icu_admit + datetime.timedelta(28):
                            sidx = (start - icu_admit).days
                            send = min(28, sidx + (stop - start).days)
                            pd_free_28d[:send] = 0
                        if start < icu_admit and stop > icu_admit + datetime.timedelta(7):
                            pd_free_7d[:] = 0

        hd_free_7d = np.sum(hd_free_7d)
        hd_free_28d = np.sum(hd_free_28d)
        crrt_free_7d = np.sum(crrt_free_7d)
        crrt_free_28d = np.sum(crrt_free_28d)
        pd_free_7d = np.sum(pd_free_7d)
        pd_free_28d = np.sum(pd_free_28d)

        # tPatients
        dem_idx = np.where(dem_m[:, 0] == idx)[0][0]
        male = dem_m[dem_idx, sex_loc]
        eth = dem_m[dem_idx, eth_loc]
        dob = get_date(dem_m[dem_idx, dob_loc])
        dod = get_date(dem_m[dem_idx, dod_loc])
        # dodl = dem_m[dem_idx, dod_locs]
        # dod = 'nan'
        # for x in dodl:
        #     dodDate = get_date(str(x))
        #     if dodDate != 'nan':
        #         dod = dodDate

        died = 0
        dtd = np.nan
        if dod != 'nan':
            dtd = (dod - icu_admit).total_seconds() / (60 * 60 * 24)
            hosp_admit = get_date(hosp_windows[i][0])
            hosp_disch = get_date(hosp_windows[i][1])
            if hosp_admit < dod:
                if hosp_disch == 'nan' or dod < hosp_disch:
                    died = 1

        tage = icu_admit - dob
        age = tage.total_seconds() / (60 * 60 * 24 * 365)

        hlos, ilos = get_los(idx, date_m, hosp_locs, icu_locs)
        hfree = 28 - hlos
        ifree = 28 - ilos
        if hfree < 0:
            hfree = 0
        if ifree < 0:
            ifree = 0

        # tAOS table
        mech_flag = 0
        mech_day = 0
        mech = 28
        ecmo = 0
        iabp = 0
        vad = 0
        organ_sup_idx = np.where(organ_sup[:, 0] == idx)[0]
        if organ_sup_idx.size > 0:
            for row in organ_sup_idx:
                mech_start = get_date(organ_sup[row, mech_vent_dates[0]])
                mech_stop = get_date(organ_sup[row, mech_vent_dates[1]])
                iabp_start = get_date(organ_sup[row, iabp_dates[0]])
                iabp_stop = get_date(organ_sup[row, iabp_dates[1]])
                vad_start = get_date(organ_sup[row, vad_dates[0]])
                vad_stop = get_date(organ_sup[row, vad_dates[1]])
                ecmo_start = get_date(organ_sup[row, ecmo_dates[0]])
                ecmo_stop = get_date(organ_sup[row, ecmo_dates[1]])

                if mech_stop != 'nan':
                    if mech_stop < icu_admit:
                        pass
                    elif (mech_start - icu_admit).days > 28:
                        pass
                    else:
                        if mech_start < icu_admit:
                            mech_day = (mech_stop - icu_admit).days
                        else:
                            mech_day = (mech_stop - mech_start).days

                        mech = 28 - mech_day
                        mech_flag = 1

                if iabp_start != 'nan' and iabp_stop != 'nan':
                    if iabp_stop < icu_admit:
                        pass
                    elif (iabp_start - icu_admit).days > 28:
                        pass
                    else:
                        iabp = 1

                if vad_start != 'nan' and iabp_stop != 'nan':
                    if vad_stop < icu_admit:
                        pass
                    elif (vad_start - icu_admit).days > 28:
                        pass
                    else:
                        vad = 1

                if ecmo_start != 'nan' and iabp_stop != 'nan':
                    if ecmo_stop < icu_admit:
                        pass
                    elif (ecmo_start - icu_admit).days > 28:
                        pass
                    else:
                        ecmo = 1

        if died:
            if mech < 28:
                mech = 0
            if hfree < 28:
                hfree = 0
            if ifree < 28:
                ifree = 0

        # icu_flw_data
        flw_idx = np.where(flw_m[:, 0] == idx)[0]
        d0_idx = flw_idx[np.where(flw_m[flw_idx, flw_day] == 'D0')[0]]
        d1_idx = flw_idx[np.where(flw_m[flw_idx, flw_day] == 'D1')[0]]
        d2_idx = flw_idx[np.where(flw_m[flw_idx, flw_day] == 'D2')[0]]

        height_idx = flw_idx[np.where(flw_m[flw_idx, flw_col] == 'HEIGHT')]
        weight_idx = flw_idx[np.where(flw_m[flw_idx, flw_col] == 'WEIGHT')]
        uop_idxs = flw_idx[np.where(flw_m[flw_idx, flw_col] == 'UOP')]
        temp_idxs = flw_idx[np.where(flw_m[flw_idx, flw_col] == 'TEMPERATURE')]
        hr_idxs = flw_idx[np.where(flw_m[flw_idx, flw_col] == 'HEART RATE')]
        map_idxs = flw_idx[np.where(flw_m[flw_idx, flw_col] == 'MAP')]
        in_idx = flw_idx[np.where(flw_m[flw_idx, flw_col] == 'IN')]
        out_idx = flw_idx[np.where(flw_m[flw_idx, flw_col] == 'Out')]
        ph_idxs = flw_idx[np.where(flw_m[flw_idx, flw_col] == 'PH')]
        glasgow_idxs = flw_idx[np.where(flw_m[flw_idx, flw_col] == 'GLASGOW SCORE')]
        resp_idxs = flw_idx[np.where(flw_m[flw_idx, flw_col] == 'RESPIRATORY RATE')]

        bmi = np.nan
        weight = np.nan
        height = np.nan
        
        height_idx = np.intersect1d(d0_idx, height_idx)
        if height_idx.size > 0:
            height = flw_m[height_idx[0], flw_min] / 100
        else:
            height_idx = np.intersect1d(d1_idx, height_idx)
            if height_idx.size > 0:
                height = flw_m[height_idx[0], flw_min] / 100
            else:
                height_idx = np.intersect1d(d2_idx, height_idx)
                if height_idx.size > 0:
                    height = flw_m[height_idx[0], flw_min] / 100

        weight_idx = np.intersect1d(d0_idx, weight_idx)
        if weight_idx.size > 0:
            weight = flw_m[weight_idx[0], flw_min]
        else:
            weight_idx = np.intersect1d(d1_idx, weight_idx)
            if weight_idx.size > 0:
                weight = flw_m[weight_idx[0], flw_min]
            else:
                weight_idx = np.intersect1d(d2_idx, weight_idx)
                if weight_idx.size > 0:
                    weight = flw_m[weight_idx[0], flw_min]

        if weight != np.nan and height != np.nan:
            bmi = weight / (height * height)

        temp = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        temp_idx = np.intersect1d(d0_idx, temp_idxs)
        if temp_idx.size > 0:
            temp[0, 0] = flw_m[temp_idx[0], flw_min]
            temp[0, 1] = flw_m[temp_idx[0], flw_max]
        temp_idx = np.intersect1d(d1_idx, temp_idxs)
        if temp_idx.size > 0:
            temp[1, 0] = flw_m[temp_idx[0], flw_min]
            temp[1, 1] = flw_m[temp_idx[0], flw_max]
        temp_idx = np.intersect1d(d2_idx, temp_idxs)
        if temp_idx.size > 0:
            temp[2, 0] = flw_m[temp_idx[0], flw_min]
            temp[2, 1] = flw_m[temp_idx[0], flw_max]

        tmap = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        tmap_idx = np.intersect1d(d0_idx, map_idxs)
        if tmap_idx.size > 0:
            tmap[0, 0] = flw_m[tmap_idx[0], flw_min]
            tmap[0, 1] = flw_m[tmap_idx[0], flw_max]
        tmap_idx = np.intersect1d(d1_idx, map_idxs)
        if tmap_idx.size > 0:
            tmap[1, 0] = flw_m[tmap_idx[0], flw_min]
            tmap[1, 1] = flw_m[tmap_idx[0], flw_max]
        tmap_idx = np.intersect1d(d2_idx, map_idxs)
        if tmap_idx.size > 0:
            tmap[2, 0] = flw_m[tmap_idx[0], flw_min]
            tmap[2, 1] = flw_m[tmap_idx[0], flw_max]

        hr = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        hr_idx = np.intersect1d(d0_idx, hr_idxs)
        if hr_idx.size > 0:
            hr[0, 0] = flw_m[hr_idx[0], flw_min]
            hr[0, 1] = flw_m[hr_idx[0], flw_max]
        hr_idx = np.intersect1d(d1_idx, hr_idxs)
        if hr_idx.size > 0:
            hr[1, 0] = flw_m[hr_idx[0], flw_min]
            hr[1, 1] = flw_m[hr_idx[0], flw_max]
        hr_idx = np.intersect1d(d2_idx, hr_idxs)
        if hr_idx.size > 0:
            hr[2, 0] = flw_m[hr_idx[0], flw_min]
            hr[2, 1] = flw_m[hr_idx[0], flw_max]

        ph = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        ph_idx = np.intersect1d(d0_idx, ph_idxs)
        if ph_idx.size > 0:
            ph[0, 0] = flw_m[ph_idx[0], flw_min]
            ph[0, 1] = flw_m[ph_idx[0], flw_max]
        ph_idx = np.intersect1d(d1_idx, ph_idxs)
        if ph_idx.size > 0:
            ph[1, 0] = flw_m[ph_idx[0], flw_min]
            ph[1, 1] = flw_m[ph_idx[0], flw_max]
        ph_idx = np.intersect1d(d2_idx, ph_idxs)
        if ph_idx.size > 0:
            ph[2, 0] = flw_m[ph_idx[0], flw_min]
            ph[2, 1] = flw_m[ph_idx[0], flw_max]

        glasgow = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        glasgow_idx = np.intersect1d(d0_idx, glasgow_idxs)
        if glasgow_idx.size > 0:
            glasgow[0, 0] = flw_m[glasgow_idx[0], flw_min]
            glasgow[0, 1] = flw_m[glasgow_idx[0], flw_max]
        glasgow_idx = np.intersect1d(d1_idx, glasgow_idxs)
        if glasgow_idx.size > 0:
            glasgow[1, 0] = flw_m[glasgow_idx[0], flw_min]
            glasgow[1, 1] = flw_m[glasgow_idx[0], flw_max]
        glasgow_idx = np.intersect1d(d2_idx, glasgow_idxs)
        if glasgow_idx.size > 0:
            glasgow[2, 0] = flw_m[glasgow_idx[0], flw_min]
            glasgow[2, 1] = flw_m[glasgow_idx[0], flw_max]

        resp = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        resp_idx = np.intersect1d(d0_idx, resp_idxs)
        if resp_idx.size > 0:
            resp[0, 0] = flw_m[resp_idx[0], flw_min]
            resp[0, 1] = flw_m[resp_idx[0], flw_max]
        resp_idx = np.intersect1d(d1_idx, resp_idxs)
        if resp_idx.size > 0:
            resp[1, 0] = flw_m[resp_idx[0], flw_min]
            resp[1, 1] = flw_m[resp_idx[0], flw_max]
        resp_idx = np.intersect1d(d2_idx, resp_idxs)
        if resp_idx.size > 0:
            resp[2, 0] = flw_m[resp_idx[0], flw_min]
            resp[2, 1] = flw_m[resp_idx[0], flw_max]

        urine_out = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        urine_flow = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        urine_idx = np.intersect1d(d0_idx, uop_idxs)
        if urine_idx.size > 0:
            urine_out[0, 0] = float(flw_m[urine_idx, flw_min])
            urine_flow[0, 0] = urine_out[0, 0] / weight / 24
            urine_out[0, 1] = float(flw_m[urine_idx, flw_max])
            urine_flow[0, 1] = urine_out[0, 1] / weight / 24
        urine_idx = np.intersect1d(d1_idx, uop_idxs)
        if urine_idx.size > 0:
            urine_out[1, 0] = float(flw_m[urine_idx, flw_min])
            urine_flow[1, 0] = urine_out[1, 0] / weight / 24
            urine_out[1, 1] = float(flw_m[urine_idx, flw_max])
            urine_flow[1, 1] = urine_out[1, 1] / weight / 24
        urine_idx = np.intersect1d(d2_idx, uop_idxs)
        if urine_idx.size > 0:
            urine_out[2, 0] = float(flw_m[urine_idx, flw_min])
            urine_flow[2, 0] = urine_out[2, 0] / weight / 24
            urine_out[2, 1] = float(flw_m[urine_idx, flw_max])
            urine_flow[2, 1] = urine_out[2, 1] / weight / 24

        d0_in_idx = np.intersect1d(d0_idx, in_idx)
        d1_in_idx = np.intersect1d(d1_idx, in_idx)
        d2_in_idx = np.intersect1d(d2_idx, in_idx)

        d0_out_idx = np.intersect1d(d0_idx, out_idx)
        d1_out_idx = np.intersect1d(d1_idx, out_idx)
        d2_out_idx = np.intersect1d(d2_idx, out_idx)
        
        try:
            net0 = float(flw_m[d0_in_idx, flw_min] - flw_m[d0_out_idx, flw_min])
        except:
             net0 = np.nan
        try:
            net1 = float(flw_m[d1_in_idx, flw_min] - flw_m[d1_out_idx, flw_min])
        except:
            net1 = np.nan
        try:
            net2 = float(flw_m[d2_in_idx, flw_min] - flw_m[d2_out_idx, flw_min])
        except:
            net2 = np.nan

        try:
            net = float(np.nansum((net0, net1, net2)))
        except:
            net = np.nan
        
        try:
            tot0 = float(flw_m[d0_in_idx, flw_min] + flw_m[d0_out_idx, flw_min])
        except:
             tot0 = np.nan
        try:
            tot1 = float(flw_m[d1_in_idx, flw_min] + flw_m[d1_out_idx, flw_min])
        except:
            tot1 = np.nan
        try:
            tot2 = float(flw_m[d2_in_idx, flw_min] + flw_m[d2_out_idx, flw_min])
        except:
            tot2 = np.nan

        try:
            tot = float(np.nansum((tot0, tot1, tot2)))
        except:
            tot = np.nan

        if tot < 0:
            net = np.nan
            tot = np.nan

        if net != np.nan and weight != np.nan:
            fo = net / weight
        else:
            fo = np.nan

        # tMedications
        med_idx = np.where(med_m[:, 0] == idx)[0]
        neph_ct = 0
        vaso_ct = 0
        if med_idx.size > 0:
            tid = med_idx[0]
            # Nephrotoxins
            if np.any([med_m[tid, x] for x in [amino_loc, nsaid_loc, acei_loc, arb_loc]]):
                neph_ct += 1
            # Vasoactive Drugs
            if med_m[tid, press_loc]:
                vaso_ct += 1

        # icu_lab_data
        # Anemia, bilirubin, and BUN (labs set 1)
        anemic = np.zeros(3)
        lab_idx = np.where(lab_m[:, 0] == idx)[0]

        d0_idx = lab_idx[np.where(lab_m[lab_idx, lab_day] == 'D0')[0]]
        d1_idx = lab_idx[np.where(lab_m[lab_idx, lab_day] == 'D1')[0]]
        d2_idx = lab_idx[np.where(lab_m[lab_idx, lab_day] == 'D2')[0]]

        alb_idxs = lab_idx[np.where(lab_m[lab_idx, lab_col] == 'ALBUMIN')[0]]
        bili_idxs = lab_idx[np.where(lab_m[lab_idx, lab_col] == 'BILIRUBIN, TOTAL')[0]]
        bun_idxs = lab_idx[np.where(lab_m[lab_idx, lab_col] == 'BUN')[0]]
        chloride_idxs = lab_idx[np.where(lab_m[lab_idx, lab_col] == 'CHLORIDE')[0]]
        hemat_idxs = lab_idx[np.where(lab_m[lab_idx, lab_col] == 'HEMATOCRIT')[0]]
        hemo_idxs = lab_idx[np.where(lab_m[lab_idx, lab_col] == 'HEMOGLOBIN')[0]]
        pltlt_idxs = lab_idx[np.where(lab_m[lab_idx, lab_col] == 'PLATELETS')[0]]
        pot_idxs = lab_idx[np.where(lab_m[lab_idx, lab_col] == 'POTASSIUM')[0]]
        sod_idxs = lab_idx[np.where(lab_m[lab_idx, lab_col] == 'SODIUM')[0]]
        wbc_idxs = lab_idx[np.where(lab_m[lab_idx, lab_col] == 'WBC')[0]]
        fio2_idxs = lab_idx[np.where(lab_m[lab_idx, lab_col] == 'WBC')[0]]
        scr_agg_idxs = lab_idx[np.where(lab_m[lab_idx, lab_col] == 'CREATININE')[0]]
        potas_idxs = lab_idx[np.where(lab_m[lab_idx, lab_col] == 'POTASSIUM')[0]]
        pco2_idxs = lab_idx[np.where(lab_m[lab_idx, lab_col] == 'PCO2')[0]]
        po2_idxs = lab_idx[np.where(lab_m[lab_idx, lab_col] == 'PO2')]

        alb = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        alb_idx = np.intersect1d(d0_idx, alb_idxs)
        if alb_idx.size > 0:
            alb[0, 0] = lab_m[alb_idx[0], lab_min]
            alb[0, 1] = lab_m[alb_idx[0], lab_max]
        alb_idx = np.intersect1d(d1_idx, alb_idxs)
        if alb_idx.size > 0:
            alb[1, 0] = lab_m[alb_idx[0], lab_min]
            alb[1, 1] = lab_m[alb_idx[0], lab_max]
        alb_idx = np.intersect1d(d2_idx, alb_idxs)
        if alb_idx.size > 0:
            alb[2, 0] = lab_m[alb_idx[0], lab_min]
            alb[2, 1] = lab_m[alb_idx[0], lab_max]

        bili = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        bili_idx = np.intersect1d(d0_idx, bili_idxs)
        if bili_idx.size > 0:
            bili[0, 0] = lab_m[bili_idx[0], lab_min]
            bili[0, 1] = lab_m[bili_idx[0], lab_max]
        bili_idx = np.intersect1d(d1_idx, bili_idxs)
        if bili_idx.size > 0:
            bili[1, 0] = lab_m[bili_idx[0], lab_min]
            bili[1, 1] = lab_m[bili_idx[0], lab_max]
        bili_idx = np.intersect1d(d2_idx, bili_idxs)
        if bili_idx.size > 0:
            bili[2, 0] = lab_m[bili_idx[0], lab_min]
            bili[2, 1] = lab_m[bili_idx[0], lab_max]

        bun = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        bun_idx = np.intersect1d(d0_idx, bun_idxs)
        if bun_idx.size > 0:
            bun[0, 0] = lab_m[bun_idx[0], lab_min]
            bun[0, 1] = lab_m[bun_idx[0], lab_max]
        bun_idx = np.intersect1d(d1_idx, bun_idxs)
        if bun_idx.size > 0:
            bun[1, 0] = lab_m[bun_idx[0], lab_min]
            bun[1, 1] = lab_m[bun_idx[0], lab_max]
        bun_idx = np.intersect1d(d2_idx, bun_idxs)
        if bun_idx.size > 0:
            bun[2, 0] = lab_m[bun_idx[0], lab_min]
            bun[2, 1] = lab_m[bun_idx[0], lab_max]

        chloride = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        chloride_idx = np.intersect1d(d0_idx, chloride_idxs)
        if chloride_idx.size > 0:
            chloride[0, 0] = lab_m[chloride_idx[0], lab_min]
            chloride[0, 1] = lab_m[chloride_idx[0], lab_max]
        chloride_idx = np.intersect1d(d1_idx, chloride_idxs)
        if chloride_idx.size > 0:
            chloride[1, 0] = lab_m[chloride_idx[0], lab_min]
            chloride[1, 1] = lab_m[chloride_idx[0], lab_max]
        chloride_idx = np.intersect1d(d2_idx, chloride_idxs)
        if chloride_idx.size > 0:
            chloride[2, 0] = lab_m[chloride_idx[0], lab_min]
            chloride[2, 1] = lab_m[chloride_idx[0], lab_max]

        hemat = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        hemat_idx = np.intersect1d(d0_idx, hemat_idxs)
        if hemat_idx.size > 0:
            hemat[0, 0] = lab_m[hemat_idx[0], lab_min]
            hemat[0, 1] = lab_m[hemat_idx[0], lab_max]
        hemat_idx = np.intersect1d(d1_idx, hemat_idxs)
        if hemat_idx.size > 0:
            hemat[1, 0] = lab_m[hemat_idx[0], lab_min]
            hemat[1, 1] = lab_m[hemat_idx[0], lab_max]
        hemat_idx = np.intersect1d(d2_idx, hemat_idxs)
        if hemat_idx.size > 0:
            hemat[2, 0] = lab_m[hemat_idx[0], lab_min]
            hemat[2, 1] = lab_m[hemat_idx[0], lab_max]

        hemo = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        hemo_idx = np.intersect1d(d0_idx, hemo_idxs)
        if hemo_idx.size > 0:
            hemo[0, 0] = lab_m[hemo_idx[0], lab_min]
            hemo[0, 1] = lab_m[hemo_idx[0], lab_max]
        hemo_idx = np.intersect1d(d1_idx, hemo_idxs)
        if hemo_idx.size > 0:
            hemo[1, 0] = lab_m[hemo_idx[0], lab_min]
            hemo[1, 1] = lab_m[hemo_idx[0], lab_max]
        hemo_idx = np.intersect1d(d2_idx, hemo_idxs)
        if hemo_idx.size > 0:
            hemo[2, 0] = lab_m[hemo_idx[0], lab_min]
            hemo[2, 1] = lab_m[hemo_idx[0], lab_max]

        pltlt = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        pltlt_idx = np.intersect1d(d0_idx, pltlt_idxs)
        if pltlt_idx.size > 0:
            pltlt[0, 0] = lab_m[pltlt_idx[0], lab_min]
            pltlt[0, 1] = lab_m[pltlt_idx[0], lab_max]
        pltlt_idx = np.intersect1d(d1_idx, pltlt_idxs)
        if pltlt_idx.size > 0:
            pltlt[1, 0] = lab_m[pltlt_idx[0], lab_min]
            pltlt[1, 1] = lab_m[pltlt_idx[0], lab_max]
        pltlt_idx = np.intersect1d(d2_idx, pltlt_idxs)
        if pltlt_idx.size > 0:
            pltlt[2, 0] = lab_m[pltlt_idx[0], lab_min]
            pltlt[2, 1] = lab_m[pltlt_idx[0], lab_max]

        pot = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        pot_idx = np.intersect1d(d0_idx, pot_idxs)
        if pot_idx.size > 0:
            pot[0, 0] = lab_m[pot_idx[0], lab_min]
            pot[0, 1] = lab_m[pot_idx[0], lab_max]
        pot_idx = np.intersect1d(d1_idx, pot_idxs)
        if pot_idx.size > 0:
            pot[1, 0] = lab_m[pot_idx[0], lab_min]
            pot[1, 1] = lab_m[pot_idx[0], lab_max]
        pot_idx = np.intersect1d(d2_idx, pot_idxs)
        if pot_idx.size > 0:
            pot[2, 0] = lab_m[pot_idx[0], lab_min]
            pot[2, 1] = lab_m[pot_idx[0], lab_max]

        sod = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        sod_idx = np.intersect1d(d0_idx, sod_idxs)
        if sod_idx.size > 0:
            sod[0, 0] = lab_m[sod_idx[0], lab_min]
            sod[0, 1] = lab_m[sod_idx[0], lab_max]
        sod_idx = np.intersect1d(d1_idx, sod_idxs)
        if sod_idx.size > 0:
            sod[1, 0] = lab_m[sod_idx[0], lab_min]
            sod[1, 1] = lab_m[sod_idx[0], lab_max]
        sod_idx = np.intersect1d(d2_idx, sod_idxs)
        if sod_idx.size > 0:
            sod[2, 0] = lab_m[sod_idx[0], lab_min]
            sod[2, 1] = lab_m[sod_idx[0], lab_max]

        wbc = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        wbc_idx = np.intersect1d(d0_idx, wbc_idxs)
        if wbc_idx.size > 0:
            wbc[0, 0] = lab_m[wbc_idx[0], lab_min]
            wbc[0, 1] = lab_m[wbc_idx[0], lab_max]
        wbc_idx = np.intersect1d(d1_idx, wbc_idxs)
        if wbc_idx.size > 0:
            wbc[1, 0] = lab_m[wbc_idx[0], lab_min]
            wbc[1, 1] = lab_m[wbc_idx[0], lab_max]
        wbc_idx = np.intersect1d(d2_idx, wbc_idxs)
        if wbc_idx.size > 0:
            wbc[2, 0] = lab_m[wbc_idx[0], lab_min]
            wbc[2, 1] = lab_m[wbc_idx[0], lab_max]

        fio2 = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        fio2_idx = np.intersect1d(d0_idx, fio2_idxs)
        if fio2_idx.size > 0:
            fio2[0, 0] = lab_m[fio2_idx[0], lab_min]
            fio2[0, 1] = lab_m[fio2_idx[0], lab_max]
        fio2_idx = np.intersect1d(d1_idx, fio2_idxs)
        if fio2_idx.size > 0:
            fio2[1, 0] = lab_m[fio2_idx[0], lab_min]
            fio2[1, 1] = lab_m[fio2_idx[0], lab_max]
        fio2_idx = np.intersect1d(d2_idx, fio2_idxs)
        if fio2_idx.size > 0:
            fio2[2, 0] = lab_m[fio2_idx[0], lab_min]
            fio2[2, 1] = lab_m[fio2_idx[0], lab_max]

        potas = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        potas_idx = np.intersect1d(d0_idx, potas_idxs)
        if potas_idx.size > 0:
            potas[0, 0] = lab_m[potas_idx[0], lab_min]
            potas[0, 1] = lab_m[potas_idx[0], lab_max]
        potas_idx = np.intersect1d(d1_idx, potas_idxs)
        if potas_idx.size > 0:
            potas[1, 0] = lab_m[potas_idx[0], lab_min]
            potas[1, 1] = lab_m[potas_idx[0], lab_max]
        potas_idx = np.intersect1d(d2_idx, potas_idxs)
        if potas_idx.size > 0:
            potas[2, 0] = lab_m[potas_idx[0], lab_min]
            potas[2, 1] = lab_m[potas_idx[0], lab_max]

        pco2 = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        pco2_idx = np.intersect1d(d0_idx, pco2_idxs)
        if pco2_idx.size > 0:
            pco2[0, 0] = lab_m[pco2_idx[0], lab_min]
            pco2[0, 1] = lab_m[pco2_idx[0], lab_max]
        pco2_idx = np.intersect1d(d1_idx, pco2_idxs)
        if pco2_idx.size > 0:
            pco2[1, 0] = lab_m[pco2_idx[0], lab_min]
            pco2[1, 1] = lab_m[pco2_idx[0], lab_max]
        pco2_idx = np.intersect1d(d2_idx, pco2_idxs)
        if pco2_idx.size > 0:
            pco2[2, 0] = lab_m[pco2_idx[0], lab_min]
            pco2[2, 1] = lab_m[pco2_idx[0], lab_max]

        # po2 = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        # ox_idx = np.intersect1d(d0_idx, ox_idx)
        # if ox_idx.size > 0:
        #     po2 = flw_m[ox_idx[0], flw_min]
        # else:
        #     ox_idx = np.intersect1d(d1_idx, ox_idx)
        #     if ox_idx.size > 0:
        #         po2 = flw_m[ox_idx[0], flw_min]
        #     else:
        #         ox_idx = np.intersect1d(d2_idx, ox_idx)
        #         if ox_idx.size > 0:
        #             po2 = flw_m[ox_idx[0], flw_min]

        po2 = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        po2_idx = np.intersect1d(d0_idx, po2_idxs)
        if po2_idx.size > 0:
            po2[0, 0] = lab_m[po2_idx[0], lab_min]
            po2[0, 1] = lab_m[po2_idx[0], lab_max]
        po2_idx = np.intersect1d(d1_idx, po2_idxs)
        if po2_idx.size > 0:
            po2[1, 0] = lab_m[po2_idx[0], lab_min]
            po2[1, 1] = lab_m[po2_idx[0], lab_max]
        po2_idx = np.intersect1d(d2_idx, po2_idxs)
        if po2_idx.size > 0:
            po2[2, 0] = lab_m[po2_idx[0], lab_min]
            po2[2, 1] = lab_m[po2_idx[0], lab_max]

        scr_agg = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        scr_agg_idx = np.intersect1d(d0_idx, scr_agg_idxs)
        if scr_agg_idx.size > 0:
            scr_agg[0, 0] = lab_m[scr_agg_idx[0], lab_min]
            scr_agg[0, 1] = lab_m[scr_agg_idx[0], lab_max]
        scr_agg_idx = np.intersect1d(d1_idx, scr_agg_idxs)
        if scr_agg_idx.size > 0:
            scr_agg[1, 0] = lab_m[scr_agg_idx[0], lab_min]
            scr_agg[1, 1] = lab_m[scr_agg_idx[0], lab_max]
        scr_agg_idx = np.intersect1d(d2_idx, scr_agg_idxs)
        if scr_agg_idx.size > 0:
            scr_agg[2, 0] = lab_m[scr_agg_idx[0], lab_min]
            scr_agg[2, 1] = lab_m[scr_agg_idx[0], lab_max]

        # definition A
        if male:
            if hemat[1, 0] < 39 or hemo[1, 0] < 18:
                anemic[0] = 1
        else:
            if hemat[1, 0] < 36 or hemo[1, 0] < 12:
                anemic[0] = 1
        # definition B
        if hemat[1, 0] < 30 or hemo[1, 0] < 10:
            anemic[1] = 1
        # definition C
        if hemat[1, 0] < 27 or hemo[1, 0] < 9:
            anemic[2] = 1

        charl = 0
        elix = 0
        smoker = 0

        admits.append(str(icu_admit))
        discharges.append(str(icu_discharge))
        ages.append(age)
        genders.append(male)
        mks_7d.append(mk7d)
        mks.append(mk)
        dieds.append(died)
        nepss.append(eps)
        hosp_days.append(hlos)
        hosp_frees.append(hfree)
        icu_days.append(ilos)
        icu_frees.append(ifree)
        sepsiss.append(sepsis)
        diabetics.append(diabetic)
        hypertensives.append(hypertensive)
        admit_scrs.append(admit_scr)
        peak_scrs.append(peak_scr)

        bmis.append(bmi)
        weights.append(weight)
        net_fluids.append(net)
        gross_fluids.append(tot)
        fos.append(fo)
        c_scores.append(charl)
        e_scores.append(elix)
        mv_flags.append(mech_flag)
        mv_days.append(mech_day)
        mv_frees.append(mech)
        ecmos.append(ecmo)
        iabps.append(iabp)
        vads.append(vad)
        eths.append(eth)
        dtds.append(dtd)
        hd_dayss.append(hd_days)
        crrt_dayss.append(crrt_days)
        hd_frees_7d.append(hd_free_7d)
        hd_frees_28d.append(hd_free_28d)
        crrt_frees_7d.append(crrt_free_7d)
        crrt_frees_28d.append(crrt_free_28d)
        pd_dayss.append(pd_days)
        pd_frees_7d.append(pd_free_7d)
        pd_frees_28d.append(pd_free_28d)

        nephrotox_cts.append(neph_ct)
        vasopress_cts.append(vaso_ct)
        anemics.append(anemic)
        urine_outs.append(urine_out)
        urine_flows.append(urine_flow)
        smokers.append(smoker)

        phs.append(ph)
        maps.append(tmap)
        albs.append(alb)
        bilis.append(bili)
        buns.append(bun)
        coos.append(pco2)
        oxs.append(po2)
        
        chlorides.append(chloride)
        hemats.append(hemat)
        hemos.append(hemo)
        pltlts.append(pltlt)
        sods.append(sod)
        wbcs.append(wbc)
        temps.append(temp)
        hrs.append(hr)
        glasgows.append(glasgow)
        fio2s.append(fio2)
        scr_aggs.append(scr_agg)
        resps.append(resp)
        potass.append(potas)

    admits = np.array(admits, dtype=str)
    discharges = np.array(discharges, dtype=str)
    ages = np.array(ages, dtype=float)
    genders = np.array(genders, dtype=bool)
    eths = np.array(eths, dtype=bool)
    bmis = np.array(bmis, dtype=float)
    weights = np.array(weights, dtype=float)
    c_scores = np.array(c_scores, dtype=float)  # Float bc possible NaNs
    e_scores = np.array(e_scores, dtype=float)  # Float bc possible NaNs
    diabetics = np.array(diabetics, dtype=bool)
    hypertensives = np.array(hypertensives, bool)

    hosp_days = np.array(hosp_days, dtype=float)
    hosp_frees = np.array(hosp_frees, dtype=float)
    icu_days = np.array(icu_days, dtype=float)
    icu_frees = np.array(icu_frees, dtype=float)
    mv_flags = np.array(mv_flags, dtype=bool)
    mv_days = np.array(mv_days, dtype=float)
    mv_frees = np.array(mv_frees, dtype=float)
    ecmos = np.array(ecmos, dtype=int)
    iabps = np.array(iabps, dtype=int)
    vads = np.array(vads, dtype=int)
    sepsiss = np.array(sepsiss, dtype=bool)
    dieds = np.array(dieds, dtype=int)

    bsln_scrs = np.array(bsln_scrs, dtype=float)
    bsln_gfrs = np.array(bsln_gfrs, dtype=float)
    admit_scrs = np.array(admit_scrs, dtype=float)
    peak_scrs = np.array(peak_scrs, dtype=float)
    mks_7d = np.array(mks_7d, dtype=int)
    mks = np.array(mks, dtype=int)
    net_fluids = np.array(net_fluids, dtype=float)
    gross_fluids = np.array(gross_fluids, dtype=float)
    fos = np.array(fos, dtype=float)
    nepss = np.array(nepss, dtype=int)
    hd_dayss = np.array(hd_dayss, dtype=int)
    pd_dayss = np.array(hd_dayss, dtype=int)
    crrt_dayss = np.array(crrt_dayss, dtype=int)
    hd_frees_7d = np.array(hd_frees_7d, dtype=int)
    hd_frees_28d = np.array(hd_frees_28d, dtype=int)
    pd_frees_7d = np.array(pd_frees_7d, dtype=int)
    pd_frees_28d = np.array(pd_frees_28d, dtype=int)
    crrt_frees_7d = np.array(crrt_frees_7d, dtype=int)
    crrt_frees_28d = np.array(crrt_frees_28d, dtype=int)

    dtds = np.array(dtds, dtype=float)

    nephrotox_cts = np.array(nephrotox_cts, dtype=int)
    vasopress_cts = np.array(vasopress_cts, dtype=int)
    anemics = np.array(anemics, dtype=bool)
    urine_outs = np.array(urine_outs, dtype=float)
    urine_flows = np.array(urine_flows, dtype=float)
    smokers = np.array(smokers, dtype=bool)

    phs = np.array(phs, dtype=float)
    maps = np.array(maps, dtype=float)
    albs = np.array(albs, dtype=float)
    bilis = np.array(bilis, dtype=float)
    buns = np.array(buns, dtype=float)
    scr_aggs = np.array(scr_aggs, dtype=float)
    glasgows = np.array(glasgows, dtype=float)
    fio2s = np.array(fio2s, dtype=float)
    resps = np.array(resps, dtype=float)
    oxs = np.array(oxs, dtype=float)

    chlorides = np.array(chlorides, dtype=float)
    hemats = np.array(hemats, dtype=float)
    hemos = np.array(hemos, dtype=float)
    pltlts = np.array(pltlts, dtype=float)
    sods = np.array(sods, dtype=float)
    wbcs = np.array(wbcs, dtype=float)
    potass = np.array(potass, dtype=float)
    coos = np.array(coos, dtype=float)

    temps = np.array(temps, dtype=float)
    hrs = np.array(hrs, dtype=float)

    try:
        f = h5py.File(out_name, 'r+')
    except:
        f = h5py.File(out_name, 'w')

    try:
        meta = f[grp_name]
    except:
        meta = f.create_group(grp_name)

    meta.create_dataset('ids', data=ids, dtype=int)
    meta.create_dataset('admit', data=admits.astype(bytes), dtype='|S20')
    meta.create_dataset('discharge', data=discharges.astype(bytes), dtype='|S20')
    meta.create_dataset('age', data=ages, dtype=float)
    meta.create_dataset('gender', data=genders, dtype=int)
    meta.create_dataset('race', data=eths, dtype=int)
    meta.create_dataset('bmi', data=bmis, dtype=float)
    meta.create_dataset('weight', data=weights, dtype=float)
    meta.create_dataset('charlson', data=c_scores, dtype=int)
    meta.create_dataset('elixhauser', data=e_scores, dtype=int)
    meta.create_dataset('diabetic', data=diabetics, dtype=int)
    meta.create_dataset('hypertensive', data=hypertensives, dtype=int)
    meta.create_dataset('hosp_los', data=hosp_days, dtype=float)
    meta.create_dataset('hosp_free_days', data=hosp_frees, dtype=float)
    meta.create_dataset('icu_los', data=icu_days, dtype=float)
    meta.create_dataset('icu_free_days', data=icu_frees, dtype=float)
    meta.create_dataset('mv_flag', data=mv_flags, dtype=int)
    meta.create_dataset('mv_days', data=mv_days, dtype=float)
    meta.create_dataset('mv_free_days', data=mv_frees, dtype=float)
    meta.create_dataset('ecmo', data=ecmos, dtype=int)
    meta.create_dataset('iabp', data=iabps, dtype=int)
    meta.create_dataset('vad', data=vads, dtype=int)
    meta.create_dataset('sepsis', data=sepsiss, dtype=int)
    meta.create_dataset('died_inp', data=dieds, dtype=int)
    meta.create_dataset('baseline_scr', data=bsln_scrs, dtype=float)
    meta.create_dataset('baseline_gfr', data=bsln_gfrs, dtype=float)
    meta.create_dataset('baseline_type', data=bsln_types.astype(bytes), dtype='|S8')
    meta.create_dataset('admit_scr', data=admit_scrs, dtype=float)
    meta.create_dataset('peak_scr', data=peak_scrs, dtype=float)
    meta.create_dataset('net_fluid', data=net_fluids, dtype=int)
    meta.create_dataset('gross_fluid', data=gross_fluids, dtype=int)
    meta.create_dataset('fluid_overload', data=fos, dtype=float)
    meta.create_dataset('hd_days', data=hd_dayss, dtype=int)
    meta.create_dataset('crrt_days', data=crrt_dayss, dtype=int)
    meta.create_dataset('hd_free_7d', data=hd_frees_7d, dtype=int)
    meta.create_dataset('crrt_free_7d', data=crrt_frees_7d, dtype=int)
    meta.create_dataset('hd_free_28d', data=hd_frees_28d, dtype=int)
    meta.create_dataset('crrt_free_28d', data=crrt_frees_28d, dtype=int)
    meta.create_dataset('pd_days', data=pd_dayss, dtype=int)
    meta.create_dataset('pd_free_7d', data=pd_frees_7d, dtype=int)
    meta.create_dataset('pd_free_28d', data=pd_frees_28d, dtype=int)

    meta.create_dataset('max_kdigo_7d', data=mks_7d, dtype=int)
    meta.create_dataset('max_kdigo', data=mks, dtype=int)
    meta.create_dataset('n_episodes', data=nepss, dtype=int)
    meta.create_dataset('days_to_death', data=dtds, dtype=float)

    meta.create_dataset('nephrotox_ct', data=nephrotox_cts, dtype=int)
    meta.create_dataset('vasopress_ct', data=vasopress_cts, dtype=int)
    meta.create_dataset('anemia', data=anemics, dtype=bool)
    meta.create_dataset('urine_out', data=urine_outs, dtype=float)
    meta.create_dataset('urine_flow', data=urine_flows, dtype=float)
    meta.create_dataset('smoker', data=smokers, dtype=bool)

    meta.create_dataset('ph', data=phs, dtype=float)
    meta.create_dataset('map', data=maps, dtype=float)
    meta.create_dataset('albumin', data=albs, dtype=float)
    meta.create_dataset('lactate', data=lacs, dtype=float)
    meta.create_dataset('bilirubin', data=bilis, dtype=float)
    meta.create_dataset('bun', data=buns, dtype=float)
    meta.create_dataset('potassium', data=potass, dtype=float)
    meta.create_dataset('pco2', data=coos, dtype=float)
    meta.create_dataset('po2', data=oxs, dtype=float)

    meta.create_dataset('chloride', data=chlorides, dtype=float)
    meta.create_dataset('hematocrit', data=hemats, dtype=float)
    meta.create_dataset('hemoglobin', data=hemos, dtype=float)
    meta.create_dataset('platelets', data=pltlts, dtype=float)
    meta.create_dataset('sodium', data=sods, dtype=float)
    meta.create_dataset('wbc', data=wbcs, dtype=float)
    meta.create_dataset('temperature', data=temps, dtype=float)
    meta.create_dataset('hr', data=hrs, dtype=float)
    meta.create_dataset('glasgow', data=glasgows, dtype=float)
    meta.create_dataset('scr_agg', data=scr_aggs, dtype=float)
    meta.create_dataset('fio2', data=fio2s, dtype=float)
    meta.create_dataset('respiration', data=resps, dtype=float)

    return f


def iqr(d, axis=None):
    m = np.nanmedian(d, axis=axis)
    q25 = np.nanpercentile(d, 25, axis=axis)
    q75 = np.nanpercentile(d, 75, axis=axis)
    return m, q25, q75


def pstring(p, cutoffs=[0.05, 0.01]):
    if p > cutoffs[0]:
        return ''
    for i in range(1, len(cutoffs)):
        if p > cutoffs[i]:
            return '*' * i
    return '*' * len(cutoffs)



# %%
def get_cstats(in_file, label_path, meta_grp='meta', data_path=None, dm=None):
    # get IDs and Clusters in order from cluster file
    # ids = np.loadtfxt(id_file, dtype=int, delimiter=',')
    if type(in_file) == str:
        got_str = True
        f = h5py.File(in_file, 'r')
    else:
        got_str = False
        f = in_file
    meta = f[meta_grp]
    all_ids = meta['ids'][:]
    ids = np.loadtxt(os.path.join(label_path, 'clusters.csv'), delimiter=',', usecols=0, dtype=int)
    sel = np.array([x in ids for x in all_ids])
    
    lbls = load_csv(os.path.join(label_path, 'clusters.csv'), ids, dt=str)

    ages = meta['age'][:][sel]
    genders = meta['gender'][:][sel]
    races = meta['race'][:][sel]
    bmis = meta['bmi'][:][sel]
    charls = meta['charlson'][:][sel]
    elixs = meta['elixhauser'][:][sel]
    diabetics = meta['diabetic'][:][sel]
    hypertensives = meta['hypertensive'][:][sel]

    sofa = meta['sofa'][:][sel]
    apache = meta['apache'][:][sel]
    hosp_days = meta['hosp_los'][:][sel]
    hosp_free = meta['hosp_free_days'][:][sel]
    icu_days = meta['icu_los'][:][sel]
    icu_free = meta['icu_free_days'][:][sel]
    mv_days = meta['mv_days'][:][sel]
    mv_free = meta['mv_free_days'][:][sel]
    mv_flag = meta['mv_flag'][:][sel]
    ecmo_flag = meta['ecmo'][:][sel]
    iabp_flag = meta['iabp'][:][sel]
    vad_flag = meta['vad'][:][sel]

    sepsis = meta['sepsis'][:][sel]
    died_inp = meta['died_inp'][:][sel]

    bsln_scr = meta['baseline_scr'][:][sel]
    bsln_gfr = meta['baseline_gfr'][:][sel]
    admit_scr = meta['admit_scr'][:][sel]
    peak_scr = meta['peak_scr'][:][sel]
    net_fluid = meta['net_fluid'][:][sel]
    gross_fluid = meta['gross_fluid'][:][sel]


    m_kdigos = meta['max_kdigo'][:][sel]
    n_eps = meta['n_episodes'][:][sel]
    hd_days = meta['hd_days'][:][sel]
    crrt_days = meta['crrt_days'][:][sel]
    rrt_tot = hd_days + crrt_days

    neph_cts = meta['nephrotox_ct'][:][sel]
    vaso_cts = meta['vasopress_ct'][:][sel]
    anemia = meta['anemia'][:][sel]
    urine_out = meta['urine_out'][:][sel] / 1000

    lbl_names = np.unique(lbls)

    cluster_data = {'count': {},
                    'died_inpatient': {},
                    'age': {},
                    'gender': {},
                    'ethnicity': {},
                    'bmi': {},
                    'charlson': {},
                    'elixhauser': {},
                    'diabetic': {},
                    'hypertensive': {},
                    'ecmo': {},
                    'iabp': {},
                    'vad': {},
                    'hospital_los': {},
                    'icu_los': {},
                    'sofa': {},
                    'apache': {},
                    'mech_vent': {},
                    'mech_vent_days': {},
                    'nephrotoxins_cts': {},
                    'vasoactive_cts': {},
                    'septic': {},
                    'anemic': {},
                    'baseline_scr': {},
                    'baseline_gfr': {},
                    'admit_scr': {},
                    'peak_scr': {},
                    'urine_output': {},
                    'cumulative_fluid_balance': {},
                    'crrt_days': {},
                    'hd_days': {},
                    'rrt_days_tot': {},
                    }
    
    # c_header = ','.join(['cluster_id', 'count', 'kdigo_0', 'kdigo_1', 'kdigo_2', 'kdigo_3', 'kdigo_3D',  'age_mean', 'age_std',
    #                      'bmi_mean', 'bmi_std', 'pct_male', 'pct_white', 'charlson_mean', 'charlson_std',
    #                      'elixhauser_mean', 'elixhauser_std', 'pct_diabetic', 'pct_hypertensive',
    #                      'sofa_med', 'sofa_25', 'sofa_75', 'apache_med', 'apache_25', 'apache_75',
    #                      'hosp_los_med', 'hosp_los_25', 'hosp_los_75', 'hosp_free_med', 'hosp_free_25', 'hosp_free_75',
    #                      'icu_los_med', 'icu_los_25', 'icu_los_75', 'icu_free_med', 'icu_free_25', 'icu_free_75',
    #                      'pct_ecmo', 'pct_iabp', 'pct_vad', 'pct_mech_vent',
    #                      'mech_vent_days_med', 'mech_vent_days_25', 'mech_vent_days_75',
    #                      'mech_vent_free_med', 'mech_vent_free_25', 'mech_vent_free_75',
    #                      'pct_septic', 'pct_inp_mort',
    #                      'bsln_scr_med', 'bsln_scr_25', 'bsln_scr_75',
    #                      'bsln_gfr_med', 'bsln_gfr_25', 'bsln_gfr_75',
    #                      'admit_scr_med', 'admit_scr_25', 'admit_scr_75',
    #                      'peak_scr_med', 'peak_scr_25', 'peak_scr_75',
    #                      'net_fluid_med', 'net_fluid_25', 'net_fluid_75',
    #                      'gross_fluid_med', 'gross_fluid_25', 'gross_fluid_75',
    #                      'crrt_days_med', 'crrt_days_25', 'crrt_days_75',
    #                      'hd_days_med', 'hd_days_25', 'hd_days_75', '\n'])
    #                      # 'record_len_med', 'record_len_25', 'record_len_75', '\n'])
    #
    # sf = open(os.path.join(label_path, 'cluster_stats.csv'), 'w')
    # sf.write(c_header)
    for i in range(len(lbl_names)):
        cluster_id = lbl_names[i]
        rows = np.where(lbls == cluster_id)[0]
        count = len(rows)
        k_counts = np.zeros(5)
        for j in range(5):
            k_counts[j] = len(np.where(m_kdigos[rows] == j)[0])

        # Outcome
        # Pct Inpatient Mortality, n (%)
        cluster_data['died_inpatient'][cluster_id] = died_inp[rows]

        # Demographic info
        cluster_data['count'][cluster_id] = len(rows)
        # Age - mean(SD)
        cluster_data['age'][cluster_id] = ages[rows]
        # Gender - male n( %)
        cluster_data['gender'][cluster_id] = genders[rows]
        # Ethnic group, white n( %)
        cluster_data['ethnicity'][cluster_id] = races[rows]
        # BMI - mean(SD)
        cluster_data['bmi'][cluster_id] = bmis[rows]

        # Comorbidity
        # Charlson Comorbidty Index Score - mean(SD)
        cluster_data['charlson'][cluster_id] = charls[rows]
        # Elixhauser Comorbidty Index Score - mean(SD)
        cluster_data['elixhauser'][cluster_id] = elixs[rows]
        # Diabetes, n( %)
        cluster_data['diabetic'][cluster_id] = diabetics[rows]
        # Hypertension, n( %)
        cluster_data['hypertensive'][cluster_id] = hypertensives[rows]
        # ECMO, n( %)
        cluster_data['ecmo'][cluster_id] = ecmo_flag[rows]
        # IABP, n( %)
        cluster_data['iabp'][cluster_id] = iabp_flag[rows]
        # VAD, n( %)
        cluster_data['vad'][cluster_id] = vad_flag[rows]

        # Acute Illness
        # Hospital length of stay - median[IQ1 - IQ3]
        cluster_data['hospital_los'][cluster_id] = hosp_days[rows]
        # ICU length of stay - median[IQ1 - IQ3]
        cluster_data['icu_los'][cluster_id] = icu_days[rows]
        # SOFA score - median[IQ1 - IQ3]
        cluster_data['sofa'][cluster_id] = sofa[rows]
        # APACHE Score, - median[IQ1 - IQ3]
        cluster_data['apache'][cluster_id] = apache[rows]
        # Mechanical ventilation, n( %)
        cluster_data['mech_vent'][cluster_id] = mv_flag[rows]
        # Days on mechanical ventilation - mean[IQ1 - IQ3]
        cluster_data['mech_vent_days'][cluster_id] = mv_days[rows][np.where(cluster_data['mech_vent'][cluster_id])]
        # Number of nephrotoxins
        cluster_data['nephrotoxins_cts'][cluster_id] = neph_cts[rows]
        # Number of pressor or inotrope - median[IQ1 - IQ3]
        cluster_data['vasoactive_cts'][cluster_id] = vaso_cts[rows]
        # Sepsis, n( %)
        cluster_data['septic'][cluster_id] = sepsis[rows]
        # Anemia, n( %)
        cluster_data['anemic'][cluster_id] = anemia[rows]

        # AKI characteristics
        # Baseline SCr - median[IQ1 - IQ3]
        cluster_data['baseline_scr'][cluster_id] = bsln_scr[rows]
        # Baseline eGFR - median[IQ1 - IQ3]
        cluster_data['baseline_gfr'][cluster_id] = bsln_gfr[rows]
        # Admit SCr - median[IQ1 - IQ3]
        cluster_data['admit_scr'][cluster_id] = admit_scr[rows]
        # Peak SCr - median[IQ1 - IQ3]
        cluster_data['peak_scr'][cluster_id] = peak_scr[rows]
        # Urine output D0-D1 - median[IQ1 - IQ3]
        cluster_data['urine_output'][cluster_id] = urine_out[rows]
        # Cumulative fluid balance at 72 h - median[IQ1 - IQ3]
        cluster_data['cumulative_fluid_balance'][cluster_id] = net_fluid[rows]

        # Inpatient RRT
        rtsel = np.where(m_kdigos[rows] == 4)[0]
        if rtsel.size > 0:
            # Total days of CRRT - median[IQ1 - IQ3]
            cluster_data['crrt_days'][cluster_id] = crrt_days[rows[rtsel]][np.where(rrt_tot[rows[rtsel]] > 0)]
            # Total days of HD - median[IQ1 - IQ3]
            cluster_data['hd_days'][cluster_id] = hd_days[rows[rtsel]][np.where(rrt_tot[rows[rtsel]] > 0)]
            # Total days of CRRT + HD - median[IQ1 - IQ3]
            cluster_data['rrt_days_tot'][cluster_id] = rrt_tot[rows[rtsel]][np.where(rrt_tot[rows[rtsel]] > 0)]
        else:
            # Total days of CRRT - median[IQ1 - IQ3]
            cluster_data['crrt_days'][cluster_id] = 0
            # Total days of HD - median[IQ1 - IQ3]
            cluster_data['hd_days'][cluster_id] = 0
            # Total days of CRRT + HD - median[IQ1 - IQ3]
            cluster_data['rrt_days_tot'][cluster_id] = 0
    #
    #     age_mean = np.mean(ages[rows])
    #     age_std = np.std(ages[rows])
    #     cluster_data['age'][cluster_id] = ages[rows]
    #
    #     bmi_mean = np.nanmean(bmis[rows])
    #     bmi_std = np.nanstd(bmis[rows])
    #     cluster_data['bmi'][cluster_id] = bmis[rows]
    #
    #     pct_male = float(len(np.where(genders[rows])[0]))
    #     if pct_male > 0:
    #         pct_male /= count
    #     pct_male *= 100
    #
    #     pct_nonwhite = float(len(np.where(races[rows])[0]))
    #     if pct_nonwhite > 0:
    #         pct_white = 1 - (pct_nonwhite / count)
    #     else:
    #         pct_white = 1
    #     pct_white *= 100
    #
    #     charl_sel = np.where(charls[rows] >= 0)[0]
    #     charl_mean = np.nanmean(charls[rows[charl_sel]])
    #     charl_std = np.nanstd(charls[rows[charl_sel]])
    #     cluster_data['charlson'][cluster_id] = charls[rows]
    #
    #     elix_sel = np.where(elixs[rows] >= 0)[0]
    #     elix_mean = np.nanmean(elixs[rows[elix_sel]])
    #     elix_std = np.nanstd(elixs[rows[elix_sel]])
    #     cluster_data['elixhauser'][cluster_id] = elixs[rows]
    #
    #     pct_diabetic = float(len(np.where(diabetics[rows])[0]))
    #     if pct_diabetic > 0:
    #         pct_diabetic /= count
    #     pct_diabetic *= 100
    #     pct_hypertensive = float(len(np.where(hypertensives[rows])[0]))
    #     if pct_hypertensive > 0:
    #         pct_hypertensive /= count
    #     pct_hypertensive *= 100
    #
    #     (sofa_med, sofa_25, sofa_75) = iqr(sofa[rows])
    #     (apache_med, apache_25, apache_75) = iqr(apache[rows])
    #     (hosp_los_med, hosp_los_25, hosp_los_75) = iqr(hosp_days[rows])
    #     (hosp_free_med, hosp_free_25, hosp_free_75) = iqr(hosp_free[rows])
    #     (icu_los_med, icu_los_25, icu_los_75) = iqr(icu_days[rows])
    #     (icu_free_med, icu_free_25, icu_free_75) = iqr(icu_free[rows])
    #
    #     cluster_data['sofa'][cluster_id] = sofa[rows]
    #     cluster_data['apache'][cluster_id] = apache[rows]
    #     cluster_data['hosp_free'][cluster_id] = hosp_free[rows]
    #     cluster_data['icu_free'][cluster_id] = icu_free[rows]
    #
    #     pct_ecmo = float(len(np.where(ecmo_flag[rows])[0]))
    #     if pct_ecmo > 0:
    #         pct_ecmo /= count
    #     pct_ecmo *= 100
    #     pct_iabp = float(len(np.where(iabp_flag[rows])[0]))
    #     if pct_iabp > 0:
    #         pct_iabp /= count
    #     pct_iabp *= 100
    #     pct_vad = float(len(np.where(vad_flag[rows])[0]))
    #     if pct_vad > 0:
    #         pct_vad /= count
    #     pct_vad *= 100
    #
    #     mv_idx = np.where(mv_flag[rows])[0]
    #     pct_mv = float(len(mv_idx))
    #     if pct_mv > 0:
    #         pct_mv /= count
    #     pct_mv *= 100
    #     (mv_days_med, mv_days_25, mv_days_75) = iqr(mv_days[rows[mv_idx]])
    #     (mv_free_med, mv_free_25, mv_free_75) = iqr(mv_free[rows[mv_idx]])
    #     cluster_data['mv_free'][cluster_id] = mv_free[rows]
    #
    #     pct_mort = float(len(np.where(died_inp[rows])[0]))
    #     if pct_mort > 0:
    #         pct_mort /= count
    #     pct_mort *= 100
    #     pct_septic = float(len(np.where(sepsis[rows])[0]))
    #     if pct_septic > 0:
    #         pct_septic /= count
    #     pct_septic *= 100
    #
    #     (bsln_scr_med, bsln_scr_25, bsln_scr_75) = iqr(bsln_scr[rows])
    #     (bsln_gfr_med, bsln_gfr_25, bsln_gfr_75) = iqr(bsln_gfr[rows])
    #     (admit_scr_med, admit_scr_25, admit_scr_75) = iqr(admit_scr[rows])
    #     (peak_scr_med, peak_scr_25, peak_scr_75) = iqr(peak_scr[rows])
    #
    #     cluster_data['bsln_scr'][cluster_id] = bsln_scr[rows]
    #     cluster_data['bsln_gfr'][cluster_id] = bsln_gfr[rows]
    #     cluster_data['admit_scr'][cluster_id] = admit_scr[rows]
    #     cluster_data['peak_scr'][cluster_id] = peak_scr[rows]
    #
    #     fluid_sel = np.where(gross_fluid[rows] >= 0)[0]
    #     (net_fluid_med, net_fluid_25, net_fluid_75) = iqr(net_fluid[rows[fluid_sel]])
    #     (gross_fluid_med, gross_fluid_25, gross_fluid_75) = iqr(gross_fluid[rows[fluid_sel]])
    #
    #     cluster_data['net_fluid'][cluster_id] = net_fluid[rows[fluid_sel]]
    #     cluster_data['gross_fluid'][cluster_id] = gross_fluid[rows[fluid_sel]]
    #
    #     rrt_idx = np.union1d(np.where(hd_days[rows] > 0)[0], np.where(crrt_days[rows] > 0)[0])
    #     (hd_days_med, hd_days_25, hd_days_75) = iqr(hd_days[rows[rrt_idx]])
    #     (crrt_days_med, crrt_days_25, crrt_days_75) = iqr(crrt_days[rows[rrt_idx]])
    #     # (record_len_med, record_len_25, record_len_75) = iqr(rec_lens[rows])
    #
    #     cluster_data['hd_days'][cluster_id] = hd_days[rows[fluid_sel]]
    #     cluster_data['crrt_days'][cluster_id] = crrt_days[rows[fluid_sel]]
    #
    #     sf.write(
    #         '%s,%d,%d,%d,%d,%d,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n' %  # ,%.5f,%.5f,%.5f\n' %
    #         (cluster_id, count, k_counts[0], k_counts[1], k_counts[2], k_counts[3], k_counts[4], age_mean, age_std,
    #          bmi_mean, bmi_std, pct_male, pct_white, charl_mean, charl_std,
    #          elix_mean, elix_std, pct_diabetic, pct_hypertensive,
    #          sofa_med, sofa_25, sofa_75, apache_med, apache_25, apache_75,
    #          hosp_los_med, hosp_los_25, hosp_los_75, hosp_free_med, hosp_free_25, hosp_free_75,
    #          icu_los_med, icu_los_25, icu_los_75, icu_free_med, icu_free_25, icu_free_75,
    #          pct_ecmo, pct_iabp, pct_vad, pct_mv,
    #          mv_days_med, mv_days_25, mv_days_75,
    #          mv_free_med, mv_free_25, mv_free_75,
    #          pct_septic, pct_mort,
    #          bsln_scr_med, bsln_scr_25, bsln_scr_75,
    #          bsln_gfr_med, bsln_gfr_25, bsln_gfr_75,
    #          admit_scr_med, admit_scr_25, admit_scr_75,
    #          peak_scr_med, peak_scr_25, peak_scr_75,
    #          net_fluid_med, net_fluid_25, net_fluid_75,
    #          gross_fluid_med, gross_fluid_25, gross_fluid_75,
    #          crrt_days_med, crrt_days_25, crrt_days_75,
    #          hd_days_med, hd_days_25, hd_days_75))
    # sf.close()
    # Generate formatted table as text
    o = np.argsort([np.sum(cluster_data['died_inpatient'][x]) for x in lbl_names])
    lbl_names = lbl_names[o]

    table_f = open(os.path.join(label_path, 'formatted_table.csv'), 'w')
    hdr = ',Total'
    for lbl in lbl_names:
        hdr += ',%s' % lbl
    table_f.write(hdr + '\n')

    row = 'Count,%d' % len(lbls)
    for lbl in lbl_names:
        row += ',%d' % cluster_data['count'][lbl]
    table_f.write(row + '\n')

    row = 'Inpatient Mortality - n (%%), %d (%.2f%%)' % (np.sum(died_inp > 0), np.sum(died_inp > 0)/len(lbls)*100)
    for lbl in lbl_names:
        _, p = ttest_ind(died_inp, cluster_data['died_inpatient'][lbl])
        row += ', %d (%.2f%%)' % (np.sum(cluster_data['died_inpatient'][lbl] > 0),
                                      np.sum(cluster_data['died_inpatient'][lbl] > 0) / np.sum(cluster_data['count'][lbl])*100)
    table_f.write(row + '\n')

    # Demographics
    table_f.write('Demographics\n')
    # Age - mean(SD)
    row = 'Age - mean(SD), %.1f (%.1f)' % (np.nanmean(ages), np.nanstd(ages))
    for lbl in lbl_names:
        _, p = ttest_ind(ages, cluster_data['age'][lbl])
        row += ', %.1f (%.1f)' % (np.nanmean(cluster_data['age'][lbl]),
                                      np.nanstd(cluster_data['age'][lbl]))
    table_f.write(row + '\n')

    # Gender - male n( %)
    row = 'Gender - male n(%%), %d (%.2f%%)' % (np.sum(genders), np.sum(genders)/len(lbls)*100)
    for lbl in lbl_names:
        _, p = ttest_ind(genders, cluster_data['gender'][lbl])
        row += ', %d (%.2f%%)' % (np.sum(cluster_data['gender'][lbl]),
                                        np.sum(cluster_data['gender'][lbl]) / np.sum(
                                            cluster_data['count'][lbl])*100)
    table_f.write(row + '\n')

    # Ethnic group - white n( %)
    row = 'Ethnic group - white n(%%), %d (%.2f%%)' % (np.sum(races), np.sum(races)/len(lbls)*100)
    for lbl in lbl_names:
        _, p = ttest_ind(races, cluster_data['ethnicity'][lbl])
        row += ', %d (%.2f%%)' % (np.sum(cluster_data['ethnicity'][lbl]),
                                    np.sum(cluster_data['ethnicity'][lbl]) / np.sum(
                                        cluster_data['count'][lbl])*100)
    table_f.write(row + '\n')

    # BMI - mean(SD)
    row = 'BMI - mean(SD), %.1f (%.1f)' % (np.nanmean(bmis), np.nanstd(bmis))
    for lbl in lbl_names:
        _, p = ttest_ind(bmis, cluster_data['bmi'][lbl])
        row += ', %.1f (%.1f)' % (np.nanmean(cluster_data['bmi'][lbl]),
                                      np.nanstd(cluster_data['bmi'][lbl]))
    table_f.write(row + '\n')


    # Comorbidity
    table_f.write('Comorbidity\n')
    # Charlson Comorbidty Index Score - mean(SD)
    row = 'Charlson Comorbidty Index Score - mean(SD), %.1f (%.1f)' % (np.nanmean(charls[np.where(charls > 0)]), np.nanstd(charls[np.where(charls > 0)]))
    all_sel = charls[np.where(charls > 0)]
    for lbl in lbl_names:
        t_sel = cluster_data['charlson'][lbl][np.where(cluster_data['charlson'][lbl] > 0)]
        _, p = ttest_ind(all_sel, t_sel)
        row += ', %.1f (%.1f)' % (np.nanmean(t_sel), np.nanstd(t_sel))
    table_f.write(row + '\n')

    # Elixhauser Comorbidty Index Score - mean(SD)
    row = 'Elixhauser Comorbidty Index Score - mean(SD), %.1f (%.1f)' % (np.nanmean(elixs[np.where(elixs > 0)]), np.nanstd(elixs[np.where(elixs > 0)]))
    all_sel = elixs[np.where(elixs > 0)]
    for lbl in lbl_names:
        t_sel = cluster_data['elixhauser'][lbl][np.where(cluster_data['elixhauser'][lbl] > 0)]
        _, p = ttest_ind(all_sel, t_sel)
        row += ', %.1f (%.1f)' % (np.nanmean(t_sel), np.nanstd(t_sel))
    table_f.write(row + '\n')

    # Diabetes - n( %)
    row = 'Diabetes - n(%%), %d (%.2f%%)' % (np.sum(diabetics), np.sum(diabetics)/len(lbls)*100)
    for lbl in lbl_names:
        _, p = ttest_ind(diabetics, cluster_data['diabetic'][lbl])
        row += ', %d (%.2f%%)' % (np.sum(cluster_data['diabetic'][lbl]),
                                    np.sum(cluster_data['diabetic'][lbl]) / np.sum(
                                        cluster_data['count'][lbl])*100)
    table_f.write(row + '\n')

    # Hypertension - n( %)
    row = 'Hypertension - n(%%), %d (%.2f%%)' % (np.sum(hypertensives), np.sum(hypertensives)/len(lbls)*100)
    for lbl in lbl_names:
        _, p = ttest_ind(hypertensives, cluster_data['hypertensive'][lbl])
        row += ', %d (%.2f%%)' % (np.sum(cluster_data['hypertensive'][lbl]),
                                    np.sum(cluster_data['hypertensive'][lbl]) / np.sum(
                                        cluster_data['count'][lbl])*100)
    table_f.write(row + '\n')

    # ECMO - n( %)
    row = 'ECMO - n(%%), %d (%.2f%%)' % (np.sum(ecmo_flag), np.sum(ecmo_flag)/len(lbls)*100)
    for lbl in lbl_names:
        _, p = ttest_ind(ecmo_flag, cluster_data['ecmo'][lbl])
        row += ', %d (%.2f%%)' % (np.sum(cluster_data['ecmo'][lbl]),
                                    np.sum(cluster_data['ecmo'][lbl]) / np.sum(
                                        cluster_data['count'][lbl])*100)
    table_f.write(row + '\n')

    # IABP - n( %)
    row = 'IABP - n(%%), %d (%.2f%%)' % (np.sum(iabp_flag), np.sum(iabp_flag)/len(lbls)*100)
    for lbl in lbl_names:
        _, p = ttest_ind(iabp_flag, cluster_data['iabp'][lbl])
        row += ', %d (%.2f%%)' % (np.sum(cluster_data['iabp'][lbl]),
                                    np.sum(cluster_data['iabp'][lbl]) / cluster_data['count'][lbl]*100)
    table_f.write(row + '\n')

    # VAD - n( %)
    row = 'VAD - n(%%), %d (%.2f%%)' % (np.sum(vad_flag), np.sum(vad_flag)/len(lbls)*100)
    for lbl in lbl_names:
        _, p = ttest_ind(vad_flag, cluster_data['vad'][lbl])
        row += ', %d (%.2f%%)' % (np.sum(cluster_data['vad'][lbl]),
                                    np.sum(cluster_data['vad'][lbl]) / cluster_data['count'][lbl]*100)
    table_f.write(row + '\n')

    # Acute Illness
    table_f.write('Acute illness characteristics\n')
    # Hospital length of stay - median[IQ1 - IQ3]
    m, q1, q3 = iqr(hosp_days)
    row = 'Hospital length of stay - median[IQ1 - IQ3], %.1f (%.1f - %.1f)' % (m, q1, q3)
    for lbl in lbl_names:
        _, p = ttest_ind(hosp_days, cluster_data['hospital_los'][lbl])
        m, q1, q3 = iqr(cluster_data['hospital_los'][lbl])
        row += ', %.1f (%.1f - %.1f)' % (m, q1, q3)
    table_f.write(row + '\n')

    # ICU length of stay - median[IQ1 - IQ3]
    m, q1, q3 = iqr(icu_days)
    row = 'ICU length of stay - median[IQ1 - IQ3], %.1f (%.1f - %.1f)' % (m, q1, q3)
    for lbl in lbl_names:
        _, p = ttest_ind(icu_days, cluster_data['icu_los'][lbl])
        m, q1, q3 = iqr(cluster_data['icu_los'][lbl])
        row += ', %.1f (%.1f - %.1f)' % (m, q1, q3)
    table_f.write(row + '\n')

    # SOFA score - median[IQ1 - IQ3]
    m, q1, q3 = iqr(sofa)
    row = 'SOFA score - median[IQ1 - IQ3], %.1f (%.1f - %.1f)' % (m, q1, q3)
    for lbl in lbl_names:
        _, p = ttest_ind(sofa, cluster_data['sofa'][lbl])
        m, q1, q3 = iqr(cluster_data['sofa'][lbl])
        row += ', %.1f (%.1f - %.1f)' % (m, q1, q3)
    table_f.write(row + '\n')

    # APACHE Score, - median[IQ1 - IQ3]
    m, q1, q3 = iqr(apache)
    row = 'APACHE score - median[IQ1 - IQ3], %.1f (%.1f - %.1f)' % (m, q1, q3)
    for lbl in lbl_names:
        _, p = ttest_ind(apache, cluster_data['apache'][lbl])
        m, q1, q3 = iqr(cluster_data['apache'][lbl])
        row += ', %.1f (%.1f - %.1f)' % (m, q1, q3)
    table_f.write(row + '\n')

    # Mechanical ventilation - n( %)
    row = 'Mechanical ventilation - n(%%), %d (%.2f%%)' % (np.sum(mv_flag), np.sum(mv_flag)/len(lbls)*100)
    for lbl in lbl_names:
        _, p = ttest_ind(mv_flag, cluster_data['mech_vent'][lbl])
        row += ', %d (%.2f%%)' % (np.sum(cluster_data['mech_vent'][lbl]),
                                    np.sum(cluster_data['mech_vent'][lbl]) / cluster_data['count'][lbl]*100)
    table_f.write(row + '\n')

    # Days on mechanical ventilation - mean[IQ1 - IQ3]
    all_mv = mv_days[np.where(mv_flag)]
    m, q1, q3 = iqr(all_mv)
    row = 'Days on mechanical ventilation - mean[IQ1 - IQ3], %.1f (%.1f - %.1f)' % (m, q1, q3)
    for lbl in lbl_names:
        _, p = ttest_ind(all_mv, cluster_data['mech_vent_days'][lbl])
        m, q1, q3 = iqr(cluster_data['mech_vent_days'][lbl])
        row += ', %.1f (%.1f - %.1f)' % (m, q1, q3)
    table_f.write(row + '\n')

    # Number w/ >=1 Nephrotoxins D0-1 - n (%)
    row = 'Number w/ >=1 Nephrotoxins D0-1 - n (%%), %d (%.2f%%)' % (np.sum(neph_cts >= 1), np.sum(neph_cts >= 1)/len(lbls)*100)
    all_ct = np.array(neph_cts >= 1)
    for lbl in lbl_names:
        tcount = np.array(cluster_data['nephrotoxins_cts'][lbl] >= 1)
        _, p = ttest_ind(all_ct, tcount)
        row += ', %d (%.2f%%)' % (np.sum(tcount),
                                    np.sum(tcount) / cluster_data['count'][lbl]*100)
    table_f.write(row + '\n')
    # Number w/ >=2 Nephrotoxins D0-1 - n (%)
    row = 'Number w/ >=2 Nephrotoxins D0-1 - n (%%), %d (%.2f%%)' % (np.sum(neph_cts >= 2), np.sum(neph_cts >= 2)/len(lbls)*100)
    all_ct = np.array(neph_cts >= 2)
    for lbl in lbl_names:
        tcount = np.array(cluster_data['nephrotoxins_cts'][lbl] >= 2)
        _, p = ttest_ind(all_ct, tcount)
        row += ', %d (%.2f%%)' % (np.sum(tcount),
                                    np.sum(tcount) / cluster_data['count'][lbl]*100)
    table_f.write(row + '\n')
    # Number w/ >=3 Nephrotoxins D0-1 - n (%)
    row = 'Number w/ >=3 Nephrotoxins D0-1 - n (%%), %d (%.2f%%)' % (np.sum(neph_cts >= 3), np.sum(neph_cts >= 3)/len(lbls)*100)
    all_ct = np.array(neph_cts >= 3)
    for lbl in lbl_names:
        tcount = np.array(cluster_data['nephrotoxins_cts'][lbl] >= 3)
        _, p = ttest_ind(all_ct, tcount)
        row += ', %d (%.2f%%)' % (np.sum(tcount),
                                    np.sum(tcount) / cluster_data['count'][lbl]*100)
    table_f.write(row + '\n')
    # Number w/ >=1 Vasoactive Medication D0-1 - n (%)
    row = 'Number w/ >=1 Vasoactive Medication D0-1 - n (%%), %d (%.2f%%)' % (np.sum(vaso_cts >= 1), np.sum(vaso_cts >= 1)/len(lbls)*100)
    all_ct = np.array(vaso_cts >= 1)
    for lbl in lbl_names:
        tcount = np.array(cluster_data['vasoactive_cts'][lbl] >= 1)
        _, p = ttest_ind(all_ct, tcount)
        row += ', %d (%.2f%%)' % (np.sum(tcount),
                                    np.sum(tcount) / cluster_data['count'][lbl]*100)
    table_f.write(row + '\n')
    # Number w/ >=2 Vasoactive Medication D0-1 - n (%)
    row = 'Number w/ >=2 Vasoactive Medication D0-1 - n (%%), %d (%.2f%%)' % (np.sum(vaso_cts >= 2), np.sum(vaso_cts >= 2)/len(lbls)*100)
    all_ct = np.array(vaso_cts >= 2)
    for lbl in lbl_names:
        tcount = np.array(cluster_data['vasoactive_cts'][lbl] >= 2)
        _, p = ttest_ind(all_ct, tcount)
        row += ', %d (%.2f%%)' % (np.sum(tcount),
                                    np.sum(tcount) / cluster_data['count'][lbl]*100)
    table_f.write(row + '\n')
    # Number w/ >=3 Vasoactive Medication D0-1 - n (%)
    row = 'Number w/ >=3 Vasoactive Medication D0-1 - n (%%), %d (%.2f%%)' % (np.sum(vaso_cts >= 3), np.sum(vaso_cts >= 3)/len(lbls)*100)
    all_ct = np.array(vaso_cts >= 3)
    for lbl in lbl_names:
        tcount = np.array(cluster_data['vasoactive_cts'][lbl] >= 3)
        _, p = ttest_ind(all_ct, tcount)
        row += ', %d (%.2f%%)' % (np.sum(tcount),
                                    np.sum(tcount) / cluster_data['count'][lbl]*100)
    table_f.write(row + '\n')

    # Sepsis - n( %)
    row = 'Sepsis - n(%%), %d (%.2f%%)' % (np.sum(sepsis), np.sum(sepsis)/len(lbls)*100)
    for lbl in lbl_names:
        _, p = ttest_ind(sepsis, cluster_data['septic'][lbl])
        row += ', %d (%.2f%%)' % (np.sum(cluster_data['septic'][lbl]),
                                    np.sum(cluster_data['septic'][lbl]) / cluster_data['count'][lbl]*100)
    table_f.write(row + '\n')

    # Anemia - n( %)
    row = 'Anemia - n(%%), %d (%.2f%%)' % (np.sum(anemia), np.sum(anemia)/len(lbls)*100)
    for lbl in lbl_names:
        _, p = ttest_ind(anemia, cluster_data['anemic'][lbl])
        row += ', %d (%.2f%%)' % (np.sum(cluster_data['anemic'][lbl]),
                                    np.sum(cluster_data['anemic'][lbl]) / cluster_data['count'][lbl]*100)
    table_f.write(row + '\n')

    # AKI Characteristics
    table_f.write('AKI characteristics\n')
    # Baseline SCr - median[IQ1 - IQ3]
    m, q1, q3 = iqr(bsln_scr)
    row = 'Baseline SCr - median[IQ1 - IQ3], %.1f (%.1f - %.1f)' % (m, q1, q3)
    for lbl in lbl_names:
        _, p = ttest_ind(bsln_scr, cluster_data['baseline_scr'][lbl])
        m, q1, q3 = iqr(cluster_data['baseline_scr'][lbl])
        row += ', %.1f (%.1f - %.1f)' % (m, q1, q3)
    table_f.write(row + '\n')

    # Baseline eGFR - median[IQ1 - IQ3]
    m, q1, q3 = iqr(bsln_gfr)
    row = 'Baseline eGFR - median[IQ1 - IQ3], %.1f (%.1f - %.1f)' % (m, q1, q3)
    for lbl in lbl_names:
        _, p = ttest_ind(bsln_gfr, cluster_data['baseline_gfr'][lbl])
        m, q1, q3 = iqr(cluster_data['baseline_gfr'][lbl])
        row += ', %.1f (%.1f - %.1f)' % (m, q1, q3)
    table_f.write(row + '\n')

    # Admit SCr - median[IQ1 - IQ3]
    m, q1, q3 = iqr(admit_scr)
    row = 'Admit SCr - median[IQ1 - IQ3], %.1f (%.1f - %.1f)' % (m, q1, q3)
    for lbl in lbl_names:
        _, p = ttest_ind(admit_scr, cluster_data['admit_scr'][lbl])
        m, q1, q3 = iqr(cluster_data['admit_scr'][lbl])
        row += ', %.1f (%.1f - %.1f)' % (m, q1, q3)
    table_f.write(row + '\n')

    # Peak SCr - median[IQ1 - IQ3]
    m, q1, q3 = iqr(peak_scr)
    row = 'Peak SCr - median[IQ1 - IQ3], %.1f (%.1f - %.1f)' % (m, q1, q3)
    for lbl in lbl_names:
        _, p = ttest_ind(peak_scr, cluster_data['peak_scr'][lbl])
        m, q1, q3 = iqr(cluster_data['peak_scr'][lbl])
        row += ', %.1f (%.1f - %.1f)' % (m, q1, q3)
    table_f.write(row + '\n')

    # Urine output D0-D1 - median[IQ1 - IQ3]
    m, q1, q3 = iqr(urine_out)
    row = 'Urine output D0-D1 - median[IQ1 - IQ3], %.1f (%.1f - %.1f)' % (m, q1, q3)
    for lbl in lbl_names:
        _, p = ttest_ind(urine_out, cluster_data['urine_output'][lbl])
        m, q1, q3 = iqr(cluster_data['urine_output'][lbl])
        row += ', %.1f (%.1f - %.1f)' % (m, q1, q3)
    table_f.write(row + '\n')

    # Cumulative fluid balance at 72 h - median[IQ1 - IQ3]
    m, q1, q3 = iqr(net_fluid/1000)
    row = 'Cumulative fluid balance at 72 h - median[IQ1 - IQ3], %.1f (%.1f - %.1f)' % (m, q1, q3)
    for lbl in lbl_names:
        _, p = ttest_ind(net_fluid, cluster_data['cumulative_fluid_balance'][lbl])
        m, q1, q3 = iqr(cluster_data['cumulative_fluid_balance'][lbl]/1000)
        row += ', %.1f (%.1f - %.1f)' % (m, q1, q3)
    table_f.write(row + '\n')

    # Inpatient RRT
    table_f.write('Inpatient RRT characteristics\n')
    # Total days of CRRT - median[IQ1 - IQ3]
    all_sel = crrt_days[np.where(rrt_tot > 0)]
    m, q1, q3 = iqr(all_sel)
    row = 'Total days of CRRT - median[IQ1 - IQ3], %.1f (%.1f - %.1f)' % (m, q1, q3)
    for lbl in lbl_names:
        _, p = ttest_ind(all_sel, cluster_data['crrt_days'][lbl])
        m, q1, q3 = iqr(cluster_data['crrt_days'][lbl])
        row += ', %.1f (%.1f - %.1f)' % (m, q1, q3)
    table_f.write(row + '\n')

    # Total days of HD - median[IQ1 - IQ3]
    all_sel = hd_days[np.where(rrt_tot > 0)]
    m, q1, q3 = iqr(all_sel)
    row = 'Total days of HD - median[IQ1 - IQ3], %.1f (%.1f - %.1f)' % (m, q1, q3)
    for lbl in lbl_names:
        _, p = ttest_ind(all_sel, cluster_data['hd_days'][lbl])
        m, q1, q3 = iqr(cluster_data['hd_days'][lbl])
        row += ', %.1f (%.1f - %.1f)' % (m, q1, q3)
    table_f.write(row + '\n')

    # Total days of CRRT + HD - median[IQ1 - IQ3]
    all_sel = rrt_tot[np.where(rrt_tot > 0)]
    m, q1, q3 = iqr(all_sel)
    row = 'Total days of CRRT + HD - median[IQ1 - IQ3], %.1f (%.1f - %.1f)' % (m, q1, q3)
    for lbl in lbl_names:
        _, p = ttest_ind(all_sel, cluster_data['rrt_days_tot'][lbl])
        m, q1, q3 = iqr(cluster_data['rrt_days_tot'][lbl])
        row += ', %.1f (%.1f - %.1f)' % (m, q1, q3)
    table_f.write(row + '\n')
    table_f.close()

    # pairwise_stats(meta, label_path)
    # if data_path is not None:
    #     if dm is None:
    #         print('Data-path provided but no distance matrix. Daily KDIGO not plotted.')
    #     else:
    #         plot_daily_kdigos(data_path, ids, in_file, dm, lbls, outpath=os.path.join(label_path, 'daily_kdigo'))


def build_str(all_data, lbls, summary_type='count', fmt='%.2f'):
    tot_size = len(lbls)
    lbl_names = np.unique(lbls)
    if len(lbl_names) == 2:
        idx = np.where(lbls == lbl_names[0])[0]
        d1 = all_data[idx].flatten()
        idx = np.where(lbls == lbl_names[1])[0]
        d2 = all_data[idx].flatten()
        d1 = d1[np.logical_not(np.isnan(d1))]
        d2 = d2[np.logical_not(np.isnan(d2))]
        p = 1
        if len(d1[np.logical_not(np.isnan(d1))]) >= 8 and len(d2) >= 8:
            _, np1 = normaltest(d1)
            _, np2 = normaltest(d2)
            if np1 < 0.05 or np2 < 0.05:
                try:
                    _, p = kruskal(d1, d2)
                except ValueError:
                    p = 1
            else:
                _, p = ttest_ind(d1, d2)
        if p < 0.01:
            pstr = '**'
        elif p < 0.05:
            pstr = '*'
        else:
            pstr = ''

    if summary_type == 'count':
        tc = len(np.where(all_data)[0])
        s = (',%d (' + fmt + ')') % (tc, float(tc) / tot_size * 100)
        for i in range(len(lbl_names)):
            idx = np.where(lbls == lbl_names[i])[0]
            tc = len(np.where(all_data[idx])[0])
            if len(lbl_names) == 2 and i == 1:
                s += (',%d%s (' + fmt + ')') % (tc, pstr, float(tc) / len(idx) * 100)
            else:
                s += (',%d (' + fmt + ')') % (tc, float(tc) / len(idx) * 100)
    elif summary_type == 'countrow':
        wc = len(np.where(all_data)[0])
        s = (',%d (' + fmt + ')') % (wc, float(wc) / tot_size * 100)
        for i in range(len(lbl_names)):
            idx = np.where(lbls == lbl_names[i])[0]
            tc = len(np.where(all_data[idx])[0])
            if len(lbl_names) == 2 and i == 1:
                if wc > 0:
                    s += (',%d%s (' + fmt + ')') % (tc, pstr, float(tc) / wc * 100)
                else:
                    s += (',%d%s (' + fmt + ')') % (0, pstr, 0)
            else:
                if wc > 0:
                    s += (',%d (' + fmt + ')') % (tc, float(tc) / wc * 100)
                else:
                    s += (',%d (' + fmt + ')') % (0, 0)
    elif summary_type == 'mean':
        s = (',' + fmt + ' (' + fmt + ')') % (np.nanmean(all_data), np.nanstd(all_data))
        for i in range(len(lbl_names)):
            idx = np.where(lbls == lbl_names[i])[0]
            if len(lbl_names) == 2 and i == 1:
                s += (',' + fmt + '%s (' + fmt + ')') % (
                      np.nanmean(all_data[idx].flatten()), pstr, np.nanstd(all_data[idx].flatten()))
            else:
                s += (',' + fmt + ' (' + fmt + ')') % (np.nanmean(all_data[idx].flatten()), np.nanstd(all_data[idx].flatten()))

    elif summary_type == 'median':
        m, iq1, iq2 = iqr(all_data)
        s = (',' + fmt + ' (' + fmt + '- ' + fmt + ')') % (m, iq1, iq2)
        for i in range(len(lbl_names)):
            idx = np.where(lbls == lbl_names[i])[0]
            m, iq1, iq2 = iqr(all_data[idx].flatten())
            if len(lbl_names) == 2 and i == 1:
                s += (',' + fmt + '%s (' + fmt + '- ' + fmt + ')') % (m, pstr, iq1, iq2)
            else:
                s += (',' + fmt + ' (' + fmt + '- ' + fmt + ')') % (m, iq1, iq2)

    if len(lbl_names) == 2:
        idx = np.where(lbls == lbl_names[0])[0]
        d1 = all_data[idx].flatten()
        idx = np.where(lbls == lbl_names[1])[0]
        d2 = all_data[idx].flatten()
        d1 = d1[np.logical_not(np.isnan(d1))]
        d2 = d2[np.logical_not(np.isnan(d2))]
        if len(d1[np.logical_not(np.isnan(d1))]) >= 8 and len(d2) >= 8:
            _, p = normaltest(all_data[np.logical_not(np.isnan(all_data))])
            s += ',%.2E' % p
            _, p = normaltest(d1)
            s += ',%.2E' % p
            _, p = normaltest(d2)
            s += ',%.2E' % p
            _, p = ttest_ind(d1, d2)
            s += ',%.2E' % p
            try:
                _, p = kruskal(d1, d2)
            except ValueError:
                p = 1
            s += ',%.2E' % p
    return s


# %%
def formatted_stats(meta, label_path):
    # get IDs and Clusters in order from cluster file
    # ids = np.loadtfxt(id_file, dtype=int, delimiter=',')
    all_ids = meta['ids'][:]
    try:
        ids = np.loadtxt(os.path.join(label_path, 'labels.csv'), delimiter=',', usecols=0, dtype=int)
        sel = np.array([x in ids for x in all_ids])
        lbls = load_csv(os.path.join(label_path, 'labels.csv'), ids, dt=str)
        lbl_names = np.unique(lbls)
        # assert len(lbl_names) == 2
    except IOError:
        ids = np.loadtxt(os.path.join(label_path, 'clusters.csv'), delimiter=',', usecols=0, dtype=int)
        sel = np.array([x in ids for x in all_ids])
        lbls = load_csv(os.path.join(label_path, 'clusters.csv'), ids, dt=str)
        lbl_names = np.unique(lbls)

    ages = meta['age'][:][sel]
    genders = meta['gender'][:][sel]
    # Switch race so that 1=white and 0=not
    races = meta['race'][:][sel]
    races[np.where(races)] = 2
    races[np.where(races == 0)] = 1
    races[np.where(races == 2)] = 0
    bmis = meta['bmi'][:][sel]
    charls = meta['charlson'][:][sel]
    elixs = meta['elixhauser'][:][sel]
    diabetics = meta['diabetic'][:][sel]
    hypertensives = meta['hypertensive'][:][sel]
    maps = meta['map'][:][sel]

    sofa = meta['sofa'][:][sel]
    apache = meta['apache'][:][sel]
    hosp_days = meta['hosp_los'][:][sel]
    hosp_free = meta['hosp_free_days'][:][sel]
    icu_days = meta['icu_los'][:][sel]
    icu_free = meta['icu_free_days'][:][sel]
    mv_days = meta['mv_days'][:][sel]
    mv_free = meta['mv_free_days'][:][sel]
    mv_flag = meta['mv_flag'][:][sel]
    ecmo_flag = meta['ecmo'][:][sel]
    iabp_flag = meta['iabp'][:][sel]
    vad_flag = meta['vad'][:][sel]
    max_kdigos = meta['max_kdigo'][:][sel]

    sepsis = meta['sepsis'][:][sel]
    died_inp = meta['died_inp'][:][sel]

    bsln_types = meta['baseline_type'][:][sel].astype('U18')
    bsln_scr = meta['baseline_scr'][:][sel]
    bsln_gfr = meta['baseline_gfr'][:][sel]
    admit_scr = meta['admit_scr'][:][sel]
    peak_scr = meta['peak_scr'][:][sel]
    net_fluid = meta['net_fluid'][:][sel]
    gross_fluid = meta['gross_fluid'][:][sel]
    fluid_overload = meta['fluid_overload'][:][sel]

    mks = meta['max_kdigo'][:][sel]
    mks7d = meta['max_kdigo_7d'][:][sel]
    n_eps = meta['n_episodes'][:][sel]
    hd_days = meta['hd_days'][:][sel]
    hd_trtmts = meta['hd_treatments'][:][sel]
    crrt_days = meta['crrt_days'][:][sel]
    rrt_tot = hd_days + crrt_days
    hd_free_7d = meta['hd_free_7d'][:][sel]
    hd_free_28d = meta['hd_free_28d'][:][sel]
    crrt_free_7d = meta['crrt_free_7d'][:][sel]
    crrt_free_28d = meta['crrt_free_28d'][:][sel]

    neph_cts = meta['nephrotox_ct'][:][sel]
    vaso_cts = meta['vasopress_ct'][:][sel]
    anemia = meta['anemia'][:][sel]
    urine_out = meta['urine_out'][:][sel] / 1000
    if urine_out.ndim == 2:
        urine_out = np.mean(urine_out, axis=1)
    urine_flow = meta['urine_flow'][:][sel]
    if urine_flow.ndim == 2:
        urine_flow = np.mean(urine_flow, axis=1)
    smokers = meta['smoker'][:][sel]

    albs = meta['albumin'][:][sel]
    lacs = meta['lactate'][:][sel]
    bilis = meta['bilirubin'][:][sel]
    buns = meta['bun'][:][sel]
    phs = meta['ph'][:][sel]

    hosp_free[np.where(died_inp)] = 0
    icu_free[np.where(died_inp)] = 0
    mv_free[np.where(died_inp)] = 0
    hd_free_7d[np.where(died_inp)] = 0
    hd_free_28d[np.where(died_inp)] = 0
    crrt_free_7d[np.where(died_inp)] = 0
    crrt_free_28d[np.where(died_inp)] = 0


    # Generate formatted table as text
    table_f = open(os.path.join(label_path, 'formatted_table.csv'), 'w')
    if len(lbl_names) == 2:
        hdr = ',Total,%s,%s,Tot_Normal,Neg_Normal,Pos_Normal,T-test_pval,Kruskal_pval' % (lbl_names[0], lbl_names[1])
    else:
        hdr = ',Total'
        for i in range(len(lbl_names)):
            hdr += ',%s' % lbl_names[i]
    table_f.write(hdr + '\n')

    row = 'Count,%d' % len(lbls)
    for lbl in lbl_names:
        row += ',%d' % len(np.where(lbls == lbl)[0])
    table_f.write(row + '\n')

    s = build_str(died_inp, lbls, 'count', '%.2f')

    row = 'Inpatient Mortality - n (%)' + s
    table_f.write(row + '\n')

    # Demographics
    table_f.write('Demographics\n')
    row = 'Age - mean(SD)'
    s = build_str(ages, lbls, 'mean', '%.1f')

    row += s
    table_f.write(row + '\n')

    # Gender - male n( %)
    row = 'Gender - male n(%)'
    s = build_str(genders, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')

    # Ethnic group - white n( %)
    row = 'Ethnic group - white n(%)'
    s = build_str(races, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')

    # BMI - mean(SD)
    row = 'BMI - mean(SD)'
    s = build_str(bmis, lbls, 'mean', '%.1f')
    row += s

    table_f.write(row + '\n')

    # Comorbidity
    table_f.write('Comorbidity\n')
    row = 'Charlson Comorbidty Index Score - mean(SD)'
    s = build_str(charls[np.where(charls > 0)], lbls[np.where(charls > 0)], 'median', '%.1f')
    row += s

    table_f.write(row + '\n')

    # Elixhauser Comorbidty Index Score - mean(SD)
    row = 'Elixhauser Comorbidty Index Score - mean(SD)'
    s = build_str(elixs[np.where(elixs > 0)], lbls[np.where(elixs > 0)], 'median', '%.1f')
    row += s

    table_f.write(row + '\n')

    # Diabetes - n( %)
    row = 'Diabetes - n(%)'
    s = build_str(diabetics, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')

    # Hypertension - n( %)
    row = 'Hypertension - n(%)'
    s = build_str(hypertensives, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')

    # Smoker status - n( %)
    row = 'Smoker status - n(%)'
    s = build_str(smokers, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')

    # Anemia Def A - n( %)
    row = 'Anemia Definition A - n(%)'
    s = build_str(anemia[:, 0], lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')
    # Anemia Def B - n( %)
    row = 'Anemia Definition B - n(%)'
    s = build_str(anemia[:, 1], lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')
    # Anemia Def C - n( %)
    row = 'Anemia Definition C - n(%)'
    s = build_str(anemia[:, 2], lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')


    # Acuity of Critical Illness
    table_f.write('Acuity of Critical Illness\n')
    # Hospital length of stay - median[IQ1 - IQ3]
    row = 'Hospital free days - median[IQ1 - IQ3]'
    s = build_str(hosp_free, lbls, 'median', '%.1f')
    row += s

    table_f.write(row + '\n')

    # ICU length of stay - median[IQ1 - IQ3]
    row = 'ICU free days - median[IQ1 - IQ3]'
    s = build_str(icu_free, lbls, 'median', '%.1f')
    row += s

    table_f.write(row + '\n')

    # SOFA score - median[IQ1 - IQ3]
    row = 'SOFA score - median[IQ1 - IQ3]'
    s = build_str(sofa, lbls, 'median', '%.1f')
    row += s

    table_f.write(row + '\n')

    # APACHE Score, - median[IQ1 - IQ3]
    row = 'APACHE score - median[IQ1 - IQ3]'
    s = build_str(apache, lbls, 'median', '%.1f')
    row += s

    table_f.write(row + '\n')

    # Mechanical ventilation - n( %)
    row = 'Mechanical ventilation - n(%)'
    s = build_str(mv_flag, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')

    # Days on mechanical ventilation - mean[IQ1 - IQ3]
    row = 'Mechanical ventilation free days - mean[IQ1 - IQ3]'
    s = build_str(mv_free[np.where(mv_flag)], lbls[np.where(mv_flag)], 'median', '%.1f')
    row += s

    table_f.write(row + '\n')

    # ECMO - n( %)
    row = 'ECMO - n(%)'
    s = build_str(ecmo_flag, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')

    # IABP - n( %)
    row = 'IABP - n(%)'
    s = build_str(iabp_flag, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')

    # VAD - n( %)
    row = 'VAD - n(%)'
    s = build_str(vad_flag, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')

    table_f.write('Nephrotoxins:\n')
    # Number w/ >=1 Nephrotoxins D0-1 - n (%)
    row = '\t>=1 - n (%)'
    s = build_str(neph_cts >= 1, lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')
    # Number w/ >=2 Nephrotoxins D0-1 - n (%)
    row = '\t>=2 - n (%)'
    s = build_str(neph_cts >= 2, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')

    # Number w/ >=3 Nephrotoxins D0-1 - n (%)
    row = '\t>=3 - n (%)'
    s = build_str(neph_cts >= 3, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')

    # Vasoactive drugs, n (%)
    row = 'Vasoactive drugs - n (%)'
    s = build_str(vaso_cts >= 1, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')

    # Cumulative fluid balance at 72 h - median[IQ1 - IQ3]
    row = 'Cumulative fluid balance at 72 h - median[IQ1 - IQ3]'
    s = build_str(net_fluid / 1000, lbls, 'median', '%.1f')
    row += s

    table_f.write(row + '\n')

    # FO% at 72 h - median[IQ1 - IQ3]
    row = 'FO% at 72 h - median[IQ1 - IQ3]'
    s = build_str(fluid_overload / 10, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    # Sepsis - n( %)
    row = 'Sepsis - n(%)'
    s = build_str(sepsis, lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')

    # pH < 7.30- n( %)
    row = 'pH < 7.3 - n(%)'
    s = build_str(phs < 7.3, lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')

    # MAP < 70 mmHg - n( %)
    row = 'MAP mmHg < 70 - n(%)'
    s = build_str(maps < 70, lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')

    # MAP < 60 mmHg - n( %)
    row = 'MAP mmHg < 60 - n(%)'
    s = build_str(maps < 60, lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')

    # Serum Albumin, g/dL - median[IQ1 - IQ3]
    row = 'Serum Albumin g/dL - median[IQ1 - IQ3]'
    s = build_str(albs, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    # Serum Lactate, mmol/L - median[IQ1 - IQ3]
    row = 'Serum Lactate mmol/L - median[IQ1 - IQ3]'
    s = build_str(lacs, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    # Serum Bilirubin, mg/dL - median[IQ1 - IQ3]
    row = 'Serum Bilirubin mg/dL - median[IQ1 - IQ3]'
    s = build_str(bilis, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    # AKI Characteristics
    table_f.write('AKI characteristics\n')

    # Baseline SCr, mg/dL - median[IQ1 - IQ3]
    row = 'ALL Baseline SCr mg/dL - median[IQ1 - IQ3]'
    s = build_str(bsln_scr, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    # Baseline eGFR, mL/min/1.73m2 - median[IQ1 - IQ3]
    row = 'ALL Baseline eGFR mL/min/1.73m2 - median[IQ1 - IQ3]'
    s = build_str(bsln_gfr, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    # Measured Baselines - n (%)
    row = 'Measured Baselines - n (%)'
    s = build_str(np.array(bsln_types == 'measured'), lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')

    # Baseline SCr, mg/dL - median[IQ1 - IQ3]
    sel = np.where(bsln_types == 'measured')[0]
    row = 'Measured Baseline SCr mg/dL - median[IQ1 - IQ3]'
    s = build_str(bsln_scr[sel], lbls[sel], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    # Baseline eGFR, mL/min/1.73m2 - median[IQ1 - IQ3]
    row = 'Measured Baseline eGFR mL/min/1.73m2 - median[IQ1 - IQ3]'
    s = build_str(bsln_gfr[sel], lbls[sel], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    # Baseline SCr, mg/dL - median[IQ1 - IQ3]
    sel = np.where(bsln_types == 'imputed')[0]
    row = 'Imputed Baseline SCr mg/dL - median[IQ1 - IQ3]'
    s = build_str(bsln_scr[sel], lbls[sel], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    # Baseline eGFR mL/min/1.73m2 - median[IQ1 - IQ3]
    row = 'Imputed Baseline eGFR mL/min/1.73m2 - median[IQ1 - IQ3]'
    s = build_str(bsln_gfr[sel], lbls[sel], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    # Admit SCr, mg/dL - median[IQ1 - IQ3]
    row = 'Admit SCr mg/dL - median[IQ1 - IQ3]'
    s = build_str(admit_scr, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    # Peak SCr, mg/dL - median[IQ1 - IQ3]
    row = 'Peak SCr mg/dL - median[IQ1 - IQ3]'
    s = build_str(peak_scr, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    # Max KDIGO stage
    table_f.write('Maximum KDIGO Stage - Whole ICU:\n')
    # Stage 1,  n (%)
    row = '\tStage 1 - n (%)'
    s = build_str(mks == 1, lbls, 'countrow', '%.1f')
    row += s
    table_f.write(row + '\n')
    # Stage 2,  n (%)
    row = '\tStage 2 - n (%)'
    s = build_str(mks == 2, lbls, 'countrow', '%.1f')
    row += s
    table_f.write(row + '\n')
    # Stage 3,  n (%)
    row = '\tStage 3 - n (%)'
    s = build_str(mks == 3, lbls, 'countrow', '%.1f')
    row += s
    table_f.write(row + '\n')
    # Stage 3D,  n (%)
    row = '\tStage 3D - n (%)'
    s = build_str(mks == 4, lbls, 'countrow', '%.1f')
    row += s
    table_f.write(row + '\n')

    table_f.write('Maximum KDIGO Stage - 7 Days:\n')
    # Stage 1,  n (%)
    row = '\tStage 1 - n (%)'
    s = build_str(mks7d == 1, lbls, 'countrow', '%.1f')
    row += s
    table_f.write(row + '\n')
    # Stage 2,  n (%)
    row = '\tStage 2 - n (%)'
    s = build_str(mks7d == 2, lbls, 'countrow', '%.1f')
    row += s
    table_f.write(row + '\n')
    # Stage 3,  n (%)
    row = '\tStage 3 - n (%)'
    s = build_str(mks7d == 3, lbls, 'countrow', '%.1f')
    row += s
    table_f.write(row + '\n')
    # Stage 3D,  n (%)
    row = '\tStage 3D - n (%)'
    s = build_str(mks7d == 4, lbls, 'countrow', '%.1f')
    row += s
    table_f.write(row + '\n')

    # Urine output, L D0-D2 - median[IQ1 - IQ3]
    row = 'Urine output L D0-D2 - median[IQ1 - IQ3]'
    s = build_str(urine_out[np.where(urine_out != 0)], lbls[np.where(urine_out != 0)[0]], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    # Urine flow, ml/kg/24h D0-D2 - median[IQ1 - IQ3]
    row = 'Urine flow ml/kg/24h D0-D2 - median[IQ1 - IQ3]'
    s = build_str(urine_flow[np.where(urine_flow != 0)], lbls[np.where(urine_flow != 0)[0]], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    # Inpatient RRT
    table_f.write('Inpatient RRT characteristics\n')
    # Number of Patients with CRRT - n (%)
    row = 'Number of Patients on CRRT - n (%)'
    s = build_str(crrt_days > 0, lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')
    # Number of Patients with HD - n (%)
    row = 'Number of Patients on HD - n (%)'
    s = build_str(hd_days > 0, lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')
    # Total Number of Patients on RRT - n (%)
    row = 'Total Number of Patients on RRT - n (%)'
    s = build_str(rrt_tot > 0, lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')

    # Total days of CRRT - median[IQ1 - IQ3]
    row = 'Total days of CRRT - median[IQ1 - IQ3]'
    s = build_str(crrt_days[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    # Total days of HD - median[IQ1 - IQ3]
    row = 'Total days of HD - median[IQ1 - IQ3]'
    s = build_str(hd_days[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    # Total number of HD treatments - median[IQ1 - IQ3]
    row = 'Total number of HD treatments - median[IQ1 - IQ3]'
    s = build_str(hd_trtmts[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    # Total days of CRRT + HD - median[IQ1 - IQ3]
    row = 'Total days of CRRT + HD - median[IQ1 - IQ3]'
    s = build_str(rrt_tot[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    # CRRT free days - 7d - median[IQ1 - IQ3]
    row = 'CRRT Free Days - 7d - median[IQ1 - IQ3]'
    s = build_str(crrt_free_7d[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')
    # CRRT free days - 28d - median[IQ1 - IQ3]
    row = 'CRRT Free Days - 28d - median[IQ1 - IQ3]'
    s = build_str(crrt_free_28d[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    # hd free days - 7d - median[IQ1 - IQ3]
    row = 'HD Free Days - 7d - median[IQ1 - IQ3]'
    s = build_str(hd_free_7d[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')
    # hd free days - 28d - median[IQ1 - IQ3]
    row = 'HD Free Days - 28d - median[IQ1 - IQ3]'
    s = build_str(hd_free_28d[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    table_f.close()

    return


# %%
def plot_daily_kdigos(datapath, ids, stat_file, sqdm, lbls, outpath='',
                      max_day=7, cutoff=None, align=False, template_min_length=7, lsize=8):
    '''
    Plot the daily KDIGO score for each cluster indicated in lbls. Plots the following figures:
        center - only plot the single patient closest to the cluster center
        all_w_mean - plot all individual patient vectors, along with a bold line indicating the cluster mean
        mean_conf - plot the cluster mean, along with the 95% confidence interval band
        mean_std - plot the cluster mean, along with a band indicating 1 standard deviation
    :param datapath: fully qualified path to the directory containing KDIGO vectors
    :param ids: list of IDs
    :param stat_file: file handle for the file containing all patient statistics
    :param sqdm: square distance matrix for all patients in ids
    :param lbls: corresponding cluster labels for all patients
    :param outpath: fully qualified base directory to save figures
    :param max_day: how many days to include in the figures
    :param cutoff: optional... if cutoff is supplied, then the mean for days 0 - cutoff will be blue, whereas days
                    cutoff - max_day will be plotted red with a dotted line
    :return:
    '''
    transition_costs = [1.00,  # [0 - 1]
                        2.95,  # [1 - 2]
                        4.71,  # [2 - 3]
                        7.62]  # [3 - 4]
    mismatch = mismatch_penalty_func(*transition_costs)
    extension = extension_penalty_func(*transition_costs)

    c_lbls = np.unique(lbls)
    n_clusters = len(c_lbls)
    if np.ndim(sqdm) == 1:
        sqdm = squareform(sqdm)
    if type(lbls) is list:
        lbls = np.array(lbls, dtype=str)

    scrs = load_csv(os.path.join(datapath, 'scr_raw.csv'), ids)
    bslns = load_csv(os.path.join(datapath, 'baselines.csv'), ids)
    dmasks = load_csv(os.path.join(datapath, 'dmasks_interp.csv'), ids, dt=int)
    kdigos = load_csv(os.path.join(datapath, 'kdigo.csv'), ids, dt=int)
    days_interp = load_csv(os.path.join(datapath, 'days_interp.csv'), ids, dt=int)
    str_admits = load_csv(os.path.join(datapath, 'patient_summary.csv'), ids, dt=str, idxs=1, skip_header=True)
    admits = []
    for i in range(len(ids)):
        admits.append(datetime.datetime.strptime('%s' % str_admits[i], '%Y-%m-%d %H:%M:%S'))

    str_dates = load_csv(datapath + 'dates.csv', ids, dt=str)
    for i in range(len(ids)):
        for j in range(len(str_dates[i])):
            str_dates[i][j] = str_dates[i][j].split('.')[0]
    dates = []
    for i in range(len(ids)):
        temp = []
        for j in range(len(str_dates[i])):
            temp.append(datetime.datetime.strptime('%s' % str_dates[i][j], '%Y-%m-%d %H:%M:%S'))
        dates.append(temp)

    center_kdigos = {}
    center_days = {}
    center_idx = []
    for lbl in c_lbls:
        idx = np.where(lbls == lbl)[0]
        sel = np.ix_(idx, idx)
        tdm = sqdm[sel]
        intra = np.sum(tdm, axis=0)
        o = np.argsort(intra)
        ct = 0
        cidx = idx[o[ct]]
        while max(days_interp[cidx]) < template_min_length:
            ct += 1
            cidx = idx[o[ct]]
        center_kdigos[lbl] = kdigos[cidx][np.where(days_interp[cidx] <= max_day)]
        center_days[lbl] = days_interp[cidx][np.where(days_interp[cidx] <= max_day)]
        center_idx.append(cidx)

    blank_daily = np.repeat(np.nan, max_day + 2)
    all_daily = np.vstack([blank_daily for x in range(len(ids))])
    for i in range(len(ids)):
        l = np.min([len(x) for x in [scrs[i], dates[i], dmasks[i]]])
        if l < 2:
            continue
        # tmax = daily_max_kdigo(scrs[i][:l], dates[i][:l], bslns[i], admits[i], dmasks[i][:l], tlim=max_day)
        # tmax = daily_max_kdigo(scrs[i][:l], days[i][:l], bslns[i], dmasks[i][:l], tlim=max_day)
        if align:
            tkdigo = kdigos[i][np.where(days_interp[i] <= max_day)]
            _, _, _, path = dtw_p(tkdigo, center_kdigos[lbls[i]], mismatch=mismatch, extension=extension)
            tkdigo = tkdigo[path[0]]
            # tmax = daily_max_kdigo_aligned(tkdigo[:len(center_kdigos[lbls[i]])])
            tmax = daily_max_kdigo_interp(tkdigo[:len(center_kdigos[lbls[i]])], center_days[lbls[i]], tlim=max_day)
        else:
            tmax = daily_max_kdigo_interp(kdigos[i], days_interp[i], tlim=max_day)
        if np.all(tmax == 0):
            print('Patient %d - All 0' % ids[i])
            print('Baseline: %.3f' % bslns[i])
            print(days_interp[i])
            print(kdigos[i])
            print('\n')
            print(tmax)
            temp = raw_input('Enter to continue (q to quit):')
            if temp == 'q':
                return
        if len(tmax) > max_day + 2:
            all_daily[i, :] = tmax[:max_day + 2]
        else:
            all_daily[i, :len(tmax)] = tmax

    centers = np.zeros(n_clusters, dtype=int)
    n_recs = np.zeros((n_clusters, max_day + 2))
    cluster_idx = {}
    for i in range(n_clusters):
        tlbl = c_lbls[i]
        idx = np.where(lbls == tlbl)[0]
        cluster_idx[tlbl] = idx
        sel = np.ix_(idx, idx)
        tdm = sqdm[sel]
        sums = np.sum(tdm, axis=0)
        center = np.argsort(sums)[0]
        centers[i] = idx[center]
        for j in range(max_day + 2):
            n_recs[i, j] = (float(len(idx) - len(np.where(np.isnan(all_daily[idx, j]))[0])) / len(idx)) * 100

    if outpath != '':
        font = {'size': 20}

        matplotlib.rc('font', **font)
        if align:
            outpath = os.path.join(outpath, 'aligned')
        f = stat_file
        all_ids = f['meta']['ids'][:]
        all_inp_death = f['meta']['died_inp'][:]
        sel = np.array([x in ids for x in all_ids])
        inp_death = all_inp_death[sel]
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        if not os.path.exists(os.path.join(outpath, 'all_w_mean')):
            os.mkdir(os.path.join(outpath, 'all_w_mean'))
        if not os.path.exists(os.path.join(outpath, 'mean_std')):
            os.mkdir(os.path.join(outpath, 'mean_std'))
        if not os.path.exists(os.path.join(outpath, 'mean_conf')):
            os.mkdir(os.path.join(outpath, 'mean_conf'))
        if not os.path.exists(os.path.join(outpath, 'center')):
            os.mkdir(os.path.join(outpath, 'center'))
        for i in range(n_clusters):
            cidx = cluster_idx[c_lbls[i]]
            ct = len(cidx)
            mort = (float(len(np.where(inp_death[cidx])[0])) / len(cidx)) * 100
            mean_daily, conf_lower, conf_upper = mean_confidence_interval(all_daily[cidx])
            std_daily = np.nanstd(all_daily[cidx], axis=0)
            # stds_upper = np.minimum(mean_daily + std_daily, 4)
            # stds_lower = np.maximum(mean_daily - std_daily, 0)
            stds_upper = mean_daily + std_daily
            stds_lower = mean_daily - std_daily

            # Plot only cluster center
            if align:
                dmax = all_daily[center_idx[i], :]
            else:
                dmax = all_daily[centers[i], :]
            tfig = plt.figure()
            tplot = tfig.add_subplot(111)
            if cutoff is not None or cutoff >= max_day:
                # Trajectory used for model
                tplot.plot(range(len(dmax))[:cutoff + 1], dmax[:cutoff + 1], color='blue')
                # Rest of trajectory
                tplot.axvline(x=cutoff, linestyle='dashed')
                tplot.plot(range(len(dmax))[cutoff:], dmax[cutoff:], color='red',
                           label='Cluster Mortality = %.2f%%' % mort)
            else:
                tplot.plot(range(len(dmax)), dmax, color='blue',
                           label='Cluster Mortality = %.2f%%' % mort)
            plt.yticks(range(5), ['0', '1', '2', '3', '3D'])
            tplot.set_xlim(-0.25, 7.25)
            tplot.set_ylim(-1.0, 5.0)
            tplot.set_xlabel('Day')
            tplot.set_ylabel('KDIGO Score')
            tplot.set_title('Cluster %s Representative' % c_lbls[i])
            plt.legend()
            if align:
                plt.savefig(os.path.join(outpath, 'center', '%s_center_aligned.png' % c_lbls[i]))
            else:
                plt.savefig(os.path.join(outpath, 'center', '%s_center.png' % c_lbls[i]))
            plt.close(tfig)

            # All patients w/ mean
            fig = plt.figure()
            for j in range(len(cidx)):
                plt.plot(range(max_day + 2), all_daily[cidx[j]], lw=1, alpha=0.3)

            if cutoff is not None or cutoff >= max_day:
                plt.plot(range(max_day + 2)[:cutoff + 1], mean_daily[:cutoff + 1], color='b',
                         lw=2, alpha=.8)
                plt.plot(range(max_day + 2)[cutoff:], mean_daily[cutoff:], color='r',
                         label='Cluster Mortality = %.2f%%\n%d Patients' % (mort, ct), lw=2, alpha=.8)
                plt.axvline(x=cutoff, linestyle='dashed')
            else:
                plt.plot(range(max_day + 2), mean_daily, color='b',
                         label='Cluster Mortality = %.2f%%\n%d Patients' % (mort, ct), lw=2, alpha=.8)

            plt.fill_between(range(max_day + 2), conf_lower, conf_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

            plt.xlim([-0.25, max_day + 0.25])
            plt.ylim([-1.0, 5.0])
            plt.xlabel('Time (Days)')
            plt.ylabel('KDIGO Score')
            plt.yticks(range(5), ['0', '1', '2', '3', '3D'])
            plt.legend()
            plt.title('Average Daily KDIGO\nCluster %s' % c_lbls[i])
            if align:
                plt.savefig(os.path.join(outpath, 'all_w_mean', '%s_all_aligned.png' % c_lbls[i]))
            else:
                plt.savefig(os.path.join(outpath, 'all_w_mean', '%s_all.png' % c_lbls[i]))
            plt.close(fig)

            # Mean and standard deviation
            fig = plt.figure()
            fig, ax1 = plt.subplots()
            if cutoff is not None or cutoff >= max_day:
                ax1.plot(range(max_day + 2)[:cutoff + 1], mean_daily[:cutoff + 1], color='b',
                         lw=2, alpha=.8)
                ax1.plot(range(max_day + 2)[cutoff:], mean_daily[cutoff:], color='r', linestyle='dashed',
                         label='Cluster Mortality = %.2f%%\n%d Patients' % (mort, ct), lw=2, alpha=.8)
                ax1.axvline(x=cutoff, linestyle='dashed')
            else:
                ax1.plot(range(max_day + 2), mean_daily, color='b',
                         label='Cluster Mortality = %.2f%%\n%d Patients' % (mort, ct), lw=2, alpha=.8)
            ax1.fill_between(range(max_day + 2), stds_lower, stds_upper, color='grey', alpha=.2,
                             label=r'+/- 1 Std. Deviation')
            plt.xlim([-0.25, max_day + 0.25])
            plt.ylim([-1.0, 5.0])
            plt.xlabel('Time (Days)')
            plt.ylabel('KDIGO Score')
            plt.yticks(range(5), ['0', '1', '2', '3', '3D'])
            plt.legend()
            # ax2 = ax1.twinx()
            # ax2.plot(range(max_day + 2), n_recs[i, :], color='black', label='# Records')
            # ax2.set_ylim((-5, 105))
            # ax2.set_ylabel('% Patients Remaining')
            # plt.legend(loc=7)
            plt.title('Average Daily KDIGO\nCluster %s' % c_lbls[i])
            if align:
                plt.savefig(os.path.join(outpath, 'mean_std', '%s_mean_std_aligned.png' % c_lbls[i]))
            else:
                plt.savefig(os.path.join(outpath, 'mean_std', '%s_mean_std.png' % c_lbls[i]))
            plt.close(fig)

            # Mean and 95% confidence interval
            fig = plt.figure()
            fig, ax1 = plt.subplots()
            if cutoff is not None or cutoff >= max_day:
                ax1.plot(range(max_day + 2)[:cutoff + 1], mean_daily[:cutoff + 1], color='b',
                         lw=2, alpha=.8)
                ax1.plot(range(max_day + 2)[cutoff:], mean_daily[cutoff:], color='r', linestyle='dashed',
                         label='Cluster Mortality = %.2f%%\n%d Patients' % (mort, ct), lw=2, alpha=.8)
                ax1.axvline(x=cutoff, linestyle='dashed')
            else:
                ax1.plot(range(max_day + 2), mean_daily, color='b',
                         label='Cluster Mortality = %.2f%%\n%d Patients' % (mort, ct), lw=2, alpha=.8)
            ax1.fill_between(range(max_day + 2), conf_lower, conf_upper, color='grey', alpha=.2,
                             label=r'95% Confidence Interval')
            plt.xlim([-0.25, max_day + 0.25])
            plt.ylim([-1.0, 5.0])
            plt.xlabel('Days')
            plt.ylabel('KDIGO')
            plt.yticks(range(5), ['0', '1', '2', '3', '3D'])
            extra = plt.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            plt.legend([extra, ], [c_lbls[i] + '\nMortality %.2f%%\n# Patients %d' % (mort, ct), ], frameon=False, prop={'size': lsize})
            # ax2 = ax1.twinx()
            # ax2.plot(range(max_day + 2), n_recs[i, :], color='black', label='# Records')
            # ax2.set_ylim((-5, 105))
            # ax2.set_ylabel('% Patients Remaining')
            # plt.legend(loc=7)
            # plt.title(c_lbls[i])
            plt.tight_layout()
            if align:
                plt.savefig(os.path.join(outpath, 'mean_conf', '%s_mean_conf_aligned.png' % c_lbls[i]), dpi=600)
            else:
                plt.savefig(os.path.join(outpath, 'mean_conf', '%s_mean_conf.png' % c_lbls[i]), dpi=600)

            plt.close(fig)
        # f.close()
    return all_daily


# %%
def mean_confidence_interval(data, confidence=0.95):
    '''
    Returns the mean confidence interval for the distribution of data
    :param data:
    :param confidence: decimal percentile, i.e. 0.95 = 95% confidence interval
    :return:
    '''
    a = 1.0 * np.array(data)
    n_ex, n_pts = a.shape
    means = np.zeros(n_pts)
    diff = np.zeros(n_pts)
    for i in range(n_pts):
        x = data[:, i]
        x = x[np.logical_not(np.isnan(x))]
        n = len(x)
        m, se = np.mean(x), sem(x)
        h = se * t.ppf((1 + confidence) / 2., n-1)
        means[i] = m
        diff[i] = h
    return means, means-diff, means+diff


# %%
def get_disch_date(idx, date_m, hosp_locs):
    rows = np.where(date_m[0] == idx)
    dd = datetime.timedelta(0)
    for row in rows:
        if date_m[row, hosp_locs[1]] > dd:
            dd = date_m[row, hosp_locs[1]]
    return dd


# %%
def get_dod(idx, date_m, outcome_m, dod_loc):
    rows = np.where(date_m[0] == idx)
    if rows.size == 0:
        return rows
    dd = datetime.timedelta(0)
    for row in rows:
        if outcome_m[row, dod_loc] > dd:
            dd = outcome_m[row, dod_loc]
    if dd == datetime.timedelta(0):
        return None
    return dd


# %%
def pairwise_stats(stats, path):
    feats = [['Age', 'age'],
             ['Gender', 'gender'],
             ['BMI', 'bmi'],
             ['Baseline GFR', 'baseline_gfr'],
             ['Baseline SCr', 'baseline_scr'],
             ['Admit SCr', 'admit_scr'],
             ['Peak SCr', 'peak_scr'],
             ['SOFA', 'sofa'],
             ['APACHE', 'apache'],
             ['Charlson Score', 'charlson'],
             ['Elixhauser Score', 'elixhauser'],
             ['Net Fluid', 'net_fluid'],
             ['Gross Fluid', 'gross_fluid'],
             ['HD Days', 'hd_days'],
             ['CRRT Days', 'crrt_days'],
             ['Hospital Free Days', 'hosp_free_days'],
             ['ICU Free Days', 'icu_free_days'],
             ['Mech Vent Free Days', 'mv_free_days']]

    clusterMembership = np.loadtxt(os.path.join(path, 'clusters.csv'), delimiter=',', dtype=str)
    labels = clusterMembership[:, 1]

    clusterSummary = pd.read_csv(os.path.join(path, 'cluster_stats.csv'))
    labelIndex = clusterSummary.columns.get_loc('cluster_id')
    mortIndex = clusterSummary.columns.get_loc('pct_inp_mort')
    clusterSummary = clusterSummary.values

    mort = clusterSummary[:, mortIndex]
    o = np.argsort(mort)

    clusterLabels = clusterSummary[:, labelIndex][o]
    nClusters = len(o)

    featHeatmap = np.zeros((nClusters, len(feats)))
    featCount = 0

    if not os.path.exists(os.path.join(path, 'formal_stats')):
        os.mkdir(os.path.join(path, 'formal_stats'))

    for fl in feats:
        title = fl[0]
        featName = fl[1]
        feat = stats[featName][:]

        plog = open(os.path.join(path, 'formal_stats', featName + '.csv'), 'w')
        s = []
        for i in range(nClusters):
            label1 = clusterLabels[i]
            idx1 = np.where(labels == label1)[0]
            feat1 = feat[idx1]
            sel = np.logical_not(np.isnan(feat1))
            feat1 = feat1[sel]
            featHeatmap[i, featCount] = np.mean(feat1)
            for j in range(i+1, nClusters):
                label2 = clusterLabels[j]
                idx2 = np.where(labels == label2)[0]
                feat2 = feat[idx2]
                sel = np.logical_not(np.isnan(feat2))
                feat2 = feat2[sel]
                _, p = ttest_ind(feat1, feat2)
                s.append(p)
                plog.write('%s,%s,%f' % (label1, label2, p))
        plog.close()

        s = np.array(s)
        sq = squareform(s)
        fig = plt.figure()
        plt.pcolormesh(sq, vmin=0, vmax=1)
        plt.xticks(np.arange(len(o)) + 0.5, clusterLabels, rotation=30)
        plt.yticks(np.arange(len(o)) + 0.5, clusterLabels)
        plt.title(title)
        plt.colorbar()
        plt.savefig(os.path.join(path, 'formal_stats', featName + '.png'))
        plt.close(fig)

        np.save(os.path.join(path, 'formal_stats', featName + '.png'), s)
        featCount += 1

    for i in range(len(feats)):
        dmin = min(0, np.min(featHeatmap[:, i]))
        dmax = np.max(featHeatmap[:, i])
        featHeatmap[:, i] = (featHeatmap[:, i] - dmin) / (dmax - dmin)

    fig = plt.figure()
    plt.pcolormesh(featHeatmap)
    plt.yticks(np.arange(nClusters) + 0.5, clusterLabels)
    plt.xticks(np.arange(len(feats)) + 0.5, [x[0] for x in feats], rotation=90)
    plt.title('Normalized Features - Mean Cluster Values')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'all_features.png'))
    plt.close(fig)




# %%
def count_eps(kdigo, t_gap=48, timescale=6):
    aki = np.where(kdigo)[0]
    if len(aki) > 0:
        count = 1
    else:
        return 0
    gap_ct = t_gap / timescale
    for i in range(1, len(aki)):
        if (aki[i] - aki[i - 1]) >= gap_ct:
            count += 1
    return count


def get_los(pid, date_m, hosp_locs, icu_locs):
    date_rows = np.where(date_m[:, 0] == pid)[0]
    hosp = []
    icu = []
    for i in range(len(date_rows)):
        idx = date_rows[i]
        add_hosp = True
        for j in range(len(hosp)):
            # new start is before saved start
            if date_m[idx, hosp_locs[0]] < hosp[j][0]:
                # new stop is after saved stop - replace interior window
                if date_m[idx, hosp_locs[1]] > hosp[j][1]:
                    hosp[j][0] = date_m[idx, hosp_locs[0]]
                    hosp[j][1] = date_m[idx, hosp_locs[1]]
                    add_hosp = False
                # new stop is after saved start - replace saved start
                elif date_m[idx, hosp_locs[1]] > hosp[j][0]:
                    hosp[j][0] = date_m[idx, hosp_locs[0]]
                    add_hosp = False
            # new start is before saved stop
            elif date_m[idx, hosp_locs[0]] < hosp[j][1]:
                add_hosp = False
                # new stop is after saved stop - replace saved stop
                if date_m[idx, hosp_locs[1]] < hosp[j][1]:
                    hosp[j][1] = date_m[idx, hosp_locs[1]]
        if add_hosp:
            hosp.append([date_m[idx, hosp_locs[0]], date_m[idx, hosp_locs[1]]])

        add_icu = True
        for j in range(len(icu)):
            # new start is before saved start
            if date_m[idx, icu_locs[0]] < icu[j][0]:
                # new stop is after saved stop - replace interior window
                if date_m[idx, icu_locs[1]] > icu[j][1]:
                    icu[j][0] = date_m[idx, icu_locs[0]]
                    icu[j][1] = date_m[idx, icu_locs[1]]
                    add_icu = False
                # new stop is after saved start - replace saved start
                elif date_m[idx, icu_locs[1]] > icu[j][0]:
                    icu[j][0] = date_m[idx, icu_locs[0]]
                    add_icu = False
            # new start is before saved stop
            elif date_m[idx, icu_locs[0]] < icu[j][1]:
                add_icu = False
                # new stop is after saved stop - replace saved stop
                if date_m[idx, icu_locs[1]] < icu[j][1]:
                    icu[j][1] = date_m[idx, icu_locs[1]]
        if add_icu:
            icu.append([date_m[idx, icu_locs[0]], date_m[idx, icu_locs[1]]])

    h_dur = datetime.timedelta(0)
    for i in range(len(hosp)):
        start_str = str(hosp[i][0]).split('.')[0]
        stop_str = str(hosp[i][1]).split('.')[0]
        start = get_date(start_str)
        stop = get_date(stop_str)
        h_dur += stop - start

    icu_dur = datetime.timedelta(0)
    for i in range(len(icu)):
        start_str = str(icu[i][0]).split('.')[0]
        stop_str = str(icu[i][1]).split('.')[0]
        start = get_date(start_str)
        stop = get_date(stop_str)
        icu_dur += stop - start

    h_dur = h_dur.total_seconds() / (60 * 60 * 24)
    icu_dur = icu_dur.total_seconds() / (60 * 60 * 24)

    return h_dur, icu_dur


# %% Summarize discharge dispositions for a file
def get_disch_summary(id_file, stat_file, ids=None):
    # Code:
    #     0 - Dead in hospital
    #     1 - Dead after 48 hrs
    #     2 - Alive
    #     3 - Transfered
    #     4 - AMA

    # get IDs in order
    f = open(id_file, 'r')
    sf = open(stat_file, 'w')

    n_alive = 0
    n_lt = 0
    n_gt = 0
    n_xfer = 0
    n_ama = 0
    n_unk = 0
    n_oth = 0
    for l in f:
        line = l.strip()
        this_id = line.split(',')[0]
        if ids is not None and this_id not in ids:
            continue
        str_disp = line.split(',')[-1].upper()
        if re.search('EXP', str_disp):
            if re.search('LESS', str_disp):
                n_lt += 1
            elif re.search('MORE', str_disp):
                n_gt += 1
        elif re.search('ALIVE', str_disp):
            n_alive += 1
        elif re.search('XFER', str_disp) or re.search('TRANS', str_disp):
            n_xfer += 1
        elif re.search('AMA', str_disp):
            n_ama += 1
        elif str_disp == '':
            n_unk += 1
        else:
            n_oth += 1
    f.close()

    sf.write('All Alive: %d\n' % (n_alive + n_xfer + n_ama))
    sf.write('Alive - Routine: %d\n' % n_alive)
    sf.write('Transferred: %d\n' % n_xfer)
    sf.write('Against Medical Advice: %d\n' % n_ama)
    sf.write('Died: %d\n' % (n_lt + n_gt))
    sf.write('   <48 hrs: %d\n' % n_lt)
    sf.write('   >48 hrs: %d\n\n' % n_gt)
    sf.write('Other: %d\n' % n_oth)
    sf.write('Unknown (not provided):\t%d\n' % n_unk)

    sf.close()


# def get_sofa(ids, scr_interp, days_interp,
#              admit_info, date,
#              blood_gas, pa_o2,
#              clinical_oth, fi_o2, g_c_s,
#              clinical_vit, m_a_p, cuff,
#              labs, bili, pltlts,
#              medications, med_name, med_date, med_dur,
#              organ_sup, mech_vent,
#              scr_agg, s_c_r,
#              out_name, v=False):
#
#     pa_o2 = pa_o2[1]
#     fi_o2 = fi_o2[1]
#     m_a_p = m_a_p[0]
#     cuff = cuff[0]
#
#     missing = np.zeros(6)
#
#     out = open(out_name, 'w')
#     sofas = np.zeros((len(ids), 6))
#     for i in range(len(ids)):
#         idx = ids[i]
#         out.write('%d' % idx)
#         admit_rows = np.where(admit_info[:, 0] == idx)[0]
#         bg_rows = np.where(blood_gas[:, 0] == idx)[0]
#         co_rows = np.where(clinical_oth[:, 0] == idx)[0]
#         cv_rows = np.where(clinical_vit[:, 0] == idx)[0]
#         med_rows = np.where(medications[:, 0] == idx)[0]
#         scr_rows = np.where(scr_agg[:, 0] == idx)[0]
#         lab_rows = np.where(labs[:, 0] == idx)[0]
#         mv_rows = np.where(organ_sup[:, 0] == idx)[:]
#
#         admit = datetime.datetime.now()
#         for did in admit_rows:
#             tstr = str(admit_info[did, date]).split('.')[0]
#             tadmit = get_date(tstr)
#             if tadmit < admit:
#                 admit = tadmit
#
#         if np.size(bg_rows) > 0:
#             s1_pa = float(blood_gas[bg_rows, pa_o2])
#         else:
#             s1_pa = np.nan
#
#         if np.size(co_rows) > 0:
#             s1_fi = float(clinical_oth[co_rows,fi_o2])
#         else:
#             s1_fi = np.nan
#         if not np.isnan(s1_pa) and not np.isnan(s1_fi) and s1_fi != 0:
#             s1_ratio = s1_pa / s1_fi
#         else:
#             s1_ratio = np.nan
#         if np.isnan(s1_ratio):
#             missing[0] += 1
#
#         try:
#             s2_gcs = float(str(clinical_oth[co_rows, g_c_s][0]).split('-')[0])
#         except:
#             s2_gcs = np.nan
#         if np.isnan(s2_gcs):
#             missing[1] += 1
#
#         s3_map = np.nan
#         if np.size(cv_rows) > 0:
#             s3_map = float(clinical_vit[cv_rows, m_a_p])
#             if np.isnan(s3_map):
#                 s3_map = float(clinical_vit[cv_rows, cuff])
#         if np.isnan(s3_map):
#             missing[2] += 1
#
#         if np.size(med_rows) > 0:
#             s3_med = str(medications[med_rows, med_name])
#             s3_date = str(medications[med_rows, med_date])
#             s3_dur = str(medications[med_rows, med_dur])
#
#         if np.size(lab_rows) > 0:
#             s4_bili = float(labs[lab_rows, bili])
#             s5_plt = float(labs[lab_rows, pltlts])
#         else:
#             s4_bili = np.nan
#             s5_plt = np.nan
#         if np.isnan(s4_bili):
#             missing[3] += 1
#         if np.isnan(s5_plt):
#             missing[4] += 1
#
#         # Find maximum value in day 0
#         s6_scr = np.nan
#         day_sel = np.where(days_interp[i] == 0)[0]
#         if day_sel.size > 0:
#             s6_scr = np.max(scr_interp[i][day_sel])
#         else:
#             day_sel = np.where(days_interp[i] == 1)[0]
#             if day_sel.size > 0:
#                 s6_scr = np.max(scr_interp[i][day_sel])
#         if np.isnan(s6_scr):
#             missing[5] += 1
#
#         score = np.zeros(6, dtype=int)
#
#         vent = 0
#         if len(mv_rows) > 0:
#             for row in range(len(mv_rows)):
#                 tmv = organ_sup[row, mech_vent]
#                 mech_start_str = str(tmv[0]).split('.')[0]
#                 mech_stop_str = str(tmv[1]).split('.')[0]
#                 mech_start = get_date(mech_start_str)
#                 mech_stop = get_date(mech_stop_str)
#                 if mech_start <= admit <= mech_stop:
#                     vent = 1
#         if vent:
#             if s1_ratio < 100:
#                 score[0] = 4
#             elif s1_ratio < 200:
#                 score[0] = 3
#         elif s1_ratio < 300:
#             score[0] = 2
#         elif s1_ratio < 400:
#             score[0] = 1
#
#         if s2_gcs is not None:
#             s2 = s2_gcs
#             if not np.isnan(s2):
#                 if s2 < 6:
#                     score[1] = 4
#                 elif s2 < 10:
#                     score[1] = 3
#                 elif s2 < 13:
#                     score[1] = 2
#                 elif s2 < 15:
#                     score[1] = 1
#
#         s3 = s3_map
#         dopa = 0
#         epi = 0
#         try:
#             admit = admit_info[admit_rows[0][0], date]
#             for i in range(len(s3_med)):
#                 med_typ = s3_med[i]
#                 dm = med_typ[0][0].lower()
#                 start = s3_date[i]
#                 stop = start + datetime.timedelta(s3_dur[i])
#                 if 'dopamine' in dm or 'dobutamine' in dm:
#                     if start <= admit:
#                         if admit <= stop:
#                             dopa = 1
#                 elif 'epinephrine' in dm:
#                     if start <= admit:
#                         if admit <= stop:
#                             epi = 1
#         except:
#             s3 = s3
#         if epi:
#             score[2] = 3
#         elif dopa:
#             score[2] = 2
#         elif s3 < 70:
#             score[2] = 1
#
#         s4 = s4_bili
#         if s4 > 12.0:
#             score[3] = 4
#         elif s4 > 6.0:
#             score[3] = 3
#         elif s4 > 2.0:
#             score[3] = 2
#         elif s4 > 1.2:
#             score[3] = 1
#
#         s5 = s5_plt
#         if s5 < 20:
#             score[4] = 4
#         elif s5 < 50:
#             score[4] = 3
#         elif s5 < 100:
#             score[4] = 2
#         elif s5 < 150:
#             score[4] = 1
#
#         s6 = s6_scr
#         if s6 > 5.0:
#             score[5] = 4
#         elif s6 > 3.5:
#             score[5] = 3
#         elif s6 > 2.0:
#             score[5] = 2
#         elif s6 > 1.2:
#             score[5] = 1
#
#         out.write(',%d,%d,%d,%d,%d,%d\n' % (score[0], score[1], score[2], score[3], score[4], score[5]))
#         sofas[i, :] = score
#         print(np.sum(score))
#     return sofas, missing

def get_sofa(ids, stats, scr_interp, days_interp, out_name, v=False):

    all_ids = stats['ids'][:]
    pt_sel = np.array([x in ids for x in all_ids])

    pao2 = stats['pao2'][:][pt_sel]
    
    fio2 = stats['fio2'][:][pt_sel]
    maps = stats['map'][:][pt_sel]
    gcs = stats['glasgow'][:][pt_sel]
    bili = stats['bilirubin'][:][pt_sel]
    pltlts = stats['platelets'][:][pt_sel]
    mv_flag = stats['mv_flag'][:][pt_sel]
    admits = stats['hosp_window'][:, 0][pt_sel]
    dopas = stats['dopa'][:, 0][pt_sel]
    epis = stats['epinephrine'][:, 0][pt_sel]
    
    fio2_mv = np.nanmedian(fio2[np.where(mv_flag)])
    fio2_nomv = np.nanmedian(fio2[np.where(mv_flag == 0)])
    pao2_mv = np.nanmedian(pao2[np.where(mv_flag)])
    pao2_nomv = np.nanmedian(pao2[np.where(mv_flag == 0)])
    gcs_mv = np.nanmedian(gcs[np.where(mv_flag)])
    gcs_nomv = np.nanmedian(gcs[np.where(mv_flag == 0)])

    out = open(out_name, 'w')
    sofas = np.zeros((len(ids), 6))
    for i in range(len(ids)):
        idx = ids[i]
        out.write('%d' % idx)

        mv = mv_flag[i]

        s1_pa = pao2[i]
        if np.isnan(s1_pa):
            if mv:
                s1_pa = pao2_mv
            else:
                s1_pa = pao2_nomv
                
        s1_fi = fio2[i]
        if np.isnan(s1_fi):
            if mv:
                s1_fi = fio2_mv
            else:
                s1_fi = fio2_nomv

        if not np.isnan(s1_pa) and not np.isnan(s1_fi) and s1_fi != 0:
            s1_ratio = s1_pa / s1_fi
        else:
            s1_ratio = np.nan

        s2_gcs = gcs[i]
        if np.isnan(s2_gcs):
            if mv:
                s2_gcs = gcs_mv
            else:
                s2_gcs = gcs_nomv

        s3_map = maps[i]

        s4_bili = bili[i]
        s5_plt = pltlts[i]

        # Find maximum value in day 0
        s6_scr = np.nan
        day_sel = np.where(days_interp[i] == 0)[0]
        if day_sel.size > 0:
            s6_scr = np.max(scr_interp[i][day_sel])
        else:
            day_sel = np.where(days_interp[i] == 1)[0]
            if day_sel.size > 0:
                s6_scr = np.max(scr_interp[i][day_sel])

        score = np.zeros(6, dtype=int)

        if mv:
            if s1_ratio < 100:
                score[0] = 4
            elif s1_ratio < 200:
                score[0] = 3
        elif s1_ratio < 300:
            score[0] = 2
        elif s1_ratio < 400:
            score[0] = 1

        if s2_gcs is not None:
            s2 = s2_gcs
            if not np.isnan(s2):
                if s2 < 6:
                    score[1] = 4
                elif s2 < 10:
                    score[1] = 3
                elif s2 < 13:
                    score[1] = 2
                elif s2 < 15:
                    score[1] = 1

        s3 = s3_map
        dopa = dopas[i]
        epi = epis[i]
        if epi:
            score[2] = 3
        elif dopa:
            score[2] = 2
        elif s3 < 70:
            score[2] = 1

        s4 = s4_bili
        if s4 > 12.0:
            score[3] = 4
        elif s4 > 6.0:
            score[3] = 3
        elif s4 > 2.0:
            score[3] = 2
        elif s4 > 1.2:
            score[3] = 1

        s5 = s5_plt
        if s5 < 20:
            score[4] = 4
        elif s5 < 50:
            score[4] = 3
        elif s5 < 100:
            score[4] = 2
        elif s5 < 150:
            score[4] = 1

        s6 = s6_scr
        if s6 > 5.0:
            score[5] = 4
        elif s6 > 3.5:
            score[5] = 3
        elif s6 > 2.0:
            score[5] = 2
        elif s6 > 1.2:
            score[5] = 1

        out.write(',%d,%d,%d,%d,%d,%d\n' % (score[0], score[1], score[2], score[3], score[4], score[5]))
        sofas[i, :] = score
        if v:
            print(np.sum(score))
    return sofas


def get_sofa_dallas(stats, out_name, v=False):

    ids = stats['ids'][:]

    out = open(out_name, 'w')
    sofas = np.zeros((len(ids), 6))
    for i in range(len(ids)):
        idx = ids[i]
        out.write('%d' % idx)

        s1_po2 = stats['po2'][i, 1, 1]
        s1_fi = stats['fio2'][i, 1, 1]
        if not np.isnan(s1_po2) and not np.isnan(s1_fi) and s1_fi != 0:
            s1_ratio = s1_po2 / s1_fi
        else:
            s1_ratio = np.nan

        s2_gcs = stats['glasgow'][i, 1, 0]

        s3_map = stats['map'][i, 1, 0]

        s4_bili = stats['bilirubin'][i, 1, 1]

        s5_plt = stats['platelets'][i, 1, 0]

        s6_scr = stats['scr_agg'][i, 1, 1]

        score = np.zeros(6, dtype=int)

        vent = stats['mv_flag'][i]
        if vent:
            if s1_ratio < 100:
                score[0] = 4
            elif s1_ratio < 200:
                score[0] = 3
        elif s1_ratio < 300:
            score[0] = 2
        elif s1_ratio < 400:
            score[0] = 1

        s2 = s2_gcs
        if not np.isnan(s2):
            if s2 < 6:
                score[1] = 4
            elif s2 < 10:
                score[1] = 3
            elif s2 < 13:
                score[1] = 2
            elif s2 < 15:
                score[1] = 1

        s3 = s3_map
        # dopa = 0
        # epi = 0
        # if epi:
        #     score[2] = 3
        # elif dopa:
        #     score[2] = 2
        press = stats['vasopress_ct'][i]
        if press:
            score[2] = 3
        elif s3 < 70:
            score[2] = 1

        s4 = s4_bili
        if s4 > 12.0:
            score[3] = 4
        elif s4 > 6.0:
            score[3] = 3
        elif s4 > 2.0:
            score[3] = 2
        elif s4 > 1.2:
            score[3] = 1

        s5 = s5_plt
        if s5 < 20:
            score[4] = 4
        elif s5 < 50:
            score[4] = 3
        elif s5 < 100:
            score[4] = 2
        elif s5 < 150:
            score[4] = 1

        s6 = s6_scr
        if s6 > 5.0:
            score[5] = 4
        elif s6 > 3.5:
            score[5] = 3
        elif s6 > 2.0:
            score[5] = 2
        elif s6 > 1.2:
            score[5] = 1

        out.write(',%d,%d,%d,%d,%d,%d\n' % (score[0], score[1], score[2], score[3], score[4], score[5]))
        sofas[i, :] = score
        if v:
            print(np.sum(score))
    return sofas


def get_apache(ids, stats, scr_interp, days_interp, out_name, v=False):
    all_ids = stats['ids'][:]
    pt_sel = np.array([x in ids for x in all_ids])
    temps = stats['temperature'][:][pt_sel]
    maps = stats['map'][:][pt_sel]
    hrs = stats['heart_rate'][:][pt_sel]
    resps = stats['respiration'][:][pt_sel]
    fio2s = stats['fio2'][:][pt_sel]
    gcss = stats['glasgow'][:][pt_sel]
    pao2s = stats['pao2'][:][pt_sel]
    paco2s = stats['paco2'][:][pt_sel]
    phs = stats['ph'][:][pt_sel]
    nas = stats['sodium'][:][pt_sel]
    pks = stats['potassium'][:][pt_sel]
    hemats = stats['hematocrit'][:][pt_sel]
    wbcs = stats['wbc'][:][pt_sel]
    ages = stats['age'][:][pt_sel]

    out = open(out_name, 'w')
    ct = 0
    apaches = np.zeros((len(ids), 13))
    for i in range(len(ids)):
        idx = ids[i]
        out.write('%d' % idx)

        s1_low = temps[i, 0]
        s1_high = temps[i, 1]

        s2_low = maps[i, 0]
        s2_high = maps[i, 1]

        s3_low = hrs[i, 0]
        s3_high = hrs[i, 1]

        s4_low = resps[i, 0]
        s4_high = resps[i, 1]
        
        s5_f = fio2s[:, 1]
        s5_po = pao2s[i, 1]
        s5_pco = paco2s[i, 1]

        s6_low = phs[i, 0]
        s6_high = phs[i, 1]

        s7_low = nas[i, 0]
        s7_high = nas[i, 1]

        s8_low = pks[i, 0]
        s8_high = pks[i, 1]

        s10_low = hemats[i, 0]
        s10_high = hemats[i, 1]

        s11_low = wbcs[i, 0]
        s11_high = wbcs[i, 1]

        # Find maximum value in day 0
        s9_scr = np.nan
        day_sel = np.where(days_interp[i] == 0)[0]
        if day_sel.size > 0:
            s9_scr = np.max(scr_interp[i][day_sel])
        else:
            day_sel = np.where(days_interp[i] == 1)[0]
            if day_sel.size > 0:
                s9_scr = np.max(scr_interp[i][day_sel])

        s12_gcs = gcss[i, 0]

        s13_age = ages[i]

        score = np.zeros(13)

        if s1_low < 30 or s1_high > 40:
            score[0] = 4
        elif s1_low < 32 or s1_high > 39:
            score[0] = 3
        elif s1_low < 34:
            score[0] = 2
        elif s1_low < 36 or s1_high > 38.5:
            score[0] = 1

        if s2_low < 49 or s2_high > 160:
            score[1] = 4
        elif s2_high > 130:
            score[1] = 3
        elif s2_low < 70 or s2_high > 110:
            score[1] = 2

        if s3_low < 49 or s3_high > 160:
            score[2] = 4
        elif s3_high > 130:
            score[2] = 3
        elif s3_low < 70 or s3_high > 110:
            score[2] = 2

        if s4_low <= 5 or s4_high >= 50:
            score[3] = 4
        elif s4_high >= 35:
            score[3] = 3
        elif s4_low <= 10:
            score[3] = 2
        elif s4_low < 12 or s4_high >= 25:
            score[3] = 1

        if s5_f >= 0.5:
            aado2 = s5_f * 713 - (s5_pco / 0.8) - s5_po
            if aado2 >= 500:
                score[4] = 4
            elif aado2 > 350:
                score[4] = 3
            elif aado2 > 200:
                score[4] = 2
        else:
            if s5_po < 55:
                score[4] = 4
            elif s5_po < 60:
                score[4] = 3
            elif s5_po < 70:
                score[4] = 1

        if s6_low <= 7.15 or s6_high >= 7.7:
            score[5] = 4
        elif s6_low < 7.25 or s6_high >= 7.6:
            score[5] = 3
        elif s6_low < 7.33:
            score[5] = 2
        elif s6_high >= 7.5:
            score[5] = 1

        if s7_low <= 110 or s7_high >= 180:
            score[6] = 4
        elif s7_low < 120 or s7_high >= 160:
            score[6] = 3
        elif s7_low < 130 or s7_high >= 155:
            score[6] = 2
        elif s7_high >= 150:
            score[6] = 1

        if s8_low < 2.5 or s8_high >= 7:
            score[7] = 4
        elif s8_high >= 6:
            score[7] = 3
        elif s8_low < 3:
            score[7] = 2
        elif s8_low < 3.5 or s7_high >= 5.5:
            score[7] = 1

        # Find maximum value in day 0
        if s9 >= 3.5:
            score[8] = 4
        elif s9 >= 2:
            score[8] = 3
        elif s9 >= 1.5 or s9 < 0.6:
            score[8] = 2

        if s10_low < 20 or s10_high >= 60:
            score[9] = 4
        elif s10_low < 30 or s10_high >= 50:
            score[9] = 2
        elif s10_high >= 46:
            score[9] = 1

        if s11_low < 1 or s11_high >= 40:
            score[10] = 4
        elif s11_low < 3 or s11_high >= 20:
            score[10] = 2
        elif s11_high >= 15:
            score[10] = 1

        s12 = s12_gcs
        if not np.isnan(s12):
            score[11] = 15 - s12

        age = s13_age
        if age >= 75:
            score[12] = 6
        elif 65 <= age < 75:
            score[12] = 5
        elif 55 <= age < 65:
            score[12] = 3
        elif 45 <= age < 55:
            score[12] = 2

        for i in range(len(score)):
            out.write(',%d' % (score[i]))
        out.write('\n')
        apaches[i, :] = score
        if v:
            print(np.sum(score))
    return apaches


def get_apache_dallas(stats, out_name):
    ids = stats['ids'][:]
    ages = stats['age'][:]

    out = open(out_name, 'w')
    ct = 0
    apaches = np.zeros((len(ids), 13))
    for i in range(len(ids)):
        idx = ids[i]
        out.write('%d' % idx)
        s1_low = float(stats['temperature'][i][1, 0])
        s1_high = float(stats['temperature'][i][1, 1])

        s2_low = float(stats['map'][i][1, 0])
        s2_high = float(stats['map'][i][1, 1])

        s3_low = float(stats['hr'][i][1, 0])
        s3_high = float(stats['hr'][i][1, 1])

        s4_low = float(stats['respiration'][i][1, 0])
        s4_high = float(stats['respiration'][i][1, 1])

        s5_f = float(stats['fio2'][i][1, 1])
        s5_po = float(stats['po2'][i][1, 1])
        s5_pco = float(stats['pco2'][i][1, 1])
        if not np.isnan(s5_po):
            s5_po = s5_po / 100
        if not np.isnan(s5_pco):
            s5_pco = s5_pco / 100

        s6_low = float(stats['ph'][i][1, 0])
        s6_high = float(stats['ph'][i][1, 1])

        s7_low = float(stats['sodium'][i][1, 0])
        s7_high = float(stats['sodium'][i][1, 1])

        s8_low = float(stats['potassium'][i][1, 0])
        s8_high = float(stats['potassium'][i][1, 1])

        s10_low = float(stats['hematocrit'][i][1, 0])
        s10_high = float(stats['hematocrit'][i][1, 1])

        s11_low = float(stats['wbc'][i][1, 0])
        s11_high = float(stats['wbc'][i][1, 1])

        s9 = float(np.max(stats['scr_agg'][i][:, 1]))

        s12_gcs = float(stats['glasgow'][i][1, 0])

        s13_age = float(stats['age'][i])

        score = np.zeros(13)

        if s1_low < 30 or s1_high > 40:
            score[0] = 4
        elif s1_low < 32 or s1_high > 39:
            score[0] = 3
        elif s1_low < 34:
            score[0] = 2
        elif s1_low < 36 or s1_high > 38.5:
            score[0] = 1

        if s2_low < 49 or s2_high > 160:
            score[1] = 4
        elif s2_high > 130:
            score[1] = 3
        elif s2_low < 70 or s2_high > 110:
            score[1] = 2

        if s3_low < 49 or s3_high > 160:
            score[2] = 4
        elif s3_high > 130:
            score[2] = 3
        elif s3_low < 70 or s3_high > 110:
            score[2] = 2

        if s4_low <= 5 or s4_high >= 50:
            score[3] = 4
        elif s4_high >= 35:
            score[3] = 3
        elif s4_low <= 10:
            score[3] = 2
        elif s4_low < 12 or s4_high >= 25:
            score[3] = 1

        if s5_f >= 0.5:
            aado2 = s5_f * 713 - (s5_pco / 0.8) - s5_po
            if aado2 >= 500:
                score[4] = 4
            elif aado2 > 350:
                score[4] = 3
            elif aado2 > 200:
                score[4] = 2
        else:
            if s5_po < 55:
                score[4] = 4
            elif s5_po < 60:
                score[4] = 3
            elif s5_po < 70:
                score[4] = 1

        if s6_low <= 7.15 or s6_high >= 7.7:
            score[5] = 4
        elif s6_low < 7.25 or s6_high >= 7.6:
            score[5] = 3
        elif s6_low < 7.33:
            score[5] = 2
        elif s6_high >= 7.5:
            score[5] = 1

        if s7_low <= 110 or s7_high >= 180:
            score[6] = 4
        elif s7_low < 120 or s7_high >= 160:
            score[6] = 3
        elif s7_low < 130 or s7_high >= 155:
            score[6] = 2
        elif s7_high >= 150:
            score[6] = 1

        if s8_low < 2.5 or s8_high >= 7:
            score[7] = 4
        elif s8_high >= 6:
            score[7] = 3
        elif s8_low < 3:
            score[7] = 2
        elif s8_low < 3.5 or s7_high >= 5.5:
            score[7] = 1

        # Find maximum value in day 0
        if s9 >= 3.5:
            score[8] = 4
        elif s9 >= 2:
            score[8] = 3
        elif s9 >= 1.5 or s9 < 0.6:
            score[8] = 2

        if s10_low < 20 or s10_high >= 60:
            score[9] = 4
        elif s10_low < 30 or s10_high >= 50:
            score[9] = 2
        elif s10_high >= 46:
            score[9] = 1

        if s11_low < 1 or s11_high >= 40:
            score[10] = 4
        elif s11_low < 3 or s11_high >= 20:
            score[10] = 2
        elif s11_high >= 15:
            score[10] = 1

        s12 = s12_gcs
        if not np.isnan(s12):
            score[11] = 15 - s12

        age = s13_age
        if age >= 75:
            score[12] = 6
        elif 65 <= age < 75:
            score[12] = 5
        elif 55 <= age < 65:
            score[12] = 3
        elif 45 <= age < 55:
            score[12] = 2

        for i in range(len(score)):
            out.write(',%d' % (score[i]))
        out.write('\n')
        apaches[i, :] = score
        if v:
            print(np.sum(score))
    return apaches


# Update dialysis so that it does not exclude patients with RRT prior to discharge
# Try 90 from admission vs. 90 from discharge
def get_MAKE90(ids, stats,
               dia_m, crrt_locs, hd_locs, pd_locs,
               scm_esrd, scm_during, scm_after,
               usrds_esrd, usrds_esrd_date_loc,
               scr_all_m, scr_val_loc, scr_date_loc,
               esrd_man_rev, man_rev_dur, man_rev_rrt,
               out, ref='disch', min_day=7, buffer=0, ct=1):

    sexes = stats['gender'][:]
    ages = stats['age'][:]
    races = stats['race'][:]
    bsln_gfrs = stats['baseline_gfr'][:]
    all_ids = stats['ids'][:]
    windows = stats['icu_dates'][:].astype('unicode')
    dods = stats['dod'][:].astype('unicode')

    # print('id,died,gfr_drop,new_dialysis')
    out.write('id,died_inp,died_in_window,esrd_manual_revision,esrd_scm,esrd_usrds,new_dialysis_manual_revision,new_dialysis,gfr_drop_25,gfr_drop_50,n_vals,delta\n')
    scores = []
    out_ids = []
    for i in range(len(ids)):
        tid = ids[i]

        idx = np.where(all_ids == tid)[0][0]
        if stats['died_inp'][idx] > 0:
            died_inp = 1
        else:
            died_inp = 0
        sex = sexes[i]
        race = races[i]
        age = ages[i]

        died = 0
        esrd = 0
        esrd_man = 0
        esrd_scm = 0
        esrd_usrds = 0
        gfr_drop_25 = 0
        gfr_drop_50 = 0
        dia_dep_man = 0
        dia_dep = 0

        bsln_gfr = bsln_gfrs[idx]

        admit = get_date(windows[idx, 0])
        disch = get_date(windows[idx, 1])

        if ref == 'disch':
            tmin = get_date(windows[idx, 1])
        elif ref == 'admit':
            tmin = get_date(windows[idx, 0])

        dod = get_date(dods[idx])
        if dod != 'nan':
            if dod - tmin < datetime.timedelta(90):
                died = 1

        # ESRD
        # Check manual revision first
        man_rev_loc = np.where(esrd_man_rev[:, 0]) == tid
        if man_rev_loc.size > 0:
            if esrd_man_rev[man_rev_loc[0], man_rev_dur] == 1:
                esrd_man = 1
            if esrd_man_rev[man_rev_loc[0], man_rev_rrt] == 1:
                dia_dep_man = 1
        # If patient not included in manual revision, check SCM and USRDS separately
        # else:
            # SCM
        esrd_dloc = np.where(scm_esrd[:, 0] == tid)[0]
        if esrd_dloc.size > 0:
            if scm_esrd[esrd_dloc[0], scm_during] == 'Y':
                esrd_scm = 1
            # if scm_esrd[esrd_dloc[0], scm_after] == 'Y':
            #     esrd_scm = 1
        # USRDS
        esrd_dloc = np.where(usrds_esrd[:, 0] == tid)[0]
        if esrd_dloc.size > 0:
            tdate = get_date(usrds_esrd[esrd_dloc[0], usrds_esrd_date_loc])
            if tdate != 'nan':
                if datetime.timedelta(0) < tdate - tmin < datetime.timedelta(90):
                    esrd_usrds = 1

        dia_locs = np.where(dia_m[:, 0] == tid)[0]
        for j in range(len(dia_locs)):
            crrt_start = get_date(dia_m[dia_locs[j], crrt_locs[0]])
            pd_start = get_date(dia_m[dia_locs[j], pd_locs[0]])
            hd_start = get_date(dia_m[dia_locs[j], hd_locs[0]])
            if str(crrt_start) != 'nan':
                tstart = get_date(dia_m[dia_locs[j], crrt_locs[0]])
                tstop = get_date(dia_m[dia_locs[j], crrt_locs[1]])
            elif str(pd_start) != 'nan':
                tstart = get_date(dia_m[dia_locs[j], pd_locs[0]])
                tstop = get_date(dia_m[dia_locs[j], pd_locs[1]])
            elif str(hd_start) != 'nan':
                tstart = get_date(dia_m[dia_locs[j], hd_locs[0]])
                tstop = get_date(dia_m[dia_locs[j], hd_locs[1]])

            if tstart <= admit:
                if tstop >= disch - datetime.timedelta(2):
                    dia_dep = 1
            elif tstart <= disch:
                if tstop >= disch - datetime.timedelta(2):
                    dia_dep = 1
            # # Must be on RRT within the 48hrs prior to discharge
            # if ref == 'admit':
            #     if tstart != 'nan':
            #         if datetime.timedelta(1) < tstart - tmin < datetime.timedelta(90):
            #             dia_dep = 1
            #     if tstop != 'nan':
            #         if datetime.timedelta(1) <= tstop - tmin < datetime.timedelta(90):
            #             dia_dep = 1
            # if ref == 'disch':
            #     if tstart != 'nan':
            #         if datetime.timedelta(-2) < tstart - tmin < datetime.timedelta(90):
            #             dia_dep = 1
            #     if tstop != 'nan':
            #         if datetime.timedelta(-2) <= tstop - tmin < datetime.timedelta(90):
            #             dia_dep = 1

        nvals = 0
        delta_str = 'not_evaluated'
        if not died and not dia_dep and not esrd:
            gfr90 = 1000
            delta = 90
            gfrs = []
            deltas = []
            delta_str = 'no_valid_records'
            scr_locs = np.where(scr_all_m[:, 0] == tid)[0]
            for j in range(len(scr_locs)):
                tdate = str(scr_all_m[scr_locs[j], scr_date_loc])
                tdate = get_date(tdate)
                if tdate != 'nan':
                    td = tdate - tmin
                    if datetime.timedelta(0) < td < datetime.timedelta(90) + datetime.timedelta(buffer):
                        if td.days <= min_day:
                            continue
                        dif = ((tmin + datetime.timedelta(90)) - tdate).days
                        # if ref == 'admit' and 0 < dif < min_day:
                        #     continue
                        tscr = scr_all_m[scr_locs[j], scr_val_loc]
                        tgfr = calc_gfr(tscr, sex, race, age)
                        gfrs.append(tgfr)
                        deltas.append(dif)
                        nvals = 1
                        if delta_str == 'no_valid_records' or abs(dif) < abs(delta):
                            tscr = scr_all_m[scr_locs[j], scr_val_loc]
                            tgfr = calc_gfr(tscr, sex, race, age)
                            gfr90 = tgfr
                            delta = dif
                            delta_str = str(delta)
            if ct > 1:
                assert len(deltas) == len(gfrs)
                if len(deltas) > 1:
                    tct = min(ct, len(deltas))
                    gfrs = np.array(gfrs)
                    deltas = np.array(deltas)
                    o = np.argsort(np.abs(deltas))
                    gfr90 = np.mean(gfrs[o[:tct]])
                    delta = deltas[o[0]]
                    delta_str = str(delta)
                    nvals = tct

            thresh = 100 - 25
            rel_pct = (gfr90 / bsln_gfr) * 100
            if rel_pct < thresh:
                gfr_drop_25 = 1
            thresh = 100 - 50
            rel_pct = (gfr90 / bsln_gfr) * 100
            if rel_pct < thresh:
                gfr_drop_50 = 1

        out_ids.append(tid)
        out.write('%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s\n' % (tid, died_inp, died, esrd_man, esrd_scm, esrd_usrds, dia_dep_man, dia_dep, gfr_drop_25, gfr_drop_50, nvals, delta_str))
        scores.append(np.array((died_inp, died, esrd_man, esrd_scm, esrd_usrds, dia_dep_man, dia_dep, gfr_drop_25, gfr_drop_50)))
        # print('%d,%d,%d,%d' % (idx, died, gfr_drop, dia_dep))
    scores = np.array(scores)
    return scores, out_ids


# %%
def get_MAKE90_dallas(ids, stats,
                      dia_m, dia_start_loc, dia_stop_loc,
                      esrd_m, esrd_during_loc, esrd_after_loc,
                      scr_all_m, scr_val_loc, scr_date_loc,
                      out, ref='disch', min_day=7, buffer=0, ct=1):

    sexes = stats['gender'][:]
    ages = stats['age'][:]
    races = stats['race'][:]
    bsln_gfrs = stats['baseline_gfr'][:]
    all_ids = stats['ids'][:]
    windows = stats['icu_dates'][:]
    dods = stats['dod'][:]

    # print('id,died,gfr_drop,new_dialysis')
    out.write('id,died,esrd,new_dialysis,gfr_drop_25,gfr_drop_50,n_vals,delta\n')
    scores = []
    out_ids = []
    for i in range(len(ids)):
        tid = ids[i]

        idx = np.where(all_ids == tid)[0][0]

        sex = sexes[i]
        race = races[i]
        age = ages[i]

        died = 0
        esrd = 0
        gfr_drop_25 = 0
        gfr_drop_50 = 0
        dia_dep = 0

        if stats['died_inp'][idx] > 0:
            died = 1

        bsln_gfr = bsln_gfrs[idx]

        admit = get_date(windows[idx, 0])
        disch = get_date(windows[idx, 1])

        if ref == 'disch':
            tmin = get_date(windows[idx, 1])
        elif ref == 'admit':
            tmin = get_date(windows[idx, 0])

        dod = get_date(dods[idx])
        if dod != 'nan':
            if dod - tmin < datetime.timedelta(90):
                died = 1

        # ESRD
        esrd_dloc = np.where(esrd_m[:, 0] == tid)[0]
        if esrd_dloc.size > 0:
            if esrd_m[esrd_dloc[0], esrd_during_loc] == 'DURING_INDEXED_ADT':
                esrd = 1
            if esrd_m[esrd_dloc[0], esrd_after_loc] == 'AFTER_INDEXED_ADT':
                esrd = 1

        dia_locs = np.where(dia_m[:, 0] == tid)[0]
        for j in range(len(dia_locs)):
            start = get_date(dia_m[dia_locs[j], dia_start_loc])
            stop = get_date(dia_m[dia_locs[j], dia_stop_loc])
            if start <= admit:
                if stop >= disch - datetime.timedelta(2):
                    dia_dep = 1
            elif start <= disch:
                if stop >= disch - datetime.timedelta(2):
                    dia_dep = 1

        nvals = 0
        delta_str = 'not_evaluated'
        if not died and not dia_dep and not esrd:
            gfr90 = 1000
            delta = 90
            gfrs = []
            deltas = []
            delta_str = 'no_valid_records'
            scr_locs = np.where(scr_all_m[:, 0] == tid)[0]
            for j in range(len(scr_locs)):
                tdate = str(scr_all_m[scr_locs[j], scr_date_loc])
                tdate = get_date(tdate)
                if tdate != 'nan':
                    td = tdate - tmin
                    if datetime.timedelta(0) < td < datetime.timedelta(90) + datetime.timedelta(buffer):
                        if td.days <= min_day:
                            continue
                        dif = ((tmin + datetime.timedelta(90)) - tdate).days
                        # if ref == 'admit' and 0 < dif < min_day:
                        #     continue
                        tscr = scr_all_m[scr_locs[j], scr_val_loc]
                        tgfr = calc_gfr(tscr, sex, race, age)
                        gfrs.append(tgfr)
                        deltas.append(dif)
                        nvals = 1
                        if delta_str == 'no_valid_records' or abs(dif) < abs(delta):
                            tscr = scr_all_m[scr_locs[j], scr_val_loc]
                            tgfr = calc_gfr(tscr, sex, race, age)
                            gfr90 = tgfr
                            delta = dif
                            delta_str = str(delta)
            if ct > 1:
                assert len(deltas) == len(gfrs)
                if len(deltas) > 1:
                    tct = min(ct, len(deltas))
                    gfrs = np.array(gfrs)
                    deltas = np.array(deltas)
                    o = np.argsort(np.abs(deltas))
                    gfr90 = np.mean(gfrs[o[:tct]])
                    delta = deltas[o[0]]
                    delta_str = str(delta)
                    nvals = tct

            thresh = 100 - 25
            rel_pct = (gfr90 / bsln_gfr) * 100
            if rel_pct < thresh:
                gfr_drop_25 = 1
            thresh = 100 - 50
            rel_pct = (gfr90 / bsln_gfr) * 100
            if rel_pct < thresh:
                gfr_drop_50 = 1

        out_ids.append(tid)
        out.write('%d,%d,%d,%d,%d,%d,%d,%s\n' % (tid, died, esrd, dia_dep, gfr_drop_25, gfr_drop_50, nvals, delta_str))
        scores.append(np.array((died, esrd, dia_dep, gfr_drop_25, gfr_drop_50)))
        # print('%d,%d,%d,%d' % (idx, died, gfr_drop, dia_dep))
    scores = np.array(scores)
    return scores, out_ids


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    return TP, FP, TN, FN


# %%
def get_even_pos_neg(target, method='under'):
    '''
    Returns even number of positive/negative examples
    :param target:
    :return:
    '''
    # If even distribution of pos/neg for training
    pos_idx = np.where(target)[0]
    neg_idx = np.where(target == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    if method == 'under':
        n_train = min(n_pos, n_neg)
        pos_idx = np.random.permutation(pos_idx)[:n_train]
        neg_idx = np.random.permutation(neg_idx)[:n_train]
    elif method == 'rand_over':
        if n_pos > n_neg:
            neg_idx = resample(neg_idx, replace=True, n_samples=n_pos, random_state=123)
        else:
            pos_udx = resample(pos_idx, replace=True, n_samples=n_neg, random_state=123)
    sel_idx = np.sort(np.concatenate((pos_idx, neg_idx)))
    return sel_idx
