#!/usr/bin/env python2
# -*- coding: utf-8 -*-`
"""
Created on Wed Nov 29 13:15:39 2017

@author: taylorsmith
"""
from __future__ import division
from kdigo_funcs import load_csv, calc_gfr, get_date
from dtw_distance import dtw_p, mismatch_penalty_func, extension_penalty_func
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
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support
from tqdm import trange
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri as n2r
from warnings import filterwarnings

def summarize_stats(f, ids, kdigos, days, scrs, icu_windows, hosp_windows, data_path, grp_name='meta', tlim=7, maxDay=1):
    try:
        meta = f[grp_name]
    except:
        meta = f.create_group(grp_name)
        meta.create_dataset('ids', data=ids, dtype=int)

    icu_windows_save = []
    hosp_windows_save = []
    for tid in ids:
        icu_windows_save.append(icu_windows[tid])
        hosp_windows_save.append(hosp_windows[tid])

    # stringType = h5py.special_dtype(vlen=np.str)
    stringType = 'S20'
    if 'icu_dates' not in list(meta):
        icu_windows_save = np.array(icu_windows_save, dtype='S20')
        meta.create_dataset('icu_dates', data=icu_windows_save, dtype=stringType)

    if 'hosp_dates' not in list(meta):
        hosp_windows_save = np.array(hosp_windows_save, dtype='S20')
        meta.create_dataset('hosp_dates', data=hosp_windows_save, dtype=stringType)

    if 'num_episodes' not in list(meta):
        n_eps = []
        for i in range(len(kdigos)):
            n_eps.append(count_eps(kdigos[i]))
        meta.create_dataset('num_episodes', data=n_eps, dtype=int)

    if 'age' not in list(meta):
        print('Getting patient demographics...')
        genders, eths, ages = get_uky_demographics(ids, hosp_windows, data_path)
        genders = np.array(genders, dtype=int)
        eths = np.array(eths, dtype=int)
        ages = np.array(ages, dtype=float)
        meta.create_dataset('age', data=ages, dtype=float)
        meta.create_dataset('gender', data=genders, dtype=int)
        meta.create_dataset('race', data=eths, dtype=int)
    else:
        genders = meta['gender'][:]
        ages = meta['age'][:]
        eths = meta['race'][:]

    if 'baseline_scr' not in list(meta):
        bvals = load_csv(os.path.join(data_path, 'all_baseline_info.csv'), ids, idxs=[1], dt=float, skip_header=True)
        btypes_temp = load_csv(os.path.join(data_path, 'all_baseline_info.csv'), ids, idxs=[2], dt=str, skip_header=True)
        gfrs = np.zeros(len(bvals))
        btypes = np.zeros(len(bvals), dtype='|S20').astype(str)
        for i in range(len(gfrs)):
            gfrs[i] = calc_gfr(bvals[i], genders[i], eths[i], ages[i])
            if btypes_temp[i] == 'mdrd':
                btypes[i] = 'imputed'
            else:
                btypes[i] = 'measured'
        meta.create_dataset('baseline_scr', data=bvals, dtype=float)
        meta.create_dataset('baseline_gfr', data=gfrs, dtype=float)
        meta.create_dataset('baseline_type', data=np.array(btypes, dtype=stringType), dtype=stringType)

    if 'map' not in list(meta):
        print('Getting patient vitals...')
        temps, maps, hrs = get_uky_vitals(ids, data_path)
        temps = np.array(temps, dtype=float)
        maps = np.array(maps, dtype=float)
        hrs = np.array(hrs, dtype=float)
        meta.create_dataset('map', data=maps, dtype=float)
        meta.create_dataset('temperature', data=temps, dtype=float)
        meta.create_dataset('heart_rate', data=hrs, dtype=float)

    if 'charlson' not in list(meta):
        print('Getting patient comorbidities...')
        c_scores, e_scores, smokers, charls, elixs = get_uky_comorbidities(ids, data_path)
        c_scores = np.array(c_scores, dtype=float)  # Float bc posible NaNs
        e_scores = np.array(e_scores, dtype=float)  # Float bc posible NaNs
        smokers = np.array(smokers, dtype=bool)
        meta.create_dataset('smoker', data=smokers, dtype=int)
        meta.create_dataset('charlson', data=c_scores, dtype=int)
        meta.create_dataset('elixhauser', data=e_scores, dtype=int)
        meta.create_dataset('charlson_components', data=charls, dtype=int)
        meta.create_dataset('elixhauser_components', data=elixs, dtype=int)

    if 'bmi' not in list(meta):
        print('Getting clinical others...')
        bmis, weights, heights, fio2s, resps, gcs = get_uky_clinical_others(ids, data_path)
        bmis = np.array(bmis, dtype=float)
        weights = np.array(weights, dtype=float)
        heights = np.array(heights, dtype=float)
        fio2s = np.array(fio2s, dtype=float)
        resps = np.array(resps, dtype=float)
        gcs = np.array(gcs, dtype=float)
        meta.create_dataset('bmi', data=bmis, dtype=float)
        meta.create_dataset('weight', data=weights, dtype=float)
        meta.create_dataset('height', data=heights, dtype=float)
        meta.create_dataset('fio2', data=fio2s, dtype=float)
        meta.create_dataset('respiration', data=resps, dtype=float)
        meta.create_dataset('glasgow', data=gcs, dtype=float)
    else:
        weights = meta['weight'][:]

    if 'septic' not in list(meta):
        print('Getting diagnoses...')
        septics, diabetics, hypertensives = get_uky_diagnoses(ids, data_path)
        septics = np.array(septics, dtype=int)
        diabetics = np.array(diabetics, dtype=int)
        hypertensives = np.array(hypertensives, int)
        meta.create_dataset('septic', data=septics, dtype=int)
        meta.create_dataset('diabetic', data=diabetics, dtype=int)
        meta.create_dataset('hypertensive', data=hypertensives, dtype=int)

    if 'net_fluid' not in list(meta):
        print('Getting fluid IO...')
        net_fluids, gros_fluids, fos, cfbs = get_uky_fluids(ids, weights, data_path)
        net_fluids = np.array(net_fluids, dtype=float)
        gros_fluids = np.array(gros_fluids, dtype=float)
        fos = np.array(fos, dtype=float)
        meta.create_dataset('net_fluid', data=net_fluids, dtype=float)
        meta.create_dataset('gross_fluid', data=gros_fluids, dtype=float)
        meta.create_dataset('fluid_overload', data=fos, dtype=float)
        meta.create_dataset('cum_fluid_balance', data=cfbs, dtype=float)

    if 'died_inp' not in list(meta):
        print('Getting mortalities...')
        died_inps, died_90d_admit, died_90d_disch, \
        died_120d_admit, died_120d_disch, dtds, dods = get_uky_mortality(ids, hosp_windows, icu_windows, data_path)
        dods = np.array(dods, dtype=stringType)
        meta.create_dataset('days_to_death', data=dtds, dtype=float)
        meta.create_dataset('died_inp', data=died_inps, dtype=int)
        meta.create_dataset('died_90d_admit', data=died_90d_admit, dtype=int)
        meta.create_dataset('died_90d_disch', data=died_90d_disch, dtype=int)
        meta.create_dataset('died_120d_admit', data=died_120d_admit, dtype=int)
        meta.create_dataset('died_120d_disch', data=died_120d_disch, dtype=int)
        meta.create_dataset('dod', data=dods, dtype=stringType)
    else:
        died_inps = meta['died_inp'][:]

    if 'rrt_flag' not in list(meta):
        print('Getting dialysis...')
        rrt_flags, hd_days, crrt_days, hd_dayss_win, \
        crrt_dayss_win, hd_frees_7d, hd_frees_28d, hd_trtmts, \
        crrt_frees_7d, crrt_frees_28d, crrt_flags, hd_flags = get_uky_rrt(ids, died_inps, icu_windows, data_path)
        rrt_flags = np.array(rrt_flags, dtype=int)
        hd_days = np.array(hd_days, dtype=int)
        crrt_days = np.array(crrt_days, dtype=int)
        hd_frees_7d = np.array(hd_frees_7d, dtype=int)
        hd_frees_28d = np.array(hd_frees_28d, dtype=int)
        crrt_frees_7d = np.array(crrt_frees_7d, dtype=int)
        hd_trtmts = np.array(hd_trtmts, dtype=int)
        hd_frees_28d = np.array(hd_frees_28d, dtype=int)
        meta.create_dataset('rrt_flag', data=rrt_flags, dtype=int)
        meta.create_dataset('crrt_flag', data=crrt_flags, dtype=int)
        meta.create_dataset('hd_flag', data=hd_flags, dtype=int)
        meta.create_dataset('hd_days', data=hd_days, dtype=float)
        meta.create_dataset('crrt_days', data=crrt_days, dtype=float)
        meta.create_dataset('hd_free_win', data=hd_frees_7d, dtype=float)
        meta.create_dataset('crrt_free_win', data=crrt_frees_7d, dtype=float)
        meta.create_dataset('hd_free_28d', data=hd_frees_28d, dtype=float)
        meta.create_dataset('crrt_free_28d', data=crrt_frees_28d, dtype=float)
        meta.create_dataset('hd_treatments', data=hd_trtmts, dtype=float)

    if 'vad' not in list(meta):
        print('Getting other organ support...')
        mv_flags, mv_days, mv_frees, ecmos, vads, iabps, mhs = get_uky_organsupp(ids, died_inps, icu_windows, data_path)
        mv_flags = np.array(mv_flags, dtype=int)
        mv_days = np.array(mv_days, dtype=float)
        mv_frees = np.array(mv_frees, dtype=float)
        ecmos = np.array(ecmos, dtype=int)
        iabps = np.array(iabps, dtype=int)
        vads = np.array(vads, dtype=int)

        meta.create_dataset('mv_flag', data=mv_flags, dtype=int)
        meta.create_dataset('mv_days', data=mv_days, dtype=float)
        meta.create_dataset('mv_free_days', data=mv_frees, dtype=float)
        meta.create_dataset('ecmo', data=ecmos, dtype=int)
        meta.create_dataset('iabp', data=iabps, dtype=int)
        meta.create_dataset('vad', data=vads, dtype=int)
        meta.create_dataset('mhs', data=mhs, dtype=int)

    if 'nephrotox_ct' not in list(meta):
        print('Getting medications...')
        nephrotox_cts, vasopres_cts, dopas, epis = get_uky_medications(ids, icu_windows, data_path)
        nephrotox_cts = np.array(nephrotox_cts, dtype=int)
        vasopres_cts = np.array(vasopres_cts, dtype=int)
        dopas = np.array(dopas)
        epis = np.array(epis)
        meta.create_dataset('nephrotox_ct', data=nephrotox_cts, dtype=int)
        meta.create_dataset('vasopress_ct', data=vasopres_cts, dtype=int)
        meta.create_dataset('dopa', data=dopas, dtype=int)
        meta.create_dataset('epinephrine', data=epis, dtype=int)

    if 'anemia' not in list(meta):
        print('Getting labs...')
        anemics, bilis, buns, \
        hemats, hemos, pltlts, \
        nas, pks, albs, lacs, phs, \
        po2s, pco2s, wbcs, bicarbs = get_uky_labs(ids, genders, data_path)
        anemics = np.array(anemics, dtype=int)
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
        po2s = np.array(po2s, dtype=float)
        pco2s = np.array(pco2s, dtype=float)
        wbcs = np.array(wbcs, dtype=float)
        bicarbs = np.array(bicarbs, dtype=float)
        meta.create_dataset('anemia', data=anemics, dtype=int)
        meta.create_dataset('albumin', data=albs, dtype=float)
        meta.create_dataset('lactate', data=lacs, dtype=float)
        meta.create_dataset('bilirubin', data=bilis, dtype=float)
        meta.create_dataset('bun', data=buns, dtype=float)
        meta.create_dataset('hematocrit', data=hemats, dtype=float)
        meta.create_dataset('hemoglobin', data=hemos, dtype=float)
        meta.create_dataset('platelets', data=pltlts, dtype=float)
        meta.create_dataset('sodium', data=nas, dtype=float)
        meta.create_dataset('potassium', data=pks, dtype=float)
        meta.create_dataset('po2', data=po2s, dtype=float)
        meta.create_dataset('pco2', data=pco2s, dtype=float)
        meta.create_dataset('wbc', data=wbcs, dtype=float)
        meta.create_dataset('bicarbonate', data=bicarbs, dtype=float)
        meta.create_dataset('ph', data=phs, dtype=float)

    if 'urine_out' not in list(meta):
        print('Getting urine flow...')
        urine_outs, urine_flows = get_uky_urine(ids, weights, data_path)
        urine_outs = np.array(urine_outs, dtype=float)
        urine_flows = np.array(urine_flows, dtype=float)
        meta.create_dataset('urine_out', data=urine_outs, dtype=float)
        meta.create_dataset('urine_flow', data=urine_flows, dtype=float)

    if 'unPlannedAdmissions' not in list(meta):
        unPlannedAdmissions = get_uky_planned_admissions(ids, icu_windows, data_path)
        meta.create_dataset('unPlannedAdmissions', data=unPlannedAdmissions, dtype=int)

    if 'hosp_los' not in list(meta):
        mks_win = []
        mks_whole = []
        neps = []
        hosp_days = []
        hosp_frees = []
        icu_days = []
        icu_frees = []

        admit_scrs = []
        peak_scrs = []

        for i in range(len(ids)):
            tid = ids[i]
            td = np.array(days[i])
            (hosp_admit, hosp_disch) = hosp_windows[tid]
            (icu_admit, icu_disch) = icu_windows[tid]
            mkwin = 0
            mk = 0

            if len(scrs[i]):
                mask = np.where(td <= tlim)[0]
                if len(mask) > 0:
                    mkwin = np.max(kdigos[i][mask])
                mk = np.max(kdigos[i])

                admit_scr = scrs[i][0]
                peak_scr = np.max(scrs[i])
            else:
                admit_scr = np.nan
                peak_scr = np.nan

            try:
                hlos = (hosp_disch - hosp_admit).total_seconds() / (60 * 60 * 24)
                ilos = (icu_disch - icu_admit).total_seconds() / (60 * 60 * 24)
                hfree = 28 - hlos
                ifree = 28 - ilos
                if hfree < 0 or died_inps[i]:
                    hfree = 0
                if ifree < 0 or died_inps[i]:
                    ifree = 0
            except TypeError:
                hlos = np.nan
                ilos = np.nan
                hfree = np.nan
                ifree = np.nan

            mks_win.append(mkwin)
            mks_whole.append(mk)
            hosp_days.append(hlos)
            hosp_frees.append(hfree)
            icu_days.append(ilos)
            icu_frees.append(ifree)
            admit_scrs.append(admit_scr)
            peak_scrs.append(peak_scr)

        mks_win = np.array(mks_win, dtype=int)
        mks_whole = np.array(mks_whole, dtype=int)
        hosp_days = np.array(hosp_days, dtype=float)
        hosp_frees = np.array(hosp_frees, dtype=float)
        icu_days = np.array(icu_days, dtype=float)
        icu_frees = np.array(icu_frees, dtype=float)
        admit_scrs = np.array(admit_scrs, dtype=float)
        peak_scrs = np.array(peak_scrs, dtype=float)

        meta.create_dataset('hosp_los', data=hosp_days, dtype=float)
        meta.create_dataset('hosp_free_days', data=hosp_frees, dtype=float)
        meta.create_dataset('icu_los', data=icu_days, dtype=float)
        meta.create_dataset('icu_free_days', data=icu_frees, dtype=float)
        meta.create_dataset('admit_scr', data=admit_scrs, dtype=float)
        meta.create_dataset('peak_scr', data=peak_scrs, dtype=float)
        meta.create_dataset('max_kdigo_win', data=mks_win, dtype=int)
        meta.create_dataset('max_kdigo', data=mks_whole, dtype=int)

    return meta


def get_uky_vitals(ids, dataPath=''):
    clinical_vit = pd.read_csv(os.path.join(dataPath, 'CLINICAL_VITALS.csv'))
    temps = np.full([len(ids), 2], np.nan)
    maps = np.full([len(ids), 2], np.nan)
    hrs = np.full([len(ids), 2], np.nan)
    for i in range(len(ids)):
        tid = ids[i]
        # Clinical vitals
        vit_idx = np.where(clinical_vit['STUDY_PATIENT_ID'].values == tid)[0]
        tmap = np.array((np.nan, np.nan))
        temp = np.array((np.nan, np.nan))
        hr = np.array((np.nan, np.nan))
        if vit_idx.size > 0:
            vit_idx = vit_idx[0]
            tmap = [clinical_vit['ART_MEAN_D1_LOW_VALUE'][vit_idx], clinical_vit['ART_MEAN_D1_HIGH_VALUE'][vit_idx]]
            if np.isnan(tmap[0]):
                tmap = [clinical_vit['CUFF_MEAN_D1_LOW_VALUE'][vit_idx], clinical_vit['CUFF_MEAN_D1_HIGH_VALUE'][vit_idx]]
            temp = [clinical_vit['TEMPERATURE_D1_LOW_VALUE'][vit_idx], clinical_vit['TEMPERATURE_D1_HIGH_VALUE'][vit_idx]]
            hr = [clinical_vit['HEART_RATE_D1_LOW_VALUE'][vit_idx], clinical_vit['HEART_RATE_D1_HIGH_VALUE'][vit_idx]]
        temps[i] = temp
        maps[i] = tmap
        hrs[i] = hr
    del clinical_vit
    return temps, maps, hrs


def get_uky_urine(ids, weights, dataPath=''):
    urine_m = pd.read_csv(os.path.join(dataPath, 'URINE_OUTPUT.csv'))
    urine_outs = np.full([len(ids), ], np.nan)
    urine_flows = np.full([len(ids), ], np.nan)
    for i in range(len(ids)):
        tid = ids[i]
        weight = weights[i]
        urine_idx = np.where(urine_m['STUDY_PATIENT_ID'].values == tid)[0]
        urine_out = np.zeros(2)
        if urine_idx.size > 0:
            urine_idx = urine_idx[0]
            for j in range(2):
                urine_out[j] = np.nansum(
                    (urine_m['U0_INDWELLING_URETHRAL_CATHTER_D%d_VALUE' % j][urine_idx],
                     urine_m['U0_VOIDED_ML_D%d_VALUE' % j][urine_idx]))
        urine_outs[i] = np.nanmean(urine_out)
        urine_flows[i] = urine_outs[i] / weight / 24
    del urine_m
    return urine_outs, urine_flows


def get_uky_medications(ids, icu_windows, dataPath='', maxDay=1):
    med_m = pd.read_csv(os.path.join(dataPath, 'MEDICATIONS_INDX.csv'))
    neph_cts = np.zeros(len(ids))
    vaso_cts = np.zeros(len(ids))
    dopas = np.zeros(len(ids))
    dobus = np.zeros(len(ids))
    milris = np.zeros(len(ids))
    epis = np.zeros(len(ids))
    phenylephs = np.zeros(len(ids))
    vasopressins = np.zeros(len(ids))
    for i in range(len(ids)):
        tid = ids[i]
        admit = icu_windows[tid][0]
        if type(admit) == str:
            admit = get_date(admit)
        med_idx = np.where(med_m['STUDY_PATIENT_ID'].values == tid)[0]
        neph_ct = 0
        vaso_ct = 0
        dopa = 0
        dobu = 0
        mili = 0
        epi = 0
        phenyl = 0
        vaso = 0
        if med_idx.size > 0:
            for tid in med_idx:
                tstr = str(med_m['ORDER_ENTERED_DATE'][tid]).split('.')[0]
                tdate = get_date(tstr)
                tname = med_m['MEDICATION_NAME'][tid].lower()
                start = tdate
                dur = float(med_m['DAYS_ON_MEDICATION'][tid])
                if not np.isnan(dur):
                    stop = start + datetime.timedelta(dur)
                    if tdate != 'nan':
                        if start <= admit:
                            if stop > admit or ((start.toordinal() - admit.toordinal()) <= maxDay):
                                # Nephrotoxins
                                if np.any(
                                        [med_m['MEDICATION_TYPE'][tid].lower() in x for x in ['acei', 'ace inhibitors',
                                                                                              'angiotensin receptor blockers',
                                                                                              'arb',
                                                                                              'aminoglycosides',
                                                                                              'nsaids']]):
                                    neph_ct = 1

                                # Vasoactive Drugs
                                if np.any([med_m['MEDICATION_TYPE'][tid].lower() in x for x in ['presor', 'inotrope',
                                                                                                'presor or inotrope']]):
                                    vaso_ct = 1

                                if 'dopamine' in tname:
                                    dopa = 1
                                    vaso_ct = 1
                                elif 'dobutamine' in tname:
                                    dobu = 1
                                    vaso_ct = 1
                                elif 'milrinone' in tname:
                                    mili = 1
                                    vaso_ct = 1
                                elif 'epinephrine':
                                    epi = 1
                                    vaso_ct = 1
                                elif 'phenylephrine' in tname:
                                    phenyl = 1
                                    vaso_ct = 1
                                elif 'vasopressin' in tname:
                                    vaso = 1
                                    vaso_ct = 1

        neph_cts[i] = neph_ct
        vaso_cts[i] = vaso_ct
        dopas[i] = dopa
        dobus[i] = dobu
        milris[i] = mili
        epis[i] = epi
        phenylephs[i] = phenyl
        vasopressins[i] = vaso
    del med_m
    return neph_cts, vaso_cts, dopas, dobus, milris, epis, phenylephs, vasopressins


def get_uky_mortality(ids, hosp_windows, icu_windows, dataPath=''):
    date_m = pd.read_csv(os.path.join(dataPath, 'ADMISSION_INDX.csv'))
    mort_m = pd.read_csv(os.path.join(dataPath, 'OUTCOMES_COMBINED.csv'))
    died_inps = np.zeros(len(ids), dtype=int)
    died_90d_admit = np.zeros(len(ids), dtype=int)
    died_90d_disch = np.zeros(len(ids), dtype=int)
    died_120d_admit = np.zeros(len(ids), dtype=int)
    died_120d_disch = np.zeros(len(ids), dtype=int)
    dtds = np.zeros(len(ids))
    dods = np.zeros(len(ids), dtype='|S20')
    dods[:] = 'nan'
    for i in range(len(ids)):
        tid = ids[i]
        died_inp = 0
        died_90a = 0
        died_90d = 0
        died_120a = 0
        died_120d = 0
        admit = icu_windows[tid][0]
        disch = hosp_windows[tid][1]
        icu_disch = icu_windows[tid][1]
        dtd = np.nan

        if type(admit) != datetime.datetime:
            admit = get_date(str(admit))
            disch = get_date(str(disch))

        mdate = 'nan'
        disch_disp = 'nan'
        didx = np.where(date_m['STUDY_PATIENT_ID'].values == tid)[0]
        if didx.size > 0:
            didx = didx[0]
            disch_disp = date_m['DISCHARGE_DISPOSITION'][didx]
            try:
                disch_disp = disch_disp.upper()
                if 'EXPIRED' in disch_disp or 'DIED' in disch_disp:
                    died_inp = 1
                    died_90d = 1
                    died_120d = 1
                    mdate = disch
            except AttributeError:
                pass

        mort_idx = np.where(mort_m['STUDY_PATIENT_ID'].values == tid)[0]
        if mort_idx.size > 0:
            mdate = get_date(mort_m['DECEASED_DATE'][mort_idx[0]])
        if mdate != 'nan':
            dods[i] = str(mdate)
            dtd = (mdate - admit).total_seconds() / (60 * 60 * 24)
            # if icu_disch == 'nan' or admit < mdate < icu_disch:
            #     died = 2
            if disch == 'nan':
                died_inp = 1
                died_90d = 1
            else:
                if mdate < disch or mdate.toordinal() == disch.toordinal():
                    died_inp = 1
                if (mdate - disch) < datetime.timedelta(90):
                    died_90d = 1
                if (mdate - disch) < datetime.timedelta(120):
                    died_120d = 1
            if (mdate - admit) < datetime.timedelta(90):
                died_90a = 1
            if (mdate - admit) < datetime.timedelta(120):
                died_120a = 1
        if died_inp and mdate == 'nan':
            mdate = disch
        died_inps[i] = died_inp
        died_90d_admit[i] = died_90a
        died_90d_disch[i] = died_90d
        died_120d_admit[i] = died_120a
        died_120d_disch[i] = died_120d
        dtds[i] = dtd
        dods[i] = str(mdate)
    del date_m, mort_m
    return died_inps, died_90d_admit, died_90d_disch, died_120d_admit, died_120d_disch, dtds, dods


def get_uky_comorbidities(ids, dataPath=''):
    charl_m = pd.read_csv(os.path.join(dataPath, 'CHARLSON_SCORE.csv'))
    elix_m = pd.read_csv(os.path.join(dataPath, 'ELIXHAUSER_SCORE.csv'))
    smoke_m = pd.read_csv(os.path.join(dataPath, 'SMOKING_HIS.csv'))
    charl_idxs = np.full(len(ids), np.nan)
    elix_idxs = np.full(len(ids), np.nan)
    charls = np.full((len(ids), 14), np.nan)
    elixs = np.full((len(ids), 31), np.nan)
    smokers = np.zeros(len(ids))
    for i in range(len(ids)):
        tid = ids[i]
        charl_idx = np.where(charl_m['STUDY_PATIENT_ID'].values == tid)[0]
        charl = np.nan
        if charl_idx.size > 0:
            charl_idx = charl_idx[0]
            if charl_m['CHARLSON_INDEX'][charl_idx] > 0:
                charl = charl_m['CHARLSON_INDEX'][charl_idx]
            for j in range(14):
                charls[i, j] = charl_m['CC_GRP_%d' % (j+1)][charl_idx]

        elix_idx = np.where(elix_m['STUDY_PATIENT_ID'].values == tid)[0]
        elix = np.nan
        if elix_idx.size > 0:
            elix_idx = elix_idx[0]
            if elix_m['ELIXHAUSER_INDEX'][elix_idx] > 0:
                elix = elix_m['ELIXHAUSER_INDEX'][elix_idx]
            for j in range(31):
                elixs[i, j] = elix_m['ELX_GRP_%d' % (j + 1)][elix_idx]
        charl_idxs[i] = charl
        elix_idxs[i] = elix

        smoke_idx = np.where(smoke_m['STUDY_PATIENT_ID'].values == tid)[0]
        smoker = 0
        if smoke_idx.size > 0:
            smoke_idx = smoke_idx[0]
            if smoke_m['SMOKING_HISTORY_STATUS'][smoke_idx] == 'FORMER SMOKER':
                smoker = 1
            if smoke_m['SMOKING_CURRENT_STATUS'][smoke_idx] == 'YES':
                smoker = 1
        smokers[i] = smoker
    del charl_m, elix_m, smoke_m
    return charl_idxs, elix_idxs, smokers, charls, elixs


def get_uky_fluids(ids, weights, dataPath='', maxDay=1):
    io_m = pd.read_csv(os.path.join(dataPath, 'IO_TOTALS.csv'))
    fos = np.full(len(ids), np.nan)
    nets = np.full(len(ids), np.nan)
    tots = np.full(len(ids), np.nan)
    cfbs = np.full(len(ids), np.nan)
    for i in range(len(ids)):
        tid = ids[i]
        weight = weights[i]
        io_idx = np.where(io_m['STUDY_PATIENT_ID'].values == tid)[0]
        if io_idx.size > 0:
            net = np.nan
            tot = np.nan
            cfb = np.nan
            invals = []
            outvals = []
            for tid in io_idx:
                for j in range(maxDay + 1):
                    inval = io_m['IO_TOTALS_D%d_INVALUE' % j][tid] / 1000
                    outval = io_m['IO_TOTALS_D%d_OUTVALUE' % j][tid] / 1000
                    if inval > 0 and outval > 0 and not np.isnan(inval) and not np.isnan(outval):
                        invals.append(inval)
                        outvals.append(outval)
                if len(invals) > 0:
                    invals = np.mean(invals)
                    outvals = np.mean(outvals)
                    cfb = np.sum(invals) - np.sum(outvals)
                    net = np.mean(invals) - np.mean(outvals)
                    tot = np.mean(invals) + np.mean(outvals)
                else:
                    net = np.nan
                    tot = np.nan
                    cfb = np.nan
        else:
            net = np.nan
            tot = np.nan
            cfb = np.nan
        nets[i] = net
        tots[i] = tot
        cfbs[i] = cfb

        if net != np.nan and weight != np.nan:
            fo = (cfb / weight) * 100
        else:
            fo = np.nan
        fos[i] = fo
    del io_m
    return nets, tots, fos, cfbs


def get_uky_organsupp(ids, dieds, windows, dataPath='', maxDay=1, tlim=14):
    # Windows assumed to correspond to ICU, not hospitalization
    organ_sup_mv = pd.read_csv(os.path.join(dataPath, 'ORGANSUPP_VENT.csv'))
    organ_sup_vad = pd.read_csv(os.path.join(dataPath, 'ORGANSUPP_VAD.csv'))
    organ_sup_ecmo = pd.read_csv(os.path.join(dataPath, 'ORGANSUPP_ECMO.csv'))
    organ_sup_iabp = pd.read_csv(os.path.join(dataPath, 'ORGANSUPP_IABP.csv'))
    mech_flags = np.zeros(len(ids))
    mech_days = np.zeros(len(ids))
    mech_frees_28d = np.zeros(len(ids))
    mech_frees_win = np.zeros(len(ids))
    ecmos = np.zeros(len(ids))
    vads = np.zeros(len(ids))
    iabps = np.zeros(len(ids))

    for i in range(len(ids)):
        tid = ids[i]
        died = dieds[i]
        admit = windows[tid][0]
        disch = windows[tid][1]
        if type(admit) == 'str':
            admit = get_date(admit)
        mech_flag = 0
        mech_day = 0
        mech_day_win = 0
        mech_idx = np.where(organ_sup_mv['STUDY_PATIENT_ID'].values == tid)[0]
        if mech_idx.size > 0:
            mech_idx = mech_idx[0]
            try:
                mech_start = get_date(organ_sup_mv['VENT_START_DATE'][mech_idx])
                mech_stop = get_date(organ_sup_mv['VENT_STOP_DATE'][mech_idx])
                maxDate = admit + datetime.timedelta(tlim)
                if mech_stop < admit:
                    pass
                elif (mech_start - admit).days > 28:
                    pass
                else:
                    if mech_start < admit:
                        mech_day = (mech_stop - admit).days
                        mech_flag = 1
                    else:
                        if mech_stop < disch:
                            mech_day = (mech_stop - mech_start).days
                            mech_day_win = (min(mech_stop, maxDate) - mech_start).days
                            if (mech_start.toordinal() - admit.toordinal()) <= maxDay:
                                mech_flag = 1
                        else:
                            mech_day = (disch - mech_start).days
                            mech_day_win = (min(disch, maxDate) - mech_start).days
                            if (mech_start.toordinal() - admit.toordinal()) <= maxDay:
                                mech_flag = 1
            except ValueError:
                pass

        if died:
            mech_free_28d = 0
            mech_free_win = 0
        else:
            mech_free_28d = max(0, 28 - mech_day)
            mech_free_win = max(0, tlim - mech_day_win)

        ecmo = 0
        ecmo_idx = np.where(organ_sup_ecmo['STUDY_PATIENT_ID'].values == tid)[0]
        if ecmo_idx.size > 0:
            ecmo_idx = ecmo_idx[0]
            try:
                ecmo_start = get_date(organ_sup_ecmo['ECMO_START_DATE'][ecmo_idx])
                ecmo_stop = get_date(organ_sup_ecmo['ECMO_STOP_DATE'][ecmo_idx])
                if ecmo_stop < admit:
                    pass
                elif ecmo_start.toordinal() - admit.toordinal() <= maxDay:
                    ecmo = 1
            except ValueError:
                pass

        iabp = 0
        iabp_idx = np.where(organ_sup_iabp['STUDY_PATIENT_ID'].values == tid)[0]
        if iabp_idx.size > 0:
            iabp_idx = iabp_idx[0]
            try:
                iabp_start = get_date(organ_sup_iabp['IABP_START_DATE'][iabp_idx])
                iabp_stop = get_date(organ_sup_iabp['IABP_STOP_DATE'][iabp_idx])
                if iabp_stop < admit:
                    pass
                elif iabp_start.toordinal() - admit.toordinal() <= maxDay:
                    iabp = 1
            except ValueError:
                pass

        vad = 0
        vad_d1 = 0
        vad_idx = np.where(organ_sup_vad['STUDY_PATIENT_ID'].values == tid)[0]
        if vad_idx.size > 0:
            vad_idx = vad_idx[0]
            try:
                vad_start = get_date(organ_sup_vad['VAD_START_DATE'][vad_idx])
                vad_stop = get_date(organ_sup_vad['VAD_STOP_DATE'][vad_idx])
                if vad_stop < admit:
                    pass
                elif vad_start.toordinal() - admit.toordinal() <= maxDay:
                    vad = 1
            except ValueError:
                pass

        mech_flags[i] = mech_flag
        mech_days[i] = mech_day
        mech_frees_win[i] = mech_free_win
        mech_frees_28d[i] = mech_free_28d
        ecmos[i] = ecmo
        vads[i] = vad
        iabps[i] = iabp

    mhs = np.max(np.vstack((ecmos, vads, iabps)), axis=0).T
    del organ_sup_mv, organ_sup_iabp, organ_sup_vad, organ_sup_ecmo
    return mech_flags, mech_days, mech_frees_win, ecmos, vads, iabps, mhs


def get_uky_clinical_others(ids, dataPath=''):
    clinical_oth = pd.read_csv(os.path.join(dataPath, 'CLINICAL_OTHERS.csv'))
    bmis = np.full([len(ids), ], np.nan)
    weights = np.full([len(ids), ], np.nan)
    heights = np.full([len(ids), ], np.nan)
    fio2s = np.full([len(ids), 2], np.nan)
    gcss = np.full([len(ids), 2], np.nan)
    resps = np.full([len(ids), 2], np.nan)
    for i in range(len(ids)):
        tid = ids[i]

        bmi = np.nan
        weight = np.nan
        fio2 = np.array((np.nan, np.nan))
        gcs = np.array((np.nan, np.nan))
        resp = np.array((np.nan, np.nan))
        co_rows = np.where(clinical_oth['STUDY_PATIENT_ID'].values == tid)[0]
        if co_rows.size > 0:
            row = co_rows[0]
            height = clinical_oth['HEIGHT_CM_VALUE'][row] / 100
            weight = clinical_oth['INITIAL_WEIGHT_KG'][row]
            if weight == 0:
                weight = np.nan
                bmi = np.nan
            elif height > 0.2:
                bmi = weight / (height * height)

            fio2 = [clinical_oth['FI02_D1_LOW_VALUE'][row], clinical_oth['FI02_D1_HIGH_VALUE'][row]]
            resp = [clinical_oth['RESP_RATE_D1_LOW_VALUE'][row], clinical_oth['RESP_RATE_D1_HIGH_VALUE'][row]]
            try:
                gcs_str = np.array(str(clinical_oth['GLASGOW_SCORE_D1_LOW_VALUE'][row]).split('-'))
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


def get_uky_planned_admissions(ids, icu_windows, dataPath=''):
    surg_m = pd.read_csv(os.path.join(dataPath, 'SURGERY_INDX.csv'))
    unPlannedAdmissions = np.ones(len(ids), dtype=int)

    for i, tid in enumerate(ids):
        if type(icu_windows) == dict:
            admit = icu_windows[tid][0]
        else:
            admit = icu_windows[i][0]

        surg_rows = np.where(surg_m['STUDY_PATIENT_ID'].values == tid)[0]
        for row in surg_rows:
            tdate = get_date(surg_m['SURGERY_PERFORMED_DATE'][row])
            if tdate != 'nan':
                if abs((tdate - admit).total_seconds() / (60 * 60)) < 24:
                    unPlannedAdmissions[i] = 0

    return unPlannedAdmissions


def get_uky_demographics(ids, hosp_windows, dataPath=''):
    dem_m = pd.read_csv(os.path.join(dataPath, 'DEMOGRAPHICS_INDX.csv'))
    dob_m = pd.read_csv(os.path.join(dataPath, 'DOB.csv'))
    males = np.full(len(ids), np.nan)
    races = np.full(len(ids), np.nan)
    ages = np.full(len(ids), np.nan)
    for i in range(len(ids)):
        tid = ids[i]
        dem_idx = np.where(dem_m['STUDY_PATIENT_ID'].values == tid)[0]
        if dem_idx.size > 0:
            dem_idx = dem_idx[0]
            if str(dem_m['GENDER'][dem_idx]).upper() == 'M':
                males[i] = 1
            elif str(dem_m['GENDER'][dem_idx]).upper() == 'F':
                males[i] = 0
            admit = get_date(hosp_windows[tid][0])
            race = str(dem_m['RACE'][dem_idx]).upper()
            if "WHITE" in race:
                races[i] = 0
            elif "BLACK" in race:
                races[i] = 1
            else:
                races[i] = 2
        dob_idx = np.where(dob_m['STUDY_PATIENT_ID'].values == tid)[0]
        if dob_idx.size > 0:
            dob_idx = dob_idx[0]
            dob = get_date(dob_m['DOB'][dob_idx])
            tage = admit - dob
            ages[i] = tage.total_seconds() / (60 * 60 * 24 * 365)
    del dem_m, dob_m
    return males, races, ages


def get_uky_rrt(ids, dieds, windows, dataPath='', t_lim=7):
    rrt_m = pd.read_csv(os.path.join(dataPath, 'RENAL_REPLACE_THERAPY.csv'))
    hd_dayss = np.zeros(len(ids))
    crrt_dayss = np.zeros(len(ids))
    hd_dayss_win = np.zeros(len(ids))
    crrt_dayss_win = np.zeros(len(ids))
    hd_frees_7d = np.zeros(len(ids))
    hd_frees_28d = np.zeros(len(ids))
    crrt_frees_7d = np.zeros(len(ids))
    crrt_frees_28d = np.zeros(len(ids))
    rrt_flags = np.zeros(len(ids))
    hd_trtmts = np.zeros(len(ids))
    crrt_flags = np.zeros(len(ids))
    hd_flags = np.zeros(len(ids))
    for i in range(len(ids)):
        tid = ids[i]
        admit = windows[tid][0]
        disch = windows[tid][1]
        died = dieds[i]
        if type(admit) == str:
            admit = get_date(admit)
            disch = get_date(disch)
        endwin = admit + datetime.timedelta(t_lim)
        end28d = admit + datetime.timedelta(28)
        hd_days = 0
        hd_days_win = 0
        crrt_days = 0
        crrt_days_win = 0
        rrt_flag = 0
        hd_trtmt = 0
        crrt_flag = 0
        hd_flag = 0
        dia_ids = np.where(rrt_m['STUDY_PATIENT_ID'].values == tid)[0]
        if np.size(dia_ids) > 0:
            for row in dia_ids:
                # CRRT
                start = get_date(rrt_m['CRRT_START_DATE'][row])
                stop = get_date(rrt_m['CRRT_STOP_DATE'][row])
                if start != 'nan' and stop != 'nan':
                    wstart = max(admit, start)
                    wstop = min(disch, stop)
                    if wstop > wstart and wstart < endwin:
                        rrt_flag = 1
                        crrt_flag = 1
                        crrt_days_win += (wstop - wstart).total_seconds() / (60 * 60 * 24)
                    if wstart < end28d:
                        crrt_days += (wstop - wstart).total_seconds() / (60 * 60 * 24)
                # hd
                start = get_date(rrt_m['HD_START_DATE'][row])
                stop = get_date(rrt_m['HD_STOP_DATE'][row])
                if start != 'nan' and stop != 'nan':
                    wstart = max(admit, start)
                    wstop = min(disch, stop)
                    if wstop > wstart and wstart < endwin:
                        rrt_flag = 1
                        hd_flag = 1
                        hd_days_win += (wstop - wstart).total_seconds() / (60 * 60 * 24)
                        hd_trtmt += rrt_m['HD_TREATMENTS'][row]
                    if wstart < end28d:
                        hd_days += (wstop - wstart).total_seconds() / (60 * 60 * 24)
    
        hd_free_win = min(0, t_lim - hd_days_win)
        hd_free_28d = min(0, 28 - hd_days)
        crrt_free_win = min(0, t_lim - crrt_days_win)
        crrt_free_28d = min(0, 28 - crrt_days)
        
        if hd_days or crrt_days:
            if died:
                hd_free_win = 0
                hd_free_28d = 0
                crrt_free_win = 0
                crrt_free_28d = 0
        
        hd_dayss[i] = hd_days
        crrt_dayss[i] = crrt_days
        hd_dayss_win[i] = hd_days_win
        crrt_dayss_win[i] = crrt_days_win
        hd_frees_7d[i] = hd_free_win
        hd_frees_28d[i] = hd_free_28d
        crrt_frees_7d[i] = crrt_free_win
        crrt_frees_28d[i] = crrt_free_28d
        rrt_flags[i] = rrt_flag
        hd_trtmts[i] = hd_trtmt
        crrt_flags[i] = crrt_flag
        hd_flags[i] = hd_flag
    del rrt_m
    return rrt_flags, hd_dayss, crrt_dayss, hd_dayss_win, crrt_dayss_win, hd_frees_7d, hd_frees_28d, hd_trtmts, crrt_frees_7d, crrt_frees_28d, crrt_flags, hd_flags


def get_uky_diagnoses(ids, dataPath=''):
    diag_m = pd.read_csv(os.path.join(dataPath, 'DIAGNOSIS.csv'))
    sepsiss = np.zeros(len(ids))
    diabetics = np.zeros(len(ids))
    hypertensives = np.zeros(len(ids))
    for i in range(len(ids)):
        tid = ids[i]
        sepsis = 0
        diabetic = 0
        hypertensive = 0
        diag_ids = np.where(diag_m['STUDY_PATIENT_ID'].values == tid)[0]
        for j in range(len(diag_ids)):
            tid = diag_ids[j]
            try:
                desc = diag_m['DIAGNOSIS_DESC'][tid].lower()
                if 'sep' in desc:
                    # if int(diag_m[tid, diag_nb_loc]) == 1:
                    sepsis = 1
                if 'diabe' in desc:
                    diabetic = 1
                if 'hypert' in desc:
                    hypertensive = 1
            except AttributeError:
                pass
        sepsiss[i] = sepsis
        diabetics[i] = diabetic
        hypertensives[i] = hypertensive
    del diag_m
    return sepsiss, diabetics, hypertensives


def get_uky_labs(ids, males, dataPath='', maxDay=1):
    labs1_m = pd.read_csv(os.path.join(dataPath, 'LABS_SET1.csv'))
    labs2_m = pd.read_csv(os.path.join(dataPath, 'LABS_SET2.csv'))
    blood_m = pd.read_csv(os.path.join(dataPath, 'BLOOD_GAS.csv'))
    anemics = np.full((len(ids), 3), np.nan)
    bilis = np.full(len(ids), np.nan)
    buns = np.full((len(ids), min(maxDay+1, 4)), np.nan)
    hemats = np.full((len(ids), 2), np.nan)
    hemos = np.full((len(ids), 2), np.nan)
    pltlts = np.full(len(ids), np.nan)
    nas = np.full((len(ids), 2), np.nan)
    pks = np.full((len(ids), 2), np.nan)
    albs = np.full(len(ids), np.nan)
    lacs = np.full(len(ids), np.nan)
    phs = np.full((len(ids), 2), np.nan)
    po2s = np.full((len(ids), 2), np.nan)
    pco2s = np.full((len(ids), 2), np.nan)
    wbcs = np.full((len(ids), 2), np.nan)
    bicarbs = np.full((len(ids), 2), np.nan)
    for i in range(len(ids)):
        tid = ids[i]
        male = males[i]
        # Anemia, bilirubin, and BUN (labs set 1)
        anemic = np.zeros(3)
        bili = np.nan
        bun = np.array((np.nan, np.nan, np.nan, np.nan))
        hemat = np.array((np.nan, np.nan))
        hemo = np.array((np.nan, np.nan))
        pltlt = np.nan
        na = np.array((np.nan, np.nan))
        pk = np.array((np.nan, np.nan))
        wbc = np.array((np.nan, np.nan))
        bicarb = np.array((np.nan, np.nan))
        lab_idx = np.where(labs1_m['STUDY_PATIENT_ID'].values == tid)[0]
        if lab_idx.size > 0:
            lab_idx = lab_idx[0]
            bili = labs1_m['BILIRUBIN_D1_HIGH_VALUE'][lab_idx]
            bun = [labs1_m['BUN_D%d_HIGH_VALUE' % x][lab_idx] for x in range(maxDay+1)]
            hemat = [labs1_m['HEMATOCRIT_D1_LOW_VALUE'][lab_idx], labs1_m['HEMATOCRIT_D1_HIGH_VALUE'][lab_idx]]
            hemo = [labs1_m['HEMOGLOBIN_D1_LOW_VALUE'][lab_idx], labs1_m['HEMOGLOBIN_D1_HIGH_VALUE'][lab_idx]]
            pltlt = labs1_m['PLATELETS_D1_LOW_VALUE'][lab_idx]
            na = [labs1_m['SODIUM_D1_LOW_VALUE'][lab_idx], labs1_m['SODIUM_D1_HIGH_VALUE'][lab_idx]]
            pk = [labs1_m['POTASSIUM_D1_LOW_VALUE'][lab_idx], labs1_m['POTASSIUM_D1_HIGH_VALUE'][lab_idx]]
            wbc = [labs1_m['WBC_D1_LOW_VALUE'][lab_idx], labs1_m['WBC_D1_HIGH_VALUE'][lab_idx]]
            bicarb = [labs1_m['CO2_D1_LOW_VALUE'][lab_idx], labs1_m['CO2_D1_HIGH_VALUE'][lab_idx]]
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
        lab_idx = np.where(labs2_m['STUDY_PATIENT_ID'].values == tid)[0]
        alb = np.nan
        lac = np.nan
        if lab_idx.size > 0:
            lab_idx = lab_idx[0]
            alb = labs2_m['ALBUMIN_VALUE'][lab_idx]
            lac = labs2_m['LACTATE_SYRINGE_ION_VALUE'][lab_idx]

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
        bicarbs[i] = bicarb

        # ph_loc, po2_loc, pco2_loc,
        # Blood gas
        blood_idx = np.where(blood_m['STUDY_PATIENT_ID'].values == tid)[0]
        ph = np.array((np.nan, np.nan))
        po2 = np.array((np.nan, np.nan))
        pco2 = np.array((np.nan, np.nan))
        if blood_idx.size > 0:
            blood_idx = blood_idx[0]
            ph = [blood_m['PH_D1_LOW_VALUE'][blood_idx], blood_m['PH_D1_HIGH_VALUE'][blood_idx]]
            po2 = [blood_m['PO2_D1_LOW_VALUE'][blood_idx], blood_m['PO2_D1_HIGH_VALUE'][blood_idx]]
            pco2 = [blood_m['PCO2_D1_LOW_VALUE'][blood_idx], blood_m['PCO2_D1_HIGH_VALUE'][blood_idx]]
        phs[i] = ph
        po2s[i] = po2
        pco2s[i] = pco2
    del labs1_m, labs2_m, blood_m
    return anemics, bilis, buns, hemats, hemos, pltlts, nas, pks, albs, lacs, phs, po2s, pco2s, wbcs, bicarbs


def get_uky_labs_V2(mrns, males, dataPath='', maxDay=3):
    labs_m = pd.read_csv(os.path.join(dataPath, 'new', 'EX706_GRP2_LABS.csv'))
    anemics = np.full((len(mrns), 3), np.nan)
    bilis = np.full(len(mrns), np.nan)
    buns = np.full((len(mrns), min(maxDay + 1, 4)), np.nan)
    hemats = np.full((len(mrns), 2), np.nan)
    hemos = np.full((len(mrns), 2), np.nan)
    pltlts = np.full(len(mrns), np.nan)
    nas = np.full((len(mrns), 2), np.nan)
    pks = np.full((len(mrns), 2), np.nan)
    albs = np.full(len(mrns), np.nan)
    lacs = np.full(len(mrns), np.nan)
    phs = np.full((len(mrns), 2), np.nan)
    po2s = np.full((len(mrns), 2), np.nan)
    pco2s = np.full((len(mrns), 2), np.nan)
    wbcs = np.full((len(mrns), 2), np.nan)
    bicarbs = np.full((len(mrns), 2), np.nan)
    for i in range(len(mrns)):
        tid = mrns[i]
        male = males[i]
        # Anemia, bilirubin, and BUN (labs set 1)
        anemic = np.zeros(3)
        bili = np.nan
        bun = np.array((np.nan, np.nan, np.nan, np.nan))
        hemat = np.array((np.nan, np.nan))
        hemo = np.array((np.nan, np.nan))
        pltlt = np.nan
        na = np.array((np.nan, np.nan))
        pk = np.array((np.nan, np.nan))
        wbc = np.array((np.nan, np.nan))
        bicarb = np.array((np.nan, np.nan))
        lab_idx = np.where(labs_m['STUDY_PATIENT_ID'].values == tid)[0]
        if lab_idx.size > 0:
            lab_idx = lab_idx[0]
            bili = labs_m['BILIRUBIN_D1_HIGH_VALUE'][lab_idx]
            bun = [labs_m['BUN_D%d_HIGH_VALUE' % x][lab_idx] for x in range(maxDay+1)]
            hemat = [labs_m['HEMATOCRIT_D1_LOW_VALUE'][lab_idx], labs_m['HEMATOCRIT_D1_HIGH_VALUE'][lab_idx]]
            hemo = [labs_m['HEMOGLOBIN_D1_LOW_VALUE'][lab_idx], labs_m['HEMOGLOBIN_D1_HIGH_VALUE'][lab_idx]]
            pltlt = labs_m['PLATELETS_D1_LOW_VALUE'][lab_idx]
            na = [labs_m['SODIUM_D1_LOW_VALUE'][lab_idx], labs_m['SODIUM_D1_HIGH_VALUE'][lab_idx]]
            pk = [labs_m['POTASSIUM_D1_LOW_VALUE'][lab_idx], labs_m['POTASSIUM_D1_HIGH_VALUE'][lab_idx]]
            wbc = [labs_m['WBC_D1_LOW_VALUE'][lab_idx], labs_m['WBC_D1_HIGH_VALUE'][lab_idx]]
            bicarb = [labs_m['CO2_D1_LOW_VALUE'][lab_idx], labs_m['CO2_D1_HIGH_VALUE'][lab_idx]]
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
        lab_idx = np.where(labs2_m['STUDY_PATIENT_ID'].values == tid)[0]
        alb = np.nan
        lac = np.nan
        if lab_idx.size > 0:
            lab_idx = lab_idx[0]
            alb = labs2_m['ALBUMIN_VALUE'][lab_idx]
            lac = labs2_m['LACTATE_SYRINGE_ION_VALUE'][lab_idx]

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
        bicarbs[i] = bicarb

        # ph_loc, po2_loc, pco2_loc,
        # Blood gas
        blood_idx = np.where(blood_m['STUDY_PATIENT_ID'].values == tid)[0]
        ph = np.array((np.nan, np.nan))
        po2 = np.array((np.nan, np.nan))
        pco2 = np.array((np.nan, np.nan))
        if blood_idx.size > 0:
            blood_idx = blood_idx[0]
            ph = [blood_m['PH_D1_LOW_VALUE'][blood_idx], blood_m['PH_D1_HIGH_VALUE'][blood_idx]]
            po2 = [blood_m['PO2_D1_LOW_VALUE'][blood_idx], blood_m['PO2_D1_HIGH_VALUE'][blood_idx]]
            pco2 = [blood_m['PCO2_D1_LOW_VALUE'][blood_idx], blood_m['PCO2_D1_HIGH_VALUE'][blood_idx]]
        phs[i] = ph
        po2s[i] = po2
        pco2s[i] = pco2
    del labs_m, labs2_m, blood_m
    return anemics, bilis, buns, hemats, hemos, pltlts, nas, pks, albs, lacs, phs, po2s, pco2s, wbcs, bicarbs


def summarize_stats_dallas(ids, kdigos, days, scrs, icu_windows, hosp_windows,
                           f, base_path, grp_name='meta', tlim=7, v=False):

    n_eps = []
    for i in range(len(kdigos)):
        n_eps.append(count_eps(kdigos[i]))

    date_m = pd.read_csv(os.path.join(base_path, 'tIndexedIcuAdmission.csv'))
    hosp_locs = [date_m.columns.get_loc('HSP_ADMSN_TIME'), date_m.columns.get_loc('HSP_DISCH_TIME')]
    icu_locs = [date_m.columns.get_loc('ICU_ADMSN_TIME'), date_m.columns.get_loc('ICU_DISCH_TIME')]
    date_m = date_m.values

    bsln_scrs = load_csv(os.path.join(base_path, 'all_baseline_info.csv'), ids, float, idxs=[1], skip_header=True)
    bsln_types = load_csv(os.path.join(base_path, 'all_baseline_info.csv'), ids, str, idxs=[2], skip_header=True)

    genders = None
    mk_wins = []
    mks = []
    dieds = []
    neps = []
    hosp_days = []
    hosp_frees_win = []
    hosp_frees_28d = []
    icu_days = []
    icu_frees_win = []
    icu_frees_28d = []

    admit_scrs = np.zeros(len(ids))
    peak_scrs = np.zeros(len(ids))

    assert type(icu_windows) == type(hosp_windows)
    if type(icu_windows) == dict:
        icu_windows_save = []
        hosp_windows_save = []
    else:
        icu_windows_save = icu_windows
        hosp_windows_save = hosp_windows

    try:
        meta = f[grp_name]
    except:
        meta = f.create_group(grp_name)
        meta.create_dataset('ids', data=ids, dtype=int)

    if 'age' not in list(meta):
        print("Getting general patient info...")
        genders, races, ages, _, _ = get_utsw_demographics(ids, hosp_windows, base_path)
        meta.create_dataset('age', data=ages, dtype=float)
        meta.create_dataset('gender', data=genders, dtype=int)
        meta.create_dataset('race', data=races, dtype=int)
        # meta.create_dataset('dod', data=dods.astype(bytes), dtype='S20')
        # meta.create_dataset('dtd', data=dtds, dtype=float)

    if 'died_inp' not in list(meta):
        died_inps, died_90d_admit, died_90d_disch, \
        died_120d_admit, died_120d_disch, dods, dtds = get_utsw_mortality(ids, icu_windows, hosp_windows, base_path)
        meta.create_dataset('died_inp', data=died_inps, dtype=int)
        meta.create_dataset('died_90d_admit', data=died_90d_admit, dtype=int)
        meta.create_dataset('died_90d_disch', data=died_90d_disch, dtype=int)
        meta.create_dataset('died_120d_admit', data=died_120d_admit, dtype=int)
        meta.create_dataset('died_120d_disch', data=died_120d_disch, dtype=int)
        meta.create_dataset('dod', data=dods.astype(bytes), dtype='|S20')
        meta.create_dataset('dtd', data=dtds, dtype=float)
    else:
        died_inps = meta['died_inp']

    if 'rrt_flag' not in list(meta):
        print("Getting dialysis info...")
        # hd_days, crrt_days, hd_free_win,\
        # hd_free_28d, crrt_free_win, crrt_free_28d = get_utsw_rrt(ids, icu_windows, base_path)
        rrt_flags, hd_days, crrt_days, hd_dayss_win, \
        crrt_dayss_win, hd_frees_win, hd_frees_28d, hd_trtmts, \
        crrt_frees_win, crrt_frees_28d, crrt_flags, hd_flags = get_utsw_rrt_v2(ids, died_inps, icu_windows, base_path)
        rrt_flags = np.array(rrt_flags, dtype=int)
        hd_days = np.array(hd_days, dtype=int)
        crrt_days = np.array(crrt_days, dtype=int)
        hd_frees_win = np.array(hd_frees_win, dtype=int)
        hd_frees_28d = np.array(hd_frees_28d, dtype=int)
        crrt_frees_win = np.array(crrt_frees_win, dtype=int)
        hd_trtmts = np.array(hd_trtmts, dtype=int)
        hd_frees_28d = np.array(hd_frees_28d, dtype=int)
        meta.create_dataset('rrt_flag', data=rrt_flags, dtype=int)
        meta.create_dataset('crrt_flag', data=crrt_flags, dtype=int)
        meta.create_dataset('hd_flag', data=hd_flags, dtype=int)
        meta.create_dataset('hd_days', data=hd_days, dtype=float)
        meta.create_dataset('crrt_days', data=crrt_days, dtype=float)
        meta.create_dataset('hd_free_win', data=hd_frees_win, dtype=float)
        meta.create_dataset('crrt_free_win', data=crrt_frees_win, dtype=float)
        meta.create_dataset('hd_free_28d', data=hd_frees_28d, dtype=float)
        meta.create_dataset('crrt_free_28d', data=crrt_frees_28d, dtype=float)
        meta.create_dataset('hd_treatments', data=hd_trtmts, dtype=float)

    if 'nephrotox_ct' not in list(meta):
        print ('Getting medications...')
        neph_cts, vaso_cts = get_utsw_medications(ids, base_path)
        meta.create_dataset('nephrotox_ct', data=neph_cts, dtype=int)
        meta.create_dataset('vasopress_ct', data=vaso_cts, dtype=int)

    if 'septic' not in list(meta):
        print('Getting diagnoses...')
        septics, diabetics, hypertensives = get_utsw_diagnoses(ids, base_path)
        meta.create_dataset('septic', data=septics, dtype=int)
        meta.create_dataset('diabetic', data=diabetics, dtype=int)
        meta.create_dataset('hypertensive', data=hypertensives, dtype=int)

    # tAOS table
    if 'mv_flag' not in list(meta):
        print('Getting organ support...')
        mech_flags, mech_days, ecmos, iabps, vads,\
        mv_frees_win, mv_frees_28d = get_utsw_organsupp(ids, icu_windows, dtds, died_inps, base_path)

        meta.create_dataset('mv_flag', data=mech_flags, dtype=int)
        meta.create_dataset('mv_days', data=mech_days, dtype=float)
        meta.create_dataset('mv_free_days_win', data=mv_frees_win, dtype=float)
        meta.create_dataset('mv_free_days_28d', data=mv_frees_28d, dtype=float)
        meta.create_dataset('ecmo', data=ecmos, dtype=int)
        meta.create_dataset('iabp', data=iabps, dtype=int)
        meta.create_dataset('vad', data=vads, dtype=int)

    if 'weight' not in list(meta):
        print('Getting clinical others...')
        heights, weights, bmis, temps, net_fluids, gros_fluids, cfbs, glasgows, resps, hrs, maps, fos, urine_outs, urine_flows = get_utsw_flw(ids, base_path)
        meta.create_dataset('bmi', data=bmis, dtype=float)
        meta.create_dataset('weight', data=weights, dtype=float)
        meta.create_dataset('height', data=heights, dtype=float)
        meta.create_dataset('temperature', data=temps, dtype=float)
        meta.create_dataset('hr', data=hrs, dtype=float)
        meta.create_dataset('glasgow', data=glasgows, dtype=float)
        meta.create_dataset('respiration', data=resps, dtype=float)
        meta.create_dataset('map', data=maps, dtype=float)
        meta.create_dataset('net_fluid', data=net_fluids, dtype=float)
        meta.create_dataset('gross_fluid', data=gros_fluids, dtype=float)
        meta.create_dataset('cum_fluid_balance', data=cfbs, dtype=float)
        meta.create_dataset('fluid_overload', data=fos, dtype=float)
        meta.create_dataset('urine_out', data=urine_outs, dtype=float)
        meta.create_dataset('urine_flow', data=urine_flows, dtype=float)

    if 'anemia' not in list(meta):
        print('Getting labs...')
        if genders is None:
            genders = meta['gender'][:]
        albs, bilis, bicarbs, buns, chlorides, fio2s, \
        hemats, hemos, phs, pco2s, po2s, pots, pltlts, sods, wbcs, anemics = get_utsw_labs(ids, genders, base_path)
        meta.create_dataset('anemia', data=anemics, dtype=int)
        meta.create_dataset('ph', data=phs, dtype=float)
        meta.create_dataset('albumin', data=albs, dtype=float)
        meta.create_dataset('lactate', data=np.zeros(len(ids)), dtype=float)
        meta.create_dataset('bilirubin', data=bilis, dtype=float)
        meta.create_dataset('bun', data=buns, dtype=float)
        meta.create_dataset('potassium', data=pots, dtype=float)
        meta.create_dataset('pco2', data=pco2s, dtype=float)
        meta.create_dataset('po2', data=po2s, dtype=float)

        meta.create_dataset('chloride', data=chlorides, dtype=float)
        meta.create_dataset('hematocrit', data=hemats, dtype=float)
        meta.create_dataset('hemoglobin', data=hemos, dtype=float)
        meta.create_dataset('platelets', data=pltlts, dtype=float)
        meta.create_dataset('sodium', data=sods, dtype=float)
        meta.create_dataset('wbc', data=wbcs, dtype=float)
        meta.create_dataset('fio2', data=fio2s, dtype=float)

    if 'icu_admit_discharge' not in list(meta):
        bsln_gfrs = np.zeros(len(ids))
        for i in range(len(ids)):
            tid = ids[i]
            if v:
                print('Summarizing Patient #%d (%d/%d)' % (tid, i+1, len(ids)))
            if len(kdigos[i]) > 1:
                td = days[i]
                mask = np.where(td <= tlim)[0]
                mkwin = int(np.max(np.array(kdigos[i])[mask]))
                mk = int(np.max(kdigos[i]))

                eps = n_eps[i]

                gfr = calc_gfr(bsln_scrs[i], genders[i], races[i], ages[i])

                admit_scrs[i] = scrs[i][0]
                peak_scrs[i] = np.max(scrs[i][np.where(days[i] <= tlim)])
            elif len(kdigos[i]) == 1:
                mkwin = kdigos[i][0]
                mk = kdigos[i][0]
                eps = 1
                gfr = calc_gfr(bsln_scrs[i], genders[i], races[i], ages[i])
                admit_scrs[i] = scrs[i][0]
                peak_scrs[i] = scrs[i][0]

            else:
                mkwin = 0
                mk = 0

                gfr = np.nan
                admit_scrs[i] = np.nan
                peak_scrs[i] = np.nan


            if type(icu_windows) == dict:
                icu_admit = get_date(icu_windows[tid][0])
                icu_disch = get_date(icu_windows[tid][1])
            else:
                icu_admit = get_date(icu_windows[i][0])
                icu_disch = get_date(icu_windows[i][1])

            if type(hosp_windows) == dict:
                hosp_admit = get_date(hosp_windows[tid][0])
                hosp_disch = get_date(hosp_windows[tid][1])
            else:
                hosp_admit = get_date(hosp_windows[i][0])
                hosp_disch = get_date(hosp_windows[i][1])

            hlos, ilos = get_los(tid, date_m, hosp_locs, icu_locs)
            hfree_win = 7 - hlos
            ifree_win = 7 - ilos
            hfree_28d = 28 - hlos
            ifree_28d = 28 - ilos

            if died_inps[i]:
                if dtds[i] <= 7:
                    hfree_win = 0
                    hfree_28d = 0
                    ifree_win = 0
                    ifree_28d = 0
                elif dtds[i] <= 28:
                    hfree_28d = 0
                    ifree_28d = 0

            if hfree_win < 0:
                hfree_win = 0
            if ifree_win < 0:
                ifree_win = 0
            if hfree_28d < 0:
                hfree_28d = 0
            if ifree_28d < 0:
                ifree_28d = 0

            if type(icu_windows) == dict:
                icu_windows_save.append([str(icu_admit), str(icu_disch)])
                hosp_windows_save.append([str(hosp_admit), str(hosp_disch)])

            bsln_gfrs[i] = gfr

            mk_wins.append(mkwin)
            mks.append(mk)
            neps.append(eps)
            hosp_days.append(hlos)
            hosp_frees_win.append(hfree_win)
            hosp_frees_28d.append(hfree_28d)
            icu_days.append(ilos)
            icu_frees_win.append(ifree_win)
            icu_frees_28d.append(ifree_28d)

        icu_windows_save = np.array(icu_windows_save, dtype=str)
        hosp_windows_save = np.array(hosp_windows_save, dtype=str)
        meta.create_dataset('icu_admit_discharge', data=icu_windows_save.astype(bytes), dtype='|S20')
        meta.create_dataset('hosp_admit_discharge', data=hosp_windows_save.astype(bytes), dtype='|S20')

        meta.create_dataset('hosp_los', data=hosp_days, dtype=float)
        meta.create_dataset('hosp_free_days_win', data=hosp_frees_win, dtype=float)
        meta.create_dataset('hosp_free_days_28d', data=hosp_frees_28d, dtype=float)
        meta.create_dataset('icu_los', data=icu_days, dtype=float)
        meta.create_dataset('icu_free_days_win', data=icu_frees_win, dtype=float)
        meta.create_dataset('icu_free_days_28d', data=icu_frees_28d, dtype=float)

        # meta.create_dataset('died_inp', data=dieds, dtype=int)
        meta.create_dataset('baseline_scr', data=bsln_scrs, dtype=float)
        meta.create_dataset('baseline_gfr', data=bsln_gfrs, dtype=float)
        meta.create_dataset('baseline_type', data=bsln_types.astype(bytes), dtype='|S8')
        meta.create_dataset('admit_scr', data=admit_scrs, dtype=float)
        meta.create_dataset('peak_scr', data=peak_scrs, dtype=float)

        meta.create_dataset('max_kdigo_win', data=mk_wins, dtype=int)
        meta.create_dataset('max_kdigo', data=mks, dtype=int)
        meta.create_dataset('n_episodes', data=neps, dtype=int)
        meta.create_dataset('days_to_death', data=dtds, dtype=float)

    if 'charlson' not in list(meta):
        meta.create_dataset('charlson', data=np.zeros(len(ids)), dtype=int)
        meta.create_dataset('elixhauser', data=np.zeros(len(ids)), dtype=int)
        meta.create_dataset('smoker', data=np.zeros(len(ids)), dtype=int)

    return f


def get_utsw_labs(ids, males, dataPath):
    lab_m = pd.read_csv(dataPath + '/icu_lab_data.csv')
    anemics = np.full((len(ids), 3), np.nan)
    albs = np.full((len(ids), 3, 2), np.nan)
    bilis = np.full((len(ids), 3, 2), np.nan)
    buns = np.full((len(ids), 3, 2), np.nan)
    chlorides = np.full((len(ids), 3, 2), np.nan)
    hemats = np.full((len(ids), 3, 2), np.nan)
    hemos = np.full((len(ids), 3, 2), np.nan)
    pltlts = np.full((len(ids), 3, 2), np.nan)
    pots = np.full((len(ids), 3, 2), np.nan)
    sods = np.full((len(ids), 3, 2), np.nan)
    wbcs = np.full((len(ids), 3, 2), np.nan)
    fio2s = np.full((len(ids), 3, 2), np.nan)
    potas = np.full((len(ids), 3, 2), np.nan)
    pco2s = np.full((len(ids), 3, 2), np.nan)
    po2s = np.full((len(ids), 3, 2), np.nan)
    phs = np.full((len(ids), 3, 2), np.nan)
    bicarbs = np.full((len(ids), 3, 2), np.nan)
    
    d0s = np.where(lab_m['DAY_NO'].values == 'D0')[0]
    d1s = np.where(lab_m['DAY_NO'].values == 'D1')[0]
    d2s = np.where(lab_m['DAY_NO'].values == 'D2')[0]

    alb_rows = np.where(lab_m['TERM_GRP_NAME'].values == 'ALBUMIN')[0]
    bili_rows = np.where(lab_m['TERM_GRP_NAME'].values == 'BILIRUBIN, TOTAL')[0]
    bun_rows = np.where(lab_m['TERM_GRP_NAME'].values == 'BUN')[0]
    chloride_rows = np.where(lab_m['TERM_GRP_NAME'].values == 'CHLORIDE')[0]
    hemat_rows = np.where(lab_m['TERM_GRP_NAME'].values ==  'HEMATOCRIT')[0]
    hemo_rows = np.where(lab_m['TERM_GRP_NAME'].values == 'HEMOGLOBIN')[0]
    pltlt_rows = np.where(lab_m['TERM_GRP_NAME'].values == 'PLATELETS')[0]
    pot_rows = np.where(lab_m['TERM_GRP_NAME'].values == 'POTASSIUM')[0]
    sod_rows = np.where(lab_m['TERM_GRP_NAME'].values == 'SODIUM')[0]
    wbc_rows = np.where(lab_m['TERM_GRP_NAME'].values == 'WBC')[0]
    fio2_rows = np.where(lab_m['TERM_GRP_NAME'].values == 'FIO2')[0]
    pco2_rows = np.where(lab_m['TERM_GRP_NAME'].values == 'PCO2')[0]
    po2_rows = np.where(lab_m['TERM_GRP_NAME'].values == 'PO2')[0]
    ph_rows = np.where(lab_m['TERM_GRP_NAME'].values == 'PH')[0]
    bicarb_rows = np.where(lab_m['TERM_GRP_NAME'].values == 'HCO3 ART')[0]

    for i, tid in enumerate(ids):
        anemic = np.zeros(3)
        lab_idx = np.where(lab_m['PATIENT_NUM'].values == tid)[0]

        d0_idx = np.intersect1d(lab_idx, d0s)
        d1_idx = np.intersect1d(lab_idx, d1s)
        d2_idx = np.intersect1d(lab_idx, d2s)

        alb = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        alb_idx = np.intersect1d(d0_idx, alb_rows)
        if alb_idx.size > 0:
            alb_idx = alb_idx[0]
            alb[0, 0] = lab_m['D_MIN_VAL'][alb_idx]
            alb[0, 1] = lab_m['D_MAX_VAL'][alb_idx]
        alb_idx = np.intersect1d(d1_idx, alb_rows)
        if alb_idx.size > 0:
            alb_idx = alb_idx[0]
            alb[1, 0] = lab_m['D_MIN_VAL'][alb_idx]
            alb[1, 1] = lab_m['D_MAX_VAL'][alb_idx]
        alb_idx = np.intersect1d(d2_idx, alb_rows)
        if alb_idx.size > 0:
            alb_idx = alb_idx[0]
            alb[2, 0] = lab_m['D_MIN_VAL'][alb_idx]
            alb[2, 1] = lab_m['D_MAX_VAL'][alb_idx]

        bili = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        bili_idx = np.intersect1d(d0_idx, bili_rows)
        if bili_idx.size > 0:
            bili_idx = bili_idx[0]
            bili[0, 0] = lab_m['D_MIN_VAL'][bili_idx]
            bili[0, 1] = lab_m['D_MAX_VAL'][bili_idx]
        bili_idx = np.intersect1d(d1_idx, bili_rows)
        if bili_idx.size > 0:
            bili_idx = bili_idx[0]
            bili[1, 0] = lab_m['D_MIN_VAL'][bili_idx]
            bili[1, 1] = lab_m['D_MAX_VAL'][bili_idx]
        bili_idx = np.intersect1d(d2_idx, bili_rows)
        if bili_idx.size > 0:
            bili_idx = bili_idx[0]
            bili[2, 0] = lab_m['D_MIN_VAL'][bili_idx]
            bili[2, 1] = lab_m['D_MAX_VAL'][bili_idx]

        bun = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        bun_idx = np.intersect1d(d0_idx, bun_rows)
        if bun_idx.size > 0:
            bun_idx = bun_idx[0]
            bun[0, 0] = lab_m['D_MIN_VAL'][bun_idx]
            bun[0, 1] = lab_m['D_MAX_VAL'][bun_idx]
        bun_idx = np.intersect1d(d1_idx, bun_rows)
        if bun_idx.size > 0:
            bun_idx = bun_idx[0]
            bun[1, 0] = lab_m['D_MIN_VAL'][bun_idx]
            bun[1, 1] = lab_m['D_MAX_VAL'][bun_idx]
        bun_idx = np.intersect1d(d2_idx, bun_rows)
        if bun_idx.size > 0:
            bun_idx = bun_idx[0]
            bun[2, 0] = lab_m['D_MIN_VAL'][bun_idx]
            bun[2, 1] = lab_m['D_MAX_VAL'][bun_idx]

        chloride = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        chloride_idx = np.intersect1d(d0_idx, chloride_rows)
        if chloride_idx.size > 0:
            chloride_idx = chloride_idx[0]
            chloride[0, 0] = lab_m['D_MIN_VAL'][chloride_idx]
            chloride[0, 1] = lab_m['D_MAX_VAL'][chloride_idx]
        chloride_idx = np.intersect1d(d1_idx, chloride_rows)
        if chloride_idx.size > 0:
            chloride_idx = chloride_idx[0]
            chloride[1, 0] = lab_m['D_MIN_VAL'][chloride_idx]
            chloride[1, 1] = lab_m['D_MAX_VAL'][chloride_idx]
        chloride_idx = np.intersect1d(d2_idx, chloride_rows)
        if chloride_idx.size > 0:
            chloride_idx = chloride_idx[0]
            chloride[2, 0] = lab_m['D_MIN_VAL'][chloride_idx]
            chloride[2, 1] = lab_m['D_MAX_VAL'][chloride_idx]

        hemat = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        hemat_idx = np.intersect1d(d0_idx, hemat_rows)
        if hemat_idx.size > 0:
            hemat_idx = hemat_idx[0]
            hemat[0, 0] = lab_m['D_MIN_VAL'][hemat_idx]
            hemat[0, 1] = lab_m['D_MAX_VAL'][hemat_idx]
        hemat_idx = np.intersect1d(d1_idx, hemat_rows)
        if hemat_idx.size > 0:
            hemat_idx = hemat_idx[0]
            hemat[1, 0] = lab_m['D_MIN_VAL'][hemat_idx]
            hemat[1, 1] = lab_m['D_MAX_VAL'][hemat_idx]
        hemat_idx = np.intersect1d(d2_idx, hemat_rows)
        if hemat_idx.size > 0:
            hemat_idx = hemat_idx[0]
            hemat[2, 0] = lab_m['D_MIN_VAL'][hemat_idx]
            hemat[2, 1] = lab_m['D_MAX_VAL'][hemat_idx]

        hemo = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        hemo_idx = np.intersect1d(d0_idx, hemo_rows)
        if hemo_idx.size > 0:
            hemo_idx = hemo_idx[0]
            hemo[0, 0] = lab_m['D_MIN_VAL'][hemo_idx]
            hemo[0, 1] = lab_m['D_MAX_VAL'][hemo_idx]
        hemo_idx = np.intersect1d(d1_idx, hemo_rows)
        if hemo_idx.size > 0:
            hemo_idx = hemo_idx[0]
            hemo[1, 0] = lab_m['D_MIN_VAL'][hemo_idx]
            hemo[1, 1] = lab_m['D_MAX_VAL'][hemo_idx]
        hemo_idx = np.intersect1d(d2_idx, hemo_rows)
        if hemo_idx.size > 0:
            hemo_idx = hemo_idx[0]
            hemo[2, 0] = lab_m['D_MIN_VAL'][hemo_idx]
            hemo[2, 1] = lab_m['D_MAX_VAL'][hemo_idx]

        pltlt = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        pltlt_idx = np.intersect1d(d0_idx, pltlt_rows)
        if pltlt_idx.size > 0:
            pltlt_idx = pltlt_idx[0]
            pltlt[0, 0] = lab_m['D_MIN_VAL'][pltlt_idx]
            pltlt[0, 1] = lab_m['D_MAX_VAL'][pltlt_idx]
        pltlt_idx = np.intersect1d(d1_idx, pltlt_rows)
        if pltlt_idx.size > 0:
            pltlt_idx = pltlt_idx[0]
            pltlt[1, 0] = lab_m['D_MIN_VAL'][pltlt_idx]
            pltlt[1, 1] = lab_m['D_MAX_VAL'][pltlt_idx]
        pltlt_idx = np.intersect1d(d2_idx, pltlt_rows)
        if pltlt_idx.size > 0:
            pltlt_idx = pltlt_idx[0]
            pltlt[2, 0] = lab_m['D_MIN_VAL'][pltlt_idx]
            pltlt[2, 1] = lab_m['D_MAX_VAL'][pltlt_idx]

        pot = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        pot_idx = np.intersect1d(d0_idx, pot_rows)
        if pot_idx.size > 0:
            pot_idx = pot_idx[0]
            pot[0, 0] = lab_m['D_MIN_VAL'][pot_idx]
            pot[0, 1] = lab_m['D_MAX_VAL'][pot_idx]
        pot_idx = np.intersect1d(d1_idx, pot_rows)
        if pot_idx.size > 0:
            pot_idx = pot_idx[0]
            pot[1, 0] = lab_m['D_MIN_VAL'][pot_idx]
            pot[1, 1] = lab_m['D_MAX_VAL'][pot_idx]
        pot_idx = np.intersect1d(d2_idx, pot_rows)
        if pot_idx.size > 0:
            pot_idx = pot_idx[0]
            pot[2, 0] = lab_m['D_MIN_VAL'][pot_idx]
            pot[2, 1] = lab_m['D_MAX_VAL'][pot_idx]

        sod = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        sod_idx = np.intersect1d(d0_idx, sod_rows)
        if sod_idx.size > 0:
            sod_idx = sod_idx[0]
            sod[0, 0] = lab_m['D_MIN_VAL'][sod_idx]
            sod[0, 1] = lab_m['D_MAX_VAL'][sod_idx]
        sod_idx = np.intersect1d(d1_idx, sod_rows)
        if sod_idx.size > 0:
            sod_idx = sod_idx[0]
            sod[1, 0] = lab_m['D_MIN_VAL'][sod_idx]
            sod[1, 1] = lab_m['D_MAX_VAL'][sod_idx]
        sod_idx = np.intersect1d(d2_idx, sod_rows)
        if sod_idx.size > 0:
            sod_idx = sod_idx[0]
            sod[2, 0] = lab_m['D_MIN_VAL'][sod_idx]
            sod[2, 1] = lab_m['D_MAX_VAL'][sod_idx]

        wbc = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        wbc_idx = np.intersect1d(d0_idx, wbc_rows)
        if wbc_idx.size > 0:
            wbc_idx = wbc_idx[0]
            wbc[0, 0] = lab_m['D_MIN_VAL'][wbc_idx]
            wbc[0, 1] = lab_m['D_MAX_VAL'][wbc_idx]
        wbc_idx = np.intersect1d(d1_idx, wbc_rows)
        if wbc_idx.size > 0:
            wbc_idx = wbc_idx[0]
            wbc[1, 0] = lab_m['D_MIN_VAL'][wbc_idx]
            wbc[1, 1] = lab_m['D_MAX_VAL'][wbc_idx]
        wbc_idx = np.intersect1d(d2_idx, wbc_rows)
        if wbc_idx.size > 0:
            wbc_idx = wbc_idx[0]
            wbc[2, 0] = lab_m['D_MIN_VAL'][wbc_idx]
            wbc[2, 1] = lab_m['D_MAX_VAL'][wbc_idx]

        fio2 = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        fio2_idx = np.intersect1d(d0_idx, fio2_rows)
        if fio2_idx.size > 0:
            fio2_idx = fio2_idx[0]
            fio2[0, 0] = lab_m['D_MIN_VAL'][fio2_idx]
            fio2[0, 1] = lab_m['D_MAX_VAL'][fio2_idx]
        fio2_idx = np.intersect1d(d1_idx, fio2_rows)
        if fio2_idx.size > 0:
            fio2_idx = fio2_idx[0]
            fio2[1, 0] = lab_m['D_MIN_VAL'][fio2_idx]
            fio2[1, 1] = lab_m['D_MAX_VAL'][fio2_idx]
        fio2_idx = np.intersect1d(d2_idx, fio2_rows)
        if fio2_idx.size > 0:
            fio2_idx = fio2_idx[0]
            fio2[2, 0] = lab_m['D_MIN_VAL'][fio2_idx]
            fio2[2, 1] = lab_m['D_MAX_VAL'][fio2_idx]

        pco2 = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        pco2_idx = np.intersect1d(d0_idx, pco2_rows)
        if pco2_idx.size > 0:
            pco2_idx = pco2_idx[0]
            pco2[0, 0] = lab_m['D_MIN_VAL'][pco2_idx]
            pco2[0, 1] = lab_m['D_MAX_VAL'][pco2_idx]
        pco2_idx = np.intersect1d(d1_idx, pco2_rows)
        if pco2_idx.size > 0:
            pco2_idx = pco2_idx[0]
            pco2[1, 0] = lab_m['D_MIN_VAL'][pco2_idx]
            pco2[1, 1] = lab_m['D_MAX_VAL'][pco2_idx]
        pco2_idx = np.intersect1d(d2_idx, pco2_rows)
        if pco2_idx.size > 0:
            pco2_idx = pco2_idx[0]
            pco2[2, 0] = lab_m['D_MIN_VAL'][pco2_idx]
            pco2[2, 1] = lab_m['D_MAX_VAL'][pco2_idx]

        ph = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        ph_idx = np.intersect1d(d0_idx, ph_rows)
        if ph_idx.size > 0:
            ph_idx = ph_idx[0]
            ph[0, 0] = lab_m['D_MIN_VAL'][ph_idx]
            ph[0, 1] = lab_m['D_MAX_VAL'][ph_idx]
        ph_idx = np.intersect1d(d1_idx, ph_rows)
        if ph_idx.size > 0:
            ph_idx = ph_idx[0]
            ph[1, 0] = lab_m['D_MIN_VAL'][ph_idx]
            ph[1, 1] = lab_m['D_MAX_VAL'][ph_idx]
        ph_idx = np.intersect1d(d2_idx, ph_rows)
        if ph_idx.size > 0:
            ph_idx = ph_idx[0]
            ph[2, 0] = lab_m['D_MIN_VAL'][ph_idx]
            ph[2, 1] = lab_m['D_MAX_VAL'][ph_idx]

        po2 = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        po2_idx = np.intersect1d(d0_idx, po2_rows)
        if po2_idx.size > 0:
            po2_idx = po2_idx[0]
            po2[0, 0] = lab_m['D_MIN_VAL'][po2_idx]
            po2[0, 1] = lab_m['D_MAX_VAL'][po2_idx]
        po2_idx = np.intersect1d(d1_idx, po2_rows)
        if po2_idx.size > 0:
            po2_idx = po2_idx[0]
            po2[1, 0] = lab_m['D_MIN_VAL'][po2_idx]
            po2[1, 1] = lab_m['D_MAX_VAL'][po2_idx]
        po2_idx = np.intersect1d(d2_idx, po2_rows)
        if po2_idx.size > 0:
            po2_idx = po2_idx[0]
            po2[2, 0] = lab_m['D_MIN_VAL'][po2_idx]
            po2[2, 1] = lab_m['D_MAX_VAL'][po2_idx]
            
        bicarb = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        bicarb_idx = np.intersect1d(d0_idx, bicarb_rows)
        if bicarb_idx.size > 0:
            bicarb_idx = bicarb_idx[0]
            bicarb[0, 0] = lab_m['D_MIN_VAL'][bicarb_idx]
            bicarb[0, 1] = lab_m['D_MAX_VAL'][bicarb_idx]
        bicarb_idx = np.intersect1d(d1_idx, bicarb_rows)
        if bicarb_idx.size > 0:
            bicarb_idx = bicarb_idx[0]
            bicarb[1, 0] = lab_m['D_MIN_VAL'][bicarb_idx]
            bicarb[1, 1] = lab_m['D_MAX_VAL'][bicarb_idx]
        bicarb_idx = np.intersect1d(d2_idx, bicarb_rows)
        if bicarb_idx.size > 0:
            bicarb_idx = bicarb_idx[0]
            bicarb[2, 0] = lab_m['D_MIN_VAL'][bicarb_idx]
            bicarb[2, 1] = lab_m['D_MAX_VAL'][bicarb_idx]

        male = males[i]
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

        anemics[i] = anemic
        albs[i] = alb
        bilis[i] = bili
        buns[i] = bun
        chlorides[i] = chloride
        hemats[i] = hemat
        hemos[i] = hemo
        pltlts[i] = pltlt
        pots[i] = pot
        sods[i] = sod
        wbcs[i] = wbc
        fio2s[i] = fio2
        pco2s[i] = pco2
        po2s[i] = po2
        phs[i] = ph
        bicarbs[i] = bicarb

    return albs, bilis, bicarbs, buns, chlorides, fio2s, hemats, hemos, phs, pco2s, po2s, pots, pltlts, sods, wbcs, anemics
            
        

def get_utsw_flw(ids, dataPath):
    flw_m = pd.read_csv(dataPath + '/icu_flw_data.csv')

    heights = np.full(len(ids), np.nan)
    weights = np.full(len(ids), np.nan)
    bmis = np.full(len(ids), np.nan)

    net_fluids = np.full(len(ids), np.nan)
    gros_fluids = np.full(len(ids), np.nan)
    cum_fluid_bals = np.full(len(ids), np.nan)
    glasgows = np.full((len(ids), 3, 2), np.nan)
    resps = np.full((len(ids), 3, 2), np.nan)
    hrs = np.full((len(ids), 3, 2), np.nan)
    maps = np.full((len(ids), 3, 2), np.nan)
    temps = np.full((len(ids), 3, 2), np.nan)
    fos = np.full((len(ids), 3), np.nan)
    uops = np.full((len(ids), 3), np.nan)
    uflows = np.full((len(ids), 3), np.nan)

    drows = np.union1d(np.union1d(np.where(flw_m['DAY_NO'].values == 'D0')[0], np.where(flw_m['DAY_NO'].values == 'D1')[0]),
                       np.union1d(np.where(flw_m['DAY_NO'].values == 'D2')[0], np.where(flw_m['DAY_NO'].values == 'D3')[0]))
    hrows = np.where(flw_m['TERM_GRP_NAME'].values == 'HEIGHT')[0]
    wrows = np.where(flw_m['TERM_GRP_NAME'].values == 'WEIGHT')[0]
    uoprows = np.where(flw_m['TERM_GRP_NAME'].values == 'UOP')[0]
    temprows = np.where(flw_m['TERM_GRP_NAME'].values == 'TEMPERATURE')[0]
    hrrows = np.where(flw_m['TERM_GRP_NAME'].values == 'HEART RATE')[0]
    maprows = np.where(flw_m['TERM_GRP_NAME'].values == 'MAP')[0]
    inrows = np.where(flw_m['TERM_GRP_NAME'].values == 'IN')[0]
    outrows = np.where(flw_m['TERM_GRP_NAME'].values == 'Out')[0]
    glasrows = np.where(flw_m['TERM_GRP_NAME'].values == 'GLASGOW SCORE')[0]
    resprows = np.where(flw_m['TERM_GRP_NAME'].values == 'RESPIRATORY RATE')[0]

    for i, tid in enumerate(ids):
        prows = np.where(flw_m['PATIENT_NUM'].values == tid)[0]
        pdrows = np.intersect1d(prows, drows)
        hidx = np.intersect1d(pdrows, hrows)
        height = np.nan
        weight = np.nan
        bmi = np.nan
        for idx in hidx:
            if not np.isnan(flw_m['D_MIN_VAL'][idx]):
                height = flw_m['D_MIN_VAL'][idx] / 39.37
                break
        widx = np.intersect1d(pdrows, wrows)
        for idx in widx:
            if not np.isnan(flw_m['D_MIN_VAL'][idx]):
                weight = flw_m['D_MIN_VAL'][idx]
                break
        bmi = weight / (height**2)

        flw_idx = np.where(flw_m['PATIENT_NUM'].values == tid)[0]
        d0_idx = flw_idx[np.where(flw_m['DAY_NO'][flw_idx].values == 'D0')[0]]
        d1_idx = flw_idx[np.where(flw_m['DAY_NO'][flw_idx].values == 'D1')[0]]
        d2_idx = flw_idx[np.where(flw_m['DAY_NO'][flw_idx].values == 'D2')[0]]

        uop_idxs = np.intersect1d(flw_idx, uoprows)
        temp_idxs = np.intersect1d(flw_idx, temprows)
        hr_idxs = np.intersect1d(flw_idx, hrrows)
        map_idxs = np.intersect1d(flw_idx, maprows)
        in_idx = np.intersect1d(flw_idx, inrows)
        out_idx = np.intersect1d(flw_idx, outrows)
        glasgow_idxs = np.intersect1d(flw_idx, glasrows)
        resp_idxs = np.intersect1d(flw_idx, resprows)

        temp = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        temp_idx = np.intersect1d(d0_idx, temp_idxs)
        if temp_idx.size > 0:
            temp_idx = temp_idx[0]
            temp[0, 0] = flw_m['D_MIN_VAL'][temp_idx]
            temp[0, 1] = flw_m['D_MAX_VAL'][temp_idx]
        temp_idx = np.intersect1d(d1_idx, temp_idxs)
        if temp_idx.size > 0:
            temp_idx = temp_idx[0]
            temp[1, 0] = flw_m['D_MIN_VAL'][temp_idx]
            temp[1, 1] = flw_m['D_MAX_VAL'][temp_idx]
        temp_idx = np.intersect1d(d2_idx, temp_idxs)
        if temp_idx.size > 0:
            temp_idx = temp_idx[0]
            temp[2, 0] = flw_m['D_MIN_VAL'][temp_idx]
            temp[2, 1] = flw_m['D_MAX_VAL'][temp_idx]

        tmap = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        tmap_idx = np.intersect1d(d0_idx, map_idxs)
        if tmap_idx.size > 0:
            tmap_idx = tmap_idx[0]
            tmap[0, 0] = flw_m['D_MIN_VAL'][tmap_idx]
            tmap[0, 1] = flw_m['D_MAX_VAL'][tmap_idx]
        tmap_idx = np.intersect1d(d1_idx, map_idxs)
        if tmap_idx.size > 0:
            tmap_idx = tmap_idx[0]
            tmap[1, 0] = flw_m['D_MIN_VAL'][tmap_idx]
            tmap[1, 1] = flw_m['D_MAX_VAL'][tmap_idx]
        tmap_idx = np.intersect1d(d2_idx, map_idxs)
        if tmap_idx.size > 0:
            tmap_idx = tmap_idx[0]
            tmap[2, 0] = flw_m['D_MIN_VAL'][tmap_idx]
            tmap[2, 1] = flw_m['D_MAX_VAL'][tmap_idx]

        hr = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        hr_idx = np.intersect1d(d0_idx, hr_idxs)
        if hr_idx.size > 0:
            hr_idx = hr_idx[0]
            hr[0, 0] = flw_m['D_MIN_VAL'][hr_idx]
            hr[0, 1] = flw_m['D_MAX_VAL'][hr_idx]
        hr_idx = np.intersect1d(d1_idx, hr_idxs)
        if hr_idx.size > 0:
            hr_idx = hr_idx[0]
            hr[1, 0] = flw_m['D_MIN_VAL'][hr_idx]
            hr[1, 1] = flw_m['D_MAX_VAL'][hr_idx]
        hr_idx = np.intersect1d(d2_idx, hr_idxs)
        if hr_idx.size > 0:
            hr_idx = hr_idx[0]
            hr[2, 0] = flw_m['D_MIN_VAL'][hr_idx]
            hr[2, 1] = flw_m['D_MAX_VAL'][hr_idx]

        glasgow = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        glasgow_idx = np.intersect1d(d0_idx, glasgow_idxs)
        if glasgow_idx.size > 0:
            glasgow_idx = glasgow_idx[0]
            glasgow[0, 0] = flw_m['D_MIN_VAL'][glasgow_idx]
            glasgow[0, 1] = flw_m['D_MAX_VAL'][glasgow_idx]
        glasgow_idx = np.intersect1d(d1_idx, glasgow_idxs)
        if glasgow_idx.size > 0:
            glasgow_idx = glasgow_idx[0]
            glasgow[1, 0] = flw_m['D_MIN_VAL'][glasgow_idx]
            glasgow[1, 1] = flw_m['D_MAX_VAL'][glasgow_idx]
        glasgow_idx = np.intersect1d(d2_idx, glasgow_idxs)
        if glasgow_idx.size > 0:
            glasgow_idx = glasgow_idx[0]
            glasgow[2, 0] = flw_m['D_MIN_VAL'][glasgow_idx]
            glasgow[2, 1] = flw_m['D_MAX_VAL'][glasgow_idx]

        resp = np.vstack((np.repeat(np.nan, 2), np.repeat(np.nan, 2), np.repeat(np.nan, 2)))
        resp_idx = np.intersect1d(d0_idx, resp_idxs)
        if resp_idx.size > 0:
            resp_idx = resp_idx[0]
            resp[0, 0] = flw_m['D_MIN_VAL'][resp_idx]
            resp[0, 1] = flw_m['D_MAX_VAL'][resp_idx]
        resp_idx = np.intersect1d(d1_idx, resp_idxs)
        if resp_idx.size > 0:
            resp_idx = resp_idx[0]
            resp[1, 0] = flw_m['D_MIN_VAL'][resp_idx]
            resp[1, 1] = flw_m['D_MAX_VAL'][resp_idx]
        resp_idx = np.intersect1d(d2_idx, resp_idxs)
        if resp_idx.size > 0:
            resp_idx = resp_idx[0]
            resp[2, 0] = flw_m['D_MIN_VAL'][resp_idx]
            resp[2, 1] = flw_m['D_MAX_VAL'][resp_idx]

        urine_out = np.repeat(np.nan, 3)
        urine_flow = np.repeat(np.nan, 3)
        urine_idx = np.intersect1d(d0_idx, uop_idxs)
        if urine_idx.size > 0:
            urine_idx = urine_idx[0]
            urine_out[0] = float(flw_m['D_SUM_VAL'][urine_idx])
            urine_flow[0] = urine_out[0] / weight / 24
        urine_idx = np.intersect1d(d1_idx, uop_idxs)
        if urine_idx.size > 0:
            urine_idx = urine_idx[0]
            urine_out[1] = float(flw_m['D_SUM_VAL'][urine_idx])
            urine_flow[1] = urine_out[1] / weight / 24
        urine_idx = np.intersect1d(d2_idx, uop_idxs)
        if urine_idx.size > 0:
            urine_idx = urine_idx[0]
            urine_out[2] = float(flw_m['D_SUM_VAL'][urine_idx])
            urine_flow[2] = urine_out[2] / weight / 24

        d0_in_idx = np.intersect1d(d0_idx, in_idx)
        d1_in_idx = np.intersect1d(d1_idx, in_idx)
        d2_in_idx = np.intersect1d(d2_idx, in_idx)

        d0_out_idx = np.intersect1d(d0_idx, out_idx)
        d1_out_idx = np.intersect1d(d1_idx, out_idx)
        d2_out_idx = np.intersect1d(d2_idx, out_idx)

        try:
            net0 = float(flw_m['D_SUM_VAL'][d0_in_idx] - flw_m['D_SUM_VAL'][d0_out_idx])
        except:
            net0 = np.nan
        try:
            net1 = float(flw_m['D_SUM_VAL'][d1_in_idx] - flw_m['D_SUM_VAL'][d1_out_idx])
        except:
            net1 = np.nan
        try:
            net2 = float(flw_m['D_SUM_VAL'][d2_in_idx] - flw_m['D_SUM_VAL'][d2_out_idx])
        except:
            net2 = np.nan

        try:
            net = float(np.nanmean((net0, net1, net2)))
        except:
            net = np.nan

        try:
            cfb = float(np.nansum((net0, net1, net2)))
        except:
            cfb = np.nan

        try:
            tot0 = float(flw_m['D_SUM_VAL'][d0_in_idx] + flw_m['D_SUM_VAL'][d0_out_idx])
        except:
            tot0 = np.nan
        try:
            tot1 = float(flw_m['D_SUM_VAL'][d1_in_idx] + flw_m['D_SUM_VAL'][d1_out_idx])
        except:
            tot1 = np.nan
        try:
            tot2 = float(flw_m['D_SUM_VAL'][d2_in_idx] + flw_m['D_SUM_VAL'][d2_out_idx])
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

        heights[i] = height
        weights[i] = weight
        bmis[i] = bmi
        temps[i] = temp
        net_fluids[i] = net
        gros_fluids[i] = tot
        cum_fluid_bals[i] = cfb
        glasgows[i] = glasgow
        resps[i] = resp
        hrs[i] = hr
        maps[i] = tmap
        fos[i] = fo
        uops[i] = urine_out
        uflows[i] = urine_flow
    return heights, weights, bmis, temps, net_fluids, gros_fluids, cum_fluid_bals, glasgows, resps, hrs, maps, fos, uops, uflows


def get_utsw_rrt_v2(ids, dieds, windows, dataPath='', t_lim=7):
    rrt_m = pd.read_csv(os.path.join(dataPath, 'tDialysis.csv'))
    hd_dayss = np.zeros(len(ids))
    crrt_dayss = np.zeros(len(ids))
    hd_dayss_win = np.zeros(len(ids))
    crrt_dayss_win = np.zeros(len(ids))
    hd_frees_7d = np.zeros(len(ids))
    hd_frees_28d = np.zeros(len(ids))
    crrt_frees_7d = np.zeros(len(ids))
    crrt_frees_28d = np.zeros(len(ids))
    rrt_flags = np.zeros(len(ids))
    hd_trtmts = np.zeros(len(ids))
    crrt_flags = np.zeros(len(ids))
    hd_flags = np.zeros(len(ids))
    for i in range(len(ids)):
        tid = ids[i]
        admit = windows[tid][0]
        disch = windows[tid][1]
        died = dieds[i]
        if type(admit) == str:
            admit = get_date(admit)
            disch = get_date(disch)
        endwin = admit + datetime.timedelta(t_lim)
        end28d = admit + datetime.timedelta(28)
        hd_days = 0
        hd_days_win = 0
        crrt_days = 0
        crrt_days_win = 0
        rrt_flag = 0
        hd_trtmt = 0
        crrt_flag = 0
        hd_flag = 0
        dia_ids = np.where(rrt_m['PATIENT_NUM'].values == tid)[0]
        if np.size(dia_ids) > 0:
            for row in dia_ids:
                start = get_date(rrt_m['START_DATE'][row])
                stop = get_date(rrt_m['END_DATE'][row])
                if start != 'nan' and stop != 'nan':
                    wstart = max(admit, start)
                    wstop = min(disch, stop)
                    tdays_win = 0
                    tdays = 0
                    tflag = 0
                    if wstop > wstart and wstart < endwin:
                        tflag = 1
                        tdays_win = (wstop - wstart).total_seconds() / (60 * 60 * 24)
                    if wstart < end28d:
                        tdays = (wstop - wstart).total_seconds() / (60 * 60 * 24)

                    if str(rrt_m['DIALYSIS_TYPE'][row]).lower() == 'crrt':
                        crrt_days_win += tdays_win
                        crrt_days += tdays
                        crrt_flag = tflag
                        rrt_flag = tflag
                    elif str(rrt_m['DIALYSIS_TYPE'][row]).lower() == 'hd':
                        hd_days_win += tdays_win
                        hd_days += tdays
                        hd_flag = tflag
                        rrt_flag = tflag

        hd_free_win = min(0, t_lim - hd_days_win)
        hd_free_28d = min(0, 28 - hd_days)
        crrt_free_win = min(0, t_lim - crrt_days_win)
        crrt_free_28d = min(0, 28 - crrt_days)

        if hd_days or crrt_days:
            if died:
                hd_free_win = 0
                hd_free_28d = 0
                crrt_free_win = 0
                crrt_free_28d = 0

        hd_dayss[i] = hd_days
        crrt_dayss[i] = crrt_days
        hd_dayss_win[i] = hd_days_win
        crrt_dayss_win[i] = crrt_days_win
        hd_frees_7d[i] = hd_free_win
        hd_frees_28d[i] = hd_free_28d
        crrt_frees_7d[i] = crrt_free_win
        crrt_frees_28d[i] = crrt_free_28d
        rrt_flags[i] = rrt_flag
        hd_trtmts[i] = hd_trtmt
        crrt_flags[i] = crrt_flag
        hd_flags[i] = hd_flag
    del rrt_m
    return rrt_flags, hd_dayss, crrt_dayss, hd_dayss_win, crrt_dayss_win, hd_frees_7d, hd_frees_28d, hd_trtmts, crrt_frees_7d, crrt_frees_28d, crrt_flags, hd_flags


def get_utsw_rrt(ids, icu_windows, dataPath):
    rrt_m = pd.read_csv(os.path.join(dataPath, 'tDialysis.csv'))
    rrt_m.sort_values(by=['PATIENT_NUM'], inplace=True)
    rrt_start_loc = rrt_m.columns.get_loc('START_DATE')
    rrt_stop_loc = rrt_m.columns.get_loc('END_DATE')
    rrt_type_loc = rrt_m.columns.get_loc('DIALYSIS_TYPE')
    rrt_m = rrt_m.values

    hd_dayss = np.zeros(len(ids))
    crrt_dayss = np.zeros(len(ids))
    hd_free_7ds = np.zeros(len(ids))
    hd_free_28ds = np.zeros(len(ids))
    crrt_free_7ds = np.zeros(len(ids))
    crrt_free_28ds = np.zeros(len(ids))

    for i in range(len(ids)):
        hd_days = 0
        crrt_days = 0
        hd_free_7d = np.ones(7)
        hd_free_28d = np.ones(28)
        crrt_free_7d = np.ones(7)
        crrt_free_28d = np.ones(28)

        tid = ids[i]
        dia_ids = np.where(rrt_m[:, 0] == tid)[0]
        if type(icu_windows) == dict:
            icu_admit = icu_windows[tid][0]
        else:
            icu_admit = icu_windows[i][0]
        if type(icu_admit) != datetime.datetime:
            icu_admit = get_date(icu_admit)
        if np.size(dia_ids) > 0:
            for row in dia_ids:
                start = get_date(rrt_m[row, rrt_start_loc])
                stop = get_date(rrt_m[row, rrt_stop_loc])
                rrt_type = rrt_m[row, rrt_type_loc]
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

        hd_free_7d = np.sum(hd_free_7d)
        hd_free_28d = np.sum(hd_free_28d)
        crrt_free_7d = np.sum(crrt_free_7d)
        crrt_free_28d = np.sum(crrt_free_28d)

        hd_dayss[i] = hd_days
        crrt_dayss[i] = crrt_days
        hd_free_7ds[i] = hd_free_7d
        hd_free_28ds[i] = hd_free_28d
        crrt_free_7ds[i] = crrt_free_7d
        crrt_free_28ds[i] = crrt_free_28d

    return hd_dayss, crrt_dayss, hd_free_7ds, hd_free_28ds, crrt_free_7ds, crrt_free_28ds


def get_utsw_medications(ids, dataPath):
    med_m = pd.read_csv(os.path.join(dataPath, 'tMedications.csv'))
    med_m.sort_values(by='PATIENT_NUM', inplace=True)
    neph_cts = np.zeros(len(ids))
    vaso_cts = np.zeros(len(ids))
    neph_cols = [med_m.columns.get_loc(x) for x in ['ACEI', 'ARB', 'NSAIDS', 'AMINOGLYCOSIDES']]
    vaso_cols = [med_m.columns.get_loc(x) for x in ['PRESSOR_INOTROPE']]
    med_m = med_m.values
    for i in range(len(ids)):
        tid = ids[i]
        idx = np.where(med_m[:, 0] == tid)[0]
        neph_ct = 0
        vaso_ct = 0
        if idx.size > 0:
            idx = idx[0]
            neph_ct = sum([med_m[idx, x] for x in neph_cols])
            vaso_ct = sum([med_m[idx, x] for x in vaso_cols])
        neph_cts[i] = neph_ct
        vaso_cts[i] = vaso_ct
    del med_m
    return neph_cts, vaso_cts


def get_utsw_diagnoses(ids, dataPath):
    # tHospitalFinalDiagnosis
    septics = np.zeros(len(ids), dtype=int)
    diabetics = np.zeros(len(ids), dtype=int)
    hypertensives = np.zeros(len(ids), dtype=int)
    diag_m = pd.read_csv(os.path.join(dataPath, 'tHospitalFinalDiagnoses.csv'))
    diag_loc = diag_m.columns.get_loc('DX_NAME')
    diag_m = diag_m.values
    for i in range(len(ids)):
        idx = ids[i]
        septic = 0
        diabetic = 0
        hypertensive = 0
        diag_ids = np.where(diag_m[:, 0] == idx)[0]
        for j in range(len(diag_ids)):
            tid = diag_ids[j]
            if 'sep' in str(diag_m[tid, diag_loc]).lower():
                # if int(diag_m[tid, diag_nb_loc]) == 1:
                septic = 1
            if 'diabe' in str(diag_m[tid, diag_loc]).lower():
                diabetic = 1
            if 'hypert' in str(diag_m[tid, diag_loc]).lower():
                hypertensive = 1
        septics[i] = septic
        diabetics[i] = diabetic
        hypertensives[i] = hypertensive
    return septics, diabetics, hypertensives


def get_utsw_mortality(ids, icu_windows, hosp_windows, dataPath):
    usrds_m = pd.read_csv(os.path.join(dataPath, 'tUSRDS.csv'))
    pat_m = pd.read_csv(os.path.join(dataPath, 'tPatients.csv'))
    died_inps = np.zeros(len(ids))
    died_90d_admit = np.zeros(len(ids))
    died_90d_disch = np.zeros(len(ids))
    died_120d_admit = np.zeros(len(ids))
    died_120d_disch = np.zeros(len(ids))
    dtds = np.zeros(len(ids))
    dods = np.zeros(len(ids), dtype='|S20')
    dods[:] = 'nan'
    for i, tid in enumerate(ids):
        died_inp = 0
        died_90a = 0
        died_90d = 0
        died_120a = 0
        died_120d = 0
        if type(icu_windows) == list:
            admit = icu_windows[i][0]
            disch = hosp_windows[i][1]
        else:
            admit = icu_windows[tid][0]
            disch = hosp_windows[tid][1]

        dtd = np.nan
        dod = 'nan'

        if type(admit) != datetime.datetime:
            admit = get_date(str(admit))
            disch = get_date(str(disch))

        idx = np.where(pat_m['PATIENT_NUM'].values == tid)[0][0]
        tm = get_date(str(pat_m['DOD_Epic'][idx]))
        if tm != 'nan':
            dod = tm
        tm = get_date(str(pat_m['DOD_NDRI'][idx]))
        if tm != 'nan':
            dod = tm
        tm = get_date(str(pat_m['DMF_DEATH_DATE'][idx]))
        if tm != 'nan':
            dod = tm

        idx = np.where(usrds_m['PATIENT_NUM'].values == tid)[0]
        if idx.size > 0:
            idx = idx[0]
            tm = get_date(usrds_m['DIED'][idx])
            if tm != 'nan':
                if dod != 'nan':
                    if tm != dod:
                        print('Patient %d has DOD mismatch' % tid)
                        print('EHR: %s\tUSRDS: %s' % (str(dod), str(tm)))
                    dod = min(dod, tm)
                dod = tm

        if dod != 'nan':
            if dod < disch or dod.toordinal() == disch.toordinal():
                died_inp = 1
            if (dod - disch) < datetime.timedelta(90):
                died_90d = 1
            if (dod - disch) < datetime.timedelta(120):
                died_120d = 1
            if (dod - admit) < datetime.timedelta(90):
                died_90a = 1
            if (dod - admit) < datetime.timedelta(120):
                died_120a = 1
            dtd = (dod - admit).total_seconds() / (60*60*24)
        died_inps[i] = died_inp
        died_90d_admit[i] = died_90a
        died_90d_disch[i] = died_90d
        died_120d_admit[i] = died_120a
        died_120d_disch[i] = died_120d
        dods[i] = str(dod)
        dtds[i] = dtd
    return died_inps, died_90d_admit, died_90d_disch, died_120d_admit, died_120d_disch, dods, dtds


def get_utsw_demographics(ids, hosp_windows, dataPath=''):
    dem_m = pd.read_csv(os.path.join(dataPath, 'tPatients.csv'))
    races = np.zeros(len(ids))
    males = np.full(len(ids), np.nan)
    ages = np.full(len(ids), np.nan)
    dods = np.zeros(len(ids), dtype='|S20').astype(str)
    dtds = np.full(len(ids), np.nan)
    for i, tid in enumerate(ids):
        dem_idx = np.where(dem_m['PATIENT_NUM'] == tid)[0]
        if dem_idx.size > 0:
            if type(hosp_windows) == dict:
                hosp_admit = hosp_windows[tid][0]
            else:
                hosp_admit = hosp_windows[i][0]
            if type(hosp_admit) != datetime.datetime:
                hosp_admit = get_date(hosp_admit)
            dem_idx = dem_idx[0]
            males[i] = dem_m['SEX_ID'][dem_idx]
            white = dem_m['RACE_WHITE'][dem_idx]
            black = dem_m['RACE_BLACK'][dem_idx]
            if white == 1:
                races[i] = 0
            elif black == 1:
                races[i] = 1
            else:
                races[i] = 2
            dob = get_date(dem_m['DOB'][dem_idx])
            if dob != 'nan' and hosp_admit != 'nan':
                ages[i] = (hosp_admit - dob).total_seconds() / (60 * 60 * 24 * 365)
            for col in ['DOD_Epic', 'DOD_NDRI', 'DMF_DEATH_DATE']:
                tdod = get_date(dem_m[col][dem_idx])
                if tdod != 'nan':
                    dod = tdod
                    dtd = (dod - hosp_admit).total_seconds() / (60 * 60 * 24)
                    dods[i] = dod
                    dtds[i] = dtd
                    break
    return males, races, ages, dods, dtds


def get_utsw_organsupp(ids, icu_windows, dtds, died_inps, dataPath):
    organ_sup = pd.read_csv(os.path.join(dataPath, 'tAOS.csv'))
    mech_flags = np.zeros(len(ids))
    mech_days = np.zeros(len(ids))
    ecmos = np.zeros(len(ids))
    iabps = np.zeros(len(ids))
    vads = np.zeros(len(ids))
    mv_frees_7d = np.zeros(len(ids))
    mv_frees_28d = np.zeros(len(ids))
    for i, tid in enumerate(ids):
        mech_flag = 0
        mech_day = 0
        ecmo = 0
        iabp = 0
        vad = 0
        mech_free_7d = 7
        mech_free_28d = 28

        if type(icu_windows) == dict:
            icu_admit = icu_windows[tid][0]
        else:
            icu_admit = icu_windows[i][0]

        organ_sup_idx = np.where(organ_sup['PATIENT_NUM'].values == tid)[0]
        if organ_sup_idx.size > 0:
            for row in organ_sup_idx:
                mech_start = get_date(organ_sup['MV_START_DATE'][row])
                mech_stop = get_date(organ_sup['MV_END_DATE'][row])
                iabp_start = get_date(organ_sup['IABP_START_DATE'][row])
                iabp_stop = get_date(organ_sup['IABP_END_DATE'][row])
                vad_start = get_date(organ_sup['VAD_START_DATE'][row])
                vad_stop = get_date(organ_sup['VAD_END_DATE'][row])
                ecmo_start = get_date(organ_sup['ECMO_START_DATE'][row])
                ecmo_stop = get_date(organ_sup['ECMO_END_DATE'][row])

                if mech_stop != 'nan':
                    if mech_stop < icu_admit:
                        pass
                    elif (mech_start - icu_admit).days > 28:
                        pass
                    else:
                        if mech_start < icu_admit:
                            tmech_day = (mech_stop - icu_admit).days
                        else:
                            tmech_day = (mech_stop - mech_start).days

                        mech_day += tmech_day
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

        dtd = dtds[i]
        died_inp = died_inps[i]
        mech_free_7d = 7 - mech_day
        mech_free_28d = 28 - mech_day
        if died_inp:
            if mech_flag and dtd < 7:
                mech_free_7d = 0
                mech_free_28d = 0
            if mech_flag and dtd < 28:
                mech_free_28d = 0
        if mech_free_7d < 0:
            mech_free_7d = 0
        if mech_free_28d < 0:
            mech_free_28d = 0
            
        mech_flags[i] = mech_flag
        mech_days[i] = mech_day
        ecmos[i] = ecmo
        iabps[i] = iabp
        vads[i] = vad
        mv_frees_7d[i] = mech_free_7d
        mv_frees_28d[i] = mech_free_28d
    return mech_flags, mech_days, ecmos, iabps, vads, mv_frees_7d, mv_frees_28d

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
    gros_fluid = meta['gros_fluid'][:][sel]


    m_kdigos = meta['max_kdigo'][:][sel]
    n_eps = meta['n_episodes'][:][sel]
    hd_days = meta['hd_days'][:][sel]
    crrt_days = meta['crrt_days'][:][sel]
    rrt_tot = hd_days + crrt_days

    neph_cts = meta['nephrotox_ct'][:][sel]
    vaso_cts = meta['vasopres_ct'][:][sel]
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
    #                      'gros_fluid_med', 'gros_fluid_25', 'gros_fluid_75',
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

        # Acute Illnes
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
        # Number of presor or inotrope - median[IQ1 - IQ3]
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
    #     fluid_sel = np.where(gros_fluid[rows] >= 0)[0]
    #     (net_fluid_med, net_fluid_25, net_fluid_75) = iqr(net_fluid[rows[fluid_sel]])
    #     (gros_fluid_med, gros_fluid_25, gros_fluid_75) = iqr(gros_fluid[rows[fluid_sel]])
    #
    #     cluster_data['net_fluid'][cluster_id] = net_fluid[rows[fluid_sel]]
    #     cluster_data['gros_fluid'][cluster_id] = gros_fluid[rows[fluid_sel]]
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
    #          gros_fluid_med, gros_fluid_25, gros_fluid_75,
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

    # Acute Illnes
    table_f.write('Acute illnes characteristics\n')
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
    filterwarnings("ignore")
    if summary_type == 'count':
        tc = len(np.where(all_data)[0])
        if tot_size > 0:
            s = (',%d (' + fmt + ')') % (tc, float(tc) / tot_size * 100)
        else:
            s = (',%d (' + fmt + ')') % (tc, 0)
        for i in range(len(lbl_names)):
            idx = np.where(lbls == lbl_names[i])[0]
            tc = len(np.where(all_data[idx])[0])
            if len(lbl_names) == 2 and i == 1:
                if len(idx) > 0:
                    s += (',%d%s (' + fmt + ')') % (tc, pstr, float(tc) / len(idx) * 100)
                else:
                    s += (',%d%s (' + fmt + ')') % (tc, pstr, 0)
            else:
                if len(idx) > 0:
                    s += (',%d (' + fmt + ')') % (tc, float(tc) / len(idx) * 100)
                else:
                    s += (',%d (' + fmt + ')') % (tc, 0)
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
                s += (',' + fmt + '%s (' + fmt + '-' + fmt + ')') % (m, pstr, iq1, iq2)
            else:
                s += (',' + fmt + ' (' + fmt + '-' + fmt + ')') % (m, iq1, iq2)

    else:
        mn, std = np.nanmean(all_data), np.nanstd(all_data)
        m, iq1, iq2 = iqr(all_data)
        s = (',' + fmt + ' (+-' + fmt + ') ' + fmt + ' (' + fmt + '-' + fmt + ')') % (mn, std, m, iq1, iq2)
        for i in range(len(lbl_names)):
            idx = np.where(lbls == lbl_names[i])[0]
            mn, std = np.nanmean(all_data[idx].flatten()), np.nanstd(all_data[idx].flatten())
            m, iq1, iq2 = iqr(all_data[idx].flatten())
            s += (',' + fmt + ' (+-' + fmt + ') ' + fmt + ' (' + fmt + '-' + fmt + ')') % (mn, std, m, iq1, iq2)

    if len(lbl_names) == 2 and summary_type != 'all':
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
    filterwarnings("default")
    return s


# %%
def formatted_stats(meta, label_path):
    all_ids = meta['ids'][:]
    if os.path.exists(os.path.join(label_path, 'labels.csv')):
        ids = np.loadtxt(os.path.join(label_path, 'labels.csv'), delimiter=',', usecols=0)
        lbls = load_csv(os.path.join(label_path, 'labels.csv'), ids, dt=str)
        lbl_names = np.unique(lbls)
    else:
        ids = np.loadtxt(os.path.join(label_path, 'clusters.csv'), delimiter=',', usecols=0)
        lbls = load_csv(os.path.join(label_path, 'clusters.csv'), ids, dt=str)
        lbl_names = np.unique(lbls)

    sel = np.array([x in ids for x in all_ids])
    ages = meta['age'][:][sel]
    genders = meta['gender'][:][sel]
    races = meta['race'][:][sel]
    bmis = meta['bmi'][:][sel]
    charls = meta['charlson'][:][sel]
    elixs = meta['elixhauser'][:][sel]
    diabetics = meta['diabetic'][:][sel]
    hypertensives = meta['hypertensive'][:][sel]
    maps = meta['map'][:][sel]
    if maps.ndim == 2:
        maps = maps[:, 0]

    sofa = meta['sofa'][:][sel]
    apache = meta['apache'][:][sel]
    hosp_days = meta['hosp_los'][:][sel]
    try:
        hosp_free = meta['hosp_free_days'][:][sel]
    except KeyError:
        hosp_free = meta['hosp_free_days_28d'][:][sel]
    icu_days = meta['icu_los'][:][sel]
    try:
        icu_free = meta['icu_free_days'][:][sel]
    except KeyError:
        icu_free = meta['icu_free_days_28d'][:][sel]
    mv_days = meta['mv_days'][:][sel]
    try:
        mv_free = meta['mv_free_days'][:][sel]
    except KeyError:
        mv_free = meta['mv_free_days_28d'][:][sel]
    mv_flag = meta['mv_flag'][:][sel]
    ecmo_flag = meta['ecmo'][:][sel]
    iabp_flag = meta['iabp'][:][sel]
    vad_flag = meta['vad'][:][sel]

    sepsis = meta['septic'][:][sel]
    died_inp = meta['died_inp'][:][sel]
    died_90d_admit = meta['died_90d_admit'][:][sel]
    died_90d_disch = meta['died_90d_disch'][:][sel]
    died_120d_admit = meta['died_120d_admit'][:][sel]
    died_120d_disch = meta['died_120d_disch'][:][sel]

    bsln_types = meta['baseline_type'][:].astype('U18')[sel]
    bsln_scr = meta['baseline_scr'][:][sel]
    bsln_gfr = meta['baseline_gfr'][:][sel]
    admit_scr = meta['admit_scr'][:][sel]
    peak_scr = meta['peak_scr'][:][sel]
    net_fluid = meta['net_fluid'][:][sel]
    cum_fluid = meta['cum_fluid_balance'][:][sel]
    fluid_overload = meta['fluid_overload'][:][sel]

    mks = meta['max_kdigo'][:][sel]
    mks7d = meta['max_kdigo_win'][:][sel]
    hd_days = meta['hd_days'][:][sel]
    hd_trtmts = meta['hd_treatments'][:][sel]
    crrt_days = meta['crrt_days'][:][sel]
    rrt_tot = hd_days + crrt_days
    hd_free_7d = meta['hd_free_win'][:][sel]
    hd_free_28d = meta['hd_free_28d'][:][sel]
    crrt_free_7d = meta['crrt_free_win'][:][sel]
    crrt_free_28d = meta['crrt_free_28d'][:][sel]
    crrt_flag = meta['crrt_flag'][:][sel]
    hd_flag = meta['hd_flag'][:][sel]
    rrt_flag = meta['rrt_flag'][:][sel]

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
    phs = meta['ph'][:, 0][sel]

    hosp_free[np.where(died_inp)] = 0
    # icu_free[np.where(died_inp)] = 0
    # mv_free[np.where(died_inp)] = 0
    # hd_free_7d[np.where(died_inp)] = 0
    # hd_free_28d[np.where(died_inp)] = 0
    # crrt_free_7d[np.where(died_inp)] = 0
    # crrt_free_28d[np.where(died_inp)] = 0

    # Generate formatted table as text
    table_f = open(os.path.join(label_path, 'formatted_table.csv'), 'w')
    stat_f = open(os.path.join(label_path, 'summary_statistics.csv'), 'w')
    if len(lbl_names) == 2:
        hdr = ',Total,%s,%s,Tot_Normal,Neg_Normal,Pos_Normal,T-test_pval,Kruskal_pval' % (lbl_names[0], lbl_names[1])
    else:
        hdr = ',Total'
        for i in range(len(lbl_names)):
            hdr += ',%s' % lbl_names[i]
    table_f.write(hdr + '\n')
    stat_f.write(hdr + '\n')

    np.seterr(invalid='ignore')
    row = 'Count,%d' % len(lbls)
    for lbl in lbl_names:
        row += ',%d' % len(np.where(lbls == lbl)[0])
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    s = build_str(died_inp, lbls, 'count', '%.2f')

    row = 'Inpatient Mortality - n (%)' + s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    s = build_str(died_90d_admit, lbls, 'count', '%.2f')

    row = 'Mortality w/in ICU Admit + 90d - n (%)' + s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    s = build_str(died_120d_admit, lbls, 'count', '%.2f')

    row = 'Mortality w/in ICU Admit + 120d - n (%)' + s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    s = build_str(died_90d_disch, lbls, 'count', '%.2f')

    row = 'Mortality w/in Hospital Discharge + 90d - n (%)' + s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    s = build_str(died_120d_disch, lbls, 'count', '%.2f')

    row = 'Mortality w/in Hospital Discharge + 120d - n (%)' + s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # Demographics
    table_f.write('Demographics\n')
    row = 'Age - mean(SD)'
    s = build_str(ages, lbls, 'mean', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Age '
    s = build_str(ages, lbls, 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # Gender - male n( %)
    row = 'Gender - male n(%)'
    s = build_str(genders, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # Ethnic group - white n( %)
    row = 'Ethnic group - white n(%)'
    s = build_str(np.array(races == 0), lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # Ethnic group - black n( %)
    row = 'Ethnic group - black n(%)'
    s = build_str(np.array(races == 1), lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # Ethnic group - other n( %)
    row = 'Ethnic group - other n(%)'
    s = build_str(np.array(races == 2), lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # BMI - mean(SD)
    row = 'BMI - mean(SD)'
    s = build_str(bmis, lbls, 'mean', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'BMI'
    s = build_str(bmis, lbls, 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # Comorbidity
    table_f.write('Comorbidity\n')
    row = 'Charlson Comorbidty Index Score - mean(SD)'
    s = build_str(charls[np.where(charls > 0)], lbls[np.where(charls > 0)], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Charlson Comorbidty Index Score'
    s = build_str(charls[np.where(charls > 0)], lbls[np.where(charls > 0)], 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')


    # Elixhauser Comorbidty Index Score - mean(SD)
    row = 'Elixhauser Comorbidty Index Score - mean(SD)'
    s = build_str(elixs[np.where(elixs > 0)], lbls[np.where(elixs > 0)], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Elixhauser Comorbidty Index Score'
    s = build_str(elixs[np.where(elixs > 0)], lbls[np.where(elixs > 0)], 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # Diabetes - n( %)
    row = 'Diabetes - n(%)'
    s = build_str(diabetics, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # Hypertension - n( %)
    row = 'Hypertension - n(%)'
    s = build_str(hypertensives, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # Smoker status - n( %)
    row = 'Smoker status - n(%)'
    s = build_str(smokers, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # Anemia Def A - n( %)
    row = 'Anemia Definition A - n(%)'
    s = build_str(anemia[:, 0], lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')
    # stat_f.write(row + '\n')
    # Anemia Def B - n( %)
    row = 'Anemia Definition B - n(%)'
    s = build_str(anemia[:, 1], lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')
    # stat_f.write(row + '\n')
    # Anemia Def C - n( %)
    row = 'Anemia Definition C - n(%)'
    s = build_str(anemia[:, 2], lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')
    # stat_f.write(row + '\n')


    # Acuity of Critical Illnes
    table_f.write('Acuity of Critical Illnes\n')
    # Hospital length of stay - median[IQ1 - IQ3]
    row = 'Hospital free days - median[IQ1 - IQ3]'
    s = build_str(hosp_free, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Hospital free days'
    s = build_str(hosp_free, lbls, 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # ICU length of stay - median[IQ1 - IQ3]
    row = 'ICU free days - median[IQ1 - IQ3]'
    s = build_str(icu_free, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'ICU free days'
    s = build_str(icu_free, lbls, 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # SOFA score - median[IQ1 - IQ3]
    row = 'SOFA score - median[IQ1 - IQ3]'
    s = build_str(sofa, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'SOFA score'
    s = build_str(sofa, lbls, 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # APACHE Score, - median[IQ1 - IQ3]
    row = 'APACHE score - median[IQ1 - IQ3]'
    s = build_str(apache, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'APACHE score'
    s = build_str(apache, lbls, 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # Mechanical ventilation - n( %)
    row = 'Mechanical ventilation - n(%)'
    s = build_str(mv_flag, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # Days on mechanical ventilation - mean[IQ1 - IQ3]
    row = 'Mechanical ventilation free days - mean[IQ1 - IQ3]'
    s = build_str(mv_free[np.where(mv_flag)], lbls[np.where(mv_flag)], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Mechanical ventilation free days'
    s = build_str(mv_free[np.where(mv_flag)], lbls[np.where(mv_flag)], 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # ECMO - n( %)
    row = 'ECMO - n(%)'
    s = build_str(ecmo_flag, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # IABP - n( %)
    row = 'IABP - n(%)'
    s = build_str(iabp_flag, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # VAD - n( %)
    row = 'VAD - n(%)'
    s = build_str(vad_flag, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    table_f.write('Nephrotoxins:\n')
    stat_f.write('Nephrotoxins:\n')
    # Number w/ >=1 Nephrotoxins D0-1 - n (%)
    row = '\t>=1 - n (%)'
    s = build_str(neph_cts >= 1, lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')
    # Number w/ >=2 Nephrotoxins D0-1 - n (%)
    row = '\t>=2 - n (%)'
    s = build_str(neph_cts >= 2, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # Number w/ >=3 Nephrotoxins D0-1 - n (%)
    row = '\t>=3 - n (%)'
    s = build_str(neph_cts >= 3, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # Vasoactive drugs, n (%)
    row = 'Vasoactive drugs - n (%)'
    s = build_str(vaso_cts >= 1, lbls, 'count', '%.1f')
    row += s

    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # Cumulative fluid balance at 72 h - median[IQ1 - IQ3]
    row = 'Cumulative fluid balance at 72 h - median[IQ1 - IQ3]'
    s = build_str(cum_fluid, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Cumulative fluid balance at 72 h'
    s = build_str(cum_fluid, lbls, 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # FO% at 72 h - median[IQ1 - IQ3]
    row = 'FO% at 72 h - median[IQ1 - IQ3]'
    s = build_str(fluid_overload * 100, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'FO% at 72 h'
    s = build_str(fluid_overload * 100, lbls, 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # Sepsis - n( %)
    row = 'Sepsis - n(%)'
    s = build_str(sepsis, lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # pH < 7.30- n( %)
    row = 'pH < 7.3 - n(%)'
    s = build_str(phs < 7.3, lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # MAP < 70 mmHg - n( %)
    row = 'MAP mmHg < 70 - n(%)'
    s = build_str(maps < 70, lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # MAP < 60 mmHg - n( %)
    row = 'MAP mmHg < 60 - n(%)'
    s = build_str(maps < 60, lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # Serum Albumin, g/dL - median[IQ1 - IQ3]
    row = 'Serum Albumin g/dL - median[IQ1 - IQ3]'
    s = build_str(albs, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Serum Albumin g/dL'
    s = build_str(albs, lbls, 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # Serum Lactate, mmol/L - median[IQ1 - IQ3]
    row = 'Serum Lactate mmol/L - median[IQ1 - IQ3]'
    s = build_str(lacs, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Serum Lactate mmol/L'
    s = build_str(lacs, lbls, 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # Serum Bilirubin, mg/dL - median[IQ1 - IQ3]
    row = 'Serum Bilirubin mg/dL - median[IQ1 - IQ3]'
    s = build_str(bilis, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Serum Bilirubin mg/dL'
    s = build_str(bilis, lbls, 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # AKI Characteristics
    table_f.write('AKI characteristics\n')

    # Baseline SCr, mg/dL - median[IQ1 - IQ3]
    row = 'ALL Baseline SCr mg/dL - median[IQ1 - IQ3]'
    s = build_str(bsln_scr, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'ALL Baseline SCr mg/dL'
    s = build_str(bsln_scr, lbls, 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # Baseline eGFR, mL/min/1.73m2 - median[IQ1 - IQ3]
    row = 'ALL Baseline eGFR mL/min/1.73m2 - median[IQ1 - IQ3]'
    s = build_str(bsln_gfr, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'ALL Baseline eGFR mL/min/1.73m2'
    s = build_str(bsln_gfr, lbls, 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # Measured Baselines - n (%)
    row = 'Measured Baselines - n (%)'
    s = build_str(np.array(bsln_types == 'measured'), lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # Baseline SCr, mg/dL - median[IQ1 - IQ3]
    sel = np.where(bsln_types == 'measured')[0]
    row = 'Measured Baseline SCr mg/dL - median[IQ1 - IQ3]'
    s = build_str(bsln_scr[sel], lbls[sel], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Measured Baseline SCr mg/dL'
    s = build_str(bsln_scr[sel], lbls[sel], 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # Baseline eGFR, mL/min/1.73m2 - median[IQ1 - IQ3]
    row = 'Measured Baseline eGFR mL/min/1.73m2 - median[IQ1 - IQ3]'
    s = build_str(bsln_gfr[sel], lbls[sel], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Measured Baseline eGFR mL/min/1.73m2'
    s = build_str(bsln_gfr[sel], lbls[sel], 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # Baseline SCr, mg/dL - median[IQ1 - IQ3]
    sel = np.where(bsln_types == 'imputed')[0]
    row = 'Imputed Baseline SCr mg/dL - median[IQ1 - IQ3]'
    s = build_str(bsln_scr[sel], lbls[sel], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Imputed Baseline SCr mg/dL'
    s = build_str(bsln_scr[sel], lbls[sel], 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # Baseline eGFR mL/min/1.73m2 - median[IQ1 - IQ3]
    row = 'Imputed Baseline eGFR mL/min/1.73m2 - median[IQ1 - IQ3]'
    s = build_str(bsln_gfr[sel], lbls[sel], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Imputed Baseline eGFR mL/min/1.73m2'
    s = build_str(bsln_gfr[sel], lbls[sel], 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # Admit SCr, mg/dL - median[IQ1 - IQ3]
    row = 'Admit SCr mg/dL - median[IQ1 - IQ3]'
    s = build_str(admit_scr, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Admit SCr mg/dL'
    s = build_str(admit_scr, lbls, 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # Peak SCr, mg/dL - median[IQ1 - IQ3]
    row = 'Peak SCr mg/dL - median[IQ1 - IQ3]'
    s = build_str(peak_scr, lbls, 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Peak SCr mg/dL'
    s = build_str(peak_scr, lbls, 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # Max KDIGO stage
    table_f.write('Maximum KDIGO Stage - Whole ICU:\n')
    stat_f.write('Maximum KDIGO Stage - Whole ICU\n')
    # Stage 1,  n (%)
    row = '\tStage 1 - n (%)'
    s = build_str(mks == 1, lbls, 'countrow', '%.1f')
    row += s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')
    # Stage 2,  n (%)
    row = '\tStage 2 - n (%)'
    s = build_str(mks == 2, lbls, 'countrow', '%.1f')
    row += s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')
    # Stage 3,  n (%)
    row = '\tStage 3 - n (%)'
    s = build_str(mks == 3, lbls, 'countrow', '%.1f')
    row += s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')
    # Stage 3D,  n (%)
    row = '\tStage 3D - n (%)'
    s = build_str(mks == 4, lbls, 'countrow', '%.1f')
    row += s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    table_f.write('Maximum KDIGO Stage - 14 Days:\n')
    stat_f.write('Maximum KDIGO Stage - 14 Days:\n')
    # Stage 1,  n (%)
    row = '\tStage 1 - n (%)'
    s = build_str(mks7d == 1, lbls, 'countrow', '%.1f')
    row += s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')
    # Stage 2,  n (%)
    row = '\tStage 2 - n (%)'
    s = build_str(mks7d == 2, lbls, 'countrow', '%.1f')
    row += s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')
    # Stage 3,  n (%)
    row = '\tStage 3 - n (%)'
    s = build_str(mks7d == 3, lbls, 'countrow', '%.1f')
    row += s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')
    # Stage 3D,  n (%)
    row = '\tStage 3D - n (%)'
    s = build_str(mks7d == 4, lbls, 'countrow', '%.1f')
    row += s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # Urine output, L D0-D2 - median[IQ1 - IQ3]
    row = 'Urine output L D0-D2 - median[IQ1 - IQ3]'
    s = build_str(urine_out[np.where(urine_out != 0)], lbls[np.where(urine_out != 0)[0]], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Urine output L D0-D2'
    s = build_str(urine_out[np.where(urine_out != 0)], lbls[np.where(urine_out != 0)[0]], 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # Urine flow, ml/kg/24h D0-D2 - median[IQ1 - IQ3]
    row = 'Urine flow ml/kg/24h D0-D2 - median[IQ1 - IQ3]'
    s = build_str(urine_flow[np.where(urine_flow != 0)], lbls[np.where(urine_flow != 0)[0]], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Urine flow ml/kg/24h D0-D2'
    s = build_str(urine_flow[np.where(urine_flow != 0)], lbls[np.where(urine_flow != 0)[0]], 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # Inpatient RRT
    table_f.write('Inpatient RRT characteristics\n')
    # Number of Patients with CRRT - n (%)
    row = 'Number of Patients on CRRT - n (%)'
    s = build_str(crrt_flag, lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')
    # Number of Patients with HD - n (%)
    row = 'Number of Patients on HD - n (%)'
    s = build_str(hd_flag, lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')
    # Total Number of Patients on RRT - n (%)
    row = 'Total Number of Patients on RRT - n (%)'
    s = build_str(rrt_flag, lbls, 'count', '%.1f')
    row += s
    table_f.write(row + '\n')
    # stat_f.write(row + '\n')

    # Total days of CRRT - median[IQ1 - IQ3]
    row = 'Total days of CRRT - median[IQ1 - IQ3]'
    s = build_str(crrt_days[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Total days of CRRT'
    s = build_str(crrt_days[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # Total days of HD - median[IQ1 - IQ3]
    row = 'Total days of HD - median[IQ1 - IQ3]'
    s = build_str(hd_days[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Total days of HD'
    s = build_str(hd_days[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # Total number of HD treatments - median[IQ1 - IQ3]
    row = 'Total number of HD treatments - median[IQ1 - IQ3]'
    s = build_str(hd_trtmts[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Total number of HD treatments'
    s = build_str(hd_trtmts[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # Total days of CRRT + HD - median[IQ1 - IQ3]
    row = 'Total days of CRRT + HD - median[IQ1 - IQ3]'
    s = build_str(rrt_tot[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'Total days of CRRT + HD'
    s = build_str(rrt_tot[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # CRRT free days - 7d - median[IQ1 - IQ3]
    row = 'CRRT Free Days - 14d'
    s = build_str(crrt_free_7d[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'CRRT Free Days - 14d'
    s = build_str(crrt_free_7d[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # CRRT free days - 28d - median[IQ1 - IQ3]
    row = 'CRRT Free Days - 28d - median[IQ1 - IQ3]'
    s = build_str(crrt_free_28d[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'CRRT Free Days - 28d'
    s = build_str(crrt_free_28d[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # hd free days - 7d - median[IQ1 - IQ3]
    row = 'HD Free Days - 7d - median[IQ1 - IQ3]'
    s = build_str(hd_free_7d[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'HD Free Days - 7d'
    s = build_str(hd_free_7d[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    # hd free days - 28d - median[IQ1 - IQ3]
    row = 'HD Free Days - 28d - median[IQ1 - IQ3]'
    s = build_str(hd_free_28d[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'median', '%.1f')
    row += s
    table_f.write(row + '\n')

    row = 'HD Free Days - 28d'
    s = build_str(hd_free_28d[np.where(rrt_tot > 0)], lbls[np.where(rrt_tot > 0)], 'all', '%.1f')
    row += s
    stat_f.write(row + '\n')

    np.seterr(invalid='warn')

    table_f.close()
    stat_f.close()

    return


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
             ['Gros Fluid', 'gros_fluid'],
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

    direction = 0
    numPeaks = 0
    for i in range(1, len(kdigo)):
        if kdigo[i] > kdigo[i - 1]:
            if direction < 1:
                numPeaks += 1
            direction = 1
        elif kdigo[i] < kdigo[i - 1]:
            direction = -1
    if direction == 1:
        numPeaks -= 1
    return count, numPeaks


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
#     mising = np.zeros(6)
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
#             mising[0] += 1
#
#         try:
#             s2_gcs = float(str(clinical_oth[co_rows, g_c_s][0]).split('-')[0])
#         except:
#             s2_gcs = np.nan
#         if np.isnan(s2_gcs):
#             mising[1] += 1
#
#         s3_map = np.nan
#         if np.size(cv_rows) > 0:
#             s3_map = float(clinical_vit[cv_rows, m_a_p])
#             if np.isnan(s3_map):
#                 s3_map = float(clinical_vit[cv_rows, cuff])
#         if np.isnan(s3_map):
#             mising[2] += 1
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
#             mising[3] += 1
#         if np.isnan(s5_plt):
#             mising[4] += 1
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
#             mising[5] += 1
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
#     return sofas, mising

def get_sofa(ids, stats, scr_interp, days_interp, out_name, v=False):

    # For medications (if "dopa" in...)
    # Use individual medications
    # Dopamine/Dobutamine/Milrinone = 2
    # Epinephrine/Norepinephrine/Phenylephrine/Vasopressin = 3

    all_ids = stats['ids'][:]
    pt_sel = np.array([x in ids for x in all_ids])

    po2 = stats['po2'][:][pt_sel]
    if po2.ndim == 3:
        po2 = np.nanmax(po2[:, :2, 1], axis=1)
    elif po2.ndim == 2:
        po2 = po2[:, 1]

    fio2 = stats['fio2'][:][pt_sel]
    if fio2.ndim == 3:
        fio2 = np.nanmax(fio2[:, :2, 1], axis=1)
    elif fio2.ndim == 2:
        fio2 = fio2[:, 1]

    maps = stats['map'][:][pt_sel]
    if maps.ndim == 3:
        maps = np.nanmax(maps[:, :2, 0], axis=1)
    elif maps.ndim == 2:
        maps = maps[:, 0]

    gcs = stats['glasgow'][:][pt_sel]
    if gcs.ndim == 3:
        gcs = np.nanmax(gcs[:, :2, 0], axis=1)
    elif gcs.ndim == 2:
        gcs = gcs[:, 0]

    bili = stats['bilirubin'][:][pt_sel]
    if bili.ndim == 3:
        bili = np.nanmax(bili[:, :2, 1], axis=1)
    elif bili.ndim == 2:
        bili = bili[:, 1]

    pltlts = stats['platelets'][:][pt_sel]
    if pltlts.ndim == 3:
        pltlts = np.nanmax(pltlts[:, :2, 1], axis=1)
    elif pltlts.ndim == 2:
        pltlts = pltlts[:, 1]

    mv_flag = stats['mv_flag'][:][pt_sel]

    if 'dopa' in list(stats):
        dopas = stats['dopa'][:][pt_sel]
        if dopas.ndim == 3:
            dopas = np.nanmax(dopas[:, :2, 1], axis=1)
        elif dopas.ndim == 2:
            dopas = dopas[:, 1]

        epis = stats['epinephrine'][:][pt_sel]
        if epis.ndim == 3:
            epis = np.nanmax(epis[:, :2, 1], axis=1)
        elif epis.ndim == 2:
            epis = epis[:, 1]
    else:
        dopas = epis = np.array(stats['vasopress_ct'][:] > 0)

    fio2_mv = np.nanmedian(fio2[np.where(mv_flag)])
    fio2_nomv = np.nanmedian(fio2[np.where(mv_flag == 0)])
    po2_mv = np.nanmedian(po2[np.where(mv_flag)])
    po2_nomv = np.nanmedian(po2[np.where(mv_flag == 0)])
    gcs_mv = np.nanmedian(gcs[np.where(mv_flag)])
    gcs_nomv = np.nanmedian(gcs[np.where(mv_flag == 0)])

    sofa_hdr = 'STUDY_PATIENT_ID,SOFA_1_MV-PaO2/FiO2,SOFA_2_GCS,SOFA_3_MAP,SOFA_4_Bili,SOFA_5_Platelets,SOFA_6_SCr'
    raw_hdr = 'STUDY_PATIENT_ID,PaO2,FiO2,MechVent,Glasgow,MAP,Dopa,Epi,Bilirubin,Platelets,Creatinine'

    out_scores = open(out_name, 'w')
    out_scores.write(sofa_hdr + '\n')

    sofas = np.zeros((len(ids), 6))
    sofas_raw = np.full((len(ids), 10), np.nan)
    for i in trange(len(ids), desc='Calculating SOFA Scores'):
        idx = ids[i]
        out_scores.write('%d' % idx)

        mv = mv_flag[i]

        s1_pa = po2[i]
        if np.isnan(s1_pa):
            if mv:
                s1_pa = po2_mv
            else:
                s1_pa = po2_nomv

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
        try:
            if day_sel.size > 0:
                s6_scr = np.max(scr_interp[i][day_sel])
            else:
                day_sel = np.where(days_interp[i] == 1)[0]
                if day_sel.size > 0:
                    s6_scr = np.max(scr_interp[i][day_sel])
        except TypeError:
            pass

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
        else:
            s2 = 15

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

        out_scores.write(',%d,%d,%d,%d,%d,%d\n' % (score[0], score[1], score[2], score[3], score[4], score[5]))
        sofas[i, :] = score
        sofas_raw[i, :] = np.array([s1_pa, s1_fi, mv, s2, s3, dopa, epi, s4, s5, s6])
        if v:
            print(np.sum(score))
    return sofas, sofa_hdr, sofas_raw, raw_hdr


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
        pres = stats['vasopres_ct'][i]
        if pres:
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

    apache_hdr = 'STUDY_PATIENT_ID,APACHE_1_Temp,APACHE_2_MAP,APACHE_3_HR,APACHE_4_Resp,' \
                 'APACHE_5_FiO2/aaDO2,APACHE_6_pH,APACHE_7_Na,APACHE_8_pK,APACHE_9_Scr,' \
                 'APACHE_10_Hemat,APACHE_11_WBC,APACHE_12_GCS,APACHE_13_Age'

    raw_hdr = 'STUDY_PATIENT_ID,Temp_Low,Temp_High,MAP_Low,MAP_High,HR_Low,HR_High,Resp_Low,Resp_High,' \
              'FiO2_High,pO2_High,pCO2_High,pH_Low,pH_High,Na_Low,Na_High,pK_Low,pK_High,Hemat_Low,Hemat_High,' \
              'WBC_Low,WBC_High,SCr_High,GCS_Low,Age'


    temps = stats['temperature'][:][pt_sel]
    if temps.ndim == 3:
        temps = np.transpose((np.nanmin(temps[:, :2, 0], axis=1), np.nanmax(temps[:, :2, 1], axis=1)))
    maps = stats['map'][:][pt_sel]
    if maps.ndim == 3:
        maps = np.transpose((np.nanmin(maps[:, :2, 0], axis=1), np.nanmax(maps[:, :2, 1], axis=1)))
    try:
        hrs = stats['heart_rate'][:][pt_sel]
    except KeyError:
        hrs = stats['hr'][:][pt_sel]
    if hrs.ndim == 3:
        hrs = np.transpose((np.nanmin(hrs[:, :2, 0], axis=1), np.nanmax(hrs[:, :2, 1], axis=1)))
    resps = stats['respiration'][:][pt_sel]
    if resps.ndim == 3:
        resps = np.transpose((np.nanmin(resps[:, :2, 0], axis=1), np.nanmax(resps[:, :2, 1], axis=1)))
    fio2s = stats['fio2'][:][pt_sel]
    if fio2s.ndim == 3:
        fio2s = np.transpose((np.nanmin(fio2s[:, :2, 0], axis=1), np.nanmax(fio2s[:, :2, 1], axis=1)))
    gcs = stats['glasgow'][:][pt_sel]
    if gcs.ndim == 3:
        gcs = np.transpose((np.nanmin(gcs[:, :2, 0], axis=1), np.nanmax(gcs[:, :2, 1], axis=1)))
    po2s = stats['po2'][:][pt_sel]
    if po2s.ndim == 3:
        po2s = np.transpose((np.nanmin(po2s[:, :2, 0], axis=1), np.nanmax(po2s[:, :2, 1], axis=1)))
    pco2s = stats['pco2'][:][pt_sel]
    if pco2s.ndim == 3:
        pco2s = np.transpose((np.nanmin(pco2s[:, :2, 0], axis=1), np.nanmax(pco2s[:, :2, 1], axis=1)))
    phs = stats['ph'][:][pt_sel]
    if phs.ndim == 3:
        phs = np.transpose((np.nanmin(phs[:, :2, 0], axis=1), np.nanmax(phs[:, :2, 1], axis=1)))
    nas = stats['sodium'][:][pt_sel]
    if nas.ndim == 3:
        nas = np.transpose((np.nanmin(nas[:, :2, 0], axis=1), np.nanmax(nas[:, :2, 1], axis=1)))
    pks = stats['potassium'][:][pt_sel]
    if pks.ndim == 3:
        pks = np.transpose((np.nanmin(pks[:, :2, 0], axis=1), np.nanmax(pks[:, :2, 1], axis=1)))
    hemats = stats['hematocrit'][:][pt_sel]
    if hemats.ndim == 3:
        hemats = np.transpose((np.nanmin(hemats[:, :2, 0], axis=1), np.nanmax(hemats[:, :2, 1], axis=1)))
    wbcs = stats['wbc'][:][pt_sel]
    if wbcs.ndim == 3:
        wbcs = np.transpose((np.nanmin(wbcs[:, :2, 0], axis=1), np.nanmax(wbcs[:, :2, 1], axis=1)))
    ages = stats['age'][:][pt_sel]

    out = open(out_name, 'w')
    out.write(apache_hdr + '\n')
    apaches = np.zeros((len(ids), 13))
    apaches_raw = np.full((len(ids), 24), np.nan)
    for i in trange(len(ids), desc='Calculating APACHE-II Scores'):
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

        s5_f = fio2s[i, 1]
        s5_po = po2s[i, 1]
        s5_pco = pco2s[i, 1]

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
        s9 = np.nan
        day_sel = np.where(days_interp[i] == 0)[0]
        try:
            if day_sel.size > 0:
                s9 = np.max(scr_interp[i][day_sel])
            else:
                day_sel = np.where(days_interp[i] == 1)[0]
                if day_sel.size > 0:
                    s9 = np.max(scr_interp[i][day_sel])
        except TypeError:
            pass

        s12_gcs = gcs[i, 0]

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
        elif s8_low < 3.5 or s8_high >= 5.5:
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

        for j in range(len(score)):
            out.write(',%d' % (score[j]))
        out.write('\n')
        apaches[i, :] = score
        apaches_raw[i, :] = np.array([s1_low, s1_high, s2_low, s2_high, s3_low, s3_high, s4_low, s4_high, s5_f, s5_po, s5_pco,
                             s6_low, s6_high, s7_low, s7_high, s8_low, s8_high, s9, s10_low, s10_high, s11_low,
                             s11_high, s12_gcs, s13_age])
        if v:
            print(np.sum(score))
    return apaches, apache_hdr, apaches_raw, raw_hdr


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

        s8_low = float(stats['potasium'][i][1, 0])
        s8_high = float(stats['potasium'][i][1, 1])

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
    return apaches


# Update dialysis so that it does not exclude patients with RRT prior to discharge
# Try 90 from admision vs. 90 from discharge
def get_MAKE90(ids, stats, datapath, out, ref='disch', min_day=7, buffer=0, ct=1, sort_id='STUDY_PATIENT_ID', label='make90', ovwt=False):

    rrt_m = pd.read_csv(os.path.join(datapath, 'RENAL_REPLACE_THERAPY.csv'))
    rrt_m.sort_values(by=sort_id, inplace=True)
    crrt_locs = [rrt_m.columns.get_loc('CRRT_START_DATE'), rrt_m.columns.get_loc('CRRT_STOP_DATE')]
    hd_locs = [rrt_m.columns.get_loc('HD_START_DATE'), rrt_m.columns.get_loc('HD_STOP_DATE')]
    rrt_m = rrt_m.values

    scr_all_m = pd.read_csv(os.path.join(datapath, 'SCR_ALL_VALUES.csv'))
    scr_all_m.sort_values(by=[sort_id, 'SCR_ENTERED'], inplace=True)
    scr_date_loc = scr_all_m.columns.get_loc('SCR_ENTERED')
    scr_val_loc = scr_all_m.columns.get_loc('SCR_VALUE')
    scr_all_m = scr_all_m.values

    usrds_esrd_m = pd.read_csv(os.path.join(datapath, 'USRDS_ESRD.csv'))
    usrds_esrd_m.sort_values(by=sort_id, inplace=True)
    usrds_esrd_date_loc = usrds_esrd_m.columns.get_loc('ESRD_DATE')
    usrds_esrd_m = usrds_esrd_m.values

    esrd_man_rev = pd.read_csv(os.path.join(datapath, 'ESRD_MANUAL_REVISION.csv'))
    esrd_man_rev.sort_values(by=sort_id, inplace=True)
    man_rev_dur = esrd_man_rev.columns.get_loc('during')
    man_rev_rrt = esrd_man_rev.columns.get_loc('rrt_dependent')
    esrd_man_rev = esrd_man_rev.values

    scm_esrd_m = pd.read_csv(os.path.join(datapath, 'ESRD_STATUS.csv'))
    scm_esrd_m.sort_values(by=sort_id, inplace=True)
    scm_esrd_during = scm_esrd_m.columns.get_loc("DURING_INDEXED_INDICATOR")
    scm_esrd_m = scm_esrd_m.values

    sexes = stats['gender'][:]
    ages = stats['age'][:]
    races = stats['race'][:]
    bsln_gfrs = stats['baseline_gfr'][:]
    all_ids = stats['ids'][:]
    windows = stats['icu_dates'][:].astype('unicode')
    dods = stats['dod'][:].astype('unicode')

    # print('id,died,gfr_drop,new_dialysis')
    out.write('id,died_inp,died90,died_in_window,esrd_manual_revision,esrd_scm,esrd_usrds_90,esrd_usrds_window,new_dialysis_manual_revision,new_dialysis,gfr_drop_25,gfr_drop_30,gfr_drop_50,n_vals,delta\n')
    scores = []
    out_ids = []
    for i in range(len(ids)):
        tid = ids[i]

        idx = np.where(all_ids == tid)[0][0]
        if stats['died_inp'][idx] > 0:
            died_inp = 1
            if ref == 'disch':
                died_buf = 1
            else:
                died_buf = 0
        else:
            died_inp = 0
        sex = sexes[i]
        race = races[i]
        age = ages[i]

        died90 = 0
        diedbuf = 0
        esrd = 0
        esrd_man = 0
        esrd_scm = 0
        esrd_usrds90 = 0
        esrd_usrdsbuf = 0
        gfr_drop_25 = 0
        gfr_drop_30 = 0
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
                died90 = 1
            if dod - tmin < datetime.timedelta(90 + buffer):
                diedbuf = 1

        # ESRD
        # Check manual revision first
        if not died90:
            man_rev_loc = np.where(esrd_man_rev[:, 0]) == tid
            if man_rev_loc.size > 0:
                if esrd_man_rev[man_rev_loc[0], man_rev_dur] == 1:
                    esrd_man = 1
                if esrd_man_rev[man_rev_loc[0], man_rev_rrt] == 1:
                    dia_dep_man = 1
            # If patient not included in manual revision, check SCM and USRDS separately
            # else:
                # SCM
            esrd_dloc = np.where(scm_esrd_m[:, 0] == tid)[0]
            if esrd_dloc.size > 0:
                if scm_esrd_m[esrd_dloc[0], scm_esrd_during] == 'Y':
                    esrd_scm = 1
                # if scm_esrd[esrd_dloc[0], scm_after] == 'Y':
                #     esrd_scm = 1
            # USRDS
            esrd_dloc = np.where(usrds_esrd_m[:, 0] == tid)[0]
            if esrd_dloc.size > 0:
                tdate = get_date(usrds_esrd_m[esrd_dloc[0], usrds_esrd_date_loc])
                if tdate != 'nan':
                    if datetime.timedelta(0) < tdate - tmin < datetime.timedelta(90):
                        esrd_usrds90 = 1
                    if datetime.timedelta(0) < tdate - tmin < datetime.timedelta(90 + buffer):
                        esrd_usrdsbuf = 1

            dia_locs = np.where(rrt_m[:, 0] == tid)[0]
            for j in range(len(dia_locs)):
                crrt_start = get_date(rrt_m[dia_locs[j], crrt_locs[0]])
                hd_start = get_date(rrt_m[dia_locs[j], hd_locs[0]])
                if str(crrt_start) != 'nan':
                    tstart = get_date(rrt_m[dia_locs[j], crrt_locs[0]])
                    tstop = get_date(rrt_m[dia_locs[j], crrt_locs[1]])
                elif str(hd_start) != 'nan':
                    tstart = get_date(rrt_m[dia_locs[j], hd_locs[0]])
                    tstop = get_date(rrt_m[dia_locs[j], hd_locs[1]])

                if tstart <= disch:
                    if tstop >= disch - datetime.timedelta(2):
                        dia_dep = 1

            nvals = 0
            delta_str = 'not_evaluated'
            # Only evaluates drop in eGFR if patient survived and is not dialysis dependent
            if not dia_dep and not esrd:
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
                        tdiffs = deltas[o[:tct]]
                        while max(tdiffs) - min(tdiffs) > 30 and tct > 1:
                            tct -= 1
                            gfr90 = np.mean(gfrs[o[:tct]])
                            tdiffs = deltas[o[:tct]]
                        delta = deltas[o[0]]
                        delta_str = str(delta)
                        nvals = tct

                thresh = 100 - 25
                rel_pct = (gfr90 / bsln_gfr) * 100
                if rel_pct < thresh:
                    gfr_drop_25 = 1
                thresh = 100 - 30
                rel_pct = (gfr90 / bsln_gfr) * 100
                if rel_pct < thresh:
                    gfr_drop_30 = 1
                thresh = 100 - 50
                rel_pct = (gfr90 / bsln_gfr) * 100
                if rel_pct < thresh:
                    gfr_drop_50 = 1

        out_ids.append(tid)
        out.write('%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s\n' % (tid, died_inp, died90, diedbuf, esrd_man, esrd_scm, esrd_usrds90, esrd_usrdsbuf, dia_dep_man, dia_dep, gfr_drop_25, gfr_drop_30, gfr_drop_50, nvals, delta_str))
        scores.append(np.array((died_inp, died90, diedbuf, esrd_man, esrd_scm, esrd_usrds90, esrd_usrdsbuf, dia_dep_man, dia_dep, gfr_drop_25, gfr_drop_30, gfr_drop_50)))
        # print('%d,%d,%d,%d' % (idx, died, gfr_drop, dia_dep))
    scores = np.array(scores)
    d25 = np.max(scores[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], axis=1)
    d30 = np.max(scores[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]], axis=1)
    d50 = np.max(scores[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11]], axis=1)
    try:
        stats[label + '_d25'] = d25
        stats[label + '_d30'] = d30
        stats[label + '_d50'] = d50
    except RuntimeError:
        stats[label + '_d25'][:] = d25
        stats[label + '_d30'][:] = d30
        stats[label + '_d50'][:] = d50

    return scores, out_ids


# %%
def get_MAKE90_dallas(ids, stats, dataPath,
                      out, ref='disch', min_day=7, buffer=0, ct=1, label='make90', ovwt=False):

    sort_id = 'PATIENT_NUM'

    esrd_m = pd.read_csv(os.path.join(dataPath, 'tESRDSummary.csv'))
    esrd_m.sort_values(by=sort_id, inplace=True)
    esrd_date_loc = esrd_m.columns.get_loc("EFFECTIVE_TIME")
    esrd_m = esrd_m.values

    scr_all_m = pd.read_csv(os.path.join(dataPath, 'all_scr_data.csv'))
    scr_all_m.sort_values(by=[sort_id, 'SPECIMN_TEN_TIME'], inplace=True)
    scr_date_loc = scr_all_m.columns.get_loc('SPECIMN_TEN_TIME')
    scr_val_loc = scr_all_m.columns.get_loc('ORD_VALUE')
    scr_all_m = scr_all_m.values

    rrt_m = pd.read_csv(os.path.join(dataPath, 'tDialysis.csv'))
    rrt_m.sort_values(by=sort_id, inplace=True)
    rrt_start_loc = rrt_m.columns.get_loc('START_DATE')
    rrt_stop_loc = rrt_m.columns.get_loc('END_DATE')
    rrt_m = rrt_m.values

    usrds_m = pd.read_csv(os.path.join(dataPath, 'tUSRDS_CORE_Patients.csv'))
    usrds_m.sort_values(by=sort_id, inplace=True)
    usrds_esrd_loc = usrds_m.columns.get_loc('FIRST_SE')
    usrds_m = usrds_m.values

    sexes = stats['gender'][:]
    ages = stats['age'][:]
    races = stats['race'][:]
    bsln_gfrs = stats['baseline_gfr'][:]
    all_ids = stats['ids'][:]
    windows = stats['icu_dates'][:].astype('unicode')
    dods = stats['dod'][:].astype('unicode')

    # print('id,died,gfr_drop,new_dialysis')
    out.write(
        'id,died_inp,died_in_window,died_in_window_PlusBuffer,scm_esrd,usrds_esrd,new_dialysis,gfr_drop_25,gfr_drop_50,n_vals,delta\n')
    scores = []
    out_ids = []
    for i in trange(len(ids), desc="Determining MAKE-90 - Reference: %s, Buffer: %ddays, # Points: %d" % (ref, buffer, ct)):
        tid = ids[i]

        idx = np.where(all_ids == tid)[0][0]
        if stats['died_inp'][idx] > 0:
            died_inp = 1
            if ref == 'disch':
                died_win = 1
        else:
            died_inp = 0
        sex = sexes[i]
        race = races[i]
        age = ages[i]

        died_win = 0
        died_buf = 0
        esrd_scm = 0
        esrd_usrds = 0
        gfr_drop_25 = 0
        gfr_drop_30 = 0
        gfr_drop_50 = 0
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
                died_win = 1
                died_buf = 1
            elif dod - tmin < datetime.timedelta(90) + datetime.timedelta(buffer):
                died_buf = 1

        if not died_buf:
            # ESRD
            # Check manual revision first
            esrd_loc = np.where(esrd_m[:, 0] == tid)[0]
            if esrd_loc.size > 0:
                for tloc in esrd_loc:
                    tdate = get_date(esrd_m[tloc, esrd_date_loc])
                    if tdate != 'nan':
                        if datetime.timedelta(0) < tdate - tmin < datetime.timedelta(90) + datetime.timedelta(buffer):
                            esrd_scm = 1

            esrd_dloc = np.where(usrds_m[:, 0] == tid)[0]
            if esrd_dloc.size > 0:
                tdate = get_date(usrds_m[esrd_dloc[0], usrds_esrd_loc])
                if tdate != 'nan':
                    if datetime.timedelta(0) < tdate - tmin < datetime.timedelta(90) + datetime.timedelta(buffer):
                        esrd_usrds = 1

            dia_locs = np.where(rrt_m[:, 0] == tid)[0]
            for j in range(len(dia_locs)):
                start = get_date(rrt_m[dia_locs[j], rrt_start_loc])
                if str(start) != 'nan':
                    tstart = start
                    tstop = get_date(rrt_m[dia_locs[j], rrt_stop_loc])

                # if tstart <= admit:
                #     if tstop >= disch - datetime.timedelta(2):
                #         dia_dep = 1
                if tstart <= disch:
                    if tstop >= disch - datetime.timedelta(2):
                        dia_dep = 1

            nvals = 0
            delta_str = 'not_evaluated'
            if not died_buf and not dia_dep and not esrd_scm and not esrd_usrds:
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
                        tdiffs = deltas[o[:tct]]
                        while max(tdiffs) - min(tdiffs) > 30 and tct > 1:
                            tct -= 1
                            gfr90 = np.mean(gfrs[o[:tct]])
                            tdiffs = deltas[o[:tct]]
                        delta = deltas[o[0]]
                        delta_str = str(delta)
                        nvals = tct

                thresh = 100 - 25
                rel_pct = (gfr90 / bsln_gfr) * 100
                if rel_pct < thresh:
                    gfr_drop_25 = 1
                thresh = 100 - 30
                rel_pct = (gfr90 / bsln_gfr) * 100
                if rel_pct < thresh:
                    gfr_drop_30 = 1
                thresh = 100 - 50
                rel_pct = (gfr90 / bsln_gfr) * 100
                if rel_pct < thresh:
                    gfr_drop_50 = 1

        out_ids.append(tid)
        out.write('%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s\n' % (tid, died_inp, died_win, died_buf, esrd_scm, esrd_usrds, dia_dep,
                                                          gfr_drop_25, gfr_drop_30, gfr_drop_50, nvals, delta_str))
        scores.append(
            np.array((died_inp, died_win, died_buf, esrd_scm, esrd_usrds, dia_dep, gfr_drop_25, gfr_drop_30, gfr_drop_50)))
        # print('%d,%d,%d,%d' % (idx, died, gfr_drop, dia_dep))
    scores = np.array(scores)
    d25 = np.max(scores[:, [0, 1, 2, 3, 4, 5, 6]], axis=1)
    d30 = np.max(scores[:, [0, 1, 2, 3, 4, 5, 7]], axis=1)
    d50 = np.max(scores[:, [0, 1, 2, 3, 4, 5, 8]], axis=1)
    try:
        stats[label + '_d25'] = d25
        stats[label + '_d30'] = d30
        stats[label + '_d50'] = d50
    except RuntimeError:
        if ovwt:
            stats[label + '_d25'][:] = d25
            stats[label + '_d30'][:] = d30
            stats[label + '_d50'][:] = d50
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
            pos_idx = resample(pos_idx, replace=True, n_samples=n_neg, random_state=123)
    sel_idx = np.sort(np.concatenate((pos_idx, neg_idx)))
    return sel_idx


# def eval_reclassification(old_probs, new_probs, ground_truth):
#     '''
#     Categorical NRI: p < 1, 1 ≤ p < 5, 5 ≤ p < 10, 10 ≤ p < 20, p ≥ 20
#     :param old_probs:
#     :param new_probs:
#     :param ground_truth:
#     :return:
#     '''
#     # IDI
#     events = np.where(ground_truth)[0]
#     non_events = np.where(ground_truth == 0)[0]
#     IDI = (np.mean(new_probs[events]) - np.mean(old_probs[events])) - \
#           (np.mean(new_probs[non_events]) - np.mean(old_probs[non_events]))
#     # c-statistic
#     old_auc = roc_auc_score(ground_truth, old_probs)
#     new_auc = roc_auc_score(ground_truth, new_probs)
#     # NRI
#     up = np.where(new_probs > old_probs)[0]
#     down = np.where(new_probs < old_probs)[0]
#     # Continuous
#     cNRI_e = (float(len(np.intersect1d(events, up))) / len(events)) - \
#             (float(len(np.intersect1d(events, down))) / len(events))
#     cNRI_ne = (float(len(np.intersect1d(non_events, down))) / len(non_events)) - \
#              (float(len(np.intersect1d(non_events, up))) / len(non_events))
#     cNRI = cNRI_e + cNRI_ne
#     # Categorical
#     return cNRI_e, cNRI_ne, cNRI, IDI, (new_auc - old_auc)


def eval_reclassification(basePath):
    lbls = np.loadtxt(os.path.join(basePath, 'labels.txt'))[:, None]
    probs1 = np.loadtxt(os.path.join(basePath, 'model1.txt'))[:, None]
    probs2 = np.loadtxt(os.path.join(basePath, 'model2.txt'))[:, None]
    h = importr('Hmisc')
    pa = importr('PredictABEL')
    base = importr('base')

    probs1 = n2r.py2ro(probs1)
    probs2 = n2r.py2ro(probs2)
    data = n2r.py2ro(np.hstack((lbls, lbls, lbls)))
    idx = base.integer(1)
    idx[0] = 0
    cutoffs = n2r.py2ro(np.array([0, 0.05, 0.1, 0.2, 1.0]))
    pa.reclassification(data, idx, probs1, probs2, cutoffs)


def eval_classification(probs, ground_truth, thresh=0.5):
    preds = np.array(probs >= thresh, dtype=int)
    tn, fp, fn, tp = confusion_matrix(ground_truth, preds).ravel()
    precision, recall, f1, support = precision_recall_fscore_support(ground_truth, preds, average='micro')
    sensitivity = tp / np.sum(ground_truth)
    specificity = tn / (len(ground_truth) - np.sum(ground_truth))
    auc = roc_auc_score(ground_truth, probs)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    return auc, sensitivity, specificity, ppv, npv, precision, recall, f1, support
