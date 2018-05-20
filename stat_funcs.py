#!/usr/bin/env python2
# -*- coding: utf-8 -*-`
"""
Created on Wed Nov 29 13:15:39 2017

@author: taylorsmith
"""
from __future__ import division
from kdigo_funcs import load_csv, get_mat, calc_gfr
import datetime
import numpy as np
import re
import matplotlib.pyplot as plt
import h5py
import pandas as pd


def summarize_stats(data_path, out_name, grp_name='meta'):
    all_ids = np.loadtxt(data_path + 'icu_valid_ids.csv', dtype=int)

    # load KDIGO and discharge dispositions
    k_ids, kdigos = load_csv(data_path + 'kdigo.csv', all_ids, dt=int)

    # load genders
    dem_m = get_mat('../DATA/KDIGO_full.xlsx', 'DEMOGRAPHICS_INDX', 'STUDY_PATIENT_ID')
    sex_loc = dem_m.columns.get_loc("GENDER")
    eth_loc = dem_m.columns.get_loc('RACE')
    dem_m = dem_m.as_matrix()

    # load dates of birth
    dob_m = get_mat('../DATA/KDIGO_full.xlsx', 'DOB', 'STUDY_PATIENT_ID')
    dob_loc = dob_m.columns.get_loc("DOB")
    dob_m = dob_m.as_matrix()

    # load diagnosis info
    diag_m = get_mat('../DATA/KDIGO_full.xlsx', 'DIAGNOSIS', 'STUDY_PATIENT_ID')
    diag_loc = diag_m.columns.get_loc("DIAGNOSIS_DESC")
    diag_nb_loc = diag_m.columns.get_loc("DIAGNOSIS_SEQ_NB")
    diag_m = diag_m.as_matrix()

    # load fluid input/output info
    io_m = get_mat('../DATA/KDIGO_full.xlsx', 'IO_TOTALS', 'STUDY_PATIENT_ID')
    io_m = io_m.as_matrix()

    # load dates of birth
    charl_m = get_mat('../DATA/KDIGO_full.xlsx', 'CHARLSON_SCORE', 'STUDY_PATIENT_ID')
    charl_loc = charl_m.columns.get_loc("CHARLSON_INDEX")
    charl_m = charl_m.as_matrix()

    # load dates of birth
    elix_m = get_mat('../DATA/KDIGO_full.xlsx', 'ELIXHAUSER_SCORE', 'STUDY_PATIENT_ID')
    elix_loc = elix_m.columns.get_loc("ELIXHAUSER_INDEX")
    elix_m = elix_m.as_matrix()

    # load mechanical ventilation
    mech_m = get_mat('../DATA/KDIGO_full.xlsx', 'ORGANSUPP_VENT', 'STUDY_PATIENT_ID')
    mech_loc = mech_m.columns.get_loc("TOTAL_DAYS")
    mech_m = mech_m.as_matrix()

    # load outcome data
    date_m = get_mat('../DATA/KDIGO_full.xlsx', 'ADMISSION_INDX', 'STUDY_PATIENT_ID')
    hosp_locs = [date_m.columns.get_loc("HOSP_ADMIT_DATE"), date_m.columns.get_loc("HOSP_DISCHARGE_DATE")]
    icu_locs = [date_m.columns.get_loc("ICU_ADMIT_DATE"), date_m.columns.get_loc("ICU_DISCHARGE_DATE")]
    date_m = date_m.as_matrix()
    date_ids = date_m[:, 0]
    date_all_ids = []
    for i in range(len(date_ids)):
        date_all_ids.append(date_ids[i])
    all_apache = np.loadtxt('../DATA/icu/7days_final/apache.csv', delimiter=',', dtype=int)
    all_sofa = np.loadtxt('../DATA/icu/7days_final/sofa.csv', delimiter=',', dtype=int)

    f = open(data_path + 'disch_disp.csv', 'r')
    dd = []
    ddids = []
    for line in f:
        l = line.rstrip().split(',')
        ddids.append(int(l[0]))
        dd.append(l[1])

    f.close()
    dd = np.array(dd)

    n_eps = []
    for i in range(len(kdigos)):
        n_eps.append(count_eps(kdigos[i]))

    ages = []
    genders = []
    mks = []
    dieds = []
    nepss = []
    hosp_frees = []
    icu_frees = []
    sofas = []
    apaches = []
    sepsiss = []
    net_fluids = []
    gross_fluids = []
    c_scores = []
    e_scores = []
    mv_frees = []
    eths = []
    for i in range(len(all_ids)):
        idx = all_ids[i]
        mk = np.max(kdigos[i])
        died = 0
        if 'EXPIRED' in dd[i]:
            died += 1
            if 'LESS' in dd[i]:
                died += 1
        hlos, ilos = get_los(idx, date_m, hosp_locs, icu_locs)
        hfree = 28 - hlos
        ifree = 28 - ilos
        if hfree < 0:
            hfree = 0
        if ifree < 0:
            ifree = 0

        mech_idx = np.where(mech_m[:, 0] == idx)[0]
        if mech_idx.size > 0:
            mech = 28 - int(mech_m[mech_idx, mech_loc])
        else:
            mech = 28
        if mech < 0:
            mech = 0

        if died:
            if mech < 28:
                mech = 0
            if hfree < 28:
                hfree = 0
            if ifree < 28:
                ifree = 0

        eps = n_eps[i]

        sepsis = 0
        diag_ids = np.where(diag_m[:, 0] == idx)[0]
        for j in range(len(diag_ids)):
            tid = diag_ids[j]
            try:
                if 'sep' in str(diag_m[tid, diag_loc]).lower():
                    if int(diag_m[tid, diag_nb_loc]) == 1:
                        sepsis = 1
                        break
            except:
                sepsis = 0

        male = 0
        dem_idx = np.where(dem_m[:, 0] == idx)[0][0]
        try:
            if dem_m[dem_idx, sex_loc].upper()[0] == 'M':
                male = 1
        except:
            male = -1

        dob_idx = np.where(dob_m[:, 0] == idx)[0]
        dob = dob_m[dob_idx, dob_loc]
        date_idx = np.where(date_m[:, 0] == idx)[0][0]
        admit = date_m[date_idx, hosp_locs[0]]
        tage = admit - dob
        age = tage[0].total_seconds() / (60 * 60 * 24 * 365)

        race = dem_m[dem_idx, eth_loc]
        if race == "BLACK/AFR AMERI":
            eth = 1
        else:
            eth = 0

        sofa_idx = np.where(all_sofa[:, 0] == idx)[0]
        sofa = np.sum(all_sofa[sofa_idx, 1:])

        apache_idx = np.where(all_apache[:, 0] == idx)[0]
        apache = np.sum(all_apache[apache_idx, 1:])

        io_idx = np.where(io_m[:, 0] == idx)[0]
        if io_idx.size > 0:
            try:
                net = np.nansum((io_m[io_idx, 1], io_m[io_idx, 3], io_m[io_idx, 5])) - \
                      np.nansum((io_m[io_idx, 2], io_m[io_idx, 4], io_m[io_idx, 6]))
                tot = np.nansum((io_m[io_idx, 1], io_m[io_idx, 3], io_m[io_idx, 5])) + \
                      np.nansum((io_m[io_idx, 2], io_m[io_idx, 4], io_m[io_idx, 6]))
            except:
                net = np.nan
                tot = np.nan
        else:
            net = np.nan
            tot = np.nan

        charl_idx = np.where(charl_m[:, 0] == idx)[0]
        if charl_idx.size > 0:
            charl = charl_m[charl_idx, charl_loc]
        else:
            charl = np.nan

        elix_idx = np.where(elix_m[:, 0] == idx)[0]
        if elix_idx.size > 0:
            elix = elix_m[elix_idx, elix_loc]
        else:
            elix = np.nan

        ages.append(age)
        genders.append(male)
        mks.append(mk)
        dieds.append(died)
        nepss.append(eps)
        hosp_frees.append(hfree)
        icu_frees.append(ifree)
        sofas.append(sofa)
        apaches.append(apache)
        sepsiss.append(sepsis)
        net_fluids.append(net)
        gross_fluids.append(tot)
        c_scores.append(charl)
        e_scores.append(elix)
        mv_frees.append(mech)
        eths.append(eth)
    ages = np.array(ages, dtype=float)
    genders = np.array(genders, dtype=bool)
    mks = np.array(mks, dtype=int)
    dieds = np.array(dieds, dtype=int)
    nepss = np.array(nepss, dtype=int)
    hosp_frees = np.array(hosp_frees, dtype=float)
    icu_frees = np.array(icu_frees, dtype=float)
    sofas = np.array(sofas, dtype=int)
    apaches = np.array(apaches, dtype=int)
    sepsiss = np.array(sepsis, dtype=bool)
    net_fluids = np.array(net_fluids, dtype=float)
    gross_fluids = np.array(gross_fluids, dtype=float)
    c_scores = np.array(c_scores, dtype=float)  # Float bc possible NaNs
    e_scores = np.array(e_scores, dtype=float)  # Float bc possible NaNs
    mv_frees = np.array(mv_frees, dtype=float)
    eths = np.array(eths, dtype=bool)
    try:
        f = h5py.File(out_name, 'r+')
    except:
        f = h5py.File(out_name, 'w')

    try:
        meta = f[grp_name]
    except:
        meta = f.create_group(grp_name)

    meta.create_dataset('ids', data=all_ids, dtype=int)
    meta.create_dataset('age', data=ages, dtype=float)
    meta.create_dataset('race', data=eths, dtype=bool)
    meta.create_dataset('gender', data=genders, dtype=bool)
    meta.create_dataset('max_kdigo', data=mks, dtype=int)
    meta.create_dataset('died_inp', data=dieds, dtype=int)
    meta.create_dataset('n_episodes', data=nepss, dtype=int)
    meta.create_dataset('hosp_free_days', data=hosp_frees, dtype=float)
    meta.create_dataset('icu_free_days', data=icu_frees, dtype=float)
    meta.create_dataset('sofa', data=sofas, dtype=int)
    meta.create_dataset('apache', data=apaches, dtype=int)
    meta.create_dataset('sepsis', data=sepsiss, dtype=bool)
    meta.create_dataset('net_fluid', data=net_fluids, dtype=int)
    meta.create_dataset('gross_fluid', data=gross_fluids, dtype=int)
    meta.create_dataset('charlson', data=c_scores, dtype=int)
    meta.create_dataset('elixhauser', data=e_scores, dtype=int)
    meta.create_dataset('mv_free_days', data=mv_frees, dtype=int)

    return meta


# %%
def get_cstats(in_file, cluster_method, n_clust, out_name, plot_hist=False, report_kdigo0=True, meta_grp='meta'):
    # get IDs and Clusters in order from cluster file
    # ids = np.loadtxt(id_file, dtype=int, delimiter=',')
    if type(in_file) == str:
        f = h5py.File(in_file, 'r')
    else:
        f = in_file
    meta = f[meta_grp]
    ids = meta['ids'][:]
    if report_kdigo0:
        ages = meta['age'][:]
        genders = meta['gender'][:]
        m_kdigos = meta['max_kdigo'][:]
        died_inp = meta['died_inp'][:]
        n_eps = meta['n_episodes'][:]
        hosp_los = meta['hosp_free_days'][:]
        icu_los = meta['icu_free_days'][:]
        sofa = meta['sofa'][:]
        apache = meta['apache'][:]
        # sepsis = meta['sepsis'][:]
        net_fluid = meta['net_fluid'][:]
        gross_fluid = meta['gross_fluid'][:]
        charlson = meta['charlson'][:]
        elixhauser = meta['elixhauser'][:]
        mech_vent = meta['mv_free_days'][:]

        cids = f['clusters'][cluster_method]['ids'][:]
        clust = f['clusters'][cluster_method][n_clust][:]
        cmin = np.min(clust)

        clusters = np.zeros(len(ids))
        for cid, c in zip(cids, clust):
            idx = np.where(ids == cid)[0][0]
            if c >= 0 and cmin == 0:
                clusters[idx] = c + 1
            else:
                clusters[idx] = c

    else:
        cids = f['clusters'][cluster_method]['ids'][:]
        clust = f['clusters'][cluster_method][n_clust][:]
        pt_mask = np.zeros(len(ids), dtype=bool)
        for i in range(len(ids)):
            if ids[i] in cids:
                pt_mask[i] = 1

        ages = meta['age'][pt_mask]
        genders = meta['gender'][pt_mask]
        m_kdigos = meta['max_kdigo'][pt_mask]
        died_inp = meta['died_inp'][pt_mask]
        n_eps = meta['n_episodes'][pt_mask]
        hosp_los = meta['hosp_free_days'][pt_mask]
        icu_los = meta['icu_free_days'][pt_mask]
        sofa = meta['sofa'][pt_mask]
        apache = meta['apache'][pt_mask]
        # sepsis = meta['sepsis'][pt_mask]
        net_fluid = meta['net_fluid'][pt_mask]
        gross_fluid = meta['gross_fluid'][pt_mask]
        charlson = meta['charlson'][pt_mask]
        elixhauser = meta['elixhauser'][pt_mask]
        mech_vent = meta['mv_free_days'][pt_mask]

        clusters = clust[np.argsort(cids)]
        if np.min(clusters) == 0:
            for i in range(len(clusters)):
                if clusters[i] >= 0:
                    clusters[i] += 1

    lbls = np.unique(clusters)

    # c_header = 'cluster_id,count,mort_pct,n_kdigo_0,n_kdigo_1,n_kdigo_2,n_kdigo_3,n_kdigo_3D,' + \
    #            'n_eps_mean,n_eps_std,hosp_free_median,hosp_free_25,hosp_free_75,icu_free_median,' + \
    #            'icu_free_25,icu_free_75,sofa_mean,sofa_std,apache_mean,apache_std,age_mean,age_std,' + \
    #            'percent_male,percent_sepsis,fluid_overload_mean,fluid_overload_std,gross_fluid_mean,gross_fluid_std,' + \
    #            'charlson_mean,charlson_std,elixhauser_mean,elixhauser_std,mech_vent_free_med,mech_vent_25,mech_vent_75\n'
    c_header = 'cluster_id,count,mort_pct,n_kdigo_0,n_kdigo_1,n_kdigo_2,n_kdigo_3,n_kdigo_3D,' + \
               'n_eps_mean,n_eps_std,hosp_free_median,hosp_free_25,hosp_free_75,icu_free_median,' + \
               'icu_free_25,icu_free_75,sofa_mean,sofa_std,apache_mean,apache_std,age_mean,age_std,' + \
               'percent_male,fluid_overload_mean,fluid_overload_std,gross_fluid_mean,gross_fluid_std,' + \
               'charlson_mean,charlson_std,elixhauser_mean,elixhauser_std,mech_vent_free_med,mech_vent_25,mech_vent_75\n'

    f = open(out_name, 'w')
    f.write(c_header)
    for i in range(len(lbls)):
        cluster_id = lbls[i]
        if cluster_id == 0 and not report_kdigo0:
            continue
        rows = np.where(clusters == cluster_id)[0]
        count = len(rows)
        mort = float(len(np.where(died_inp[rows])[0])) / count
        k_counts = np.zeros(5)
        for i in range(5):
            k_counts[i] = len(np.where(m_kdigos[rows] == i)[0])
        n_eps_avg = np.mean(n_eps[rows])
        n_eps_std = np.std(n_eps[rows])
        hosp_los_med = np.median(hosp_los[rows])
        hosp_los_25 = np.percentile(hosp_los[rows], 25)
        hosp_los_75 = np.percentile(hosp_los[rows], 75)
        icu_los_med = np.median(icu_los[rows])
        icu_los_25 = np.percentile(icu_los[rows], 25)
        icu_los_75 = np.percentile(icu_los[rows], 75)
        sofa_avg = np.mean(sofa[rows])
        sofa_std = np.std(sofa[rows])
        apache_avg = np.mean(apache[rows])
        apache_std = np.std(apache[rows])
        age_mean = np.mean(ages[rows])
        age_std = np.std(ages[rows])
        pct_male = float(len(np.where(genders[rows])[0])) / count
        # pct_septic = float(len(np.where(sepsis[rows])[0]))/count
        net_mean = np.nanmean(net_fluid[rows])
        net_std = np.nanstd(net_fluid[rows])
        gross_mean = np.nanmean(gross_fluid[rows])
        gross_std = np.nanstd(gross_fluid[rows])
        charl_mean = np.nanmean(charlson[rows])
        charl_std = np.nanstd(charlson[rows])
        elix_mean = np.nanmean(elixhauser[rows])
        elix_std = np.nanstd(elixhauser[rows])
        mech_med = np.nanmedian(mech_vent[rows])
        mech_25 = np.nanpercentile(mech_vent[rows], 25)
        mech_75 = np.nanpercentile(mech_vent[rows], 75)
        if plot_hist:
            plt.figure()
            plt.subplot(3, 2, 1)
            plt.hist(m_kdigos[rows])
            plt.ylabel('# of Patients')
            plt.xlabel('KDIGO Score')
            plt.title('Max KDIGO')

            plt.subplot(3, 2, 2)
            plt.hist(hosp_los[rows])
            plt.ylabel('# of Patients')
            plt.xlabel('Days')
            plt.title('Hospital LOS')

            plt.subplot(3, 2, 3)
            plt.hist(icu_los[rows])
            plt.ylabel('# of Patients')
            plt.xlabel('Days')
            plt.title('ICU LOS')

            plt.subplot(3, 2, 4)
            plt.hist(sofa[rows])
            plt.ylabel('# of Patients')
            plt.xlabel('SOFA Score')
            plt.title('SOFA')

            plt.subplot(3, 2, 5)
            plt.hist(apache[rows])
            plt.ylabel('# of Patients')
            plt.xlabel('APACHE Score')
            plt.title('APACHE II')

            plt.subplot(3, 2, 6)
            plt.hist(net_fluid[rows])
            plt.ylabel('# of Patients')
            plt.xlabel('Mililiters')
            plt.title('Fluid Overload')
            plt.suptitle('Cluster ' + str(cluster_id) + ' Distributions')
            plt.tight_layout()
            plt.savefig('cluster' + str(cluster_id) + 'dist.png')
        # f.write('%d,%d,%.3f,%.3f,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n' %
        #         (cluster_id, count, mort, k_counts[0], k_counts[1], k_counts[2], k_counts[3], k_counts[4],
        #          n_eps_avg, n_eps_std, hosp_los_med, hosp_los_25, hosp_los_75, icu_los_med, icu_los_25, icu_los_75,
        #          sofa_avg, sofa_std, apache_avg, apache_std, age_mean, age_std, pct_male, pct_septic, net_mean, net_std,
        #          gross_mean, gross_std, charl_mean, charl_std, elix_mean, elix_std, mech_med, mech_25, mech_75))
        f.write(
            '%d,%d,%.3f,%.3f,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n' %
            (cluster_id, count, mort, k_counts[0], k_counts[1], k_counts[2], k_counts[3], k_counts[4],
             n_eps_avg, n_eps_std, hosp_los_med, hosp_los_25, hosp_los_75, icu_los_med, icu_los_25, icu_los_75,
             sofa_avg, sofa_std, apache_avg, apache_std, age_mean, age_std, pct_male, net_mean, net_std,
             gross_mean, gross_std, charl_mean, charl_std, elix_mean, elix_std, mech_med, mech_25, mech_75))


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
        h_dur += hosp[i][1].to_pydatetime() - hosp[i][0].to_pydatetime()

    icu_dur = datetime.timedelta(0)
    for i in range(len(icu)):
        icu_dur += icu[i][1].to_pydatetime() - icu[i][0].to_pydatetime()

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


def get_sofa(id_file, in_name, out_name):
    ids = np.loadtxt(id_file, dtype=int)

    admit_info = get_mat(in_name, 'ADMISSION_INDX', 'STUDY_PATIENT_ID')
    date = admit_info.columns.get_loc('HOSP_ADMIT_DATE')
    admit_info = admit_info.as_matrix()

    blood_gas = get_mat(in_name, 'BLOOD_GAS', 'STUDY_PATIENT_ID')
    pa_o2 = blood_gas.columns.get_loc('PO2_D1_HIGH_VALUE')
    blood_gas = blood_gas.as_matrix()

    organ_sup = get_mat(in_name, 'ORGANSUPP_VENT', 'STUDY_PATIENT_ID')
    mech_vent = [organ_sup.columns.get_loc('VENT_START_DATE'),
                 organ_sup.columns.get_loc('VENT_STOP_DATE')]
    organ_sup = organ_sup.as_matrix()

    clinical_oth = get_mat(in_name, 'CLINICAL_OTHERS', 'STUDY_PATIENT_ID')
    # fi_o2 = clinical_oth.columns.get_loc('FI02_D1_HIGH_VALUE')
    g_c_s = clinical_oth.columns.get_loc('GLASGOW_SCORE_D1_LOW_VALUE')
    clinical_oth = clinical_oth.as_matrix()

    clinical_vit = get_mat(in_name, 'CLINICAL_VITALS', 'STUDY_PATIENT_ID')
    m_a_p = clinical_vit.columns.get_loc('ART_MEAN_D1_LOW_VALUE')
    clinical_vit = clinical_vit.as_matrix()

    labs = get_mat(in_name, 'LABS_SET1', 'STUDY_PATIENT_ID')
    bili = labs.columns.get_loc('BILIRUBIN_D1_HIGH_VALUE')
    pltlts = labs.columns.get_loc('PLATELETS_D1_LOW_VALUE')
    labs = labs.as_matrix()

    medications = get_mat(in_name, 'MEDICATIONS_INDX', 'STUDY_PATIENT_ID')
    med_name = medications.columns.get_loc('MEDICATION_TYPE')
    med_date = medications.columns.get_loc('ORDER_ENTERED_DATE')
    med_dur = medications.columns.get_loc('DAYS_ON_MEDICATION')
    medications = medications.as_matrix()

    scr_agg = get_mat(in_name, 'SCR_INDX_AGG', 'STUDY_PATIENT_ID')
    s_c_r = scr_agg.columns.get_loc('DAY1_MAX_VALUE')
    scr_agg = scr_agg.as_matrix()

    out = open(out_name, 'w')
    for idx in ids:
        out.write('%d' % idx)
        admit_rows = np.where(admit_info[:, 0] == idx)[0]
        bg_rows = np.where(blood_gas[:, 0] == idx)[0]
        co_rows = np.where(clinical_oth[:, 0] == idx)[0]
        cv_rows = np.where(clinical_vit[:, 0] == idx)[0]
        lab_rows = np.where(labs[:, 0] == idx)[0][0]
        med_rows = np.where(medications[:, 0] == idx)[0]
        scr_rows = np.where(scr_agg[:, 0] == idx)[0]
        mv_rows = np.where(organ_sup[:, 0] == idx)[0]

        admit = datetime.datetime.now()
        for did in admit_rows:
            if admit_info[did, date] < admit:
                admit = admit_info[did, date]
        if type(admit) == np.ndarray:
            admit = admit[0]

        s1_pa = blood_gas[bg_rows, pa_o2]
        # s1_fi = clinical_oth[co_rows,fi_o2]
        # s1_vent = organ_sup[mv_rows,mech_vent]

        s2_gcs = clinical_oth[co_rows, g_c_s]

        s3_map = clinical_vit[cv_rows, m_a_p]
        s3_med = medications[med_rows, med_name]
        s3_date = medications[med_rows, med_date]
        s3_dur = medications[med_rows, med_dur]

        s4_bili = labs[lab_rows, bili]

        s5_plt = labs[lab_rows, pltlts]

        s6_scr = scr_agg[scr_rows, s_c_r]

        score = np.zeros(6, dtype=int)

        vent = 0
        if len(mv_rows) > 0:
            for row in range(len(mv_rows)):
                tmv = organ_sup[row, mech_vent]
                if tmv[0] <= admit <= tmv[1]:
                    vent = 1
        if vent:
            if s1_pa < 100:
                score[0] = 4
            elif s1_pa < 200:
                score[0] = 3
        elif s1_pa < 300:
            score[0] = 2
        elif s1_pa < 400:
            score[0] = 1

        if s2_gcs.size > 0:
            s2_gcs = s2_gcs[0]
            s2 = float(str(s2_gcs).split('-')[0])
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
        dopa = 0
        epi = 0
        try:
            admit = admit_info[admit_rows[0][0], date]
            for i in range(len(s3_med)):
                med_typ = s3_med[i]
                dm = med_typ[0][0].lower()
                start = s3_date[i]
                stop = start + datetime.timedelta(s3_dur[i])
                if 'dopamine' in dm or 'dobutamine' in dm:
                    if start <= admit:
                        if admit <= stop:
                            dopa = 1
                elif 'epinephrine' in dm:
                    if start <= admit:
                        if admit <= stop:
                            epi = 1
        except:
            s3 = s3
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
        if s6 > 3.5:
            score[5] = 3
        if s6 > 2.0:
            score[5] = 2
        if s6 > 1.2:
            score[5] = 1

        out.write(',%d,%d,%d,%d,%d,%d\n' % (score[0], score[1], score[2], score[3], score[4], score[5]))
        print(np.sum(score))


def get_apache(id_file, in_name, out_name):
    ids = np.loadtxt(id_file, dtype=int)

    clinical_vit = get_mat(in_name, 'CLINICAL_VITALS', 'STUDY_PATIENT_ID')
    temp = [clinical_vit.columns.get_loc('TEMPERATURE_D1_LOW_VALUE'),
            clinical_vit.columns.get_loc('TEMPERATURE_D1_HIGH_VALUE')]
    m_ap = [clinical_vit.columns.get_loc('ART_MEAN_D1_LOW_VALUE'),
            clinical_vit.columns.get_loc('ART_MEAN_D1_HIGH_VALUE')]
    h_r = [clinical_vit.columns.get_loc('HEART_RATE_D1_LOW_VALUE'),
           clinical_vit.columns.get_loc('HEART_RATE_D1_HIGH_VALUE')]
    clinical_vit = clinical_vit.as_matrix()

    clinical_oth = get_mat(in_name, 'CLINICAL_OTHERS', 'STUDY_PATIENT_ID')
    resp = [clinical_oth.columns.get_loc('RESP_RATE_D1_LOW_VALUE'),
            clinical_oth.columns.get_loc('RESP_RATE_D1_HIGH_VALUE')]
    fi_o2 = [clinical_oth.columns.get_loc('FI02_D1_LOW_VALUE'),
             clinical_oth.columns.get_loc('FI02_D1_HIGH_VALUE')]
    gcs = clinical_oth.columns.get_loc('GLASGOW_SCORE_D1_LOW_VALUE')
    clinical_oth = clinical_oth.as_matrix()

    blood_gas = get_mat(in_name, 'BLOOD_GAS', 'STUDY_PATIENT_ID')
    pa_o2 = [blood_gas.columns.get_loc('PO2_D1_LOW_VALUE'),
             blood_gas.columns.get_loc('PO2_D1_HIGH_VALUE')]
    pa_co2 = [blood_gas.columns.get_loc('PCO2_D1_LOW_VALUE'),
              blood_gas.columns.get_loc('PCO2_D1_HIGH_VALUE')]
    p_h = [blood_gas.columns.get_loc('PH_D1_LOW_VALUE'),
           blood_gas.columns.get_loc('PH_D1_HIGH_VALUE')]
    blood_gas = blood_gas.as_matrix()

    labs = get_mat(in_name, 'LABS_SET1', 'STUDY_PATIENT_ID')
    na = [labs.columns.get_loc('SODIUM_D1_LOW_VALUE'),
          labs.columns.get_loc('SODIUM_D1_HIGH_VALUE')]
    p_k = [labs.columns.get_loc('POTASSIUM_D1_LOW_VALUE'),
           labs.columns.get_loc('POTASSIUM_D1_HIGH_VALUE')]
    hemat = [labs.columns.get_loc('HEMATOCRIT_D1_LOW_VALUE'),
             labs.columns.get_loc('HEMATOCRIT_D1_HIGH_VALUE')]
    w_b_c = [labs.columns.get_loc('WBC_D1_LOW_VALUE'),
             labs.columns.get_loc('WBC_D1_HIGH_VALUE')]
    labs = labs.as_matrix()

    _, ages = load_csv('../DATA/icu/7days_inc_death_051918/ages.csv', ids)

    scr_agg = get_mat(in_name, 'SCR_INDX_AGG', 'STUDY_PATIENT_ID')
    s_c_r = scr_agg.columns.get_loc('DAY1_MAX_VALUE')
    scr_agg = scr_agg.as_matrix()

    out = open(out_name, 'w')
    ct = 0
    for idx in ids:
        out.write('%d' % idx)

        bg_rows = np.where(blood_gas[:, 0] == idx)[0]
        co_rows = np.where(clinical_oth[:, 0] == idx)[0]
        cv_rows = np.where(clinical_vit[:, 0] == idx)[0]
        lab_rows = np.where(labs[:, 0] == idx)[0]
        scr_rows = np.where(scr_agg[:, 0] == idx)[0]

        s1_low = clinical_vit[cv_rows, temp[0]]
        s1_high = clinical_vit[cv_rows, temp[1]]

        s2_low = clinical_vit[cv_rows, m_ap[0]]
        s2_high = clinical_vit[cv_rows, m_ap[1]]

        s3_low = clinical_vit[cv_rows, h_r[0]]
        s3_high = clinical_vit[cv_rows, h_r[1]]

        s4_low = clinical_oth[co_rows, resp[0]]
        s4_high = clinical_oth[co_rows, resp[1]]

        s5_po = blood_gas[bg_rows[0], pa_o2[1]]
        s5_pco = blood_gas[bg_rows[0], pa_co2[1]]
        s5_f = blood_gas[bg_rows[0], fi_o2[1]]
        if type(s5_po) != str and type(s5_po) != unicode:
            if not np.isnan(s5_po):
                s5_po = float(s5_po) / 100
        else:
            s5_po = np.nan

        if type(s5_pco) != str and type(s5_pco) != unicode:
            if not np.isnan(s5_pco):
                s5_pco = float(s5_pco) / 100
        else:
            s5_pco = np.nan

        if type(s5_f) != str and type(s5_f) != unicode:
            if not np.isnan(s5_f):
                s5_f = float(s5_f) / 100
        else:
            s5_f = np.nan

        s6_low = labs[lab_rows, p_h[0]]
        s6_high = labs[lab_rows, p_h[1]]

        s7_low = labs[lab_rows, na[0]]
        s7_high = labs[lab_rows, na[1]]

        s8_low = labs[lab_rows, p_k[0]]
        s8_high = labs[lab_rows, p_k[1]]

        s9 = scr_agg[scr_rows, s_c_r]

        s10_low = labs[lab_rows, hemat[0]]
        s10_high = labs[lab_rows, hemat[1]]

        s11_low = labs[lab_rows, w_b_c[0]]
        s11_high = labs[lab_rows, w_b_c[1]]

        s12_gcs = clinical_oth[co_rows, gcs]

        s13_age = float(ages[ct])

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
            if aado2 > 4:
                score[4] = 4
            elif aado2 > 3.5:
                score[4] = 3
            elif aado2 > 2:
                score[4] = 2
        else:
            if s5_po < 0.55:
                score[4] = 4
            elif s5_po < 0.60:
                score[4] = 3
            elif s5_po < 0.70:
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
        elif s8_high <= 6:
            score[7] = 3
        elif s8_low < 3:
            score[7] = 2
        elif s8_low < 3.5 or s7_high >= 5.5:
            score[7] = 1

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

        if s12_gcs.size > 0:
            s12_gcs = s12_gcs[0]
            s12 = float(str(s12_gcs).split('-')[0])
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
        print(np.sum(score))


def get_MAKE90(ids, in_name, stats, bsln_file, out_file, pct_lim=25):
    # load outcome data
    date_m = get_mat(in_name, 'ADMISSION_INDX', 'STUDY_PATIENT_ID')
    disch_loc = date_m.columns.get_loc("HOSP_DISCHARGE_DATE")
    date_m = date_m.as_matrix()

    # load death data
    mort_m = get_mat(in_name, 'OUTCOMES', 'STUDY_PATIENT_ID')
    mdate_loc = mort_m.columns.get_loc("DECEASED_DATE")
    mort_m = mort_m.as_matrix()

    # All SCR
    print('Loading SCr values (may take a while)...')
    scr_all_m = get_mat(in_name, 'SCR_ALL_VALUES', ['STUDY_PATIENT_ID', 'SCR_ENTERED'])
    scr_date_loc = scr_all_m.columns.get_loc('SCR_ENTERED')
    scr_val_loc = scr_all_m.columns.get_loc('SCR_VALUE')
    scr_all_m = scr_all_m.as_matrix()

    # Dialysis dates
    print('Loading dialysis dates...')
    dia_m = get_mat(in_name, 'RENAL_REPLACE_THERAPY', ['STUDY_PATIENT_ID'])
    crrt_locs = [dia_m.columns.get_loc('CRRT_START_DATE'), dia_m.columns.get_loc('CRRT_STOP_DATE')]
    hd_locs = [dia_m.columns.get_loc('HD_START_DATE'), dia_m.columns.get_loc('HD_STOP_DATE')]
    pd_locs = [dia_m.columns.get_loc('PD_START_DATE'), dia_m.columns.get_loc('PD_STOP_DATE')]
    dia_m = dia_m.as_matrix()

    # Baseline data
    bsln_m = pd.read_csv(bsln_file)
    bsln_val_loc = bsln_m.columns.get_loc('bsln_val')
    bsln_m = bsln_m.as_matrix()

    ages = stats['age'][:]
    races = stats['race'][:]
    sexes = stats['gender'][:]

    print('MAKE-90: GFR Thresh = %d\%' % pct_lim)
    print('id,died,gfr_drop,new_dialysis')
    out = open(out_file, 'w')
    out.write('MAKE-90: GFR Thresh = %d\%\n' % pct_lim)
    out.write('id,died,gfr_drop,new_dialysis\n')
    for i in range(len(ids)):
        idx = ids[i]
        scr_locs = np.where(scr_all_m[:, 0] == idx)[0]
        dia_locs = np.where(dia_m[:, 0] == idx)[0]
        mort_locs = np.where(mort_m[:, 0] == idx)[0]
        date_locs = np.where(date_m[:, 0] == idx)[0]
        bsln_loc = np.where(bsln_m[:, 0] == idx)[0][0]

        died = 0
        gfr_drop = 0
        new_dia = 0

        bsln_scr = bsln_m[bsln_loc, bsln_val_loc]
        age = ages[i]
        race = races[i]
        sex = sexes[i]
        bsln_gfr = calc_gfr(bsln_scr, sex, race, age)

        disch = datetime.datetime(1000, 1, 1)
        for j in range(len(date_locs)):
            tid = date_locs[j]
            tdate = str(date_m[tid, disch_loc])
            if len(tdate) > 3:
                tdate = datetime.datetime.strptime(tdate.split('.')[0], '%Y-%m-%d %H:%M:%S')
                if tdate > disch:
                    disch = tdate

        min_gfr = 1000
        for j in range(len(scr_locs)):
            tdate = str(scr_all_m[scr_locs[j], scr_date_loc])
            tdate = datetime.datetime.strptime(tdate.split('.')[0], '%Y-%m-%d %H:%M:%S')
            if datetime.timedelta(0) < tdate - disch < datetime.timedelta(90):
                tscr = scr_all_m[scr_locs[j], scr_val_loc]
                tgfr = calc_gfr(tscr, sex, race, age)
                if tgfr < min_gfr:
                    min_gfr = tgfr

        thresh = 100 - pct_lim
        rel_pct = (min_gfr / bsln_gfr) * 100
        if rel_pct < thresh:
            gfr_drop = 1

        for j in range(len(mort_locs)):
            m = mort_m[mort_locs[j], mdate_loc]
            try:
                m = datetime.datetime.strptime(m.split('.')[0], '%Y-%m-%d %H:%M:%S')
                if datetime.timedelta(0) < m - disch < datetime.timedelta(90):
                    died = 1
            except:
                continue

        for j in range(len(dia_locs)):
            tstart = dia_m[dia_locs[j], crrt_locs[0]]
            try:
                tstart = datetime.datetime.strptime(tstart.split('.')[0], '%Y-%m-%d %H:%M:%S')
                if tstart < disch:
                    new_dia = -1
                    break
                elif datetime.timedelta(0) < tstart - disch < datetime.timedelta(90):
                    new_dia = 1
            except:
                tstart = dia_m[dia_locs[j], pd_locs[0]]
                try:
                    tstart = datetime.datetime.strptime(tstart.split('.')[0], '%Y-%m-%d %H:%M:%S')
                    if tstart < disch:
                        new_dia = -1
                        break
                    elif datetime.timedelta(0) < tstart - disch < datetime.timedelta(90):
                        new_dia = 1
                except:
                    tstart = dia_m[dia_locs[j], hd_locs[0]]
                    try:
                        tstart = datetime.datetime.strptime(tstart.split('.')[0], '%Y-%m-%d %H:%M:%S')
                        if tstart < disch:
                            new_dia = -1
                            break
                        elif datetime.timedelta(0) < tstart - disch < datetime.timedelta(90):
                            new_dia = 1
                    except:
                        continue

        print('%d,%d,%d,%d' % (idx, died, gfr_drop, new_dia))
        out.write('%d,%d,%d,%d\n' % (idx, died, gfr_drop, new_dia))
