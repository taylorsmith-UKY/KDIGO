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


def cm_rem_ids(all_data_file, bad_id_file, out_file):
    bad_ids = np.loadtxt(bad_id_file, dtype=int)

    inf = open(all_data_file, 'r')
    out = open(out_file, 'w')

    for l in inf:
        id1, id2, d = l.rstrip().split(',')
        id1 = int(id1)
        id2 = int(id2)
        d = float(d)
        if id1 in bad_ids or id2 in bad_ids:
            continue
        else:
            out.write('%d,%d,%.3f\n' % (id1, id2, d))
    return


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


def post_dm_process(h5_fname, csv_fname, output_base_path, dm_tag='_norm_norm_a1', eps=0.015, p_thresh=1e-3, no_save=False, interactive=False):
    try:
        if no_save:
            f = h5py.File(h5_fname, 'r')
        else:
            f = h5py.File(h5_fname, 'r+')
    except:
        f = h5py.File(h5_fname, 'w')
    try:
        dm = f['dm' + dm_tag][:]
    except:
        dm = np.loadtxt(csv_fname, usecols=2, dtype=float, delimiter=',')
        _ = f.create_dataset('dm' + dm_tag, data=dm)
    stats = f['meta']
    ids = stats['ids'][:]

    sqdm = squareform(dm)
    print('Distance Matrix Loaded')
    print('Starting DBSCAN')

    if not no_save:
        try:
            cg = f['clusters' + dm_tag]
        except:
            cg = f.create_group('clusters' + dm_tag)
        try:
            dbg = cg['dbscan']
        except:
            dbg = cg.create_group('dbscan')
            dbg.create_dataset('ids', data=ids, dtype=int)
            os.makedirs(output_base_path + 'dbscan')
    print('\n')
    db = DBSCAN(eps=eps, metric='precomputed', n_jobs=-1)
    cont = True
    while cont:
        db.fit(sqdm)
        lbls = np.array(db.labels_, dtype=int)
        nclust = len(np.unique(lbls)) - 1
        print('Number of Clusters = %d' % nclust)
        if interactive:
            eps = raw_input('New epsilon (non-numeric to continue): ')
            try:
                eps = float(eps)
                db = DBSCAN(eps=eps, metric='precomputed', n_jobs=-1)
            except:
                cont = False
        else:
            cont = False
    lbls[np.where(lbls >= 0)] += 1  # because dbscan starts with cluster label 0
    lbls = lbls.astype(str)
    grp_name = '%d_clusters' % nclust
    if not no_save:
        dbg.create_dataset(grp_name, data=lbls)
    plt.close('all')
    if not no_save:
        os.makedirs(output_base_path + 'dbscan/' + grp_name)
        os.makedirs(output_base_path + 'dbscan/' + grp_name + '/max_dist/')
    dpath = output_base_path + 'dbscan/' + grp_name + '/max_dist/'
    plt.close('all')
    if not no_save:
        np.savetxt(output_base_path + 'dbscan/' + grp_name + '/clusters.txt', fmt='%s')
        inter_intra_dist(f, sqdm, 'dbscan', grp_name, op='max', out_path=dpath, plot='both')
        get_cstats(f, output_base_path + 'dbscan/' + grp_name + '/', meta_grp='meta')
        np.savetxt(output_base_path + 'dbscan/' + grp_name + '/clusters.txt', lbls, fmt='%s')
    print('Breaking down DBSCAN cluster 1')
#    lbls = ward_breakdown(f, sqdm, lbls, ids, 1, cg, 'dbscan1',
#                              output_base_path + 'dbscan/', (1,))
    lbls = ward_breakdown(sqdm, lbls, '1', p_thresh=p_thresh, min_size=20)
    return lbls



def ward_breakdown(sqdm, lbls, bkdn_lbl, p_thresh=1e-3, min_size=20):
    sel = np.where(lbls == bkdn_lbl)[0]
    if len(sel) < min_size:
        return lbls

    idx = np.ix_(sel, sel)
    dm = squareform(sqdm[idx])  # squareform acts as its own inverse so dm is condensed

    dm = np.array(dm, copy=True)
    try:
        _, p = normaltest(dm)
        print(p)
    except:
        print('Normaltest did not work.')
        p = 0
    if p < p_thresh:
        print('Splitting cluster %s into %s and %s' % (bkdn_lbl, bkdn_lbl + '-1', bkdn_lbl + '-2'))

        link = ward(dm)

        nlbls = fcluster(link, 2, criterion='maxclust')
        sel1 = np.where(nlbls == 1)[0]
        sel2 = np.where(nlbls == 2)[0]

        if len(sel1) < min_size or len(sel2) < min_size:
            return lbls

        lbls[sel[sel1]] = bkdn_lbl + '-1'
        lbls[sel[sel2]] = bkdn_lbl + '-2'
        lbls = ward_breakdown(sqdm, lbls, bkdn_lbl + '-1', p_thresh, min_size)
        lbls = ward_breakdown(sqdm, lbls, bkdn_lbl + '-2', p_thresh, min_size)

    return lbls


# def ward_breakdown(f, sqdm, lbls, prev_ids, bkdn_num, clust_grp, base_name,
#                    output_base_path, parents=(), lbl_list=(), no_save=False, pthresh=5e-2):
#     sel = np.where(lbls == bkdn_num)[0]
#     ids = prev_ids[sel]
#     idx = np.ix_(sel, sel)
#     sqdm = sqdm[idx]
#     condensed = squareform(sqdm)
#     # f.create_dataset(base_name+'_dm', data=condensed)
#     cgrp_name = base_name + '_ward'
#     outpath = output_base_path + '/ward' + str(bkdn_num) + '/'
#     if not no_save:
#         os.makedirs(outpath)
#         cgrp = clust_grp.create_group(cgrp_name)
#         cgrp.create_dataset('ids', data=ids)
#     link = ward(condensed)
#     plt.close('all')
#     dendrogram(link, 50, truncate_mode='lastp')
#     plt.title(cgrp_name+'\nWard\'s Method')
#     if not no_save:
#         plt.savefig(outpath+'/dendrogram.png')
#     plt.show()
#     cont = 1
#     while cont:
#         nclust = input('How many clusters to generate: ')
#         nlbls = fcluster(link, nclust, criterion='maxclust')
#         clust_name = str(nclust)+'_clusters'
#         if not no_save:
#             cgrp.create_dataset(clust_name, data=nlbls, dtype=int)
#         plt.close('all')
#         inter_intra_dist(f, sqdm, cgrp_name, clust_name, op='max', out_path='', plot='nosave')
#         ctext = raw_input('Try different number of clusters? ')
#         if ctext.lower()[0] == 'n':
#             cont = 0
#
#     dpath = outpath + '/' + clust_name + '/'
#     if not no_save:
#         os.makedirs(dpath)
#         os.makedirs(dpath + 'max_dist/')
#         np.savetxt(dpath + 'clusters.txt', nlbls)
#     plt.close('all')
#     if not no_save:
#         inter_intra_dist(f, sqdm, cgrp_name, clust_name, op='max', out_path=dpath + '/max_dist/', plot='both')
#         get_cstats(f, cluster_method=cgrp_name, n_clust=clust_name,
#                    out_name=dpath + '/cluster_stats.csv', report_kdigo0=False, meta_grp='meta')
#
#     ctext = raw_input('Cluster IDs to break-down(id separated by comma (no space); N/n to quit): ')
#     lbl_list.append((parents, nlbls))
#     if ctext.lower()[0] == 'n':
#         return lbl_list
#     if len(ctext) == 1:
#         bkdn_nums = np.array((int(ctext),))
#     else:
#         bkdn_nums = np.array(ctext.split(','), dtype=int)
#     for i in range(len(bkdn_nums)):
#         base_temp = base_name + '_ward' + str(bkdn_nums[i])
#         lbl_list = ward_breakdown(f, sqdm, nlbls, ids, bkdn_nums[i], clust_grp, base_temp,
#                                   outpath, parents + (bkdn_nums[i],), lbl_list, no_save=no_save)
#     return lbl_list

def get_ref(lbls, stats, baseline_file, data_path, out_name):
    clbls = np.unique(lbls)
    n_clust = len(clbls)
    c_idx = {}
    for i in range(n_clust):
        idx = np.where(lbls == clbls[i])[0]
        c_idx[clbls[i]] = idx

    mk = stats['max_kdigo'][:]
    mk_lbls = np.unique(mk)
    nmk = len(mk_lbls)
    k_idx = {}
    for i in range(nmk):
        idx = np.where(mk == i + 1)[0]
        k_idx[mk_lbls[i]] = idx

    ids = stats['ids'][:]
    _, b_types = load_csv(baseline_file, ids, dt=str, skip_header=True, sel=2)
    b_types = np.array(b_types)
    bt = np.unique(b_types)
    nbt = len(bt)
    b_idx = {}
    for i in range(nbt):
        b_idx[bt[i]] = np.where(b_types == bt[i])[0]\

    _, b_vals = load_csv(baseline_file, ids, skip_header=True, sel=1)
    _, admits = load_csv(baseline_file, ids, skip_header=True, sel=4, dt=str)
    _, scrs_raw = load_csv(data_path + 'scr_raw.csv', ids)
    _, dates_raw = load_csv(data_path + 'dates.csv', ids, dt=str)
    _, dmasks = load_csv(data_path + 'dialysis.csv', ids, dt=int)
    sids, sofas = load_csv(data_path + 'sofa.csv', ids, dt=int)
    aids, apaches = load_csv(data_path + 'apache.csv', ids, dt=int)
    ages = stats['age']
    races = stats['race']
    genders = stats['gender']

    print(clbls)
    print(mk_lbls)
    print(bt)
    out = []
    f = open(out_name, 'w')
    for i in range(n_clust):
        csel = c_idx[clbls[i]]
        cmk = mk[csel]
        nmk = len(np.unique(cmk))
        nex = int(10 / nmk)
        for mid in np.unique(cmk):
            for k in range(nbt):
                idx = np.intersect1d(np.intersect1d(c_idx[clbls[i]], k_idx[mid]), np.intersect1d(c_idx[clbls[i]], b_idx[bt[k]]))
                if len(idx) >= nex:
                    tids = np.random.permutation(idx)[:nex]
                else:
                    tids = idx
                for tid in tids:
                    my_id = ids[tid]
                    out.append(my_id)
                    tc = lbls[tid]
                    bsln = b_vals[tid]
                    admit = datetime.datetime.strptime(str(admits[tid]).split('.')[0], '%Y-%m-%d %H:%M:%S')
                    tdates = [datetime.datetime.strptime(x.split('.')[0], '%Y-%m-%d %H:%M:%S') for x in dates_raw[tid]]
                    scrs = scrs_raw[tid]
                    dmask = dmasks[tid]
                    sofa = sofas[tid]
                    apache = apaches[tid]
                    kdigos = daily_max_kdigo(scrs, tdates, bsln, admit, dmask)
                    sex = genders[tid]
                    age = ages[tid]
                    race = races[tid]
                    gfr = calc_gfr(bsln, sex, race, age)
                    tstr = '%d,%d,%s,%.4f,%.4f' % (my_id, tc, bt[k], bsln, gfr)
                    for m in range(8):
                        if m < len(kdigos):
                            tstr += ',%d' % kdigos[m]
                        else:
                            tstr += ',M'
                    for m in range(len(sofa)):
                        tstr += ',%d' % sofa[m]
                    for m in range(len(apache)):
                        tstr += ',%d' % apache[m]
                    print(tstr)
    return out


def pickle_dms(path, dm_tags):
    for dm_tag in dm_tags:
        dm = np.loadtxt(path + 'kdigo_dm%s.csv' % dm_tag, delimiter=',', usecols=2)
        np.save(path + 'kdigo_dm%s' % dm_tag, dm)
