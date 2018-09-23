"""
Created on Fri Jan 19 14:31:03 2018

@author: taylorsmith
"""
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, fcluster, to_tree
from scipy.stats.mstats import normaltest
from scipy.stats import mode, sem, t
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score,\
    v_measure_score, fowlkes_mallows_score, silhouette_score, calinski_harabaz_score
import fastcluster as fc
from kdigo_funcs import arr2csv, load_csv, daily_max_kdigo, daily_max_kdigo_interp
import datetime
from stat_funcs import get_cstats
import matplotlib.pyplot as plt
import h5py
import os


# %%
def dist_cut_cluster(h5_fname, dm, ids, mk, meta_grp='meta', path='', eps=0.015, p_thresh=0.05,
                     min_size=20, height_lim=5, interactive=True, save=True, max_noise=100,
                     max_clust=None, hom_thresh=0.9, max_size_pct=0.2):
    f = h5py.File(h5_fname, 'r')
    # mk = f[meta_grp]['max_kdigo'][:]
    # if len(mk) != len(ids):
    #     all_ids = f[meta_grp]['ids'][:]
    #     all_mk = mk
    #     mk = []
    #     for i in range(len(all_ids)):
    #         if all_ids[i] in ids:
    #             mk.append(all_mk[i])
    #     mk = np.array(mk)
    n_pts = len(ids)
    if dm.ndim == 1:
        sqdm = squareform(dm)
    else:
        sqdm = np.array(dm, copy=True)
    db = DBSCAN(eps=eps, metric='precomputed', n_jobs=-1)
    cont = True
    date_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')
    while cont:
        db.fit(sqdm)
        lbls = np.array(db.labels_, dtype=int)
        lbls[np.where(lbls >= 0)] += 1
        lbl_names = np.unique(lbls)
        nclust = len(lbl_names) - 1
        for i in range(1, nclust + 1):
            tlbl = lbl_names[i]
            idx = np.where(lbls == tlbl)[0]
            if len(idx) < min_size:
                lbls[idx] = -1
        lbl_names = np.unique(lbls)
        nclust = len(lbl_names) - 1
        print('Number of DBSCAN Clusters = %d' % nclust)
        if interactive:
            lbl_names = np.unique(lbls)
            for i in range(len(lbl_names)):
                print('Cluster %d: %d members' % (lbl_names[i], len(np.where(lbls == lbl_names[i])[0])))
            t = raw_input('New epsilon (non-numeric to continue): ')
            try:
                eps = float(t)
                db = DBSCAN(eps=eps, metric='precomputed', n_jobs=-1)
            except:
                cont = False
        else:
            cont = False
    n_clusters = len(np.unique(lbls)) - 1
    if max_clust:
        if n_clusters > max_clust:
            f.close()
            return
    if len(np.where(lbls == -1)[0]) > max_noise * n_pts:
        print('%d patients designated as noise...' % len(np.where(lbls == -1)[0]))
        print('eps\tmin_size')
        print('%.3f\t%d' % (eps, min_size))
        f.close()
        return
    db_pvals = None
    if save:
        if not os.path.exists(path + 'dbscan'):
            os.makedirs(path + 'dbscan')
        if not os.path.exists(path + 'dbscan/%d_clusters' % n_clusters):
            os.makedirs(path + 'dbscan/%d_clusters' % n_clusters)
            arr2csv(path + 'dbscan/%d_clusters/clusters.txt' % n_clusters, lbls, ids, fmt='%d')
            # if not os.path.exists(path + 'dbscan/%d_clusters/mean_dist/' % n_clusters):
            #     os.makedirs(path + 'dbscan/%d_clusters/mean_dist/' % n_clusters)
            all_inter, all_intra, db_pvals = inter_intra_dist(sqdm, lbls,
                                                              out_path=path + 'dbscan/%d_clusters/mean_dist/' %
                                                              n_clusters, op='mean', plot=None)
            get_cstats(f, path + 'dbscan/%d_clusters/' % n_clusters, meta_grp=meta_grp, ids=ids)
            log = open(path + 'dbscan/%d_clusters/cluster_settings.txt' % n_clusters, 'w')
            log.write('DBSCAN Epsilon:\t\t%.4f\n' % eps)
            log.write('NormalTest p-thresh:\t%.2E\n' % p_thresh)
            log.write('Ward Height Lim:\t%d\n' % height_lim)
            log.write('Min Cluster Size:\t%d\n' % min_size)
            log.close()

            # arr2csv(path + 'dbscan/%d_clusters/mean_dist/all_intra.txt' % n_clusters, all_intra, ids,
            #         fmt='%.3f')
            # arr2csv(path + 'dbscan/%d_clusters/mean_dist/all_inter.txt' % n_clusters, all_inter, ids,
            #         fmt='%.3f')

        else:  # if folder already exists, append and create new
            ref = load_csv(path + 'dbscan/%d_clusters/clusters.txt' % n_clusters, ids).astype(int)
            if np.all(ref == lbls):
                cont = False
            else:
                cont = True
                tag = 'a'
            while cont:
                if os.path.exists(path + 'dbscan/%d_clusters_%s' % (n_clusters, tag)):
                    ref = load_csv(path + 'dbscan/%d_clusters_%s/clusters.txt'
                                   % (n_clusters, tag), ids).astype(int)
                    if np.all(ref == lbls):
                        cont = False
                    else:
                        tag = chr(ord(tag) + 1)
                else:
                    os.makedirs(path + 'dbscan/%d_clusters_%s' % (n_clusters, tag))
                    cont = False
                    arr2csv(path + 'dbscan/%d_clusters_%s/clusters.txt' % (n_clusters, tag), lbls, ids, fmt='%d')
                    # if not os.path.exists(path + 'dbscan/%d_clusters_%s/mean_dist/' % (n_clusters, tag)):
                    #     os.makedirs(path + 'dbscan/%d_clusters_%s/mean_dist/' % (n_clusters, tag))
                    all_inter, all_intra, db_pvals = inter_intra_dist(sqdm, lbls,
                                                                      out_path=path + 'dbscan/%d_clusters_%s/mean_dist/' %
                                                                               (n_clusters, tag),
                                                                      op='mean', plot=None)
                    get_cstats(f, path + 'dbscan/%d_clusters_%s/' % (n_clusters, tag), meta_grp=meta_grp, ids=ids)
                    log = open(path + 'dbscan/%d_clusters_%s/cluster_settings.txt' % (n_clusters, tag), 'w')
                    log.write('DBSCAN Epsilon:\t\t%.4f\n' % eps)
                    log.write('NormalTest p-thresh:\t%.2E\n' % p_thresh)
                    log.write('Ward Height Lim:\t%d\n' % height_lim)
                    log.write('Min Cluster Size:\t%d\n' % min_size)
                    log.close()
                    #
                    # arr2csv(path + 'dbscan/%d_clusters_%s/mean_dist/all_intra.txt' % (n_clusters, tag), all_intra,
                    #         ids,
                    #         fmt='%.3f')
                    # arr2csv(path + 'dbscan/%d_clusters_%s/mean_dist/all_inter.txt' % (n_clusters, tag), all_inter,
                    #         ids,
                    #         fmt='%.3f')
    if db_pvals is None:
        all_inter, all_intra, db_pvals = inter_intra_dist(sqdm, lbls, out_path='', op='mean', plot='none')
    db_het = np.zeros(len(db_pvals))
    lbl_names = np.unique(lbls)
    for i in range(len(lbl_names)):
        tlbl = lbl_names[i]
        sel = np.where(lbls == tlbl)[0]
        mks = mk[sel]
        tk, nm = mode(mks)
        db_het[i] = float(nm) / len(mks)
    max_size = int(max_size_pct * len(ids))
    temp = lbls
    cont = True
    while cont:
        if interactive:
            try:
                p_thresh = input('Enter p-value threshold (current is %.2E):' % p_thresh)
            except SyntaxError:
                pass
            try:
                min_size = input('Enter minimum size (current is %d):' % min_size)
            except SyntaxError:
                pass
            try:
                height_lim = input('Enter height limit (current is %d):' % height_lim)
            except SyntaxError:
                pass
            try:
                hom_thresh = input('Enter max KDIGO homogeneity threshold (as fractional %%... i.e. 1.0 means uniform\ncurrent is %.3f):' % hom_thresh)
            except SyntaxError:
                pass
            try:
                max_size_pct = input('Enter maximum cluster size (as fractional %%, current is %.3f, <0 to quit):' % max_size_pct)
                if max_size_pct < 0:
                    cont = False
            except SyntaxError:
                pass
            max_size = int(max_size_pct * len(ids))
        lbls = np.array(temp).astype(str)
        lbl_names = np.unique(lbls)
        for i in range(len(lbl_names)):
            p_val = db_pvals[i]
            het = db_het[i]
            tlbl = lbl_names[i]
            if tlbl == '-1':
                continue
            idx = np.where(lbls == tlbl)[0]
            tsize = len(idx)
            if p_val < p_thresh or het < hom_thresh or tsize > max_size:
                sel = np.ix_(idx, idx)
                tsqdm = sqdm[sel]
                tdm = squareform(tsqdm)
                link = fc.ward(tdm)
                root = to_tree(link)
                tlbls = lbls[idx]
                nlbls = dist_cut_tree(root, tlbls, tlbl, tsqdm, p_thresh, mk[idx], min_size=min_size, height_lim=height_lim, hom_thresh=hom_thresh, max_size=max_size)
                lbls[idx] = nlbls
        n_clusters = len(np.unique(lbls))
        print('Final number of clusters: %d' % n_clusters)
        if max_clust:
            if n_clusters > max_clust:
                continue
                f.close()
                return lbls, eps, p_thresh, height_lim, max_size_pct, hom_thresh, min_size
        if save:
            if not os.path.exists(path + 'composite/%d_clusters' % n_clusters):
                os.makedirs(path + 'composite/%d_clusters' % n_clusters)
                n_clusters = len(np.unique(lbls))
                arr2csv(path + 'composite/%d_clusters/clusters.txt' % n_clusters, lbls, ids, fmt='%s')
                get_cstats(f, path + 'composite/%d_clusters/' % n_clusters, meta_grp=meta_grp, ids=ids)
                log = open(path + 'composite/%d_clusters/cluster_settings.txt' % n_clusters, 'w')
                log.write('DBSCAN Epsilon:\t\t%.4f\n' % eps)
                log.write('NormalTest p-thresh:\t%.2E\n' % p_thresh)
                log.write('Ward Height Lim:\t%d\n' % height_lim)
                log.write('Min Cluster Size:\t%d\n' % min_size)
                log.write('Homogeneity threshold:\t%.2f\n' % hom_thresh)
                log.write('Maximum cluster size (as fractional %% of total size):\t%.2f\n' % max_size_pct)
                log.close()

            else:
                ref = load_csv(path + 'composite/%d_clusters/clusters.txt' % n_clusters, ids, str)
                if np.all(ref == lbls):
                    cont = False
                else:
                    cont = True
                    tag = 'a'
                while cont:
                    if os.path.exists(path + 'composite/%d_clusters_%s' % (n_clusters, tag)):
                        ref = load_csv(path + 'composite/%d_clusters_%s/clusters.txt'
                                       % (n_clusters, tag), ids, str)
                        if np.all(ref == lbls):
                            cont = False
                        else:
                            cont = True
                            tag = chr(ord(tag) + 1)
                    else:
                        os.makedirs(path + 'composite/%d_clusters_%s' % (n_clusters, tag))
                        cont = False
                        n_clusters = len(np.unique(lbls))
                        arr2csv(path + 'composite/%d_clusters_%s/clusters.txt' % (n_clusters, tag), lbls, ids,
                                fmt='%s')
                        get_cstats(f, path + 'composite/%d_clusters_%s/' % (n_clusters, tag), meta_grp=meta_grp, ids=ids)
                        log = open(path + 'composite/%d_clusters_%s/cluster_settings.txt' % (n_clusters, tag), 'w')
                        log.write('DBSCAN Epsilon:\t\t%.4f\n' % eps)
                        log.write('NormalTest p-thresh:\t%.2E\n' % p_thresh)
                        log.write('Ward Height Lim:\t%d\n' % height_lim)
                        log.write('Min Cluster Size:\t%d\n' % min_size)
                        log.write('Homogeneity threshold:\t%.2f\n' % hom_thresh)
                        log.write('Maximum cluster size (as fractional %% of total size):\t%.2f\n' % max_size_pct)
                        log.close()
        if interactive:
            t = raw_input('Try a different configuration? (y/n)')
            if 'y' in t:
                cont = True
            else:
                cont = False
        else:
            cont = False
    f.close()
    return lbls, eps, p_thresh, height_lim, max_size_pct, hom_thresh, min_size


def dist_cut_tree(node, lbls, base_name, sqdm, p_thresh, max_kdigo, min_size=20, height_lim=5, hom_thresh=0.9, max_size=0.2):
    height = len(base_name.split('-'))
    if height > height_lim:
        print('Height limit reached for node: %s' % base_name)
        return lbls
    if type(max_size) == float:
        max_size = max_size * len(lbls)
    left = node.get_left()
    right = node.get_right()
    left_name = base_name + '-l'
    right_name = base_name + '-r'
    left_idx = left.pre_order()
    left_sel = np.ix_(left_idx, left_idx)
    left_intra = np.mean(sqdm[left_sel], axis=0)

    right_idx = right.pre_order()
    right_sel = np.ix_(right_idx, right_idx)
    right_intra = np.mean(sqdm[right_sel], axis=0)

    if len(left_idx) < min_size or len(right_idx) < min_size:
        print('Splitting current node creates children < min_size: %s' % base_name)
        return lbls
    lbls[left_idx] = base_name + '-l'
    lbls[right_idx] = base_name + '-r'

    _, left_p = normaltest(left_intra)
    left_mks = max_kdigo[left_idx]
    left_tk, left_nm = mode(left_mks)
    left_hom = float(left_nm) / len(left_mks)
    if left_p < p_thresh or left_hom < hom_thresh or len(left_idx) > max_size:
        print('Splitting node %s: p-value=%.2E\tsize=%d\thomogeneity=%.2f%%' % (
              left_name, left_p, len(left_idx), left_hom))
        lbls = dist_cut_tree(left, lbls, left_name, sqdm, p_thresh, max_kdigo, min_size=min_size, height_lim=height_lim, hom_thresh=hom_thresh, max_size=max_size)
    else:
            print('Node %s final: p-value=%.2E\tsize=%d\thomogeneity=%.2f%%' % (
                  left_name, left_p, len(left_idx), left_hom))

    _, right_p = normaltest(right_intra)
    right_mks = max_kdigo[right_idx]
    right_tk, right_nm = mode(right_mks)
    right_hom = float(right_nm) / len(right_mks)
    if right_p < p_thresh or right_hom < hom_thresh or len(right_idx) > max_size:
        print('Splitting node %s: p-value=%.2E\tsize=%d\thomogeneity=%.2f%%' % (
              right_name, right_p, len(right_idx), right_hom))
        lbls = dist_cut_tree(right, lbls, right_name, sqdm, p_thresh, max_kdigo, min_size=min_size, height_lim=height_lim, hom_thresh=hom_thresh, max_size=max_size)
    else:
        print('Node %s final: p-value=%.2E\tsize=%d\thomogeneity=%.2f%%' % (
              right_name, right_p, len(right_idx), right_hom))
    return lbls


def eval_clusters(data_file, cluster_method, cg_name, sel=None, labels_true=None):
    if type(data_file) == str:
        f = h5py.File(data_file, 'r')
    else:
        f = data_file
    sqdm = f['square'][:]
    if sel is not None:
        cidx = np.ix_(sel, sel)
        sqdm = sqdm[cidx]
    clusters = f['clusters'][cluster_method][cg_name]
    ss = silhouette_score(sqdm, clusters, metric='precomputed')
    chs = calinski_harabaz_score(sqdm, clusters)
    if labels_true is None:
        print('Silhouette Score: %.4f' % ss)
        print('Calinski Harabaz Score: %.4f' % chs)
        return ss, chs
    else:
        ars = adjusted_rand_score(clusters, labels_true)
        nmi = normalized_mutual_info_score(clusters, labels_true)
        hs = homogeneity_score(clusters, labels_true)
        cs = completeness_score(clusters, labels_true)
        vms = v_measure_score(clusters, labels_true)
        fms = fowlkes_mallows_score(clusters, labels_true)

        print('Silhouette Score: %.4f' % ss)
        print('Calinski Harabaz Score: %.4f' % chs)
        print('Adjusted Rand Index: %.4f' % ars)
        print('Normalize Mutual Information: %.4f' % nmi)
        print('Homogeneity Score: %.4f' % hs)
        print('Completeness Score: %.4f' % cs)
        print('V Measure Score: %.4f' % vms)
        print('Fowlkes-Mallows Score: %.4f' % fms)
        return ss, chs, ars, nmi, hs, cs, vms, fms


def inter_intra_dist(dm, lbls, op='mean', out_path='', plot='both'):
    if dm.ndim == 1:
        sqdm = squareform(dm)
    else:
        sqdm = np.array(dm, copy=True)

    np.fill_diagonal(sqdm, np.nan)

    lbl_names = np.unique(lbls)
    n_clusters = len(lbl_names)

    if op == 'mean':
        func = lambda x: np.nanmean(x)
    elif op == 'max':
        func = lambda x: np.nanmax(x)
    elif op == 'min':
        func = lambda x: np.nanmin(x)

    all_inter = np.zeros(len(lbls))
    all_intra = np.zeros(len(lbls))
    for i in range(len(lbls)):
        clust = lbls[i]
        idx = np.array((i,))
        cids = np.where(lbls == clust)[0]
        intra = np.ix_(idx, np.setdiff1d(cids, idx))
        inter = np.ix_(idx, np.setdiff1d(np.arange(len(lbls)), cids))
        all_intra[i] = func(sqdm[intra])
        all_inter[i] = func(sqdm[inter])

    print('Normal tests:')
    p_vals = np.zeros(n_clusters)
    for i in range(n_clusters):
        clust = lbl_names[i]
        cidx = np.where(lbls == clust)[0]
        if len(cidx) < 8:
            print('Cluster ' + str(clust) + ':\tLess Than 8 Samples')
        else:
            _, p = normaltest(all_intra[cidx].flatten())
            p_vals[i] = p
            print('Cluster ' + str(clust) + ':\t' + str(p))
        plt.figure()
        if plot == 'both':
            plt.subplot(121)
        if plot == 'both' or plot == 'inter':
            plt.hist(all_inter[cidx], bins=50)
            plt.xlim((0, 1))
            plt.title('Inter-Cluster')
        if plot == 'both':
            plt.subplot(122)
        if plot == 'both' or plot == 'intra':
            plt.hist(all_intra[cidx], bins=50)
            plt.xlim((0, 1))
            plt.title('Intra-Cluster')
        if plot == 'nosave':
            plt.subplot(121)
            plt.hist(all_inter[cidx], bins=50)
            plt.xlim((0, 1))
            plt.title('Inter-Cluster')
            plt.subplot(122)
            plt.hist(all_intra[cidx], bins=50)
            plt.xlim((0, 1))
            plt.title('Intra-Cluster')
            plt.suptitle('Cluster ' + str(clust) + ' Separation')
            plt.show()
        if plot == 'both':
            if clust >= 0:
                plt.suptitle('Cluster ' + str(clust) + ' Separation')
                plt.savefig(out_path+'cluster' + str(clust) + '_separation_hist.png')
            else:
                plt.suptitle(out_path+'Noise Point Separation')
                plt.savefig(out_path+'noise_separation_hist.png')
        elif plot == 'inter':
            plt.savefig(out_path + 'cluster' + str(clust) + '_inter_dist_hist.png')
        elif plot == 'intra':
            plt.savefig(out_path + 'cluster' + str(clust) + '_intra_dist_hist.png')
        plt.close()

    return all_inter, all_intra, p_vals


def assign_feature_vectors(lbls, reference_vectors):
    n_pts = len(lbls)
    all_lbls = np.unique(lbls)
    n_feats = reference_vectors.shape[1]
    features = np.zeros((n_pts, n_feats))
    for i in range(len(all_lbls)):
        idx = np.where(lbls == all_lbls[i])[0]
        features[idx, :] = reference_vectors[i, :]
    return features


def dm_to_sim(dist_file, out_name, beta=1, eps=1e-6):
    dm = np.loadtxt(dist_file, delimiter=',')
    dm_std = np.std(dm[:, 2])
    out = open(out_name, 'w')
    for i in range(np.shape(dm)[0]):
        sim = np.exp(-beta * dm[i, 2] / dm_std) + eps
        out.write('%d,%d,%.4f\n' % (dm[i, 0], dm[i, 1], sim))
    out.close()


def mean_confidence_interval(data, confidence=0.95):
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


def plot_daily_kdigos(datapath, ids, h5_name, sqdm, lbls, outpath='', max_day=7, cutoff=None):
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
    str_admits = load_csv(os.path.join(datapath, 'patient_summary.csv'), ids, dt=str, sel=1, skip_header=True)
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

    blank_daily = np.repeat(np.nan, max_day + 2)
    all_daily = np.vstack([blank_daily for x in range(len(ids))])
    for i in range(len(ids)):
        l = np.min([len(x) for x in [scrs[i], dates[i], dmasks[i]]])
        if l < 2:
            continue
        # tmax = daily_max_kdigo(scrs[i][:l], dates[i][:l], bslns[i], admits[i], dmasks[i][:l], tlim=max_day)
        # tmax = daily_max_kdigo(scrs[i][:l], days[i][:l], bslns[i], dmasks[i][:l], tlim=max_day)
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
        f = h5py.File(h5_name, 'r')
        all_ids = f['meta']['ids'][:]
        all_inp_death = f['meta']['died_inp'][:]
        sel = np.array([x in ids for x in all_ids])
        inp_death = all_inp_death[sel]
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
            plt.savefig(os.path.join(outpath, 'mean_conf', '%s_mean_conf.png' % c_lbls[i]))
            plt.close(fig)
        f.close()
    return all_daily


def return_centers(ids, dm_f, lbl_f):
    dm = np.load(dm_f)
    sqdm = squareform(dm)
    lbls = load_csv(lbl_f, ids, str)
    lbl_names = np.unique(lbls)
    centers = np.zeros(len(lbl_names), dtype=int)
    for i in range(len(lbl_names)):
        tlbl = lbl_names[i]
        idx = np.where(lbls == tlbl)[0]
        sel = np.ix_(idx, idx)
        tdm = sqdm[sel]
        all_intra = np.sum(tdm, axis=0)
        cid = np.argsort(all_intra)[0]
        centers[i] = ids[idx[cid]]
    return centers, lbl_names


from dtw import dtw
from scipy.spatial import distance
def find_from_centers(centers, lbl_names, kdigos):
    lbls = np.zeros(len(kdigos), dtype=lbl_names.dtype)
    for i in range(len(kdigos)):
        p1 = kdigos[i]
        min_dist = None
        center_id = None
        for j in range(len(centers)):
            p2 = centers[j]
            _, _, _, path = dtw(p1, p2, lambda y, yy: np.abs(y - yy))
            if len(p1) > 1 and len(p2) > 1:
                dist, _, _, path = dtw(p1, p2, lambda y, yy: np.abs(y - yy))
                p1_path = path[0]
                p2_path = path[1]
                p1_m = [p1[p1_path[x]] for x in range(len(p1_path))]
                p2_m = [p2[p2_path[x]] for x in range(len(p2_path))]
            elif len(p1) == 1:
                p1_m = np.repeat(p1[0], len(p2))
                p2_m = p2
            elif len(p2) == 1:
                p1_m = p1
                p2_m = np.repeat(p2[0], len(p1))
            if np.all(p1_m == p2_m):
                dist = 0
            else:
                dist = distance.braycurtis(p1_m, p2_m)
            if min_dist is None:
                min_dist = dist
                center_id = j
            else:
                if dist < min_dist:
                    min_dist = dist
                    center_id = j
        lbls[i] = lbl_names[center_id]
    return lbls


