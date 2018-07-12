"""
Created on Fri Jan 19 14:31:03 2018

@author: taylorsmith
"""
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, fcluster, to_tree
from scipy.stats.mstats import normaltest
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score,\
    v_measure_score, fowlkes_mallows_score, silhouette_score, calinski_harabaz_score
import fastcluster as fc
from kdigo_funcs import arr2csv, load_csv, daily_max_kdigo
import datetime
from stat_funcs import get_cstats
import matplotlib.pyplot as plt
import h5py
import os


# %%
def dist_cut_cluster(h5_fname, dm, meta_grp='meta', path='', eps=0.015, p_thresh=0.05,
                     min_size=20, height_lim=5, interactive=True, save=True):
    f = h5py.File(h5_fname, 'r')
    if type(dm) == str:
        dm = f[dm]
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
    lbls[np.where(lbls >= 0)] += 1
    lbl_names = np.unique(lbls)
    for i in range(1, nclust + 1):
        tlbl = lbl_names[i]
        idx = np.where(lbls == tlbl)[0]
        if len(idx) < min_size:
            lbls[idx] = -1
    n_clusters = len(np.unique(lbls))
    if save:
        if not os.path.exists(path + 'dbscan'):
            os.makedirs(path + 'dbscan')
        if not os.path.exists(path + 'dbscan/%d_clusters-%s' % (n_clusters, date_str)):
            os.makedirs(path + 'dbscan/%d_clusters-%s' % (n_clusters, date_str))
        np.savetxt(path + 'dbscan/%d_clusters-%s/clusters.txt' % (n_clusters, date_str), lbls, fmt='%s')
        if not os.path.exists(path + 'dbscan/%d_clusters-%s/max_dist/' % (n_clusters, date_str)):
            os.makedirs(path + 'dbscan/%d_clusters-%s/max_dist/' % (n_clusters, date_str))
        all_inter, all_intra, db_pvals = inter_intra_dist(sqdm, lbls,
                                                          out_path=path + 'dbscan/%d_clusters-%s/max_dist/' % (
                                                          n_clusters, date_str),
                                                          op='max', plot='both')
    else:
        all_inter, all_intra, db_pvals = inter_intra_dist(sqdm, lbls,
                                                          out_path=path + 'dbscan/%d_clusters-%s/max_dist/' % (n_clusters, date_str),
                                                          op='max', plot='nosave')
    if save:
        np.savetxt(path + 'dbscan/%d_clusters-%s/max_dist/all_intra.txt' % (n_clusters, date_str), all_intra, fmt='%.3f')
        np.savetxt(path + 'dbscan/%d_clusters-%s/max_dist/all_inter.txt' % (n_clusters, date_str), all_inter, fmt='%.3f')
    lbls = np.array(lbls)
    lbl_names = np.unique(lbls).astype(str)
    lbls = lbls.astype(str)
    for i in range(len(lbl_names)):
        p_val = db_pvals[i]
        if p_val < p_thresh:
            tlbl = lbl_names[i]
            if tlbl == '-1':
                continue
            idx = np.where(lbls == tlbl)[0]
            sel = np.ix_(idx, idx)
            tdm = squareform(sqdm[sel])
            link = fc.ward(tdm)
            root = to_tree(link)
            tlbls = lbls[idx]
            nlbls = dist_cut_tree(root, tlbls, tlbl, all_intra[idx], p_thresh, min_size=min_size, height_lim=height_lim)
            lbls[idx] = nlbls
    n_clusters = len(np.unique(lbls))
    print('Final number of clusters: %d' % n_clusters)
    if save:
        if not os.path.exists(path + 'composite/%d_clusters-%s' % (n_clusters, date_str)):
            os.makedirs(path + 'composite/%d_clusters-%s' % (n_clusters, date_str))
        n_clusters = len(np.unique(lbls))
        np.savetxt(path + 'composite/%d_clusters-%s/clusters.txt' % (n_clusters, date_str), lbls, fmt='%s')
        get_cstats(f, path + 'composite/%d_clusters-%s/' % (n_clusters, date_str), meta_grp=meta_grp)
    return lbls


def dist_cut_tree(node, lbls, base_name, feat, p_thresh, min_size=20, height_lim=5):
    height = len(base_name.split('-'))
    if height > height_lim:
        print('Height limit reached for node: %s' % base_name)
        return lbls
    left = node.get_left()
    right = node.get_right()
    left_name = base_name + '-l'
    right_name = base_name + '-r'
    left_idx = left.pre_order()
    right_idx = right.pre_order()
    for idx in left_idx:
        lbls[left_idx] = base_name + '-l'
    for idx in right_idx:
        lbls[right_idx] = base_name + '-r'
    if len(left_idx) < min_size:
        print('Node %s minimum size' % left_name)
    else:
        _, left_p = normaltest(feat[left_idx])
        if left_p < p_thresh:
            print('Splitting node %s: p-value=%.2E' % (left_name, left_p))
            lbls = dist_cut_tree(left, lbls, left_name, feat, p_thresh, min_size=min_size, height_lim=height_lim)
        else:
            print('Node %s final: p-value=%.2E' % (left_name, left_p))

    if len(right_idx) < min_size:
        print('Node %s minimum size' % right_name)
    else:
        _, right_p = normaltest(feat[right_idx])
        if right_p < p_thresh:
            print('Splitting node %s: p-value=%.2E' % (right_name, right_p))
            lbls = dist_cut_tree(right, lbls, right_name, feat, p_thresh, min_size=min_size, height_lim=height_lim)
        else:
            print('Node %s final: p-value=%.2E' % (right_name, right_p))
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


def plot_cluster_centers(datapath, ids, sqdm, lbls, outpath=''):
    c_lbls = np.unique(lbls)
    n_clusters = len(c_lbls)
    if np.ndim(sqdm) == 1:
        sqdm = squareform(sqdm)
    centers = np.zeros(n_clusters, dtype=int)
    for i in range(n_clusters):
        tlbl = c_lbls[i]
        idx = np.where(lbls == tlbl)[0]
        sel = np.ix_(idx, idx)
        tdm = sqdm[sel]
        mins = np.min(tdm, axis=0)
        center = np.argsort(mins)[0]
        centers[i] = ids[idx[center]]

    _, scrs = load_csv(datapath + 'scr_raw.csv', centers)
    _, bslns = load_csv(datapath + 'baselines.csv', centers)
    _, dmasks = load_csv(datapath + 'dialysis.csv', centers, dt=int)
    _, str_admits = load_csv(datapath + 'patient_summary.csv', centers, dt=str, sel=1, skip_header=True)
    admits = []
    for i in range(n_clusters):
        admits.append(datetime.datetime.strptime('%s' % str_admits[i], '%Y-%m-%d %H:%M:%S'))

    _, str_dates = load_csv(datapath + 'dates.csv', centers, dt=str)
    for i in range(n_clusters):
        for j in range(len(str_dates[i])):
            str_dates[i][j] = str_dates[i][j].split('\'')[1].split('.')[0]
    dates = []
    for i in range(n_clusters):
        temp = []
        for j in range(len(str_dates[i])):
            temp.append(datetime.datetime.strptime('%s' % str_dates[i][j], '%Y-%m-%d %H:%M:%S'))
        dates.append(temp)

    daily_max = []
    for i in range(n_clusters):
        dmax = daily_max_kdigo(scrs[i], dates[i], bslns[i], admits[i], dmasks[i])
        daily_max.append(dmax)
        if outpath != '':
            tfig = plt.figure()
            tplot = tfig.add_subplot(111)
            tplot.plot(range(len(dmax)), dmax)
            tplot.set_xlabel('Day')
            tplot.set_ylabel('KDIGO Score')
            tplot.set_title('Cluster %s Representative' % c_lbls[i])
            plt.savefig(outpath + '%s.png' % c_lbls[i])

    return daily_max

