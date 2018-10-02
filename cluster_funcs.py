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
def cluster_trajectories(f, ids, mk, dm, meta_grp='meta', eps=0.015, p_thresh=0.05, min_size=20, hom_thresh=0.9,
                         max_size_pct=0.2, n_clusters=2, height_lim=10, hmethod='dynamic', data_path=None,
                         interactive=True, save=False):
    '''
    Provided a pre-computed distance matrix, cluster the corresponding trajectories using the specified methods.
    First applies DBSCAN to identify and remove any patients designated as noise. Next applies a dynamic hierarchical
    clustering algorithm (see function dynamic_tree_cut below for more details).

    :param stat_file: hdf5 file containing groups of metadata corresponding to different subsets of the full data
    :param dm: pre-computed distance matrix (can be condensed or square_
    :param meta_grp: name of the group in the hdf5 file corresponding to the desired subset
    :param eps: epsilon threshold for DBSCAN clustering
    :param p_thresh: p-value threshold for normaltest to split a cluster
    :param min_size: minimum cluster size allowed... if any DBSCAN clusters are smaller than min_size, designate as
                     noise, while for hierarchical clustering, do not split node if either child is smaller than
                     min_size
    :param hom_thresh: homogeneity threshold for hierarchical clustering (see dynamic_tree_cut for more)
    :param max_size_pct: maximum cluster size as percentage of total subset size
    :param hmethod: hierarchical clustering method... 'dynamic' calls dynamic algorithm, 'flat' applies flat
                    clustering with a uniform distance threshold
    :param interactive: T/F indicates whether the algorithm will prompt user for input
    :param save: None or base-path for saving output
    :return:
    '''

    # Ensure that we have both the square distance matrix for DBSCAN and the condensed matrix for hierarchical
    if dm.ndim == 1:
        # Provided dm is condensed
        sqdm = squareform(dm)
    else:
        # Provided dm is square
        sqdm = dm
        dm = squareform(dm)

    # Initialize DBSCAN model
    db = DBSCAN(eps=eps, metric='precomputed', n_jobs=-1)
    cont = True
    while cont:
        db.fit(sqdm)
        lbls = np.array(db.labels_, dtype=int)
        # DBSCAN labels start at 0... increment all but noise (-1)
        lbls[np.where(lbls >= 0)] += 1
        lbl_names = np.unique(lbls)
        # Don't count the noise in the number of clusters
        nclust = len(lbl_names) - 1

        # # If any DBSCAN clusters are smaller than min_size, designate as noise
        # for i in range(1, nclust + 1):
        #     tlbl = lbl_names[i]
        #     idx = np.where(lbls == tlbl)[0]
        #     if len(idx) < min_size:
        #         lbls[idx] = -1

        lbl_names = np.unique(lbls)
        nclust = len(lbl_names) - 1
        print('Number of DBSCAN Clusters = %d' % nclust)

        # If interactive let the user provide a new epsilon
        if interactive:
            for i in range(len(lbl_names)):
                print('Cluster %d: %d members' % (lbl_names[i], len(np.where(lbls == lbl_names[i])[0])))
            eps_str = raw_input('New epsilon (non-numeric to continue): ')
            # If numeric, recompute, otherwise continue
            try:
                eps = float(eps_str)
                db = DBSCAN(eps=eps, metric='precomputed', n_jobs=-1)
            except ValueError:
                cont = False
        else:
            cont = False
    out_lbls = []
    n_clust_str = '%d_clusters' % nclust
    if save:
        tag = ''
        if not os.path.exists(os.path.join(save, 'dbscan')):
            os.mkdir(os.path.join(save, 'dbscan'))
        if not os.path.exists(os.path.join(save, 'dbscan', n_clust_str)):
            os.mkdir(os.path.join(save, 'dbscan', n_clust_str))
            exists = False
        else:
            # check to see if labels are the same, otherwise generate new folder
            old_lbls = load_csv(os.path.join(save, 'dbscan', n_clust_str + tag, 'clusters.csv'), ids, int)
            if np.all(old_lbls == lbls):
                exists = True
            else:
                exists = False
                tag = '_a'
                while os.path.exists(os.path.join(save, 'dbscan', n_clust_str + tag)) and not exists:
                    tag = '_' + chr(ord(tag[1]) + 1)
                    old_lbls = load_csv(os.path.join(save, 'dbscan', n_clust_str + tag, 'clusters.csv'), ids, int)
                    if np.all(old_lbls == lbls):
                        exists = True
                if not exists:
                    os.mkdir(os.path.join(save, 'dbscan', n_clust_str + tag))
        if not exists:
            arr2csv(os.path.join(save, 'dbscan', n_clust_str + tag, 'clusters.csv'), lbls, ids, fmt='%d')
            save_path = os.path.join(save, 'dbscan', n_clust_str + tag)
            get_cstats(f, save_path, meta_grp=meta_grp, ids=ids)
            setting_f = open(os.path.join(save_path, 'settings.csv'), 'w')
            setting_f.write('DBSCAN\nepsilon = %.4E\n' % eps)
            setting_f.close()
            dkpath = os.path.join(save_path, 'daily_kdigo')
            os.mkdir(dkpath)
            plot_daily_kdigos(data_path, ids, f, sqdm, lbls, outpath=dkpath)

    if type(hmethod) == str:
        hmethods = [hmethod, ]
    else:
        hmethods = hmethod

    for hmethod in hmethods:
        method_lbls = lbls.astype('|S30')
        # Run dynamic tree cut on each of the non-noise clusters
        for tlbl in lbl_names:
            if tlbl == -1:
                continue
            pt_sel = np.where(lbls == tlbl)[0]
            sq_sel = np.ix_(pt_sel, pt_sel)
            ids_sel = ids[pt_sel]
            mk_sel = mk[pt_sel]

            sqdm_sel = sqdm[sq_sel]
            dm_sel = squareform(sqdm_sel)

            # Get linkage matrix for remaining patients using Ward's method
            link = fc.ward(dm_sel)

            # Generate corresponding dendrogram
            fig = plt.figure()
            dendrogram(link, 50, truncate_mode='lastp')
            plt.xlabel('Patients')
            plt.ylabel('Distance')
            plt.title('DBSCAN Cluster %d\n%d Patients' % (tlbl, len(pt_sel)))
            plt.show()
            if save:
                if not os.path.exists(os.path.join(save, hmethod)):
                    os.mkdir(os.path.join(save, hmethod))
                if not os.path.exists(os.path.join(save, hmethod, '%d_dbscan_clusters' % nclust)):
                    os.mkdir(os.path.join(save, hmethod, '%d_dbscan_clusters' % nclust))
                if not os.path.exists(os.path.join(save, hmethod, '%d_dbscan_clusters' % nclust, 'dbscan_cluster%d.png' % tlbl)):
                    fig.savefig(os.path.join(save, hmethod, '%d_dbscan_clusters' % nclust, 'dbscan_cluster%d.png' % tlbl))
            plt.close(fig)

            if hmethod == 'dynamic':
                # Convert linkage matrix to tree structure similar to that shown in dendrogram
                tree = to_tree(link)
                cont = True

                while cont:
                    if interactive:
                        # Update all clustering parameters
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
                            hom_thresh = input(
                                'Enter max KDIGO homogeneity threshold (as fractional %%... i.e. 1.0 means uniform\ncurrent is %.3f):' % hom_thresh)
                        except SyntaxError:
                            pass
                        try:
                            max_size_pct = input(
                                'Enter maximum cluster size (as fractional %%, current is %.3f, <0 to quit):' % max_size_pct)
                            if max_size_pct < 0:
                                cont = False
                        except SyntaxError:
                            pass

                    lbls_sel = dynamic_tree_cut(tree, sqdm_sel, ids_sel, mk_sel, p_thresh, min_size, hom_thresh,
                                                max_size_pct, height_lim, base_name='%d' % tlbl)
                    lbl_names_sel = np.unique(lbls_sel)
                    n_clusters = len(lbl_names_sel)
                    print('DBSCAN Cluster %d split into %d clusters' % (tlbl, n_clusters))
                    if save:
                        save_path = os.path.join(save, hmethod, '%d_dbscan_clusters' % nclust, 'cluster_%d' % tlbl)
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        save_path = os.path.join(save_path, '%d_clusters' % n_clusters)
                        tag = None
                        exists = False
                        while os.path.exists(save_path) and not exists:
                            old_lbls = load_csv(os.path.join(save_path, 'clusters.csv'), ids_sel, str)
                            if np.all(lbls_sel == old_lbls):
                                exists = True
                            if not exists:
                                if tag is None:
                                    tag = '_a'
                                else:
                                    tag = '_' + chr(ord(tag[1]) + 1)
                                save_path = os.path.join(save, hmethod, '%d_dbscan_clusters' % nclust,
                                                         'cluster_%d' % tlbl, '%d_clusters%s' % (n_clusters, tag))
                        if not exists:
                            os.mkdir(save_path)
                            arr2csv(os.path.join(save_path, 'clusters.csv'), lbls_sel, ids_sel, fmt='%s')
                            get_cstats(f, save_path, meta_grp=meta_grp, ids=ids_sel)
                            dkpath = os.path.join(save_path, 'daily_kdigo')
                            os.mkdir(dkpath)
                            plot_daily_kdigos(data_path, ids_sel, f, sqdm, lbls_sel, outpath=dkpath)
                            setting_f = open(os.path.join(save_path, 'settings.csv'), 'w')
                            setting_f.write('NormalTest p-val threshold: %.4E\n' % p_thresh)
                            setting_f.write('Minimum size: %d\n' % min_size)
                            setting_f.write('Height limit: %d\n' % height_lim)
                            setting_f.write('Max KDIGO Homogeneity threshold: %.3F\n' % hom_thresh)
                            setting_f.write('Maximum cluster size: %.3F (%d total)\n' %
                                            (max_size_pct, int(max_size_pct * len(ids_sel))))
                            setting_f.close()

                    if interactive:
                        t = raw_input('Try a different configuration? (y/n)')
                        if 'y' in t:
                            cont = True
                        else:
                            cont = False
                    else:
                        cont = False
                method_lbls[pt_sel] = lbls_sel
            else:  # flat clustering with Ward's method
                cont = True
                while cont:
                    if interactive:
                        try:
                            n_clusters = input('Enter desired number of clusters (current is %d):' % n_clusters)
                        except SyntaxError:
                            pass
                    lbls_sel = fcluster(link, n_clusters, criterion='maxclust')
                    tlbls = lbls_sel.astype('|S30')
                    for i in range(len(tlbls)):
                        tlbls[i] = '%d-%s' % (tlbl, tlbls[i])
                    if save:
                        save_path = os.path.join(save, hmethod, '%d_dbscan_clusters' % nclust, 'cluster_%d' % tlbl)
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        save_path = os.path.join(save_path, '%d_clusters' % n_clusters)
                        tag = None
                        if os.path.exists(save_path):
                            print('%d clusters has already been saved' % n_clusters)
                            if not interactive:
                                cont = False
                            continue
                        os.mkdir(save_path)
                        arr2csv(os.path.join(save_path, 'clusters.csv'), lbls_sel, ids_sel, fmt='%d')
                        get_cstats(f, save_path, meta_grp=meta_grp, ids=ids_sel)
                        if data_path is not None:
                            dkpath = os.path.join(save_path, 'daily_kdigo')
                            os.mkdir(dkpath)
                            plot_daily_kdigos(data_path, ids_sel, f, sqdm, lbls_sel, outpath=dkpath)
                    if interactive:
                        t = raw_input('Try a different configuration? (y/n)')
                        if 'y' in t:
                            cont = True
                        else:
                            cont = False
                    else:
                        cont = False
                method_lbls[pt_sel] = tlbls
        out_lbls.append(method_lbls)
    return out_lbls


# %%
def dynamic_tree_cut(node, sqdm, ids, mk, p_thresh=0.05, min_size=20, hom_thresh=0.9, max_size_pct=0.2, height_lim=None,
                     base_name='1', lbls=None, v=False):
    '''
    Iterative clustering algorithm that utilizes the tree structure derived from the linkage matrix using Ward's method
    along with additional statistical and logical tests to determine where to cut the tree.

    :param node: tree structure corresponding to the current node and it's children (initially root)
    :param sqdm: square distance matrix
    :param ids: all corresponding patient IDs in order
    :param mk: maximum KDIGO scores in the analysis window for patients 'ids'
    :param p_thresh: p-value threshold for splitting current node based on normaltest
    :param min_size: minimum cluster size... will not split any nodes whose children have < min_size leaves
    :param hom_thresh: homogeneity threshold... if < hom_thresh% of patients belonging to the node have the same max
                       KDIGO as the mode of the patients in this node, split it
    :param max_size_pct: if a node size is > max_size_pct of the full cohort size, split it
    :param base_name: name of the current node (children will be base_name-l and base_name-r)
    :param lbls: individual cluster labels for each patient. Only used by iterative calls... user should not specify
    :param v: verbose... whether or not to print extra information
    :return:
    '''
    if lbls is None:
        lbls = np.repeat(base_name, len(ids)).astype('|S30')

    max_size = max_size_pct * len(lbls)
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
        if v:
            print('Splitting current node creates children < min_size: %s' % base_name)
        return lbls

    if height_lim is not None:
        current_height = len(base_name.split('-'))
        if current_height >= height_lim:
            if v:
                print('Current height is %d, >= height_lim %d: %s' % (current_height, height_lim, base_name))
            return lbls


    lbls[left_idx] = left_name
    lbls[right_idx] = right_name

    _, left_p = normaltest(left_intra)
    left_mks = mk[left_idx]
    left_tk, left_nm = mode(left_mks)
    left_hom = float(left_nm) / len(left_mks)
    if left_p < p_thresh or left_hom < hom_thresh or len(left_idx) > max_size:
        if v:
            print('Splitting node %s: p-value=%.2E\tsize=%d\thomogeneity=%.2f%%' % (
                  left_name, left_p, len(left_idx), left_hom))

        lbls = dynamic_tree_cut(left, sqdm, ids, mk, p_thresh, min_size, hom_thresh,
                                max_size_pct, height_lim, left_name, lbls)
    else:
        if v:
            print('Node %s final: p-value=%.2E\tsize=%d\thomogeneity=%.2f%%' % (
                  left_name, left_p, len(left_idx), left_hom))

    _, right_p = normaltest(right_intra)
    right_mks = mk[right_idx]
    right_tk, right_nm = mode(right_mks)
    right_hom = float(right_nm) / len(right_mks)
    if right_p < p_thresh or right_hom < hom_thresh or len(right_idx) > max_size:
        if v:
            print('Splitting node %s: p-value=%.2E\tsize=%d\thomogeneity=%.2f%%' % (
                  right_name, right_p, len(right_idx), right_hom))
        lbls = dynamic_tree_cut(right, sqdm, ids, mk, p_thresh, min_size, hom_thresh,
                                max_size_pct, height_lim, right_name, lbls)
    else:
        if v:
            print('Node %s final: p-value=%.2E\tsize=%d\thomogeneity=%.2f%%' % (
                  right_name, right_p, len(right_idx), right_hom))

    return lbls


# %%
def eval_clusters(dm, ids, label_path, labels_true=None):
    '''
    Evaluate clustering performance. If no true labels provided, will only evaluate using Silhouette Score and
    Calinski Harabaz score.
    :param data_file: filename or file handle for hdf5 file containing patient statistics
    :param label_path: fully qualified path to directory containing cluster labels
    :param sel:
    :param labels_true:
    :return:
    '''
    if dm.ndim == 1:
        sqdm = squareform(dm)
    else:
        sqdm = dm
    clusters = load_csv(os.path.join(label_path, 'clusters.csv'), ids, str)
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
    '''
    Returns the distance within each cluster (intra) and between clusters (inter), aggregated by the function op, in
    addition to evaluating whether the aggregate intra-distance reflects a normal distribution, as determined by the
    function normaltest from scipy.stats. For each cluster, returns the aggregate intra-cluster distance, inter-cluster
    distance, and the p-value of the normaltest.
    Note: low p-value indicates that the distribution is NOT normal, where high p-value indicates it IS normal.
    :param dm:
    :param lbls:
    :param op:
    :param out_path:
    :param plot:
    :return:
    '''
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
    '''
    Assign individual reference vectors to each label and return the full list
    :param lbls: list of N individual labels
    :param reference_vectors: matrix of dimension M x p, where p is the length of the reference vectors
    :return features: a matrix of dimension N x p
    '''
    n_pts = len(lbls)
    all_lbls = np.unique(lbls)
    n_feats = reference_vectors.shape[1]
    features = np.zeros((n_pts, n_feats))
    for i in range(len(all_lbls)):
        idx = np.where(lbls == all_lbls[i])[0]
        features[idx, :] = reference_vectors[i, :]
    return features


def dm_to_sim(dm, beta=1):
    '''
    Convert distance to similarity by applying the Gaussian kernel
    :param dm: vector of N distances to convert
    :return sim: vector of N similarities
    '''
    if dm.ndim == 2:
        dm = squareform(dm)
    dm_std = np.std(dm)
    sim = np.exp(-beta * np.power(dm, 2) / np.power(dm_std, 2))
    return sim


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


def plot_daily_kdigos(datapath, ids, stat_file, sqdm, lbls, outpath='', max_day=7, cutoff=None):
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
        f = stat_file
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
        # f.close()
    return all_daily


def return_centers(ids, dm_f, lbl_f):
    '''
    Returns indices of of the patient closest to the cluster center, as defined by the minimum intra-cluster distance
    :param ids:
    :param dm_f:
    :param lbl_f:
    :return:
    '''
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
