"""
Created on Fri Jan 19 14:31:03 2018

@author: taylorsmith
"""
import numpy as np
from scipy.spatial.distance import squareform, pdist, braycurtis
from scipy.cluster.hierarchy import dendrogram, fcluster, to_tree
from scipy.stats.mstats import normaltest
from scipy.stats import mode, sem, t, ttest_ind
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score,\
    v_measure_score, fowlkes_mallows_score, silhouette_score, calinski_harabaz_score
from sklearn.preprocessing import MinMaxScaler
import fastcluster as fc
from kdigo_funcs import arr2csv, load_csv, daily_max_kdigo, daily_max_kdigo_interp, get_date, dtw_p
import datetime
from stat_funcs import get_cstats, formatted_stats
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import os

# %%
def cluster_trajectories(f, ids, mk, dm, meta_grp='meta', eps=0.015, p_thresh_l=[], min_size_l=[], hom_thresh_l=[],
                         max_size_pct=0.2, n_clusters_l=[2, ], height_lim_l=[], data_path=None,
                         interactive=True, save=False, v=False, plot_daily=False, only_flat=False, skip_db=False):
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
    if not skip_db:
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

            # If any DBSCAN clusters are smaller than 10% of the population, designate as noise
            for i in range(1, nclust + 1):
                tlbl = lbl_names[i]
                idx = np.where(lbls == tlbl)[0]
                if len(idx) < (0.1 * len(lbls)):
                    lbls[idx] = -1

            lbl_names = np.unique(lbls)
            nclust = len(lbl_names) - 1
            print('Number of DBSCAN Clusters = %d' % nclust)

            # If interactive let the user provide a new epsilon
            if interactive:
                for i in range(len(lbl_names)):
                    print('Cluster %d: %d members' % (lbl_names[i], len(np.where(lbls == lbl_names[i])[0])))
                eps_str = raw_input('New epsilon (current: %.2E...non-numeric to continue): ' % eps)
                # If numeric, recompute, otherwise continue
                try:
                    eps = float(eps_str)
                    db = DBSCAN(eps=eps, metric='precomputed', n_jobs=-1)
                except ValueError:
                    cont = False
            else:
                cont = False

        interactive = False
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
                    else:
                        print('%d clusters has already been saved' % n_clust_str + tag)
            if not exists:
                arr2csv(os.path.join(save, 'dbscan', n_clust_str + tag, 'clusters.csv'), lbls, ids, fmt='%d')
                save_path = os.path.join(save, 'dbscan', n_clust_str + tag)
                # get_cstats(f, save_path, meta_grp=meta_grp)
                formatted_stats(f['meta'], save_path)
                setting_f = open(os.path.join(save_path, 'settings.csv'), 'w')
                setting_f.write('DBSCAN\nepsilon = %.4E\n' % eps)
                setting_f.close()
                dkpath = os.path.join(save_path, 'daily_kdigo')
                os.mkdir(dkpath)
                if plot_daily:
                    plot_daily_kdigos(data_path, ids, f, sqdm, lbls, outpath=dkpath)

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
            if save and not only_flat:
                if not os.path.exists(os.path.join(save, 'dynamic')):
                    os.mkdir(os.path.join(save, 'dynamic'))
                if not os.path.exists(os.path.join(save, 'dynamic', '%d_dbscan_clusters' % nclust)):
                    os.mkdir(os.path.join(save, 'dynamic', '%d_dbscan_clusters' % nclust))
                if not os.path.exists(os.path.join(save, 'dynamic', '%d_dbscan_clusters' % nclust, 'dbscan_cluster%d.png' % tlbl)):
                    fig.savefig(os.path.join(save, 'dynamic', '%d_dbscan_clusters' % nclust, 'dbscan_cluster%d.png' % tlbl))
            else:
                plt.show()
            plt.close(fig)

            # Dynamic cut
            if not only_flat:
                for p_thresh in p_thresh_l:
                    for height_lim in height_lim_l:
                        for min_size in min_size_l:
                            for hom_thresh in hom_thresh_l:
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
                                    # Convert linkage matrix to tree structure similar to that shown in dendrogram
                                    tree = to_tree(link)

                                    lbls_sel = dynamic_tree_cut(tree, sqdm_sel, ids_sel, mk_sel, p_thresh, min_size, hom_thresh,
                                                                max_size_pct, height_lim, base_name='%d' % tlbl, v=v)
                                    lbl_names_sel = np.unique(lbls_sel)
                                    n_clusters = len(lbl_names_sel)
                                    print('DBSCAN Cluster %d split into %d clusters' % (tlbl, n_clusters))
                                    if save:
                                        save_path = os.path.join(save, 'dynamic', '%d_dbscan_clusters' % nclust, 'cluster_%d' % tlbl)
                                        if not os.path.exists(save_path):
                                            os.mkdir(save_path)
                                        save_path = os.path.join(save_path, '%d_clusters' % n_clusters)
                                        tag = None
                                        exists = False
                                        while os.path.exists(save_path) and not exists:
                                            old_lbls = load_csv(os.path.join(save_path, 'clusters.csv'), ids_sel, str)
                                            # REMOVE THE LINE BELOW LATER
                                            # get_cstats(f, save_path, meta_grp=meta_grp, ids=ids_sel)
                                            formatted_stats(f['meta'], save_path)
                                            if np.all(lbls_sel == old_lbls):
                                                exists = True
                                            if not exists:
                                                if tag is None:
                                                    tag = '_a'
                                                else:
                                                    tag = '_' + chr(ord(tag[1]) + 1)
                                                save_path = os.path.join(save, 'dynamic', '%d_dbscan_clusters' % nclust,
                                                                         'cluster_%d' % tlbl, '%d_clusters%s' % (n_clusters, tag))
                                        if not exists:
                                            os.mkdir(save_path)
                                            arr2csv(os.path.join(save_path, 'clusters.csv'), lbls_sel, ids_sel, fmt='%s')
                                            # get_cstats(f, save_path, meta_grp=meta_grp)
                                            formatted_stats(f['meta'], save_path)
                                            if plot_daily:
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
    # Flat cut
    cont = True
    if skip_db:
        link = fc.ward(dm)
    # Generate corresponding dendrogram
    fig = plt.figure()
    dendrogram(link, 50, truncate_mode='lastp')
    plt.xlabel('Patients')
    plt.ylabel('Distance')
    if not skip_db:
        plt.title('DBSCAN Cluster %d\n%d Patients' % (tlbl, len(pt_sel)))
    if save:
        if not os.path.exists(os.path.join(save, 'flat')):
            os.mkdir(os.path.join(save, 'flat'))
        if not skip_db:
            if not os.path.exists(os.path.join(save, 'flat', '%d_dbscan_clusters' % nclust)):
                os.mkdir(os.path.join(save, 'flat', '%d_dbscan_clusters' % nclust))
            if not os.path.exists(
                    os.path.join(save, 'flat', '%d_dbscan_clusters' % nclust, 'dbscan_cluster%d.png' % tlbl)):
                fig.savefig(
                    os.path.join(save, 'flat', '%d_dbscan_clusters' % nclust, 'dbscan_cluster%d.png' % tlbl))
        else:
            fig.savefig(
                os.path.join(save, 'flat', 'dendrogram.png'))
    else:
        plt.show()
    plt.close(fig)

    n_clusters = 1
    while cont:
        if interactive:
            try:
                n_clusters = input('Enter desired number of clusters (current is %d):' % n_clusters)
            except SyntaxError:
                pass
        for n_clusters in n_clusters_l:
            lbls_sel = fcluster(link, n_clusters, criterion='maxclust')
            tlbls = lbls_sel.astype('|S30')
            if not skip_db:
                for i in range(len(tlbls)):
                    tlbls[i] = '%d-%s' % (tlbl, tlbls[i])
            if save:
                if not skip_db:
                    save_path = os.path.join(save, 'flat', '%d_dbscan_clusters' % nclust, 'cluster_%d' % tlbl)
                else:
                    save_path = os.path.join(save, 'flat')
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_path = os.path.join(save_path, '%d_clusters' % n_clusters)
                tag = None
                if os.path.exists(save_path):
                    formatted_stats(f['meta'], save_path)
                    print('%d clusters has already been saved' % n_clusters)
                    if not interactive:
                        cont = False
                    continue
                os.mkdir(save_path)
                if not skip_db:
                    arr2csv(os.path.join(save_path, 'clusters.csv'), lbls_sel, ids_sel, fmt='%d')
                else:
                    arr2csv(os.path.join(save_path, 'clusters.csv'), lbls_sel, ids, fmt='%d')
                formatted_stats(f['meta'], save_path)
                if data_path is not None and plot_daily:
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
        if not skip_db:
            method_lbls[pt_sel] = tlbls
    return eps


def getFlatClusters(f, ids, dm, n_clust_rng=(2,20), data_path=None, save=False, plot_daily=False):
    '''
    Provided a pre-computed distance matrix, cluster the corresponding trajectories using the specified methods.
    First applies DBSCAN to identify and remove any patients designated as noise. Next applies a dynamic hierarchical
    clustering algorithm (see function dynamic_tree_cut below for more details).

    :param stat_file: hdf5 file containing groups of metadata corresponding to different subsets of the full data
    :param dm: pre-computed distance matrix (can be condensed or square_
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
    # Flat cut
    link = fc.ward(dm)
    # Generate corresponding dendrogram
    fig = plt.figure()
    dendrogram(link, 50, truncate_mode='lastp')
    plt.xlabel('Patients')
    plt.ylabel('Distance')
    if save:
        if not os.path.exists(os.path.join(save, 'flat')):
            os.mkdir(os.path.join(save, 'flat'))
        fig.savefig(
            os.path.join(save, 'flat', 'dendrogram.png'))
    else:
        plt.show()
    plt.close(fig)

    for n_clusters in range(*n_clust_rng):
        lbls = fcluster(link, n_clusters, criterion='maxclust')
        lbls = lbls.astype('|S30')
        if save:
            save_path = os.path.join(save_path, '%d_clusters' % n_clusters)
            if os.path.exists(save_path):
                formatted_stats(f['meta'], save_path)
                print('%d clusters has already been saved' % n_clusters)
                continue
            os.mkdir(save_path)
            arr2csv(os.path.join(save_path, 'clusters.csv'), lbls, ids, fmt='%s')
            formatted_stats(f['meta'], save_path)
            if data_path is not None and plot_daily:
                dkpath = os.path.join(save_path, 'daily_kdigo')
                os.mkdir(dkpath)
                plot_daily_kdigos(data_path, ids, f, sqdm, lbls, outpath=dkpath)
    return eps


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
                                max_size_pct, height_lim, left_name, lbls, v=v)
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
                                max_size_pct, height_lim, right_name, lbls, v=v)
    else:
        if v:
            print('Node %s final: p-value=%.2E\tsize=%d\thomogeneity=%.2f%%' % (
                  right_name, right_p, len(right_idx), right_hom))

    return lbls


# %%
def clusterCategorizer(max_kdigos, kdigos, days, lbls):
    lblIds = np.unique(lbls)
    lblNames = []
    for lbl in lblIds:
        idx = np.where(lbls == lbl)[0]
        tmk = mode(max_kdigos[idx]).mode[0]
        if tmk == 1:
            mkstr = '1'
        elif tmk == 2:
            mkstr = '2'
        elif tmk == 3:
            mkstr = '3'
        else:
            mkstr = '3D'
        # Cluster Center
        # Mean of whole cluster
        starts = np.zeros(len(idx))
        stops = np.zeros(len(idx))
        for i in range(len(idx)):
            minDay = max(0, min(days[idx[i]]))
            starts[i] = np.max(kdigos[idx[i]][np.where(days[idx[i]] == minDay)])
            maxDay = min(7, max(days[idx[i]]))
            stops[i] = np.max(kdigos[idx[i]][np.where(days[idx[i]] == maxDay)])
        start = np.mean(starts)
        stop = np.mean(stops)
        if stop > start + 0.5:
            thisName = '%s-Ws' % mkstr
        elif stop < start - 0.5:
            thisName = '%s-Im' % mkstr
        else:
            thisName = '%s-St' % mkstr
        lblNames.append(thisName)
    return lblNames


# %%
def getTreeLocs(lbls, tree):
    lblNames = np.unique(lbls)
    treeLocs = {}
    for lbl in lblNames:
        idx = np.where(lbls == lbl)[0]
        curr = tree
        name = ''
        while curr.count > len(idx):
            left_idx = sorted(curr.left.pre_order())
            right_idx = sorted(curr.right.pre_order())
            if idx[0] in left_idx:
                if name == '':
                    name = 'l'
                else:
                    name += '-l'
                curr = curr.left
            elif idx[0] in right_idx:
                curr = curr.right
                if name == '':
                    name = 'r'
                else:
                    name += '-r'
            else:
                raise ValueError
        treeLocs[lbl] = name
    return treeLocs


def evaluateDmClusters(lbls, mk_7d, mk_w, died_inp, days, sqdm):
    lblNames = np.unique(lbls)
    lblIdxs = {}
    lblGrps = [[], [], []]
    for lbl in lblNames:
        idx = np.where(lbls == lbl)[0]
        lblIdxs[lbl] = idx
        tmk = mk_7d[idx]
        mkv = mode(tmk).mode[0]
        if mkv == 4:
            mkv = 3
        lblGrps[mkv - 1].append(lbl)
    # lblGrps[2] = lbl

    for i in range(len(lblGrps)):
        lblGrps[i] = np.array(lblGrps[i])

    intra_clust_dists = np.zeros(sqdm.shape[0])

    # For sample i, store the mean distance of the second closest
    # cluster in inter_clust_dists[i]
    inter_clust_dists_all = np.inf + intra_clust_dists
    inter_clust_dists_grp = np.inf + intra_clust_dists

    morts = np.vstack([np.repeat(np.nan, 2) for x in range(3)])
    morts[:, 0] = 100
    # progs = np.vstack([np.repeat(np.nan, 2) for x in range(3)])
    # progs[:, 0] = 100
    pct_progs = [[], [], []]
    sils = [[], [], []]
    ind_morts = [[], [], []]
    n_gt7d = {}
    all_progs = {}
    all_morts = {}
    all_sils = {}
    all_mk = {}
    for i in range(len(lblGrps)):
        lblGrp = lblGrps[i]
        for lbl in lblGrp:
            mask = lbls == lbl
            current_distances = sqdm[mask]

            n_samples_curr_lab = len(np.where(mask)[0])
            if n_samples_curr_lab != 0:
                intra_clust_dists[mask] = np.sum(
                    current_distances[:, mask], axis=1) / n_samples_curr_lab

            for other_lbl in lblNames:
                if other_lbl != lbl:
                    other_mask = lbls == other_lbl
                    other_distances = np.nanmean(
                        current_distances[:, other_mask], axis=1)
                    inter_clust_dists_all[mask] = np.nanmin((inter_clust_dists_all[mask], other_distances), axis=0)
                    if other_lbl in lblGrp:
                        inter_clust_dists_grp[mask] = np.nanmin((inter_clust_dists_grp[mask], other_distances),
                                                                axis=0)

            inter_clust_dists_grp[np.where(inter_clust_dists_grp == np.inf)] = np.nan

            pct_died = float(len(np.where(died_inp[lblIdxs[lbl]])[0])) / len(lblIdxs[lbl]) * 100

            gt7d = np.where([max(days[x]) > 7 for x in lblIdxs[lbl]])[0]
            no_mk3d = np.where(mk_7d < 4)[0]
            prog_idxs = np.intersect1d(gt7d, no_mk3d)
            n_gt7d[lbl] = (len(gt7d), float(len(gt7d))/len(lblIdxs[lbl]) * 100)
            prog_sel = np.where(mk_7d[lblIdxs[lbl]][prog_idxs] != mk_w[lblIdxs[lbl]][prog_idxs])[0]
            if len(prog_idxs) > 10:
                pct_prog = float(len(prog_sel)) / len(prog_idxs) * 100
                pct_progs[i].append(pct_prog)
            else:
                pct_progs[i].append(np.nan)
                pct_prog = np.nan

            sil_samples_grp = inter_clust_dists_grp - intra_clust_dists
            sil_samples_grp /= np.maximum(intra_clust_dists, inter_clust_dists_grp)

            morts[i, 0] = np.nanmin((morts[i, 0], pct_died))
            morts[i, 1] = np.nanmax((morts[i, 1], pct_died))
            # progs[i, 0] = np.nanmin((progs[i, 0], pct_prog))
            # progs[i, 1] = np.nanmax((progs[i, 1], pct_prog))
            sils[i].append(sil_samples_grp[mask])
            ind_morts[i].append(pct_died)
            all_morts[lbl] = pct_died
            all_progs[lbl] = pct_prog
            all_sils[lbl] = np.mean(sil_samples_grp[mask])
            tmk = mk_7d[lblIdxs[lbl]]
            mkv = mode(tmk).mode[0]
            all_mk[lbl] = mkv
        if len(sils[i]) > 1:
            sils[i] = float(np.nanmean(np.hstack(sils[i])))
        elif len(sils[i]) == 1:
            sils[i] = float(np.nanmean(sils[i]))
        else:
            sils[i] = np.nan
    mort_l = [np.max(morts) - np.min(morts), ]
    # prog_l = [np.max(progs) - np.min(progs), ]
    all_prog = np.array(pct_progs[0] + pct_progs[1] + pct_progs[2]).reshape(-1, 1)
    all_mort = np.array(ind_morts[0] + ind_morts[1] + ind_morts[2]).reshape(-1, 1)
    min_prod_diff_l = [np.nanmin(pdist(all_prog)), ]
    max_prod_diff_l = [np.nanmax(pdist(all_prog)), ]
    rel_prod_diff_l = [max_prod_diff_l[0] / (100 - min_prod_diff_l[0]), ]
    min_mort_diff_l = [np.nanmin(pdist(all_mort)), ]
    max_mort_diff_l = [np.nanmax(pdist(all_mort)), ]
    rel_mort_diff_l = [max_mort_diff_l[0] / (100 - min_mort_diff_l[0]), ]
    sil_l = [np.nanmean(sil_samples_grp), ]
    for i in range(3):
        if len(lblGrps[i]) > 1:
            mort_l.append(morts[i, 1] - morts[i, 0])
            min_prod_diff_l.append(np.nanmin(pdist(np.array(pct_progs[i]).reshape(-1, 1))))
            max_prod_diff_l.append(np.nanmax(pdist(np.array(pct_progs[i]).reshape(-1, 1))))
            rel_prod_diff_l.append(max_prod_diff_l[i + 1] / (100 - min_prod_diff_l[i + 1]))
            min_mort_diff_l.append(np.nanmin(pdist(np.array(ind_morts[i]).reshape(-1, 1))))
            max_mort_diff_l.append(np.nanmax(pdist(np.array(ind_morts[i]).reshape(-1, 1))))
            rel_mort_diff_l.append(max_mort_diff_l[i + 1] / (100 - min_mort_diff_l[i + 1]))
        else:
            mort_l.append(np.nan)
            min_prod_diff_l.append(np.nan)
            max_prod_diff_l.append(np.nan)
            rel_prod_diff_l.append(np.nan)
            min_mort_diff_l.append(np.nan)
            max_mort_diff_l.append(np.nan)
            rel_mort_diff_l.append(np.nan)
        sil_l.append(sils[i])

    l1 = '%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,' \
        '%.2f,%.2f,%.2f,%.2f,%.2f,%.2f' % (# min_mort_diff_l[0], max_mort_diff_l[0], min_prod_diff_l[0],
                                                 # max_prod_diff_l[0], sil_l[0],
                                                 min_mort_diff_l[1], min_mort_diff_l[2], min_mort_diff_l[3],
                                                 max_mort_diff_l[1], max_mort_diff_l[2], max_mort_diff_l[3],
                                                 rel_mort_diff_l[1], rel_mort_diff_l[2], rel_mort_diff_l[3],
                                                 min_prod_diff_l[1], min_prod_diff_l[2], min_prod_diff_l[3],
                                                 max_prod_diff_l[1], max_prod_diff_l[2], max_prod_diff_l[3],
                                                 rel_prod_diff_l[1], rel_prod_diff_l[2], rel_prod_diff_l[3],
                                                 sil_l[1], sil_l[2], sil_l[3])
    l2 = '%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f' % (min_mort_diff_l[0], max_mort_diff_l[0], rel_mort_diff_l[0], min_prod_diff_l[0],
                                                 max_prod_diff_l[0], rel_prod_diff_l[0], sil_l[0])
    return l1, l2, n_gt7d, all_morts, all_progs, all_sils, all_mk


def getMeanParentDist(tree, tree_locs):
    l = []
    lbls = list(tree_locs)
    for i in range(len(lbls)):
        c1_loc = tree_locs[lbls[i]].split('-')
        c1 = tree
        for s in c1_loc:
            if s == 'l':
                c1 = c1.left
            else:
                c1 = c1.right
        c1_height = c1.dist
        for j in range(i + 1, len(lbls)):
            c2_loc = tree_locs[lbls[j]].split('-')
            c2 = tree
            for s in c2_loc:
                if s == 'l':
                    c2 = c2.left
                else:
                    c2 = c2.right
            c2_height = c2.dist
            parent = tree
            for k in range(min(len(c1_loc), len(c2_loc))):
                if c1_loc[k] == c2_loc[k]:
                    if c1_loc[k] == 'l':
                        parent = parent.left
                    else:
                        parent = parent.right
                else:
                    break
            l.append(parent.dist - np.mean((c1_height, c2_height)))
    return squareform(l)


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


def plot_daily_kdigos(datapath, ids, stat_file, sqdm, lbls, outpath='', max_day=7, cutoff=None, types=['center', 'mean_conf', 'mean_std', 'all_w_mean']):
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
    lblNames = np.unique(lbls)
    n_clusters = len(lblNames)
    if np.ndim(sqdm) == 1:
        sqdm = squareform(sqdm)
    if type(lbls) is list:
        lbls = np.array(lbls, dtype=str)

    scrs = load_csv(os.path.join(datapath, 'scr_raw.csv'), ids)
    bslns = load_csv(os.path.join(datapath, 'baselines.csv'), ids)
    dmasks = load_csv(os.path.join(datapath, 'dmasks_interp.csv'), ids, dt=int)
    kdigos = load_csv(os.path.join(datapath, 'kdigo.csv'), ids, dt=int)
    days_interp = load_csv(os.path.join(datapath, 'days_interp.csv'), ids, dt=int)

    str_dates = load_csv(datapath + 'dates.csv', ids, dt=str)
    for i in range(len(ids)):
        for j in range(len(str_dates[i])):
            str_dates[i][j] = str_dates[i][j].split('.')[0]
    dates = []
    for i in range(len(ids)):
        temp = []
        for j in range(len(str_dates[i])):
            temp.append(get_date(str_dates[i][j]))
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
        tlbl = lblNames[i]
        idx = np.where(lbls == tlbl)[0]
        cluster_idx[tlbl] = idx
        days = [days_interp[x] for x in idx]
        sel = np.ix_(idx, idx)
        tdm = sqdm[sel]
        sums = np.sum(tdm, axis=0)
        o = np.argsort(sums)
        ccount = 0
        center = o[ccount]
        while max(days[center]) < 7:
            ccount += 1
            center = o[ccount]
        centers[i] = idx[center]
        for j in range(max_day + 2):
            n_recs[i, j] = (float(len(idx) - len(np.where(np.isnan(all_daily[idx, j]))[0])) / len(idx)) * 100

    if outpath != '':
        f = stat_file
        all_ids = f['meta']['ids'][:]
        all_inp_death = f['meta']['died_inp'][:]
        sel = np.array([x in ids for x in all_ids])
        inp_death = all_inp_death[sel]
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        if not os.path.exists(os.path.join(outpath, 'all_w_mean')) and 'all_w_mean' in types:
            os.mkdir(os.path.join(outpath, 'all_w_mean'))
        if not os.path.exists(os.path.join(outpath, 'mean_std')) and 'mean_std' in types:
            os.mkdir(os.path.join(outpath, 'mean_std'))
        if not os.path.exists(os.path.join(outpath, 'mean_conf')) and 'mean_conf' in types:
            os.mkdir(os.path.join(outpath, 'mean_conf'))
        if not os.path.exists(os.path.join(outpath, 'center')) and 'center' in types:
            os.mkdir(os.path.join(outpath, 'center'))
        for i in range(n_clusters):
            cidx = np.where(lbls == lblNames[i])[0]
            ct = len(cidx)
            mort = (float(len(np.where(inp_death[cidx])[0])) / len(cidx)) * 100
            mean_daily, conf_lower, conf_upper = mean_confidence_interval(all_daily[cidx])
            std_daily = np.nanstd(all_daily[cidx], axis=0)
            # stds_upper = np.minimum(mean_daily + std_daily, 4)
            # stds_lower = np.maximum(mean_daily - std_daily, 0)
            stds_upper = mean_daily + std_daily
            stds_lower = mean_daily - std_daily

            if 'center' in types:
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
                tplot.set_title('Cluster %s Representative' % lblNames[i])
                plt.legend()
                plt.savefig(os.path.join(outpath, 'center', '%s_center.png' % lblNames[i]))
                plt.close(tfig)

            if 'all_w_mean' in types:
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
                plt.title('Average Daily KDIGO\nCluster %s' % lblNames[i])
                plt.savefig(os.path.join(outpath, 'all_w_mean', '%s_all.png' % lblNames[i]))
                plt.close(fig)

            if 'mean_std' in types:
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
                plt.title('Average Daily KDIGO\nCluster %s' % lblNames[i])
                plt.savefig(os.path.join(outpath, 'mean_std', '%s_mean_std.png' % lblNames[i]))
                plt.close(fig)

            if 'mean_conf' in types:
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
                plt.title('Average Daily KDIGO\nCluster %s' % lblNames[i])
                plt.savefig(os.path.join(outpath, 'mean_conf', '%s_mean_conf.png' % lblNames[i]))
                plt.close(fig)
        # f.close()
    return all_daily


def getClusterCenters(kdigos, days, lbls, sqdm):
    lblNames = np.unique(lbls)
    n_clusters = len(lblNames)
    cids = np.zeros(n_clusters, dtype=int)
    center_kdigos = []
    center_days = []
    for i in range(n_clusters):
        tlbl = lblNames[i]
        idx = np.where(lbls == tlbl)[0]
        tdays = [days[x] for x in idx]
        sel = np.ix_(idx, idx)
        tdm = sqdm[sel]
        sums = np.sum(tdm, axis=0)
        o = np.argsort(sums)
        ccount = 0
        center = o[ccount]
        while len(np.where(tdays[center] == 7)[0]) < 2 and ccount < len(o):
            ccount += 1
            center = o[ccount]
        cids[i] = idx[center]
        center_kdigos.append(kdigos[idx[center]])
        center_days.append(days[idx[center]])
    return cids, center_kdigos, center_days


def getClusterMembership(ids, kdigos, days, lbls, center_kdigos, center_days, mismatch=lambda x, y: abs(x-y), extension=lambda x: 0, dist=braycurtis, t_lim=7, logFileName='cluster_membership.csv'):
    lblNames = np.unique(lbls)
    assert len(lblNames) == len(center_kdigos) == len(center_days)

    log = open(logFileName, 'w')
    hdr = 'pt_id'
    for lbl in lblNames:
        hdr += ',' + str(lbl)
    log.write(hdr + '\n')
    nlbls = np.zeros(len(ids), dtype='|S50')
    for i in range(len(kdigos)):
        kdigo = kdigos[i]
        tdays = days[i]
        kdigo = kdigo[np.where(tdays <= t_lim)]
        d = np.zeros(len(lblNames))
        for j in range(len(lblNames)):
            ckdigo = center_kdigos[j]
            cdays = center_days[j]
            ckdigo = ckdigo[np.where(cdays <= t_lim)]
            _, _, _, path = dtw_p(kdigo, ckdigo, mismatch=mismatch, extension=extension)
            k1 = [kdigo[x] for x in path[0]]
            k2 = [ckdigo[x] for x in path[1]]
            d[j] = dist(k1, k2)
        nlbls[i] = lblNames[np.argsort(d)[0]]
        s = '%d' % ids[i]
        for val in d:
            s += ',%f' % val
        log.write(s + '\n')
    log.close()
    return nlbls





def plot_daily_kdigos_2(datapath, ids, stat_file, sqdm, lbls, outpath='', max_day=7, cutoff=None, types=['center', 'mean_conf', 'mean_std', 'all_w_mean'], aligned=False, normalized=False, mismatch_func=None, extension_func=None, debug=False, ticks=True, centers=None):
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
    lblNames = np.unique(lbls)
    n_clusters = len(lblNames)
    if np.ndim(sqdm) == 1:
        sqdm = squareform(sqdm)
    if type(lbls) is list:
        lbls = np.array(lbls, dtype=str)

    scrs = load_csv(os.path.join(datapath, 'scr_raw.csv'), ids)
    bslns = load_csv(os.path.join(datapath, 'baselines.csv'), ids)
    dmasks = load_csv(os.path.join(datapath, 'dmasks_interp.csv'), ids, dt=int)
    kdigos = load_csv(os.path.join(datapath, 'kdigo.csv'), ids, dt=int)
    days_interp = load_csv(os.path.join(datapath, 'days_interp.csv'), ids, dt=int)

    str_dates = load_csv(datapath + 'dates.csv', ids, dt=str)
    for i in range(len(ids)):
        for j in range(len(str_dates[i])):
            str_dates[i][j] = str_dates[i][j].split('.')[0]
    dates = []
    for i in range(len(ids)):
        temp = []
        for j in range(len(str_dates[i])):
            temp.append(get_date(str_dates[i][j]))
        dates.append(temp)

    if centers is None:
        centers = np.zeros(n_clusters, dtype=int)
        # n_recs = np.zeros((n_clusters, max_day + 2))
        for i in range(n_clusters):
            tlbl = lblNames[i]
            idx = np.where(lbls == tlbl)[0]
            days = [days_interp[x] for x in idx]
            sel = np.ix_(idx, idx)
            tdm = sqdm[sel]
            sums = np.sum(tdm, axis=0)
            o = np.argsort(sums)
            ccount = 0
            center = o[ccount]
            while len(np.where(days[center] == 7)[0]) < 2 and ccount < len(o):
                ccount += 1
                center = o[ccount]
            centers[i] = idx[center]
            # for j in range(max_day + 2):
            #     n_recs[i, j] = (float(len(idx) - len(np.where(np.isnan(all_daily[idx, j]))[0])) / len(idx)) * 100

    blank_daily = np.repeat(np.nan, max_day + 2)
    all_daily = np.vstack([blank_daily for x in range(len(ids))])
    for i in range(len(ids)):
        l = np.min([len(x) for x in [scrs[i], dates[i], dmasks[i]]])
        if l < 2:
            continue
        # tmax = daily_max_kdigo(scrs[i][:l], dates[i][:l], bslns[i], admits[i], dmasks[i][:l], tlim=max_day)
        # tmax = daily_max_kdigo(scrs[i][:l], days[i][:l], bslns[i], dmasks[i][:l], tlim=max_day)
        if aligned:
            tkdigo = kdigos[i][np.where(days_interp[i] <= max_day)]
            try:
                assert len(centers[0]) == 2
                tidx = np.where(lblNames == lbls[i])[0][0]
                center_kdigo = centers[tidx][0][np.where(centers[tidx][1] <= max_day)]
                _, _, _, path = dtw_p(center_kdigo, tkdigo, mismatch=mismatch_func, extension=extension_func)
                tmax = daily_max_kdigo_interp(tkdigo[path[1]],
                                              centers[tidx][1][np.where(centers[tidx][1] <= max_day)], tlim=max_day)
            except TypeError:
                tlbl = lbls[i]
                cnum = np.where(lblNames == tlbl)[0][0]
                cidx = centers[cnum]
                center_kdigo = kdigos[cidx][np.where(days_interp[cidx] <= max_day)]
                _, _, _, path = dtw_p(center_kdigo, tkdigo, mismatch=mismatch_func, extension=extension_func)
                tmax = daily_max_kdigo_interp(tkdigo[path[1]], days_interp[cidx][np.where(days_interp[cidx] <= max_day)], tlim=max_day)
        else:
            tmax = daily_max_kdigo_interp(kdigos[i], days_interp[i], tlim=max_day)
        if normalized:
            tmax = tmax.astype(float) / np.max(tmax)
        if debug and np.all(tmax == 0):
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

    if outpath != '':
        f = stat_file
        all_ids = f['meta']['ids'][:]
        all_inp_death = f['meta']['died_inp'][:]
        sel = np.array([x in ids for x in all_ids])
        inp_death = all_inp_death[sel]
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        if not os.path.exists(os.path.join(outpath, 'all_w_mean')) and 'all_w_mean' in types:
            os.mkdir(os.path.join(outpath, 'all_w_mean'))
        if not os.path.exists(os.path.join(outpath, 'mean_std')) and 'mean_std' in types:
            os.mkdir(os.path.join(outpath, 'mean_std'))
        if not os.path.exists(os.path.join(outpath, 'mean_conf')) and 'mean_conf' in types:
            os.mkdir(os.path.join(outpath, 'mean_conf'))
        if not os.path.exists(os.path.join(outpath, 'center')) and 'center' in types:
            os.mkdir(os.path.join(outpath, 'center'))
        for i in range(n_clusters):
            cidx = np.where(lbls == lblNames[i])[0]
            ct = len(cidx)
            mort = (float(len(np.where(inp_death[cidx])[0])) / len(cidx)) * 100
            mean_daily, conf_lower, conf_upper = mean_confidence_interval(all_daily[cidx])
            std_daily = np.nanstd(all_daily[cidx], axis=0)
            # stds_upper = np.minimum(mean_daily + std_daily, 4)
            # stds_lower = np.maximum(mean_daily - std_daily, 0)
            stds_upper = mean_daily + std_daily
            stds_lower = mean_daily - std_daily

            if 'center' in types:
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
                tplot.set_xlim(-0.25, 7.25)
                if not normalized:
                    plt.yticks(range(5), ['0', '1', '2', '3', '3D'])
                    tplot.set_ylim(-1.0, 5.0)
                else:
                    tplot.set_ylim(-0.5, 1.5)
                tplot.set_xlabel('Day')
                tplot.set_ylabel('KDIGO Score')
                tplot.set_title('Cluster %s Representative' % lblNames[i])
                plt.legend()
                plt.savefig(os.path.join(outpath, 'center', '%s_center.png' % lblNames[i]))
                plt.close(tfig)

            if 'all_w_mean' in types:
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
                if not normalized:
                    plt.yticks(range(5), ['0', '1', '2', '3', '3D'])
                    plt.ylim((-1.0, 5.0))
                else:
                    plt.ylim((-0.5, 1.5))
                plt.xlabel('Time (Days)')
                plt.ylabel('KDIGO Score')
                plt.legend()
                plt.title('Average Daily KDIGO\nCluster %s' % lblNames[i])
                plt.savefig(os.path.join(outpath, 'all_w_mean', '%s_all.png' % lblNames[i]))
                plt.close(fig)

            if 'mean_std' in types:
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
                if not normalized:
                    plt.yticks(range(5), ['0', '1', '2', '3', '3D'])
                    plt.ylim((-1.0, 5.0))
                else:
                    plt.ylim((-0.5, 1.5))
                plt.xlabel('Time (Days)')
                plt.ylabel('KDIGO Score')
                plt.legend()
                # ax2 = ax1.twinx()
                # ax2.plot(range(max_day + 2), n_recs[i, :], color='black', label='# Records')
                # ax2.set_ylim((-5, 105))
                # ax2.set_ylabel('% Patients Remaining')
                # plt.legend(loc=7)
                plt.title('Average Daily KDIGO\nCluster %s' % lblNames[i])
                plt.savefig(os.path.join(outpath, 'mean_std', '%s_mean_std.png' % lblNames[i]))
                plt.close(fig)

            if 'mean_conf' in types:
                # Mean and 95% confidence interval
                fig, ax1 = plt.subplots()
                if cutoff is not None or cutoff >= max_day:
                    ax1.plot(range(max_day + 2)[:cutoff + 1], mean_daily[:cutoff + 1], color='b',
                             lw=2, alpha=.8)
                    ax1.plot(range(max_day + 2)[cutoff:], mean_daily[cutoff:], color='r', linestyle='dashed',
                             label=None, lw=2, alpha=.8)
                    ax1.axvline(x=cutoff, linestyle='dashed')
                else:
                    ax1.plot(range(max_day + 2), mean_daily, color='b',
                             label=None, lw=2, alpha=.8)
                ax1.fill_between(range(max_day + 2), conf_lower, conf_upper, color='grey', alpha=.2,
                                 label=None)
                plt.xlim([-0.25, max_day + 0.25])
                if ticks:
                    if not normalized:
                        plt.yticks(range(5), ['0', '1', '2', '3', '3D'])
                    for tick in ax1.xaxis.get_major_ticks():
                        tick.label.set_fontsize(26)
                    for tick in ax1.yaxis.get_major_ticks():
                        tick.label.set_fontsize(26)
                    plt.xlabel('Time (Days)', fontsize=26)
                    plt.ylabel('KDIGO Score', fontsize=26)
                else:
                    plt.xticks(())
                    plt.yticks(())
                if not normalized:
                    plt.ylim((-1.0, 5.0))
                else:
                    plt.ylim((-0.5, 1.5))
                extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
                ax1.legend([extra], ['%s\nMortality %.2f%%\n# Patients %d' % (lblNames[i], mort, ct), ], prop={'size': 26}, frameon=False)


                # ax2 = ax1.twinx()
                # ax2.plot(range(max_day + 2), n_recs[i, :], color='black', label='# Records')
                # ax2.set_ylim((-5, 105))
                # ax2.set_ylabel('% Patients Remaining')
                # plt.legend(loc=7)
                # plt.title('Average Daily KDIGO\nCluster %s' % lblNames[i])
                plt.tight_layout()
                plt.savefig(os.path.join(outpath, 'mean_conf', '%s_mean_conf.png' % lblNames[i]))
                plt.close(fig)
        # f.close()
    return all_daily


def kdigo_cluster_silhouette(sqdm, lbls, lbl_grps):
    if sqdm.ndim == 1:
        sqdm = squareform(sqdm)
    scores = np.zeros(len(lbls))
    lbl_idxs = {}
    for lbl in np.unique(lbls):
        lbl_idxs[lbl] = np.where(lbls == lbl)[0]
    for i in range(len(lbls)):
        tlbl = lbls[i]
        grp_idx = np.where(np.array([np.any(x == tlbl) for x in lbl_grps]))[0][0]
        lbl_grp = lbl_grps[grp_idx]
        inter = 100
        a = 0
        b = 0
        for lbl in lbl_grp:
            idx = lbl_idxs[lbl]
            dist = np.mean(sqdm[i, idx])
            if lbl == tlbl:
                a = dist
            else:
                b = np.min((inter, dist))
        scores[i] = (b - a) / max(a, b)
    return scores


def inter_intra_dist(sqdm, lbls, lbl_grps):
    intras = np.zeros(len(lbls))
    inters_all = np.zeros(len(lbls))
    inters_grp = np.zeros(len(lbls))
    ref_dms = {}
    for lbl in np.unique(lbls):
        idx = np.where(lbls == lbl)[0]
        sel = np.ix_(np.arange(len(lbls)), idx)
        ref_dms[lbl] = sqdm[sel]
    for i in range(len(lbls)):
        tlbl = lbls[i]
        grp_idx = int(np.where(np.array([tlbl in x for x in lbl_grps]))[0])
        lbl_grp = lbl_grps[grp_idx]
        inter_all = 100
        inter_grp = 100
        for lbl in np.unique(lbls):
            if lbl == tlbl:
                intras[i] = np.mean(ref_dms[tlbl][i, :])
            inter = np.mean(ref_dms[tlbl][i, :])
            inter_all = min(inter_all, inter)
            if np.any([x == tlbl for x in lbl_grp]):
                inter_grp = min(inter_grp, inter)
        inters_all[i] = inter_all
        inters_grp[i] = inter_grp
    return intras, inters_all, inters_grp




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
