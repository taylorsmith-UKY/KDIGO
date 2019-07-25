"""
Created on Fri Jan 19 14:31:03 2018

@author: taylorsmith
"""
import numpy as np
from scipy.spatial.distance import squareform, pdist, braycurtis
from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.stats.mstats import normaltest
from scipy.stats import mode, sem, t
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score,\
    v_measure_score, fowlkes_mallows_score, silhouette_score, calinski_harabaz_score
import fastcluster as fc
from dtw_distance import dtw_p
from stat_funcs import formatted_stats
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from DBA import performDBA
import os
import copy
from utility_funcs import get_date, load_csv, arr2csv
from fastdtw import dtw


# %%
def cluster_trajectories(f, ids, mk, dm, eps=0.015, n_clusters=2, data_path=None, interactive=True,
                         save=False, plot_daily=False, kdigos=None, days=None):
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

    # Flat cut
    cont = True
    link = fc.ward(dm)
    # Generate corresponding dendrogram
    fig = plt.figure()
    dendrogram(link, 50, truncate_mode='lastp')
    plt.xlabel('Patients')
    plt.ylabel('Distance')
    if save:
        if not os.path.exists(os.path.join(save, 'flat')):
            os.mkdir(os.path.join(save, 'flat'))
        plt.savefig(
            os.path.join(save, 'flat', 'dendrogram.png'), dpi=600)
    else:
        plt.show()
    plt.close(fig)

    while cont:
        if interactive:
            try:
                n_clusters = input('Enter desired number of clusters (current is %d):' % n_clusters)
            except SyntaxError:
                pass

        lbls_sel = fcluster(link, n_clusters, criterion='maxclust')
        tlbls = lbls_sel.astype('|S30')
        if save:
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
            arr2csv(os.path.join(save_path, 'clusters.csv'), lbls_sel, ids, fmt='%d')
            formatted_stats(f['meta'], save_path)
            if not os.path.exists(os.path.join(save_path, 'rename')) and kdigos is not None:
                os.mkdir(os.path.join(save_path, 'rename'))
            if kdigos is not None:
                nlbls, clustCats = clusterCategorizer(mk, kdigos, days, lbls_sel)
                arr2csv(os.path.join(save_path, 'rename', 'clusters.csv'), nlbls, ids, fmt='%s')
                formatted_stats(f['meta'], os.path.join(save_path, 'rename'))
            if data_path is not None and plot_daily:
                dkpath = os.path.join(save_path, 'daily_kdigo')
                os.mkdir(dkpath)
                # plot_daily_kdigos(data_path, ids_sel, f, sqdm, lbls_sel, outpath=dkpath)
        if interactive:
            t = input('Try a different configuration? (y/n)')
            if 'y' in t:
                cont = True
            else:
                cont = False
        else:
            cont = False
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
    lblNames = {}
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
        if thisName in lblNames.values():
            thisName += '-1'
        while thisName in lblNames.values():
            tag = int(thisName.split('-')[-1])
            thisName = '-'.join(thisName.split('-')[:-1] + [str(tag + 1)])
        lblNames[lbl] = thisName
    maxNameLen = max([len(lblNames[x]) for x in list(lblNames)])
    out = np.zeros(len(lbls), dtype='|S%d' % maxNameLen).astype(str)
    for lbl in lblIds:
        out[np.where(lbls == lbl)] = lblNames[lbl]
    return out, lblNames


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
    # lblGrps = [[], [], []]
    lblGrps = {'1': {'Im': [], 'St': [], 'Ws':[]},
               '2': {'Im': [], 'St': [], 'Ws':[]},
               '3': {'Im': [], 'St': [], 'Ws':[]},
               '3D': {'Im': [], 'St': [], 'Ws':[]}}
    for lbl in lblNames:
        idx = np.where(lbls == lbl)[0]
        lblIdxs[lbl] = idx
        # tmk = mk_7d[idx]
        # mkv = mode(tmk).mode[0]
        # if mkv == 4:
        #     mkv = 3
        # lblGrps[mkv - 1].append(lbl)
        mk = lbl.split('-')[0]
        trend = lbl.split('-')[1]
        lblGrps[mk][trend].append(lbl)
    # lblGrps[2] = lbl

    for mk in ['1', '2', '3', '3D']:
        for trend in ['Im', 'St', 'Ws']:
            lblGrps[mk][trend] = np.array(lblGrps[mk][trend])

    intra_clust_dists = np.zeros(sqdm.shape[0])

    # For sample i, store the mean distance of the second closest
    # cluster in inter_clust_dists[i]
    inter_clust_dists_all = np.inf + intra_clust_dists
    inter_clust_dists_grp = np.inf + intra_clust_dists

    morts = np.vstack([np.repeat(np.nan, 2) for x in range(3)])
    morts[:, 0] = 100
    # progs = np.vstack([np.repeat(np.nan, 2) for x in range(3)])
    # progs[:, 0] = 100
    pct_progs = {'1': {'Im': [], 'St': [], 'Ws':[]},
               '2': {'Im': [], 'St': [], 'Ws':[]},
               '3': {'Im': [], 'St': [], 'Ws':[]},
               '3D': {'Im': [], 'St': [], 'Ws':[]}}
    sils = {'1': {'Im': [], 'St': [], 'Ws':[]},
               '2': {'Im': [], 'St': [], 'Ws':[]},
               '3': {'Im': [], 'St': [], 'Ws':[]},
               '3D': {'Im': [], 'St': [], 'Ws':[]}}
    ind_morts = {'1': {'Im': [], 'St': [], 'Ws':[]},
               '2': {'Im': [], 'St': [], 'Ws':[]},
               '3': {'Im': [], 'St': [], 'Ws':[]},
               '3D': {'Im': [], 'St': [], 'Ws':[]}}
    n_gt7d = {'1': {'Im': [], 'St': [], 'Ws':[]},
               '2': {'Im': [], 'St': [], 'Ws':[]},
               '3': {'Im': [], 'St': [], 'Ws':[]},
               '3D': {'Im': [], 'St': [], 'Ws':[]}}
    all_progs = {'1': {'Im': [], 'St': [], 'Ws':[]},
               '2': {'Im': [], 'St': [], 'Ws':[]},
               '3': {'Im': [], 'St': [], 'Ws':[]},
               '3D': {'Im': [], 'St': [], 'Ws':[]}}
    all_morts = {'1': {'Im': [], 'St': [], 'Ws':[]},
               '2': {'Im': [], 'St': [], 'Ws':[]},
               '3': {'Im': [], 'St': [], 'Ws':[]},
               '3D': {'Im': [], 'St': [], 'Ws':[]}}
    all_sils = {'1': {'Im': [], 'St': [], 'Ws':[]},
               '2': {'Im': [], 'St': [], 'Ws':[]},
               '3': {'Im': [], 'St': [], 'Ws':[]},
               '3D': {'Im': [], 'St': [], 'Ws':[]}}
    all_mk = {'1': {'Im': [], 'St': [], 'Ws':[]},
               '2': {'Im': [], 'St': [], 'Ws':[]},
               '3': {'Im': [], 'St': [], 'Ws':[]},
               '3D': {'Im': [], 'St': [], 'Ws':[]}}
    for mk in ['1', '2', '3', '3D']:
        for trend in ['Im', 'St', 'Ws']:
            lblGrp = lblGrps[mk][trend]
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
                    pct_progs[mk][trend].append(pct_prog)
                else:
                    pct_progs[mk][trend].append(np.nan)
                    pct_prog = np.nan
                ind_morts[mk][trend].append(pct_died)

    min_prod_diff_l = {'1': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '2': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '3': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '3D': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan}}
    max_prod_diff_l = {'1': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '2': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '3': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '3D': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan}}
    rel_prod_diff_l = {'1': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '2': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '3': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '3D': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan}}
    min_mort_diff_l = {'1': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '2': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '3': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '3D': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan}}
    max_mort_diff_l = {'1': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '2': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '3': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '3D': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan}}
    rel_mort_diff_l = {'1': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '2': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '3': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '3D': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan}}
    sil_l = {'1': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '2': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '3': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan},
             '3D': {'Im': np.nan, 'St': np.nan, 'Ws': np.nan}}
    for mk in ['1', '2', '3', '3D']:
        for trend in ['Im', 'St', 'Ws']:
            if len(lblGrps[mk][trend]) > 1:
                min_prod_diff_l[mk][trend] = np.nanmin(pdist(np.array(pct_progs[mk][trend]).reshape(-1, 1)))
                max_prod_diff_l[mk][trend] = np.nanmax(pdist(np.array(pct_progs[mk][trend]).reshape(-1, 1)))
                rel_prod_diff_l[mk][trend] = max_prod_diff_l[mk][trend] / (100 - min_prod_diff_l[mk][trend])
                min_mort_diff_l[mk][trend] = np.nanmin(pdist(np.array(ind_morts[mk][trend]).reshape(-1, 1)))
                max_mort_diff_l[mk][trend] = np.nanmax(pdist(np.array(ind_morts[mk][trend]).reshape(-1, 1)))
                rel_mort_diff_l[mk][trend] = max_mort_diff_l[mk][trend] / (100 - min_mort_diff_l[mk][trend])

    l1 = ''
    for mk in ['1', '2', '3', '3D']:
        for trend in ['Im', 'St', 'Ws']:
            if l1 == '':
                l1 = '%.2f' % min_mort_diff_l[mk][trend]
            else:
                l1 += ',%.2f' % min_mort_diff_l[mk][trend]
    for mk in ['1', '2', '3', '3D']:
        for trend in ['Im', 'St', 'Ws']:
            l1 += ',%.2f' % max_mort_diff_l[mk][trend]
    for mk in ['1', '2', '3', '3D']:
        for trend in ['Im', 'St', 'Ws']:
            l1 += ',%.2f' % rel_mort_diff_l[mk][trend]
    for mk in ['1', '2', '3', '3D']:
        for trend in ['Im', 'St', 'Ws']:
            l1 += ',%.2f' % min_prod_diff_l[mk][trend]
    for mk in ['1', '2', '3', '3D']:
        for trend in ['Im', 'St', 'Ws']:
            l1 += ',%.2f' % max_prod_diff_l[mk][trend]
    for mk in ['1', '2', '3', '3D']:
        for trend in ['Im', 'St', 'Ws']:
            l1 += ',%.2f' % rel_prod_diff_l[mk][trend]

    l2 = '%.2f,%.2f,%.2f,%.2f,%.2f,%.2f' % (min_mort_diff_l[0], max_mort_diff_l[0], rel_mort_diff_l[0], min_prod_diff_l[0],
                                                 max_prod_diff_l[0], rel_prod_diff_l[0])
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


def getMedoids(kdigos, days, lbls, sqdm, minLen=None, targLen=None):
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
        if targLen is not None:
            while len(center) != targLen and ccount < len(o):
                ccount += 1
                center = o[ccount]
        elif minLen is not None:
            while len(center) < minLen and ccount < len(o):
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


def countCategories(lbls):
    lblNames = np.unique(lbls)
    counts = np.zeros(12)
    counts[0] = np.sum(np.array(['1-Im' in x for x in lblNames]))
    counts[1] = np.sum(np.array(['1-St' in x for x in lblNames]))
    counts[2] = np.sum(np.array(['1-Ws' in x for x in lblNames]))
    counts[3] = np.sum(np.array(['2-Im' in x for x in lblNames]))
    counts[4] = np.sum(np.array(['2-St' in x for x in lblNames]))
    counts[5] = np.sum(np.array(['2-Ws' in x for x in lblNames]))
    counts[6] = np.sum(np.array(['3-Im' in x for x in lblNames]))
    counts[7] = np.sum(np.array(['3-St' in x for x in lblNames]))
    counts[8] = np.sum(np.array(['3-Ws' in x for x in lblNames]))
    counts[9] = np.sum(np.array(['3D-Im' in x for x in lblNames]))
    counts[10] = np.sum(np.array(['3D-St' in x for x in lblNames]))
    counts[11] = np.sum(np.array(['3D-Ws' in x for x in lblNames]))
    return counts


def merge_clusters(ids, kdigos, max_kdigos, days, dm, lblPath, meta, mismatch=lambda x,y: abs(x-y), extension=lambda x:0, dist=braycurtis, t_lim=7, mergeType='dba', folderName='merged', normalDTW=False, alpha=1.0, category='all', aggExt=False, plot_centers=False):
    ids = np.loadtxt(os.path.join(lblPath, 'clusters.csv'), delimiter=',', usecols=0, dtype=int)
    assert len(ids) == len(kdigos)
    lbls = load_csv(os.path.join(lblPath, 'clusters.csv'), ids, str)
    lbls = np.array(lbls, dtype='|S100').astype(str)
    catLbls, lblNames = clusterCategorizer(max_kdigos, kdigos, days, lbls)
    centers = {}
    centerName = 'centers'
    if normalDTW:
        centerName += '_normDTW'
    else:
        centerName += '_popDTW'
        centerName += '_alpha%.0E' % alpha
        if aggExt:
            centerName += '_aggExt'

    if not os.path.exists(os.path.join(lblPath, centerName + '.csv')):
        for i in range(len(kdigos)):
            kdigos[i] = kdigos[i][np.where(days[i] <= t_lim)]
        with PdfPages(os.path.join(lblPath, centerName + '.pdf')) as pdf:
            for lbl in np.unique(lbls):
                idx = np.where(lbls == lbl)[0]
                dm_sel = np.ix_(idx, idx)
                tkdigos = [kdigos[x] for x in idx]
                tdm = squareform(squareform(dm)[dm_sel])
                center, stds, confs = performDBA(tkdigos, tdm, mismatch=mismatch, extension=extension, extraDesc=' for cluster ' + lbl, alpha=alpha, aggExt=aggExt)
                centers[lbl] = center
                plt.figure()
                plt.plot(center)
                plt.fill_between(range(len(center)), center-confs, center+confs)
                plt.xticks(range(0, len(center), 4), ['%d' % x for x in range(len(center))])
                plt.yticks(range(5), ['0', '1', '2', '3', '3D'])
                plt.ylim((-0.5, 4.5))
                plt.title(lbl)
                pdf.savefig(dpi=600)
            arr2csv(os.path.join(lblPath, 'centers.csv'), list(centers.values()), list(centers.keys()))
    else:
        center_vecs, centerNames = load_csv(os.path.join(lblPath, 'centers.csv'), None)
        for tname in list(centerNames):
            centers[tname] = center_vecs[np.where(centerNames == tname)[0][0]]
    if not os.path.exists(os.path.join(lblPath, folderName)):
        os.mkdir(os.path.join(lblPath, folderName))

    if category == 'all':
        merge_group(meta, ids, kdigos, dm, lbls, lblNames, centers, lblPath, cat='1-Im', mismatch=mismatch,
                    extension=extension, mergeType=mergeType, dist=dist, folderName=folderName, normalDTW=normalDTW,
                    alpha=alpha, aggExt=aggExt, plot_centers=plot_centers)
        merge_group(meta, ids, kdigos, dm, lbls, lblNames, centers, lblPath, cat='1-St', mismatch=mismatch,
                    extension=extension, mergeType=mergeType, dist=dist, folderName=folderName, normalDTW=normalDTW,
                    alpha=alpha, aggExt=aggExt, plot_centers=plot_centers)
        merge_group(meta, ids, kdigos, dm, lbls, lblNames, centers, lblPath, cat='1-Ws', mismatch=mismatch,
                    extension=extension, mergeType=mergeType, dist=dist, folderName=folderName, normalDTW=normalDTW,
                    alpha=alpha, aggExt=aggExt, plot_centers=plot_centers)
        merge_group(meta, ids, kdigos, dm, lbls, lblNames, centers, lblPath, cat='2-Im', mismatch=mismatch,
                    extension=extension, mergeType=mergeType, dist=dist, folderName=folderName, normalDTW=normalDTW,
                    alpha=alpha, aggExt=aggExt, plot_centers=plot_centers)
        merge_group(meta, ids, kdigos, dm, lbls, lblNames, centers, lblPath, cat='2-St', mismatch=mismatch,
                    extension=extension, mergeType=mergeType, dist=dist, folderName=folderName, normalDTW=normalDTW,
                    alpha=alpha, aggExt=aggExt, plot_centers=plot_centers)
        merge_group(meta, ids, kdigos, dm, lbls, lblNames, centers, lblPath, cat='2-Ws', mismatch=mismatch,
                    extension=extension, mergeType=mergeType, dist=dist, folderName=folderName, normalDTW=normalDTW,
                    alpha=alpha, aggExt=aggExt, plot_centers=plot_centers)
        merge_group(meta, ids, kdigos, dm, lbls, lblNames, centers, lblPath, cat='3-Im', mismatch=mismatch,
                    extension=extension, mergeType=mergeType, dist=dist, folderName=folderName, normalDTW=normalDTW,
                    alpha=alpha, aggExt=aggExt, plot_centers=plot_centers)
        merge_group(meta, ids, kdigos, dm, lbls, lblNames, centers, lblPath, cat='3-St', mismatch=mismatch,
                    extension=extension, mergeType=mergeType, dist=dist, folderName=folderName, normalDTW=normalDTW,
                    alpha=alpha, aggExt=aggExt, plot_centers=plot_centers)
        merge_group(meta, ids, kdigos, dm, lbls, lblNames, centers, lblPath, cat='3-Ws', mismatch=mismatch,
                    extension=extension, mergeType=mergeType, dist=dist, folderName=folderName, normalDTW=normalDTW,
                    alpha=alpha, aggExt=aggExt, plot_centers=plot_centers)
        merge_group(meta, ids, kdigos, dm, lbls, lblNames, centers, lblPath, cat='3D-Im', mismatch=mismatch,
                    extension=extension, mergeType=mergeType, dist=dist, folderName=folderName, normalDTW=normalDTW,
                    alpha=alpha, aggExt=aggExt, plot_centers=plot_centers)
        merge_group(meta, ids, kdigos, dm, lbls, lblNames, centers, lblPath, cat='3D-St', mismatch=mismatch,
                    extension=extension, mergeType=mergeType, dist=dist, folderName=folderName, normalDTW=normalDTW,
                    alpha=alpha, aggExt=aggExt, plot_centers=plot_centers)
        merge_group(meta, ids, kdigos, dm, lbls, lblNames, centers, lblPath, cat='3D-Ws', mismatch=mismatch,
                    extension=extension, mergeType=mergeType, dist=dist, folderName=folderName, normalDTW=normalDTW,
                    alpha=alpha, aggExt=aggExt, plot_centers=plot_centers)
    else:
        merge_group(meta, ids, kdigos, dm, lbls, lblNames, centers, lblPath, cat=category, mismatch=mismatch,
                    extension=extension, mergeType=mergeType, dist=dist, folderName=folderName, normalDTW=normalDTW,
                    alpha=alpha, aggExt=aggExt, plot_centers=plot_centers)


def merge_group(meta, ids, kdigos, dm, lbls, lblNames, centers, lblPath, cat='1-Im',
                mismatch=lambda x,y: abs(x-y), extension=lambda x:0, dist=braycurtis, mergeType='dba', folderName='merged', extWeight=1, normalDTW=False, alpha=1, aggExt=False, plot_centers=False):
    lblgrp = [x for x in np.unique(lbls) if cat in lblNames[x]]
    if len(lblgrp) < 2:
        return
    if not os.path.exists(os.path.join(lblPath, folderName, cat)):
        os.mkdir(os.path.join(lblPath, folderName, cat))
    idx = np.where(lbls == lblgrp[0])[0]
    for i in range(1, len(lblgrp)):
        idx = np.union1d(idx, np.where(lbls == lblgrp[i])[0])
    dmidx = np.ix_(idx, idx)
    grpLbls = lbls[idx]
    grpCenters = {}
    for lbl in lblgrp:
        grpCenters[lbl] = centers[lbl]

    allProgs = []
    allMorts = []
    for i in range(len(idx)):
        tidx = np.where(meta['ids'][:] == ids[i])[0][0]
        if meta['max_kdigo'][tidx] > meta['max_kdigo_7d'][tidx]:
            allProgs.append(1)
        else:
            allProgs.append(0)
        allMorts.append(meta['died_inp'][tidx])
    allProgs = np.array(allProgs)
    allMorts = np.array(allMorts)

    grpProgs = {}
    grpMorts = {}
    for lbl in lblgrp:
        tidx = np.where(grpLbls == lbl)[0]
        grpProgs[lbl] = np.mean(allProgs[tidx])
        grpMorts[lbl] = np.mean(allMorts[tidx])

    # grpCenters = [centers[x] for x in lblgrp]
    grpKdigos = [kdigos[x] for x in idx]
    grpIds = ids[idx]
    grpDm = squareform(squareform(dm)[dmidx])
    nClust = len(lblgrp)
    progScores = []
    mortScores = []
    sils = []
    with PdfPages(os.path.join(lblPath, folderName, cat, cat + '_merge_visualization.pdf')) as pdf:
        if not os.path.exists(os.path.join(lblPath, folderName, cat, '%d_clusters' % nClust)):
            os.mkdir(os.path.join(lblPath, folderName, cat, '%d_clusters' % nClust))
            arr2csv(os.path.join(lblPath, folderName, cat, '%d_clusters' % nClust, 'clusters.csv'), grpLbls, grpIds, fmt='%s')
            formatted_stats(meta, os.path.join(lblPath, folderName, cat, '%d_clusters' % nClust))
            with PdfPages(os.path.join(lblPath, folderName, cat, '%d_clusters' % nClust, 'centers.pdf')) as pdf1:
                for i in range(len(lblgrp)):
                    tlbl = lblgrp[i]
                    fig = plt.figure()
                    plt.plot(grpCenters[tlbl])
                    plt.xticks(range(0, len(grpCenters[tlbl]), 4), ['%d' % x for x in range(len(grpCenters[tlbl]))])
                    plt.ylim((-0.2, 4.2))
                    plt.xlabel('Days')
                    plt.ylabel('KDIGO')
                    plt.title('Cluster ' + tlbl)
                    pdf1.savefig(dpi=600)
                    plt.close(fig)
        mergeCt = 0
        mergeDist = []
        allDist = []
        progs = np.array([x for x in grpProgs.values()])[:, None]
        morts = np.array([x for x in grpMorts.values()])[:, None]
        progDist = pdist(progs, metric='cityblock')
        mortDist = pdist(morts, metric='cityblock')
        try:
            progScores.append([np.min(progDist), np.max(progDist), np.max(progDist) / (100 - np.min(progDist))])
        except ValueError:
            progScores.append([0, 0, 0])
        try:
            mortScores.append([np.min(mortDist), np.max(mortDist), np.max(mortDist) / (100 - np.min(mortDist))])
        except ValueError:
            mortScores.append([0, 0, 0])
        try:
            sils.append(silhouette_score(squareform(grpDm), grpLbls, metric='precomputed'))
        except ValueError:
            sils.append(0)
        mergeLabels = ['Original']
        extl = []
        curExt = {}
        xext_cum = 0
        yext_cum = 0
        while len(lblgrp) > 2:
            minDist = 1000
            mergeGrp = [0, 0]
            tdist = []
            xpenalties = []
            ypenalties = []
            xipenalties = []
            yipenalties = []
            imismatches = []
            for i in range(len(lblgrp)):
                for j in range(i + 1, len(lblgrp)):
                    lbl1 = lblgrp[i]
                    lbl2 = lblgrp[j]
                    c1 = grpCenters[lbl1]
                    c2 = grpCenters[lbl2]
                    if normalDTW:
                        _, paths = dtw(c1, c2, dist=mismatch)
                        path1 = np.array(paths)[:, 0]
                        path2 = np.array(paths)[:, 1]
                        paths = [path1, path2]
                        xext = 0
                        yext = 0
                        for pi in range(len(paths[0])):
                            xidx = np.where(path1 == pi)[0]
                            yidx = np.where(path2 == pi)[0]
                            for xi in range(1, len(xidx)):
                                xext += (xi) * extension(c1[pi])
                            for yi in range(1, len(yidx)):
                                yext += (yi) * extension(c2[pi])
                    else:
                       _, _, _, paths, xext, yext = dtw_p(c1, c2, mismatch=mismatch, extension=extension, alpha=alpha, aggExt=aggExt)
                    mism = np.sum([np.abs(c1[paths[0][x]] - c2[paths[1][x]]) for x in range(len(paths[0]))])
                    imismatches.append(mism)
                    xipenalties.append(xext)
                    yipenalties.append(yext)
                    if lbl1 in list(curExt):
                        xext += curExt[lbl1]
                    if lbl2 in list(curExt):
                        yext += curExt[lbl2]
                    c1 = c1[paths[0]]
                    c2 = c2[paths[1]]
                    d = dist(c1, c2)
                    tdist.append(d)
                    xpenalties.append(xext)
                    ypenalties.append(yext)
                    if d < minDist:
                        minDist = d
                        mergeGrp = [i, j]
            allDist.append(tdist)
            mergeDist.append(minDist)
            idx1 = np.where(grpLbls == lblgrp[mergeGrp[0]])[0]
            idx2 = np.where(grpLbls == lblgrp[mergeGrp[1]])[0]
            if lblgrp[mergeGrp[0]] not in grpLbls:
                print(lblgrp[mergeGrp[0]] + 'not in labels')
            if lblgrp[mergeGrp[1]] not in grpLbls:
                print(lblgrp[mergeGrp[1]] + 'not in labels')
            plotGrp = copy.deepcopy(lblgrp)
            nlbl = '-'.join((lblgrp[mergeGrp[0]], lblgrp[mergeGrp[1]]))
            mergeLabels.append(lblgrp[mergeGrp[0]] + ' + ' + lblgrp[mergeGrp[1]] + ' -> ' + nlbl)
            vdm = squareform(tdist)
            grpLbls[idx1] = nlbl
            grpLbls[idx2] = nlbl
            sils.append(silhouette_score(squareform(grpDm), grpLbls, metric='precomputed'))
            c2 = grpCenters[lblgrp[mergeGrp[1]]]
            c1 = grpCenters[lblgrp[mergeGrp[0]]]
            del grpCenters[lblgrp[mergeGrp[1]]], grpCenters[lblgrp[mergeGrp[0]]]
            lbl2 = lblgrp.pop(mergeGrp[1])
            lbl1 = lblgrp.pop(mergeGrp[0])
            idx = np.sort(np.concatenate((idx1, idx2)))
            dmidx = np.ix_(idx, idx)
            tkdigos = [grpKdigos[x] for x in idx]
            tdm = squareform(squareform(grpDm)[dmidx])
            pidx = sum([len(lblgrp) - i - 1 for i in range(mergeGrp[0] - 1)]) + (mergeGrp[1] - mergeGrp[0])
            xext_cum = xpenalties[pidx]
            yext_cum = ypenalties[pidx]
            xext_ind = xipenalties[pidx]
            yext_ind = yipenalties[pidx]
            curExt[nlbl] = max(xext_cum, yext_cum)
            if lbl1 in list(curExt):
                del curExt[lbl1]
            if lbl2 in list(curExt):
                del curExt[lbl2]
            del grpProgs[lbl1]
            del grpProgs[lbl2]
            del grpMorts[lbl1]
            del grpMorts[lbl2]
            grpProgs[nlbl] = np.mean(allProgs[idx])
            grpMorts[nlbl] = np.mean(allMorts[idx])

            progs = np.array([x for x in grpProgs.values()])[:, None]
            morts = np.array([x for x in grpMorts.values()])[:, None]
            progDist = pdist(progs, metric='cityblock')
            mortDist = pdist(morts, metric='cityblock')
            progScores.append([np.min(progDist), np.max(progDist), np.max(progDist) / (100 - np.min(progDist))])
            mortScores.append([np.min(mortDist), np.max(mortDist), np.max(mortDist) / (100 - np.min(mortDist))])
            if mergeType == 'dba':
                center, stds, confs = performDBA(tkdigos, tdm, mismatch=mismatch, extension=extension, n_iterations=10, alpha=alpha)
            elif 'mean' in mergeType:
                if normalDTW:
                    _, paths = dtw(c1, c2, dist=mismatch)
                    path1 = np.array(paths)[:, 0]
                    path2 = np.array(paths)[:, 1]
                    path = [path1, path2]
                else:
                    _, _, _, path, xext, yext = dtw_p(c1, c2, mismatch=mismatch, extension=extension, alpha=alpha)
                c1p = c1[path[0]]
                c2p = c2[path[1]]
                if 'weighted' in mergeType:
                    count1 = len(idx1)
                    count2 = len(idx2)
                    w1 = count1 / (count1 + count2)
                    w2 = count2 / (count2 + count1)
                    center = np.array([(w1 * c1p[x]) + (w2 * c2p[x]) for x in range(len(c1p))])
                else:
                    center = np.array([((c1p[x]/2) + (c2p[x]) / 2) for x in range(len(c1p))])

            extl.append(np.max([x for x in curExt.values()]))

            grpCenters[nlbl] = center
            lblgrp.append(nlbl)
            nClust = len(lblgrp)
            os.mkdir(os.path.join(lblPath, folderName, cat, '%d_clusters' % nClust))
            arr2csv(os.path.join(lblPath, folderName, cat, '%d_clusters' % nClust, 'clusters.csv'), grpLbls, grpIds, fmt='%s')
            arr2csv(os.path.join(lblPath, folderName, cat, '%d_clusters' % nClust, 'centers.csv'), list(grpCenters.values()), list(grpCenters.keys()))
            formatted_stats(meta, os.path.join(lblPath, folderName, cat, '%d_clusters' % nClust))
            mergeCt += 1
            fig = plt.figure(figsize=[16, 8])
            gs = GridSpec(4, 4)
            ax = fig.add_subplot(gs[:2, 0])
            _ = ax.plot(c1, label='%d Patients' % len(idx1))
            ax.set_xticks(range(0, len(c1), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(c1))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            ax.legend([extra], ['%d Patients' % len(idx1)])
            ax.set_title('Original Sequences')
            ax.set_ylim((-0.2, 4.2))
            # plt.legend()
            ax = fig.add_subplot(gs[2:, 0])
            p = ax.plot(c2, label='%d Patients' % len(idx2))
            ax.set_xticks(range(0, len(c2), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(c2))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            ax.legend([extra], ['%d Patients' % len(idx2),])
            # ax.set_title('Cluster ' + lbl2, wrap=True)
            ax.set_ylim((-0.2, 4.2))
            ax = fig.add_subplot(gs[:2, 1])
            p = ax.plot(c1p, label='%d Patients' % len(idx1))
            ax.set_xticks(range(0, len(c1p), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(c1p))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            ax.legend([extra, extra], ['%d Patients' % len(idx1), 'Extension: %.1f + %.1f = %.1f' % (xext_ind, xext_cum-xext_ind, xext_cum)])
            ax.set_title('After Alignment')
            ax.set_ylim((-0.2, 4.2))
            # plt.legend()
            ax = fig.add_subplot(gs[2:, 1])
            p = ax.plot(c2p, label='%d Patients' % len(idx2))
            ax.set_xticks(range(0, len(c2p), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(c2p))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            ax.legend([extra, extra], ['%d Patients' % len(idx2),
                                   'Extension: %.1f + %.1f = %.1f' % (yext_ind, yext_cum - yext_ind, yext_cum)])
            # ax.set_title('Cluster ' + lbl2, wrap=True)
            ax.set_ylim((-0.2, 4.2))
            ax.set_title('Merged')
            # plt.legend()
            ax = fig.add_subplot(gs[1:3, 2])
            ax.plot(center, label='%d Patients' % len(idx))
            ax.set_xticks(range(0, len(center), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(center))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            # ax.set_title('Cluster ' + nlbl, wrap=True)
            ax.set_ylim((-0.2, 4.2))
            ax = fig.add_subplot(gs[:2, 3])
            plt.pcolormesh(vdm, linewidth=0, rasterized=True)
            ax.set_xticks(np.arange(vdm.shape[0]) + 0.5)
            ax.set_xticklabels([str(x) for x in plotGrp], rotation=30, ha='right')
            ax.set_yticks(np.arange(vdm.shape[0]) + 0.5)
            ax.set_yticklabels([str(x) for x in plotGrp])
            ax.set_title('Distance Matrix')
            if vdm.shape[0] <= 5:
                fs = 8.
                rot = 0
            elif vdm.shape[0] < 10:
                fs = 4.5
                rot = 0
            elif vdm.shape[0] < 15:
                fs = 3.5
                rot = 0
            else:
                fs = 2
                rot = 45
            for y in range(vdm.shape[0]):
                for x in range(vdm.shape[1]):
                    plt.text(x + 0.5, y + 0.5, '%.3f' % vdm[y, x],
                             horizontalalignment='center',
                             verticalalignment='center',
                             fontsize=fs,
                             rotation=rot
                             )
            ax = fig.add_subplot(gs[2:, 3])
            vdm = np.max(np.dstack((squareform(np.array(xpenalties)), squareform(np.array(ypenalties)))), axis=-1)
            plt.pcolormesh(vdm, linewidth=0, rasterized=True)
            ax.set_xticks(np.arange(vdm.shape[0]) + 0.5)
            ax.set_xticklabels([str(x) for x in plotGrp], rotation=30, ha='right')
            ax.set_yticks(np.arange(vdm.shape[0]) + 0.5)
            ax.set_yticklabels([str(x) for x in plotGrp])
            ax.set_title('Extension Penalties')
            if vdm.shape[0] <= 5:
                fs = 10.7
                rot = 0
            elif vdm.shape[0] < 10:
                fs = 6
                rot = 0
            elif vdm.shape[0] < 15:
                fs = 4.7
                rot = 0
            else:
                fs = 2.6
                rot = 45
            for y in range(vdm.shape[0]):
                for x in range(vdm.shape[1]):
                    plt.text(x + 0.5, y + 0.5, '%.1f' % vdm[y, x],
                             horizontalalignment='center',
                             verticalalignment='center',
                             fontsize=fs,
                             rotation=rot
                             )
            plt.suptitle("Merge #%d (%d Clusters)\n" % (mergeCt, len(lblgrp)) + r"$\bf{%s}$ + $\bf{%s}$" % (lbl1, lbl2), y=0.95, x=0.625, ha='center')
            # plt.colorbar(ax=ax)
            plt.tight_layout()
            pdf.savefig(dpi=600)
            plt.close(fig)
            if plot_centers:
                with PdfPages(os.path.join(lblPath, folderName, cat, '%d_clusters' % nClust, 'centers.pdf')) as pdf1:
                    for i in range(len(lblgrp)):
                        tlbl = lblgrp[i]
                        fig = plt.figure()
                        plt.plot(grpCenters[tlbl])
                        plt.xticks(range(0, len(grpCenters[tlbl]), 4), ['%d' % x for x in range(len(grpCenters[tlbl]))])
                        plt.ylim((-0.2, 4.2))
                        plt.xlabel('Days')
                        plt.ylabel('KDIGO')
                        plt.title('Cluster ' + tlbl, wrap=True)
                        pdf1.savefig(dpi=600)
                        plt.close(fig)
        fig = plt.figure()
        plt.plot(mergeDist)
        plt.xticks(range(len(mergeDist)), ['%d' % (x + 1) for x in range(len(mergeDist))])
        plt.xlabel('Merge Number')
        plt.ylabel('Distance')
        plt.title('Individual Merge Distances')
        pdf.savefig(dpi=600)
        plt.close(fig)
        fig = plt.figure()
        plt.plot(extl)
        plt.xticks(range(len(extl)), ['%d' % (x + 1) for x in range(len(mergeDist))])
        plt.xlabel('Merge Number')
        plt.ylabel('Extension Penalty')
        plt.title('Maximum Cumulative Extension Penalty')
        pdf.savefig(dpi=600)
        plt.close(fig)
    np.savetxt(os.path.join(lblPath, folderName, cat, 'merge_distances.txt'), mergeDist)
    arr2csv(os.path.join(lblPath, folderName, cat, 'all_distances.csv'), allDist, mergeLabels)
    arr2csv(os.path.join(lblPath, folderName, cat, 'prog_scores.csv'), progScores, mergeLabels,
            header='Merge,MinProgDiff,MaxProgDiff,RelProgDiff')
    arr2csv(os.path.join(lblPath, folderName, cat, 'mort_scores.csv'), mortScores, mergeLabels,
            header='Merge,MinMortDiff,MaxMortDiff,RelMortDiff')
    arr2csv(os.path.join(lblPath, folderName, cat, 'silhouette_scores.csv'), sils, mergeLabels,
            header='Merge,Silhouette')
    progScores = np.array(progScores)
    mortScores = np.array(mortScores)
    with PdfPages(os.path.join(lblPath, folderName, cat, cat + '_merge_evaluation.pdf')) as pdf:
        fig = plt.figure()
        plt.plot(progScores[:, 0])
        plt.xlabel('Merge #')
        plt.ylabel('Value')
        plt.title('MinProgDiff')
        pdf.savefig(dpi=600)
        plt.close(fig)
        fig = plt.figure()
        plt.plot(progScores[:, 1])
        plt.xlabel('Merge #')
        plt.ylabel('Value')
        plt.title('MaxProgDiff')
        pdf.savefig(dpi=600)
        plt.close(fig)
        fig = plt.figure()
        plt.plot(progScores[:, 2])
        plt.xlabel('Merge #')
        plt.ylabel('Value')
        plt.title('RelProgDiff')
        pdf.savefig(dpi=600)
        plt.close(fig)
        fig = plt.figure()
        plt.plot(mortScores[:, 0])
        plt.xlabel('Merge #')
        plt.ylabel('Value')
        plt.title('MinMortDiff')
        pdf.savefig(dpi=600)
        plt.close(fig)
        fig = plt.figure()
        plt.plot(mortScores[:, 1])
        plt.xlabel('Merge #')
        plt.ylabel('Value')
        plt.title('MaxMortDiff')
        pdf.savefig(dpi=600)
        plt.close(fig)
        fig = plt.figure()
        plt.plot(mortScores[:, 2])
        plt.xlabel('Merge #')
        plt.ylabel('Value')
        plt.title('RelMortDiff')
        pdf.savefig(dpi=600)
        plt.close(fig)
