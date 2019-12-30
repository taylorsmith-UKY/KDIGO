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
from dtw_distance import dtw_p, evalExtension, get_custom_distance_discrete
from stat_funcs import formatted_stats
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from DBA import performDBA
import os
import copy
from utility_funcs import load_csv, arr2csv, get_dm_tag
from fastdtw import dtw
from matplotlib import rcParams
import tqdm
import subprocess


# %%
def cluster_trajectories(f, stats, ids, mk, dm, eps=0.015, n_clusters=2, data_path=None, interactive=True,
                         save=None, plot_daily=False, kdigos=None, days=None):
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

    if save is not None:
        if not os.path.exists(os.path.join(save, 'flat')):
            fig = plt.figure()
            dendrogram(link, 50, truncate_mode='lastp')
            plt.xlabel('Patients')
            plt.ylabel('Distance')
            plt.tight_layout()
            os.mkdir(os.path.join(save, 'flat'))
            plt.savefig(os.path.join(save, 'flat', 'dendrogram.png'), dpi=600)
            plt.close(fig)
    # else:
    #     plt.show()
    #     pass

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
                formatted_stats(stats, save_path)
                print('%d clusters has already been saved' % n_clusters)
                if not interactive:
                    cont = False
                continue
            os.mkdir(save_path)
            arr2csv(os.path.join(save_path, 'clusters.csv'), lbls_sel, ids, fmt='%d')
            formatted_stats(stats, save_path)
            if not os.path.exists(os.path.join(save_path, 'rename')) and kdigos is not None:
                os.mkdir(os.path.join(save_path, 'rename'))
            if kdigos is not None:
                nlbls, clustCats = clusterCategorizer(mk, kdigos, days, lbls_sel)
                arr2csv(os.path.join(save_path, 'rename', 'clusters.csv'), nlbls, ids, fmt='%s')
                formatted_stats(stats, os.path.join(save_path, 'rename'))
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
def clusterCategorizer(max_kdigos, kdigos, days, lbls, t_lim=7):
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
            maxDay = min(t_lim, max(days[idx[i]]))
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


def centerCategorizer(centers, useTransient=False, stratifiedRecovery=False):
    if type(centers) == list or type(centers) == np.ndarray:
        cats = []
    else:
        cats = {}
    for i, center in enumerate(centers):
        # if i >= 100:
        #     print('Reached 1-Im')
        if type(centers) == list or type(centers) == np.ndarray:
            c = center
        else:
            c = centers[center]
        v = round(max(c))
        if v == 4:
            s = '3D-'
        else:
            s = '%d-' % v
        if c[-1] - c[0] > 0.5:
            s += 'Ws'
        elif c[0] - c[-1] > 0.5:

            if stratifiedRecovery:
                if c[-1] < 0.5:
                    s += 'compIm'
                else:
                    s += 'partIm'
            else:
                s += 'Im'
        elif max(c) - c[0] > 0.5 and useTransient:
            s += 'Tr'
        else:
            s += 'St'
        if type(centers) == list or type(centers) == np.ndarray:
            cats.append(s)
        else:
            cats[center] = s
    return cats


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
def eval_clusters(ids, max_kdigos, kdigos, days, inp_death, basePath, minClust=36, maxClust=96):
    cats = ["%s-%s" % (x, y) for x in ["1", "2", "3", "3D"] for y in ["Im", "St", "Ws"]]
    evals = {}
    for cat in cats:
        evals[cat] = []

    for nClust in range(minClust, maxClust + 1):
        labels = load_csv(os.path.join(basePath, "%d_clusters" % nClust, "clusters.csv"), ids, int)
        catLabels, _ = clusterCategorizer(max_kdigos, kdigos, days, labels, t_lim=14)
        catNames = np.unique(catLabels)
        for cat in cats:
            dcounts = []
            for tcat in catNames:
                if cat in tcat:
                    idx = np.where(catLabels == tcat)[0]
                    dcounts.append(np.sum(inp_death[idx]) / len(idx) * 100)
            if len(dcounts) > 0:
                evals[cat].append(max(dcounts) - min(dcounts))
            else:
                evals[cat].append(0)
    return evals



# def eval_clusters(dm, ids, label_path, labels_true=None):
#     '''
#     Evaluate clustering performance. If no true labels provided, will only evaluate using Silhouette Score and
#     Calinski Harabaz score.
#     :param data_file: filename or file handle for hdf5 file containing patient statistics
#     :param label_path: fully qualified path to directory containing cluster labels
#     :param sel:
#     :param labels_true:
#     :return:
#     '''
#     if dm.ndim == 1:
#         sqdm = squareform(dm)
#     else:
#         sqdm = dm
#     clusters = load_csv(os.path.join(label_path, 'clusters.csv'), ids, str)
#     ss = silhouette_score(sqdm, clusters, metric='precomputed')
#     chs = calinski_harabaz_score(sqdm, clusters)
#     if labels_true is None:
#         print('Silhouette Score: %.4f' % ss)
#         print('Calinski Harabaz Score: %.4f' % chs)
#         return ss, chs
#     else:
#         ars = adjusted_rand_score(clusters, labels_true)
#         nmi = normalized_mutual_info_score(clusters, labels_true)
#         hs = homogeneity_score(clusters, labels_true)
#         cs = completeness_score(clusters, labels_true)
#         vms = v_measure_score(clusters, labels_true)
#         fms = fowlkes_mallows_score(clusters, labels_true)
#
#         print('Silhouette Score: %.4f' % ss)
#         print('Calinski Harabaz Score: %.4f' % chs)
#         print('Adjusted Rand Index: %.4f' % ars)
#         print('Normalize Mutual Information: %.4f' % nmi)
#         print('Homogeneity Score: %.4f' % hs)
#         print('Completeness Score: %.4f' % cs)
#         print('V Measure Score: %.4f' % vms)
#         print('Fowlkes-Mallows Score: %.4f' % fms)
#         return ss, chs, ars, nmi, hs, cs, vms, fms


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


def merge_clusters(kdigos, days, dm, lblPath, meta, args, mismatch=lambda x,y: abs(x-y),
                   extension=lambda x:0, folderName='merged', dist=braycurtis):
    ids = np.loadtxt(os.path.join(lblPath, 'clusters.csv'), delimiter=',', usecols=0, dtype=int)
    assert len(ids) == len(kdigos)
    lbls = load_csv(os.path.join(lblPath, 'clusters.csv'), ids, str)
    lbls = np.array(lbls, dtype='|S100').astype(str)
    # catLbls, lblNames = clusterCategorizer(max_kdigos, kdigos, days, lbls, t_lim=args.t_lim)
    lblPath1 = os.path.join(lblPath, "dba_centers")
    if not os.path.exists(lblPath1):
        os.mkdir(lblPath1)

    lblPath1 = os.path.join(lblPath1, folderName)
    if not os.path.exists(lblPath1):
        os.mkdir(lblPath1)
    lblPath1 = os.path.join(lblPath1, args.seedType)
    if not os.path.exists(lblPath1):
        os.mkdir(lblPath1)
    lblPath1 = os.path.join(lblPath1, '%ddays' % args.clen)
    if not os.path.exists(lblPath1):
        os.mkdir(lblPath1)

    _, dbaTag = get_dm_tag(args.pdtw, args.malpha, False, True, 'braycurtis', 0.0, 'none')

    if os.path.exists(os.path.join(lblPath1, 'centers.csv')):
        centers = load_csv(os.path.join(lblPath1, 'centers.csv'), np.unique(lbls), struct='dict', id_dtype=str)
        stds = load_csv(os.path.join(lblPath1, 'stds.csv'), np.unique(lbls), struct='dict', id_dtype=str)
        confs = load_csv(os.path.join(lblPath1, 'confs.csv'), np.unique(lbls), struct='dict', id_dtype=str)
    else:
        centers = {}
        stds = {}
        confs = {}
    for i in range(len(kdigos)):
        kdigos[i] = kdigos[i][np.where(days[i] <= args.t_lim)]

    if len(centers) < len(np.unique(lbls)):
        if os.path.exists(os.path.join(lblPath1, 'centers.csv')):
            cf = open(os.path.join(lblPath1, 'centers.csv'), 'a')
            cof = open(os.path.join(lblPath1, 'confs.csv'), 'a')
            sf = open(os.path.join(lblPath1, 'stds.csv'), 'a')
        else:
            cf = open(os.path.join(lblPath1, 'centers.csv'), 'w')
            cof = open(os.path.join(lblPath1, 'confs.csv'), 'w')
            sf = open(os.path.join(lblPath1, 'stds.csv'), 'w')

        pdfName = 'centers'
        if os.path.exists(os.path.join(lblPath1, pdfName + '.pdf')):
            pdfName += '_a'
            while os.path.exists(os.path.join(lblPath1, pdfName + '.pdf')):
                pdfName = pdfName[:-1] + chr(ord(pdfName[-1])+1)

        with PdfPages(os.path.join(lblPath1, pdfName + '.pdf')) as pdf:
            for lbl in np.unique(lbls):
                if lbl in centers.keys():
                    print('Center for ' + lbl + ' already computed.')
                    continue
                idx = np.where(lbls == lbl)[0]
                dm_sel = np.ix_(idx, idx)
                tkdigos = [kdigos[x] for x in idx]
                tdm = squareform(squareform(dm)[dm_sel])
                center, std, conf, paths = performDBA(tkdigos, tdm, mismatch=mismatch, extension=extension,
                                                      extraDesc=' for cluster %s (%d patients)' % (lbl, len(idx)),
                                                      alpha=args.alpha, aggExt=False, targlen=args.clen,
                                                      seedType=args.seedType, n_iterations=args.dbaiter)
                centers[lbl] = center
                stds[lbl] = std
                confs[lbl] = conf
                fig = plt.figure()
                plt.plot(center)
                plt.fill_between(range(len(center)), center-conf, center+conf, alpha=0.4)
                plt.xticks(range(0, len(center), 4), ['%d' % x for x in range(len(center))])
                plt.yticks(range(5), ['0', '1', '2', '3', '3D'])
                plt.ylim((-0.5, 4.5))
                plt.title(lbl)
                pdf.savefig(dpi=600)
                cf.write('%s,' % lbl + ','.join(['%.3g' % x for x in center]) + '\n')
                cof.write('%s,' % lbl + ','.join(['%.3g' % x for x in conf]) + '\n')
                sf.write('%s,' % lbl + ','.join(['%.3g' % x for x in std]) + '\n')
                plt.close(fig)
        cf.close()
        cof.close()
        sf.close()

    dbaTag = 'merged_' + dbaTag

    if args.cat[0] == 'all':
        for tcat in ['1-Im', '1-St', '1-Ws', '2-Im', '2-St', '2-Ws', '3-Im', '3-St', '3-Ws', '3D-Im', '3D-St', '3D-Ws']:
            merge_group(meta, ids, kdigos, dm, lbls, centers, lblPath, args, cat=tcat, mismatch=mismatch,
                        extension=extension, dist=dist)
    elif args.cat[0] == 'allk':
        for tcat in ['1', '2', '3', '3D']:
            merge_group(meta, ids, kdigos, dm, lbls, centers, lblPath, args, cat=tcat, mismatch=mismatch,
                        extension=extension, dist=dist)
    elif args.cat[0] == 'none':
        return
    else:
        for tcat in args.cat:
            merge_group(meta, ids, kdigos, dm, lbls, centers, lblPath, args, cat=tcat, mismatch=mismatch,
                        extension=extension, dist=dist)


def merge_simulated_sequences(ids, sequences, labels, args, mismatch=lambda x,y: abs(x-y), extension=lambda x:0, dist=braycurtis, basePath=''):

    lblPath = os.path.join(basePath, 'merged')
    if not os.path.exists(lblPath):
        os.mkdir(lblPath)

    nameTag = basePath.split("/")[-1]

    if not os.path.exists(lblPath):
        os.mkdir(lblPath)

    lblgrp = list(ids)
    grpLbls = copy.copy(ids)
    grpIds = copy.copy(ids)

    grpSequences = {}
    for i, tid in enumerate(ids):
        grpSequences[tid] = sequences[i]

    nClust = len(lblgrp)

    with PdfPages(os.path.join(lblPath, 'merge_visualization.pdf')) as pdf:
        pdf2 = PdfPages(os.path.join(lblPath, 'ordered_merge_trajectories.pdf'))

        if not os.path.exists(os.path.join(lblPath, '%d_clusters' % nClust)):
            os.mkdir(os.path.join(lblPath, '%d_clusters' % nClust))
            arr2csv(os.path.join(lblPath, '%d_clusters' % nClust, 'clusters.csv'), grpLbls, grpIds, fmt='%s')
            arr2csv(os.path.join(lblPath, '%d_clusters' % nClust, 'sequences.csv'),
                    list(grpSequences.values()), list(grpSequences.keys()), fmt='%.3f')
            with PdfPages(os.path.join(lblPath, '%d_clusters' % nClust, 'centers.pdf')) as pdf1:
                for i in range(len(lblgrp)):
                    tlbl = lblgrp[i]
                    fig = plt.figure()
                    plt.plot(grpSequences[tlbl])
                    plt.xticks(range(0, len(grpSequences[tlbl]), 4), ['%d' % x for x in range(len(grpSequences[tlbl]))])
                    plt.ylim((-0.2, 4.2))
                    plt.xlabel('Days')
                    plt.ylabel('KDIGO')
                    plt.title('Cluster ' + tlbl)
                    pdf1.savefig(dpi=600)
                    plt.close(fig)

        mergeLabels = ['Original']
        mextl = []
        cextl = []
        iextl = []
        curExt = {}
        cellsPerRow = 17
        cellsPerCol = 8
        nBigRows = int(np.ceil((len(lblgrp) - 1) / 5))
        bigFig = plt.figure(figsize=[22, (nBigRows * 6.0)])
        aBigFig = plt.figure(figsize=[22, (nBigRows * 6.0)])
        bigGS = GridSpec(nBigRows * cellsPerRow, 5 * cellsPerCol)
        rcParams['font.size'] = 10
        # t = tqdm.tqdm(total=len(lblgrp), desc='Merging %d clusters' % len(lblgrp), unit='merge')

        mergeDistMat = squareform(np.load(os.path.join(basePath, "mergeDist_%s.npy" % nameTag)))
        xExtMat = squareform(np.load(os.path.join(basePath, "xExt_%s.npy" % nameTag)))
        yExtMat = squareform(np.load(os.path.join(basePath, "yExt_%s.npy" % nameTag)))
        pureDistMat = squareform(np.load(os.path.join(basePath, "pureDist_%s.npy" % nameTag)))

        extMat = np.max(np.dstack((xExtMat, yExtMat)), axis=2)
        cumXExtMat = np.array(xExtMat)
        cumYExtMat = np.array(yExtMat)

        # np.save(os.path.join(setPath, "mergeDist%s.npy" % nameTag), mergeDists)
        # np.save(os.path.join(setPath, "pureDist%s.npy" % nameTag), pureDists)
        # np.save(os.path.join(setPath, "xExt%s.npy" % nameTag), xexts)
        # np.save(os.path.join(setPath, "yExt%s.npy" % nameTag), yexts)
        # np.save(os.path.join(setPath, "mergeExt%s.npy" % nameTag), mergeExt)

        mergeCt = 0
        indMergeDist = []  # Individual Distance + ExtDistWeight * Extension
        cumMergeDist = []  # Cumulative Distance + ExtDistWeight * Extension
        indPureDist = []  # Individual Distance
        cumPureDist = []  # Cumulative Distance
        allPairwiseDist = []
        t = tqdm.tqdm(total=len(lblgrp) - 5, desc="Merging %s" % nameTag, unit="merge")
        while len(lblgrp) > len(np.unique(labels)):
            mergeGrp = [0, 0]
            pureDist = []
            cumXpenalties = []
            cumYpenalties = []
            xPenalties = []
            yPenalties = []
            pwiseIterationDist = []
            ct = 0
            minDist = 1000000
            # Add call to C++ program to perform DTW
            for i in range(len(lblgrp)):
                for j in range(i + 1, len(lblgrp)):
                    lbl1 = lblgrp[i]
                    lbl2 = lblgrp[j]
                    c1 = grpSequences[lbl1]
                    c2 = grpSequences[lbl2]
                    if (mergeDistMat is not None and (i < len(lblgrp) - 1) and (j < len(lblgrp) - 1)) or (mergeCt == 0):
                        d = mergeDistMat[i, j]
                        thisPureDist = pureDistMat[i, j]
                    else:
                        if not args.pdtw:
                            _, paths = dtw(c1, c2)
                            path1 = np.array(paths)[:, 0]
                            path2 = np.array(paths)[:, 1]
                            paths = [path1, path2]
                        else:
                            _, _, _, paths, xext, yext = dtw_p(c1, c2, mismatch, extension, args.malpha)

                        cxext = copy.copy(xext)
                        cyext = copy.copy(yext)
                        if lbl1 in list(curExt):
                            cxext += curExt[lbl1]
                        if lbl2 in list(curExt):
                            cyext += curExt[lbl2]

                        d = dist(c1[paths[0]], c2[paths[1]])
                        thisPureDist = copy.copy(d)
                        if args.extDistWeight > 0:
                            if args.cumExtDist:
                                d += args.extDistWeight * (cxext + cyext)
                            else:
                                d += args.extDistWeight * (xext + yext)

                        if mergeDistMat is not None:
                            mergeDistMat[i, j] = d
                            mergeDistMat[j, i] = d

                            extMat[i, j] = max(cxext, cyext)
                            extMat[j, i] = max(cxext, cyext)

                            xExtMat[i, j] = xext
                            xExtMat[j, i] = xext

                            yExtMat[i, j] = yext
                            yExtMat[j, i] = yext

                            cumXExtMat[i, j] = cxext
                            cumXExtMat[j, i] = cxext

                            cumYExtMat[i, j] = cyext
                            cumYExtMat[j, i] = cyext

                        else:
                            cumXpenalties.append(cxext)
                            cumYpenalties.append(cyext)
                            xPenalties.append(xext)
                            yPenalties.append(yext)

                    pwiseIterationDist.append(d)
                    if len(pwiseIterationDist) == 1 or d < minDist:
                        if args.maxExt < 0 or max(xext, yext) < args.maxExt:
                            mergeGrp = [i, j]
                            minDist = d
                    ct += 1
                    pureDist.append(thisPureDist)
            t.update()
            allPairwiseDist.append(pwiseIterationDist)
            indMergeDist.append(minDist)

            if len(cumMergeDist) == 0:
                cumMergeDist.append(minDist)
            else:
                cumMergeDist.append(cumMergeDist[-1] + minDist)

            indPureDist.append(pureDistMat[mergeGrp[0], mergeGrp[1]])

            if len(cumPureDist) == 0:
                cumPureDist.append(pureDistMat[mergeGrp[0], mergeGrp[1]])
            else:
                cumPureDist.append(cumPureDist[-1] + pureDistMat[mergeGrp[0], mergeGrp[1]])

            idx1 = np.where(grpLbls == lblgrp[mergeGrp[0]])[0]
            idx2 = np.where(grpLbls == lblgrp[mergeGrp[1]])[0]
            if lblgrp[mergeGrp[0]] not in grpLbls:
                print(lblgrp[mergeGrp[0]] + 'not in labels')
            if lblgrp[mergeGrp[1]] not in grpLbls:
                print(lblgrp[mergeGrp[1]] + 'not in labels')
            plotGrp = copy.deepcopy(lblgrp)
            nlbl = '-'.join((lblgrp[mergeGrp[0]], lblgrp[mergeGrp[1]]))
            mergeLabels.append(lblgrp[mergeGrp[0]] + ' + ' + lblgrp[mergeGrp[1]] + ' -> ' + nlbl)
            grpLbls[idx1] = nlbl
            grpLbls[idx2] = nlbl
            c2 = grpSequences[lblgrp[mergeGrp[1]]]
            c1 = grpSequences[lblgrp[mergeGrp[0]]]

            try:
                del grpSequences[lblgrp[mergeGrp[1]]], grpSequences[lblgrp[mergeGrp[0]]]
            except KeyError:
                pass
            lbl2 = lblgrp.pop(mergeGrp[1])
            lbl1 = lblgrp.pop(mergeGrp[0])

            xext_cum = cumXExtMat[mergeGrp[0], mergeGrp[1]]
            yext_cum = cumYExtMat[mergeGrp[0], mergeGrp[1]]
            xext_ind = xExtMat[mergeGrp[0], mergeGrp[1]]
            yext_ind = yExtMat[mergeGrp[0], mergeGrp[1]]

            curExt[nlbl] = max(xext_cum, yext_cum)
            if lbl1 in list(curExt):
                del curExt[lbl1]
            if lbl2 in list(curExt):
                del curExt[lbl2]

            if args.pdtw:
                _, _, _, path, xext, yext = dtw_p(c1, c2, mismatch=mismatch, extension=extension, alpha=args.alpha)
            else:
                _, paths = dtw(c1, c2, dist=mismatch)
                path1 = np.array(paths)[:, 0]
                path2 = np.array(paths)[:, 1]
                path = [path1, path2]

            c1p = c1[path[0]]
            c2p = c2[path[1]]

            if args.mergeType == 'dba':
                center, stds, confs, paths = performDBA(tkdigos, tdm, mismatch=mismatch, extension=extension,
                                                        n_iterations=args.dbaiter, seedType=args.seedType)
            elif 'mean' in args.mergeType:
                if 'weighted' in args.mergeType:
                    count1 = len(idx1)
                    count2 = len(idx2)
                    w1 = count1 / (count1 + count2)
                    w2 = count2 / (count2 + count1)
                    center = np.array([(w1 * c1p[x]) + (w2 * c2p[x]) for x in range(len(c1p))])
                else:
                    center = np.array([((c1p[x] / 2) + (c2p[x]) / 2) for x in range(len(c1p))])

            mextl.append(np.max([x for x in curExt.values()]))
            cextl.append(np.sum([x for x in curExt.values()]))
            iextl.append(curExt[nlbl])

            grpSequences[nlbl] = center
            lblgrp.append(nlbl)
            nClust = len(lblgrp)
            os.mkdir(os.path.join(lblPath, '%d_clusters' % nClust))
            arr2csv(os.path.join(lblPath, '%d_clusters' % nClust, 'clusters.csv'), grpLbls, grpIds, fmt='%s')
            arr2csv(os.path.join(lblPath, '%d_clusters' % nClust, 'centers.csv'), list(grpSequences.values()),
                    list(grpSequences.keys()))

            # Plot merge summary document. Shows trajectories before and after alignment and then the resulting new
            # center trajectory.
            fig = plt.figure(figsize=[16, 8])
            gs = GridSpec(4, 4)

            # First original sequence
            ax = fig.add_subplot(gs[:2, 0])
            _ = ax.plot(c1)
            ax.set_xticks(range(0, len(c1), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(c1))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            ax.set_title('Original Sequences')
            ax.set_ylim((-0.2, 4.2))

            # Second original sequence
            ax = fig.add_subplot(gs[2:, 0])
            _ = ax.plot(c2)
            ax.set_xticks(range(0, len(c2), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(c2))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            # ax.set_title('Cluster ' + lbl2, wrap=True)
            ax.set_ylim((-0.2, 4.2))

            # First sequence aligned
            ax = fig.add_subplot(gs[:2, 1])
            _ = ax.plot(c1p)
            ax.set_xticks(range(0, len(c1p), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(c1p))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            ax.legend([extra], ['Extension: %.1f + %.1f = %.1f' %
                                       (xext_ind, xext_cum - xext_ind, xext_cum)])
            ax.set_title('After Alignment')
            ax.set_ylim((-0.2, 4.2))

            ax = fig.add_subplot(gs[2:, 1])
            _ = ax.plot(c2p)
            ax.set_xticks(range(0, len(c2p), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(c2p))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            ax.legend([extra], ['Extension: %.1f + %.1f = %.1f' % (yext_ind, yext_cum - yext_ind, yext_cum)])
            # ax.set_title('Cluster ' + lbl2, wrap=True)
            ax.set_ylim((-0.2, 4.2))
            ax.set_title('Merged')

            # New center after merging
            ax = fig.add_subplot(gs[1:3, 2])
            ax.plot(center)
            ax.set_xticks(range(0, len(center), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(center))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            # ax.set_title('Cluster ' + nlbl, wrap=True)
            ax.set_ylim((-0.2, 4.2))

            # Current distance matrix
            ax = fig.add_subplot(gs[:2, 3])
            plt.pcolormesh(mergeDistMat, linewidth=0, rasterized=True)
            ax.set_xticks(np.arange(mergeDistMat.shape[0]) + 0.5)
            ax.set_xticklabels([str(x) for x in plotGrp], rotation=30, ha='right')
            ax.set_yticks(np.arange(mergeDistMat.shape[0]) + 0.5)
            ax.set_yticklabels([str(x) for x in plotGrp])
            ax.set_title('Distance Matrix')
            if mergeDistMat.shape[0] <= 5:
                fs = 8.
                rot = 0
            elif mergeDistMat.shape[0] < 10:
                fs = 4.5
                rot = 0
            elif mergeDistMat.shape[0] < 15:
                fs = 3.5
                rot = 0
            else:
                fs = 2
                rot = 45
            for y in range(mergeDistMat.shape[0]):
                for x in range(mergeDistMat.shape[1]):
                    plt.text(x + 0.5, y + 0.5, '%.3f' % mergeDistMat[y, x],
                             horizontalalignment='center',
                             verticalalignment='center',
                             fontsize=fs,
                             rotation=rot
                             )

            # Extension penalty matrix

            ax = fig.add_subplot(gs[2:, 3])
            plt.pcolormesh(extMat, linewidth=0, rasterized=True)
            ax.set_xticks(np.arange(extMat.shape[0]) + 0.5)
            ax.set_xticklabels([str(x) for x in plotGrp], rotation=30, ha='right')
            ax.set_yticks(np.arange(extMat.shape[0]) + 0.5)
            ax.set_yticklabels([str(x) for x in plotGrp])
            ax.set_title('Cumulative Extension Penalties')
            if extMat.shape[0] <= 5:
                fs = 10.7
                rot = 0
            elif extMat.shape[0] < 10:
                fs = 6
                rot = 0
            elif extMat.shape[0] < 15:
                fs = 4.7
                rot = 0
            else:
                fs = 2.6
                rot = 45
            for y in range(extMat.shape[0]):
                for x in range(extMat.shape[1]):
                    plt.text(x + 0.5, y + 0.5, '%.1f' % extMat[y, x],
                             horizontalalignment='center',
                             verticalalignment='center',
                             fontsize=fs,
                             rotation=rot
                             )
            plt.suptitle(
                "Merge #%d (%d Clusters)\n" % (mergeCt + 1, len(lblgrp)) + r"$\bf{%s}$ + $\bf{%s}$" % (lbl1, lbl2),
                y=0.95, x=0.625, ha='center')
            # plt.colorbar(ax=ax)
            plt.tight_layout()
            pdf.savefig(dpi=600)
            plt.close(fig)

            # First original sequence
            tv = int(np.floor(mergeCt / 5)) * cellsPerRow
            colStart = (mergeCt % 5) * cellsPerCol
            ax = bigFig.add_subplot(bigGS[tv:tv + 5, colStart:colStart + cellsPerCol - 2])
            _ = ax.plot(c1)
            ax.set_xticks(range(0, len(c1), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(c1))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            ax.set_title("Merge #%d" % (mergeCt + 1))
            ax.set_ylim((-0.2, 4.2))

            # Second original sequence
            ax = bigFig.add_subplot(bigGS[tv + 7:tv + 13, colStart:colStart + cellsPerCol - 2])
            _ = ax.plot(c2)
            ax.set_xticks(range(0, len(c2), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(c2))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            ax.set_title(" ")
            ax.set_ylim((-0.2, 4.2))

            # First sequence alignedtv = int(np.floor(mergeCt / 5)) * cellsPerRow
            ax = aBigFig.add_subplot(bigGS[tv:tv + 5, colStart:colStart + cellsPerCol - 2])
            _ = ax.plot(c1p)
            ax.set_xticks(range(0, len(c1p), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(c1p))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            ax.legend([extra], ['Extension: %.1f + %.1f = %.1f' % (xext_ind, xext_cum - xext_ind, xext_cum)])
            ax.set_title('Merge #%d' % (mergeCt + 1))
            ax.set_ylim((-0.2, 4.2))

            # Second sequence after alignment
            ax = aBigFig.add_subplot(bigGS[tv + 7:tv + 13, colStart:colStart + cellsPerCol - 2])
            _ = ax.plot(c2p)
            ax.set_xticks(range(0, len(c2p), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(c2p))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            ax.set_title(" ")
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            ax.legend([extra], ['Extension: %.1f + %.1f = %.1f' % (yext_ind, yext_cum - yext_ind, yext_cum)])
            ax.set_ylim((-0.2, 4.2))
            mergeCt += 1

            mergeDistMat = newMatrixAfterRemoving(mergeDistMat, mergeGrp)
            extMat = newMatrixAfterRemoving(extMat, mergeGrp)
            pureDistMat = newMatrixAfterRemoving(pureDistMat, mergeGrp)
            xExtMat = newMatrixAfterRemoving(xExtMat, mergeGrp)
            yExtMat = newMatrixAfterRemoving(yExtMat, mergeGrp)
            cumXExtMat = newMatrixAfterRemoving(cumXExtMat, mergeGrp)
            cumYExtMat = newMatrixAfterRemoving(cumYExtMat, mergeGrp)

            if args.plot_centers:
                with PdfPages(os.path.join(lblPath, '%d_clusters' % nClust, 'centers.pdf')) as pdf1:
                    for i in range(len(lblgrp)):
                        tlbl = lblgrp[i]
                        fig = plt.figure()
                        plt.plot(grpSequences[tlbl])
                        plt.xticks(range(0, len(grpSequences[tlbl]), 4),
                                   ['%d' % x for x in range(len(grpSequences[tlbl]))])
                        plt.ylim((-0.2, 4.2))
                        plt.xlabel('Days')
                        plt.ylabel('KDIGO')
                        plt.title('Cluster ' + tlbl, wrap=True)
                        pdf1.savefig(dpi=600)
                        plt.close(fig)
        t.close()
        bigFig.text(0.5, 0.975, "Merges In Order - Original Sequences", ha='center', fontsize=20, fontweight='bold')
        bigFig.subplots_adjust(top=0.95, bottom=0.0, left=0.055, right=1.0)
        pdf2.savefig(bigFig, dpi=600)
        plt.close(bigFig)

        aBigFig.text(0.5, 0.975, "Merges In Order - After Alignment", ha='center', fontsize=20, fontweight='bold')
        aBigFig.subplots_adjust(top=0.95, bottom=0.0, left=0.055, right=1.0)
        pdf2.savefig(aBigFig, dpi=600)
        plt.close(aBigFig)

        # Finally, plot the distance and extension penalty vs the number of merges.
        # Individual Merge Distance (may include extension)
        if args.extDistWeight > 0:
            fig = plt.figure()
            plt.plot(indMergeDist)
            plt.xticks(range(len(indMergeDist)), ['%d' % (x + 1) for x in range(len(indMergeDist))])
            plt.xlabel('Merge Number')
            plt.ylabel('Distance')
            if args.scaleExt:
                plt.title('Individual Merge Distances + %.3g * Extension Penalty (scaled by length)' %
                          args.extDistWeight)
            else:
                plt.title('Individual Merge Distances + %.3g * Extension Penalty' % args.extDistWeight)
            pdf.savefig(fig, dpi=600)
            pdf2.savefig(fig, dpi=600)
            plt.close(fig)

            # Cumulative Distance (may include extension)
            fig = plt.figure()
            plt.plot(cumMergeDist)
            plt.xticks(range(len(cumMergeDist)), ['%d' % (x + 1) for x in range(len(cumMergeDist))])
            plt.xlabel('Merge Number')
            plt.ylabel('Distance')
            if args.scaleExt:
                plt.title('Cumulative Merge Distance + %.3g * Extension Penalty (scaled by length)' %
                          args.extDistWeight)
            else:
                plt.title('Cumulative Merge Distance + %.3g * Extension Penalty' % args.extDistWeight)
            pdf.savefig(fig, dpi=600)
            pdf2.savefig(fig, dpi=600)
            plt.close(fig)

        # Individual Merge Distance (does not include extension)
        fig = plt.figure()
        plt.plot(indPureDist)
        plt.xticks(range(len(indPureDist)), ['%d' % (x + 1) for x in range(len(indPureDist))])
        plt.xlabel('Merge Number')
        plt.ylabel('Distance')
        plt.title('Individual Pure Merge Distances')
        pdf.savefig(fig, dpi=600)
        pdf2.savefig(fig, dpi=600)
        plt.close(fig)

        # Cumulative Distance (does not include extension)
        fig = plt.figure()
        plt.plot(cumPureDist)
        plt.xticks(range(len(cumPureDist)), ['%d' % (x + 1) for x in range(len(cumPureDist))])
        plt.xlabel('Merge Number')
        plt.ylabel('Distance')
        plt.title('Cumulative Pure Merge Distance')
        pdf.savefig(fig, dpi=600)
        pdf2.savefig(fig, dpi=600)
        plt.close(fig)

        # Maximum Extension
        fig = plt.figure()
        plt.plot(mextl)
        plt.xticks(range(len(mextl)), ['%d' % (x + 1) for x in range(len(indMergeDist))])
        plt.xlabel('Merge Number')
        plt.ylabel('Extension Penalty')
        plt.title('Maximum Cumulative Extension Penalty')
        pdf.savefig(fig, dpi=600)
        pdf2.savefig(fig, dpi=600)
        plt.close(fig)

        # Total Cumulative Extension
        fig = plt.figure()
        plt.plot(cextl)
        plt.xticks(range(len(cextl)), ['%d' % (x + 1) for x in range(len(indMergeDist))])
        plt.xlabel('Merge Number')
        plt.ylabel('Extension Penalty')
        plt.title('Total Sum of Cumulative Extension Penalties')
        pdf.savefig(fig, dpi=600)
        pdf2.savefig(fig, dpi=600)
        plt.close(fig)

        # Maximum Extension
        fig = plt.figure()
        plt.plot(iextl)
        plt.xticks(range(len(iextl)), ['%d' % (x + 1) for x in range(len(indMergeDist))])
        plt.xlabel('Merge Number')
        plt.ylabel('Extension Penalty')
        plt.title('Individual Merge Extension Penalties')
        pdf.savefig(fig, dpi=600)
        pdf2.savefig(fig, dpi=600)
        plt.close(fig)
        pdf2.close()
    np.savetxt(os.path.join(lblPath, 'merge_distances.txt'), indMergeDist)
    arr2csv(os.path.join(lblPath, 'all_distances.csv'), allPairwiseDist, mergeLabels)

    return


def newMatrixAfterRemoving(matrix, idxs):
    assert matrix.shape[0] == matrix.shape[1]
    condensed = squareform(matrix)

    ct = 0
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            if i in idxs or j in idxs:
                condensed[ct] = np.nan
            ct += 1

    newNum = len(matrix) - len(idxs) + 1
    out = np.zeros((newNum, newNum))
    out[:-1, :-1] = squareform(condensed[np.logical_not(np.isnan(condensed))])
    return out


def merge_group(meta, ids, kdigos, dm, lbls, centers, lblPath, args, cat='1-Im',
                mismatch=lambda x, y: abs(x-y), extension=lambda x: 0, dist=braycurtis):
    lblNames = centerCategorizer(centers)
    if len(cat.split('-')) == 2:
        lblgrp = [x for x in np.unique(lbls) if cat in lblNames[x]]
    elif len(cat.split('-')) == 1:
        lblgrp = [x for x in np.unique(lbls) if cat == lblNames[x].split('-')[0]]
    else:
        raise ValueError("The provided category, '{}', is not valid. Must be '1', '2', '3', '3D', "
                         "or any of those followed by '-Im', '-Tr', '-St', or 'Ws'. Please try again.".format(cat))
    if len(lblgrp) <= 2:
        return

    _, dbaTag = get_dm_tag(args.pdtw, args.malpha, False, True, 'braycurtis', args.lapVal, args.mlapType)
    lblPath = os.path.join(lblPath, 'merged_' + dbaTag)
    if not os.path.exists(lblPath):
        os.mkdir(lblPath)
    lblPath = os.path.join(lblPath, 'center_%s' % args.seedType)
    if not os.path.exists(lblPath):
        os.mkdir(lblPath)

    nameTag = ""
    maxExt = -1
    if args.extDistWeight > 0:
        if args.extDistWeight >= 1:
            nameTag = 'extWeight_%d' % args.extDistWeight
        else:
            nameTag = 'extWeight_%dE-02' % (args.extDistWeight * 100)
    else:
        nameTag = "noExt"
    if args.mlapType == "aggregated":
        nameTag += "_aggLap_1"
    elif args.mlapType == "individual":
        nameTag += "_indLap_1"
    if args.cumExtDist:
        nameTag += "_cumExt"
    if args.maxExt > 0:
        nameTag += '_maxExt_%d' % args.maxExt
        maxK = cat.split("-")[0]
        if maxK != "3D":
            maxK = int(maxK)
        else:
            maxK = 4
        maxExt = evalExtension([maxK], np.zeros(int(args.maxExt) * 4).astype(int), extension, args.malpha)

    lblPath = os.path.join(lblPath, nameTag)
    if not os.path.exists(lblPath):
        os.mkdir(lblPath)

    lblPath = os.path.join(lblPath, cat)
    if not os.path.exists(lblPath):
        os.mkdir(lblPath)

    lblPath = os.path.join(lblPath, args.mergeType)
    if not os.path.exists(lblPath):
        os.mkdir(lblPath)

    if os.path.exists(os.path.join(lblPath, 'merge_distances.txt')):
        return

    print('Merging clusters in category %s ' % cat)

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
        if meta['max_kdigo'][tidx] > meta['max_kdigo_win'][tidx]:
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
    with PdfPages(os.path.join(lblPath, cat + '_merge_visualization.pdf')) as pdf:
        pdf2 = PdfPages(os.path.join(lblPath, cat + '_ordered_merge_trajectories.pdf'))

        if not os.path.exists(os.path.join(lblPath, '%d_clusters' % nClust)):
            os.mkdir(os.path.join(lblPath, '%d_clusters' % nClust))
            arr2csv(os.path.join(lblPath, '%d_clusters' % nClust, 'clusters.csv'), grpLbls, grpIds, fmt='%s')
            formatted_stats(meta, os.path.join(lblPath, '%d_clusters' % nClust))
            with PdfPages(os.path.join(lblPath, '%d_clusters' % nClust, 'centers.pdf')) as pdf1:
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
        indMergeDist = []
        cumMergeDist = []
        indPureDist = []
        cumPureDist = []
        allPairwiseDist = []
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
        mextl = []
        cextl = []
        iextl = []
        curExt = {}
        cellsPerRow = 17
        cellsPerCol = 8
        nBigRows = int(np.ceil((len(lblgrp) - 1) / 5))
        bigFig = plt.figure(figsize=[22, (nBigRows * 6.0)])
        aBigFig = plt.figure(figsize=[22, (nBigRows * 6.0)])
        bigGS = GridSpec(nBigRows * cellsPerRow, 5 * cellsPerCol)
        rcParams['font.size'] = 10
        
        mergeDistMat = np.zeros((len(lblgrp), len(lblgrp)))
        xExtMat = np.zeros((len(lblgrp), len(lblgrp)))
        yExtMat = np.zeros((len(lblgrp), len(lblgrp)))
        pureDistMat = np.zeros((len(lblgrp), len(lblgrp)))
        extMat = np.zeros((len(lblgrp), len(lblgrp)))
        cumXExtMat = np.zeros((len(lblgrp), len(lblgrp)))
        cumYExtMat = np.zeros((len(lblgrp), len(lblgrp)))
        
        t = tqdm.tqdm(total=len(lblgrp), desc='Merging %d clusters' % len(lblgrp), unit='merge')
        while len(lblgrp) > 2:
            mergeGrp = [None, None]
            tdist = []
            pureDist = []
            pwiseIterationDist = []
            ct = 0
            minDist = 10000000
            for i in range(len(lblgrp)):
                for j in range(i + 1, len(lblgrp)):
                    lbl1 = lblgrp[i]
                    lbl2 = lblgrp[j]
                    c1 = grpCenters[lbl1]
                    c2 = grpCenters[lbl2]
                    if (i < (len(lblgrp) - 1)) and (j < (len(lblgrp) - 1)) and mergeCt > 0:
                        d = mergeDistMat[i, j]
                        thisPureDist = pureDistMat[i, j]
                    else:
                        if not args.pdtw:
                            _, paths = dtw(c1, c2)
                            path1 = np.array(paths)[:, 0]
                            path2 = np.array(paths)[:, 1]
                            paths = [path1, path2]
                        else:
                            _, _, _, paths, xext, yext = dtw_p(c1, c2, mismatch, extension, args.malpha)

                        cxext = copy.copy(xext)
                        cyext = copy.copy(yext)
                        if lbl1 in list(curExt):
                            cxext += curExt[lbl1]
                        if lbl2 in list(curExt):
                            cyext += curExt[lbl2]

                        d = dist(c1[paths[0]], c2[paths[1]])
                        thisPureDist = copy.copy(d)
                        if args.extDistWeight > 0:
                            if args.cumExtDist:
                                d += args.extDistWeight * (cxext + cyext)
                            else:
                                d += args.extDistWeight * (xext + yext)

                        mergeDistMat[i, j] = d
                        mergeDistMat[j, i] = d

                        pureDistMat[i, j] = thisPureDist
                        pureDistMat[j, i] = thisPureDist

                        extMat[i, j] = (xext + yext)
                        extMat[j, i] = (xext + yext)

                        xExtMat[i, j] = xext
                        xExtMat[j, i] = xext

                        yExtMat[i, j] = yext
                        yExtMat[j, i] = yext

                        cumXExtMat[i, j] = cxext
                        cumXExtMat[j, i] = cxext

                        cumYExtMat[i, j] = cyext
                        cumYExtMat[j, i] = cyext

                    pwiseIterationDist.append(d)
                    pureDist.append(thisPureDist)

                    if len(pwiseIterationDist) == 1 or d < minDist:
                        if maxExt < 0:
                            mergeGrp = [i, j]
                            minDist = d
                        else:
                            if args.cumExtDist:
                                if max(cxext, cyext) < maxExt:
                                    mergeGrp = [i, j]
                                    minDist = d
                            else:
                                if max(xext, yext) < maxExt:
                                    mergeGrp = [i, j]
                                    minDist = d
                    ct += 1

            if mergeGrp[0] is None:
                print("No more valid merges. Maximum extension reached.")
                break
            allPairwiseDist.append(pwiseIterationDist)
            indMergeDist.append(minDist)
            if len(cumMergeDist) == 0:
                cumMergeDist.append(minDist)
            else:
                cumMergeDist.append(cumMergeDist[-1] + minDist)

            indPureDist.append(pureDistMat[mergeGrp[0], mergeGrp[1]])

            if len(cumPureDist) == 0:
                cumPureDist.append(indPureDist[-1])
            else:
                cumPureDist.append(cumPureDist[-1] + indPureDist[-1])

            idx1 = np.where(grpLbls == lblgrp[mergeGrp[0]])[0]
            idx2 = np.where(grpLbls == lblgrp[mergeGrp[1]])[0]
            if lblgrp[mergeGrp[0]] not in grpLbls:
                print(lblgrp[mergeGrp[0]] + 'not in labels')
            if lblgrp[mergeGrp[1]] not in grpLbls:
                print(lblgrp[mergeGrp[1]] + 'not in labels')
            plotGrp = copy.deepcopy(lblgrp)
            nlbl = '-'.join((lblgrp[mergeGrp[0]], lblgrp[mergeGrp[1]]))
            mergeLabels.append(lblgrp[mergeGrp[0]] + ' + ' + lblgrp[mergeGrp[1]] + ' -> ' + nlbl)
            vdm = squareform(np.array(tdist))
            grpLbls[idx1] = nlbl
            grpLbls[idx2] = nlbl
            c2 = grpCenters[lblgrp[mergeGrp[1]]]
            c1 = grpCenters[lblgrp[mergeGrp[0]]]
            try:
                del grpCenters[lblgrp[mergeGrp[1]]], grpCenters[lblgrp[mergeGrp[0]]]
            except KeyError:
                pass

            lbl2 = lblgrp.pop(mergeGrp[1])
            lbl1 = lblgrp.pop(mergeGrp[0])

            idx = np.sort(np.concatenate((idx1, idx2)))
            dmidx = np.ix_(idx, idx)
            tkdigos = [grpKdigos[x] for x in idx]
            tdm = squareform(squareform(grpDm)[dmidx])
            
            xext_cum = cumXExtMat[mergeGrp[0], mergeGrp[1]]
            yext_cum = cumYExtMat[mergeGrp[0], mergeGrp[1]]
            xext_ind = xExtMat[mergeGrp[0], mergeGrp[1]]
            yext_ind = yExtMat[mergeGrp[0], mergeGrp[1]]
            
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

            if args.pdtw:
                _, _, _, path, xext, yext = dtw_p(c1, c2, mismatch=mismatch, extension=extension, alpha=args.alpha)
            else:
                _, paths = dtw(c1, c2, dist=mismatch)
                path1 = np.array(paths)[:, 0]
                path2 = np.array(paths)[:, 1]
                path = [path1, path2]

            c1p = c1[path[0]]
            c2p = c2[path[1]]

            if args.mergeType == 'dba':
                center, stds, confs, paths = performDBA(tkdigos, tdm, mismatch=mismatch, extension=extension,
                                                        n_iterations=args.dbaiter, seedType=args.seedType)
            elif 'mean' in args.mergeType:
                if 'weighted' in args.mergeType:
                    count1 = len(idx1)
                    count2 = len(idx2)
                    w1 = count1 / (count1 + count2)
                    w2 = count2 / (count2 + count1)
                    center = np.array([(w1 * c1p[x]) + (w2 * c2p[x]) for x in range(len(c1p))])
                else:
                    center = np.array([((c1p[x]/2) + (c2p[x]) / 2) for x in range(len(c1p))])

            mextl.append(np.max([x for x in curExt.values()]))
            cextl.append(np.sum([x for x in curExt.values()]))
            iextl.append(curExt[nlbl])

            grpCenters[nlbl] = center
            lblgrp.append(nlbl)
            nClust = len(lblgrp)
            os.mkdir(os.path.join(lblPath, '%d_clusters' % nClust))
            arr2csv(os.path.join(lblPath, '%d_clusters' % nClust, 'clusters.csv'), grpLbls, grpIds, fmt='%s')
            arr2csv(os.path.join(lblPath, '%d_clusters' % nClust, 'centers.csv'), list(grpCenters.values()),
                    list(grpCenters.keys()))
            formatted_stats(meta, os.path.join(lblPath, '%d_clusters' % nClust))

            # Plot merge summary document. Shows trajectories before and after alignment and then the resulting new
            # center trajectory.
            fig = plt.figure(figsize=[16, 8])
            gs = GridSpec(4, 4)

            # First original sequence
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

            # Second original sequence
            ax = fig.add_subplot(gs[2:, 0])
            _ = ax.plot(c2, label='%d Patients' % len(idx2))
            ax.set_xticks(range(0, len(c2), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(c2))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            ax.legend([extra], ['%d Patients' % len(idx2), ])
            # ax.set_title('Cluster ' + lbl2, wrap=True)
            ax.set_ylim((-0.2, 4.2))

            # First sequence aligned
            ax = fig.add_subplot(gs[:2, 1])
            _ = ax.plot(c1p, label='%d Patients' % len(idx1))
            ax.set_xticks(range(0, len(c1p), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(c1p))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            ax.legend([extra, extra], ['%d Patients' % len(idx1), 'Extension: %.1f + %.1f = %.1f' %
                                       (xext_ind, xext_cum - xext_ind, xext_cum)])
            ax.set_title('After Alignment')
            ax.set_ylim((-0.2, 4.2))

            ax = fig.add_subplot(gs[2:, 1])
            _ = ax.plot(c2p, label='%d Patients' % len(idx2))
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

            # New center after merging
            ax = fig.add_subplot(gs[1:3, 2])
            ax.plot(center, label='%d Patients' % len(idx))
            ax.set_xticks(range(0, len(center), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(center))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            # ax.set_title('Cluster ' + nlbl, wrap=True)
            ax.set_ylim((-0.2, 4.2))

            # Current distance matrix
            ax = fig.add_subplot(gs[:2, 3])
            plt.pcolormesh(mergeDistMat, linewidth=0, rasterized=True)
            ax.set_xticks(np.arange(mergeDistMat.shape[0]) + 0.5)
            ax.set_xticklabels([str(x) for x in plotGrp], rotation=30, ha='right')
            ax.set_yticks(np.arange(mergeDistMat.shape[0]) + 0.5)
            ax.set_yticklabels([str(x) for x in plotGrp])
            ax.set_title('Distance Matrix')
            if mergeDistMat.shape[0] <= 5:
                fs = 8.
                rot = 0
            elif mergeDistMat.shape[0] < 10:
                fs = 4.5
                rot = 0
            elif mergeDistMat.shape[0] < 15:
                fs = 3.5
                rot = 0
            else:
                fs = 2
                rot = 45
            for y in range(mergeDistMat.shape[0]):
                for x in range(mergeDistMat.shape[1]):
                    plt.text(x + 0.5, y + 0.5, '%.3f' % mergeDistMat[y, x],
                             horizontalalignment='center',
                             verticalalignment='center',
                             fontsize=fs,
                             rotation=rot
                             )

            # Extension penalty matrix

            ax = fig.add_subplot(gs[2:, 3])
            plt.pcolormesh(extMat, linewidth=0, rasterized=True)
            ax.set_xticks(np.arange(extMat.shape[0]) + 0.5)
            ax.set_xticklabels([str(x) for x in plotGrp], rotation=30, ha='right')
            ax.set_yticks(np.arange(extMat.shape[0]) + 0.5)
            ax.set_yticklabels([str(x) for x in plotGrp])
            ax.set_title('Cumulative Extension Penalties')
            if extMat.shape[0] <= 5:
                fs = 10.7
                rot = 0
            elif extMat.shape[0] < 10:
                fs = 6
                rot = 0
            elif extMat.shape[0] < 15:
                fs = 4.7
                rot = 0
            else:
                fs = 2.6
                rot = 45
            for y in range(extMat.shape[0]):
                for x in range(extMat.shape[1]):
                    plt.text(x + 0.5, y + 0.5, '%.1f' % extMat[y, x],
                             horizontalalignment='center',
                             verticalalignment='center',
                             fontsize=fs,
                             rotation=rot
                             )
            plt.suptitle("Merge #%d (%d Clusters)\n" % (mergeCt + 1, len(lblgrp)) + r"$\bf{%s}$ + $\bf{%s}$" % (lbl1, lbl2),
                         y=0.95, x=0.625, ha='center')
            # plt.colorbar(ax=ax)
            plt.tight_layout()
            pdf.savefig(dpi=600)
            plt.close(fig)

            # First original sequence
            tv = int(np.floor(mergeCt / 5)) * cellsPerRow
            colStart = (mergeCt % 5) * cellsPerCol
            ax = bigFig.add_subplot(bigGS[tv:tv + 5, colStart:colStart + cellsPerCol - 2])
            _ = ax.plot(c1)
            ax.set_xticks(range(0, len(c1), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(c1))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            ax.legend([extra, extra], [lbl1, '%d Patients' % len(idx1)])
            ax.set_title("Merge #%d" % (mergeCt + 1))
            ax.set_ylim((-0.2, 4.2))

            # Second original sequence
            ax = bigFig.add_subplot(bigGS[tv+7:tv+13, colStart:colStart+cellsPerCol-2])
            _ = ax.plot(c2)
            ax.set_xticks(range(0, len(c2), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(c2))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            ax.legend([extra, extra], [lbl2, '%d Patients' % len(idx2)])
            ax.set_title(" ")
            ax.set_ylim((-0.2, 4.2))

            # First sequence alignedtv = int(np.floor(mergeCt / 5)) * cellsPerRow
            ax = aBigFig.add_subplot(bigGS[tv:tv+5, colStart:colStart+cellsPerCol-2])
            _ = ax.plot(c1p)
            ax.set_xticks(range(0, len(c1p), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(c1p))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            ax.legend([extra, extra, extra], [lbl1, '%d Patients' % len(idx1), 'Extension: %.1f + %.1f = %.1f' %
                                              (xext_ind, xext_cum - xext_ind, xext_cum)])
            ax.set_title('Merge #%d' % (mergeCt + 1))
            ax.set_ylim((-0.2, 4.2))

            # Second sequence after alignment
            ax = aBigFig.add_subplot(bigGS[tv+7:tv+13, colStart:colStart+cellsPerCol-2])
            _ = ax.plot(c2p)
            ax.set_xticks(range(0, len(c2p), 4))
            ax.set_xticklabels(['%d' % x for x in range(len(c2p))], wrap=True)
            ax.set_xlabel('Days')
            ax.set_ylabel('KDIGO')
            ax.set_title(" ")
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            ax.legend([extra, extra, extra], [lbl2, '%d Patients' % len(idx2),
                       'Extension: %.1f + %.1f = %.1f' % (yext_ind, yext_cum - yext_ind, yext_cum)])
            ax.set_ylim((-0.2, 4.2))
            mergeCt += 1

            mergeDistMat = newMatrixAfterRemoving(mergeDistMat, mergeGrp)
            extMat = newMatrixAfterRemoving(extMat, mergeGrp)
            pureDistMat = newMatrixAfterRemoving(pureDistMat, mergeGrp)
            xExtMat = newMatrixAfterRemoving(xExtMat, mergeGrp)
            yExtMat = newMatrixAfterRemoving(yExtMat, mergeGrp)
            cumXExtMat = newMatrixAfterRemoving(cumXExtMat, mergeGrp)
            cumYExtMat = newMatrixAfterRemoving(cumYExtMat, mergeGrp)

            if args.plot_centers:
                with PdfPages(os.path.join(lblPath, '%d_clusters' % nClust, 'centers.pdf')) as pdf1:
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
            t.update()
        t.close()
        bigFig.text(0.5, 0.975, "Merges In Order - Original Sequences", ha='center', fontsize=20, fontweight='bold')
        bigFig.subplots_adjust(top=0.95, bottom=0.0, left=0.055, right=1.0)
        pdf2.savefig(bigFig, dpi=600)
        plt.close(bigFig)

        aBigFig.text(0.5, 0.975, "Merges In Order - After Alignment", ha='center', fontsize=20, fontweight='bold')
        aBigFig.subplots_adjust(top=0.95, bottom=0.0, left=0.055, right=1.0)
        pdf2.savefig(aBigFig, dpi=600)
        plt.close(aBigFig)

        # Finally, plot the distance and extension penalty vs the number of merges.
        # Individual Merge Distance (may include extension)
        if args.extDistWeight > 0:
            fig = plt.figure()
            plt.plot(indMergeDist)
            plt.xticks(range(len(indMergeDist)), ['%d' % (x + 1) for x in range(len(indMergeDist))])
            plt.xlabel('Merge Number')
            plt.ylabel('Distance')
            plt.title('Individual Merge Distances + %.3g * Extension Penalty' % args.extDistWeight)
            pdf.savefig(fig, dpi=600)
            pdf2.savefig(fig, dpi=600)
            plt.close(fig)

            # Cumulative Distance (may include extension)
            fig = plt.figure()
            plt.plot(cumMergeDist)
            plt.xticks(range(len(cumMergeDist)), ['%d' % (x + 1) for x in range(len(cumMergeDist))])
            plt.xlabel('Merge Number')
            plt.ylabel('Distance')
            plt.title('Cumulative Merge Distance + %.3g * Extension Penalty' % args.extDistWeight)
            pdf.savefig(fig, dpi=600)
            pdf2.savefig(fig, dpi=600)
            plt.close(fig)

        # Individual Merge Distance (does not include extension)
        fig = plt.figure()
        plt.plot(indPureDist)
        plt.xticks(range(len(indPureDist)), ['%d' % (x + 1) for x in range(len(indPureDist))])
        plt.xlabel('Merge Number')
        plt.ylabel('Distance')
        plt.title('Individual Pure Merge Distances')
        pdf.savefig(fig, dpi=600)
        pdf2.savefig(fig, dpi=600)
        plt.close(fig)

        # Cumulative Distance (does not include extension)
        fig = plt.figure()
        plt.plot(cumPureDist)
        plt.xticks(range(len(cumPureDist)), ['%d' % (x + 1) for x in range(len(cumPureDist))])
        plt.xlabel('Merge Number')
        plt.ylabel('Distance')
        plt.title('Cumulative Pure Merge Distance')
        pdf.savefig(fig, dpi=600)
        pdf2.savefig(fig, dpi=600)
        plt.close(fig)

        # Maximum Extension
        fig = plt.figure()
        plt.plot(mextl)
        plt.xticks(range(len(mextl)), ['%d' % (x + 1) for x in range(len(indMergeDist))])
        plt.xlabel('Merge Number')
        plt.ylabel('Extension Penalty')
        plt.title('Maximum Cumulative Extension Penalty')
        pdf.savefig(fig, dpi=600)
        pdf2.savefig(fig, dpi=600)
        plt.close(fig)

        # Total Cumulative Extension
        fig = plt.figure()
        plt.plot(cextl)
        plt.xticks(range(len(cextl)), ['%d' % (x + 1) for x in range(len(indMergeDist))])
        plt.xlabel('Merge Number')
        plt.ylabel('Extension Penalty')
        plt.title('Total Sum of Cumulative Extension Penalties')
        pdf.savefig(fig, dpi=600)
        pdf2.savefig(fig, dpi=600)
        plt.close(fig)

        # Maximum Extension
        fig = plt.figure()
        plt.plot(iextl)
        plt.xticks(range(len(iextl)), ['%d' % (x + 1) for x in range(len(indMergeDist))])
        plt.xlabel('Merge Number')
        plt.ylabel('Extension Penalty')
        plt.title('Individual Merge Extension Penalties')
        pdf.savefig(fig, dpi=600)
        pdf2.savefig(fig, dpi=600)
        plt.close(fig)
        pdf2.close()
    np.savetxt(os.path.join(lblPath, 'merge_distances.txt'), indMergeDist)
    arr2csv(os.path.join(lblPath, 'all_distances.csv'), allPairwiseDist, mergeLabels)
    arr2csv(os.path.join(lblPath, 'prog_scores.csv'), progScores, mergeLabels,
            header='Merge,MinProgDiff,MaxProgDiff,RelProgDiff')
    arr2csv(os.path.join(lblPath, 'mort_scores.csv'), mortScores, mergeLabels,
            header='Merge,MinMortDiff,MaxMortDiff,RelMortDiff')
    progScores = np.array(progScores)
    mortScores = np.array(mortScores)
    with PdfPages(os.path.join(lblPath, cat + '_merge_evaluation.pdf')) as pdf:
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


def plotGroupedCenters(lblPath, centerPath, plotConf=True):
    ids = np.loadtxt(os.path.join(lblPath, 'clusters.csv'), delimiter=',', usecols=0)
    lbls = load_csv(os.path.join(lblPath, 'clusters.csv'), ids, int)

    centerlbls = np.loadtxt(os.path.join(centerPath, 'centers.csv'), delimiter=',', usecols=0, dtype=int)
    centers = load_csv(os.path.join(centerPath, 'centers.csv'), centerlbls)
    if plotConf:
        confs = load_csv(os.path.join(centerPath, 'confs.csv'), centerlbls)

    lblNames = np.array(centerCategorizer(centers))

    rcParams['font.size'] = 15
    with PdfPages(os.path.join(centerPath, 'grouped_centers.pdf')) as pdf:
        for lblName in np.unique(lblNames):
            idx = np.where(lblNames == lblName)[0]
            grpSizes = np.array([len(np.where(lbls == centerlbls[x])[0]) for x in idx])
            tlbls = centerlbls[idx]
            if len(idx) > 12:
                nCols = 4
                nRows = int(np.ceil(len(idx) / nCols))
            else:
                nCols = 3
                nRows = int(np.ceil(len(idx) / nCols))
            if plotConf:
                fig = plotMultiKdigo([centers[x] for x in idx], uncertainties=[confs[x] for x in idx],
                                     nRows=nRows, nCols=nCols, grpSizes=grpSizes,
                                     labels=tlbls, hgap=2, wgap=1, cellsPerCol=7, cellsPerRow=6)
            else:
                fig = plotMultiKdigo([centers[x] for x in idx], uncertainties=None,
                                     nRows=nRows, nCols=nCols, grpSizes=grpSizes,
                                     labels=tlbls, hgap=2, wgap=1, cellsPerCol=7, cellsPerRow=6)
            plt.tight_layout()
            plt.suptitle("%s Clusters" % lblName, fontsize=30)
            pdf.savefig(dpi=600)
            plt.close(fig)


def plotMultiKdigo(seqs, uncertainties=None, nRows=1, nCols=1, grpSizes=[], legendEntries=[], labels=[],
                   cellsPerRow=5, cellsPerCol=5, hgap=0, wgap=0, ptsPerDay=4):
    if len(legendEntries) == 0:
        if len(grpSizes) == len(seqs):
            if len(labels) == 0:
                legendEntries = [[['Sequence #%s', '%s Patients', ], ['%d' % i, '%d' % grpSizes[i], ]] for i in range(len(seqs))]
            else:
                legendEntries = [[['%s Patients', ], ['%d' % grpSizes[i], ]] for i in
                                 range(len(seqs))]

    for i in range(len(legendEntries)):
        temp = []
        for j in range(len(legendEntries[i][0])):
            temp.append(legendEntries[i][0][j] % legendEntries[i][1][j])
        legendEntries[i] = temp

    fig = plt.figure(figsize=[nCols * 6.4, nRows * 4.8])

    if nRows == 1:
        gs = GridSpec(int(np.ceil(nRows * cellsPerRow) + hgap),
                      int((nCols * cellsPerCol) + (nCols - 1) * wgap))
    else:
        gs = GridSpec(int(np.ceil(nRows * cellsPerRow) + (nRows - 1) * hgap),
                      int((nCols * cellsPerCol) + (nCols - 1) * wgap))
    row = 0
    col = 0
    for i in range(len(seqs)):
        colStart = (col * cellsPerCol) + (col * wgap)
        colEnd = colStart + cellsPerCol

        if nRows == 1:
            rowStart = (row * cellsPerRow) + hgap
            rowEnd = rowStart + cellsPerRow
        else:
            rowStart = (row * cellsPerRow) + (row * hgap)
            rowEnd = rowStart + cellsPerRow

        loc = gs[rowStart:rowEnd, colStart:colEnd]

        if len(labels) > 0:
            if uncertainties is not None:
                _ = plotKdigo(fig, loc, seqs[i], uncertainty=uncertainties[i], title=labels[i],
                              legendEntries=legendEntries[i], ptsPerDay=ptsPerDay)
            else:
                _ = plotKdigo(fig, loc, seqs[i], uncertainty=None, title=labels[i],
                              legendEntries=legendEntries[i], ptsPerDay=ptsPerDay)

        # Generally go left to right, so column increases first
        col += 1
        # If no more columns, reset column to 0 and increase the
        # row by the number of instepidual plots included in each subplot
        if col == nCols:
            col = 0
            row += 1
    return fig


# Second sequence after alignment
def plotKdigo(fig, loc, seq, uncertainty=None, title='', legendEntries=[], ptsPerDay=4):
    ax = fig.add_subplot(loc)
    _ = ax.plot(seq)
    if uncertainty is not None:
        ax.fill_between(range(len(seq)), seq - uncertainty, seq + uncertainty, alpha=0.4)
    # ax.plot()
    ax.set_xticks(range(0, len(seq), ptsPerDay))
    ax.set_xticklabels(['%d' % x for x in range(int(np.floor(len(seq) / ptsPerDay)))], wrap=True)
    ax.set_xlabel('Days')
    ax.set_ylabel('KDIGO')
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    ax.legend([extra for _ in range(len(legendEntries))], [entry for entry in legendEntries])
    ax.set_ylim((-0.2, 4.2))
    if title != '':
        ax.set_title(title)
    return ax


def genSimulationData(lengthInDays=14, ptsPerDay=4, numVariants=15):
    simulated = []
    labels = []
    totLen = (lengthInDays * ptsPerDay) + ptsPerDay
    label = [0, 0, 0, 0, 0, 0, 0]
    for startKDIGO in range(5):
        label[0] = startKDIGO
        for longStart in [0, 1]:
            label[1] = longStart
            for stopKDIGO in range(5):
                label[2] = stopKDIGO
                for longStop in [0, 1]:
                    label[3] = longStop
                    for baseKDIGO in range(5):
                        label[4] = baseKDIGO
                        if baseKDIGO == startKDIGO and longStart:
                            continue
                        if baseKDIGO == stopKDIGO and longStop:
                            continue
                        for numPeaks in [0, 1, 2]:
                            if startKDIGO == 0 and stopKDIGO == 0 and baseKDIGO == 0 and numPeaks == 0:
                                continue
                            label[5] = numPeaks
                            if numPeaks == 0:
                                pStop = baseKDIGO + 2
                            else:
                                pStop = 5
                            for peakVal in range(baseKDIGO + 1, pStop):
                                label[6] = peakVal
                                for vnum in range(numVariants):
                                    kdigo = np.zeros(totLen) + baseKDIGO
                                    maxPeakLen = 0
                                    ct = 0
                                    startLen = 1
                                    stopLen = 1
                                    while maxPeakLen <= 2 and ct < 10:
                                        if longStart:
                                            startLen = np.random.randint(int(totLen / 4), int(totLen / 2))
                                        else:
                                            startLen = np.random.randint(1, int(totLen / 5))
                                        if longStop:
                                            stopLen = np.random.randint(int(totLen / 4), int(totLen / 2))
                                        else:
                                            stopLen = np.random.randint(1, int(totLen / 5))
                                        maxPeakLen = int((totLen - startLen - stopLen) / (numPeaks + 1))
                                        ct += 1

                                    kdigo[:startLen] = startKDIGO
                                    kdigo[-stopLen:] = stopKDIGO
                                    if maxPeakLen < 2:
                                        continue
                                    for peak in range(numPeaks):
                                        dur = np.random.randint(2, maxPeakLen)
                                        # start = np.random.randint(startLen + (peak * maxPeakLen))
                                        start = np.random.randint(maxPeakLen - dur) + startLen + (peak * maxPeakLen) + ptsPerDay
                                        kdigo[start:start + dur] = peakVal
                                        if peakVal > 1:
                                            frontSlopeOffset = np.random.randint(ptsPerDay)
                                            backSlopeOffset = np.random.randint(ptsPerDay)
                                            kdigo[start - frontSlopeOffset:start] = int((peakVal + baseKDIGO) / 2)
                                            kdigo[start + dur:start + dur + backSlopeOffset] = int(
                                                (peakVal + baseKDIGO) / 2)
                                    for i in range(len(kdigo)):
                                        if kdigo[i] < 4:    # Allow negative noise
                                            neg = np.random.randint(2)
                                            if neg:
                                                kdigo[i] -= np.random.rand() * (float(kdigo[i]) * 0.1)
                                            else:
                                                kdigo[i] += np.random.rand() * (float(kdigo[i]) * 0.1)
                                        else:
                                            kdigo[i] -= np.random.rand() * (float(kdigo[i]) * 0.1)
                                    simulated.append(kdigo)
                                    labels.append(np.array(label))
    return simulated, labels


def genSelectedSimulations(selected, nSamples=5, targLen=14, ptsPerDay=4):
    simulated = []
    labels = []
    totLen = targLen * 4
    for i in range(len(selected)):
        startKDIGO, longStart, stopKDIGO, longStop, baseKDIGO, numPeaks, peakVal = selected[i]
        for vnum in range(nSamples):
            kdigo = np.zeros(totLen) + baseKDIGO
            maxPeakLen = 0
            ct = 0
            startLen = 1
            stopLen = 1
            while maxPeakLen <= 2 and ct < 10:
                if longStart:
                    startLen = np.random.randint(int(totLen / 4), int(totLen / 2))
                else:
                    startLen = np.random.randint(1, int(totLen / 5))
                if longStop:
                    stopLen = np.random.randint(int(totLen / 4), int(totLen / 2))
                else:
                    stopLen = np.random.randint(1, int(totLen / 5))
                maxPeakLen = int((totLen - startLen - stopLen) / (numPeaks + 1))
                ct += 1

            kdigo[:startLen] = startKDIGO
            kdigo[-stopLen:] = stopKDIGO
            if maxPeakLen < 2:
                continue
            for peak in range(numPeaks):
                dur = np.random.randint(2, maxPeakLen)
                # start = np.random.randint(startLen + (peak * maxPeakLen))
                start = np.random.randint(maxPeakLen - dur) + startLen + (peak * maxPeakLen) + ptsPerDay
                kdigo[start:start + dur] = peakVal
                if peakVal > 1:
                    frontSlopeOffset = np.random.randint(ptsPerDay)
                    backSlopeOffset = np.random.randint(ptsPerDay)
                    kdigo[start - frontSlopeOffset:start] = int((peakVal + baseKDIGO) / 2)
                    kdigo[start + dur:start + dur + backSlopeOffset] = int(
                        (peakVal + baseKDIGO) / 2)
            for j in range(len(kdigo)):
                if kdigo[j] < 4:  # Allow negative noise
                    neg = np.random.randint(2)
                    if neg:
                        kdigo[j] -= np.random.rand() * (float(kdigo[j]) * 0.1)
                    else:
                        kdigo[j] += np.random.rand() * (float(kdigo[j]) * 0.1)
                else:
                    kdigo[j] -= np.random.rand() * (float(kdigo[j]) * 0.1)
            simulated.append(kdigo)
            labels.append(selected[i])
    return simulated, labels


def selectedSimulationSubsets(sequences, labels, selected, nSamples=5):
    outSeqs = []
    outLbls = []
    outIds = []
    for label in selected:
        idxs = np.where(labels == label)[0]
        idxs = np.sort(np.random.permutation(idxs)[:nSamples])
        for idx in idxs:
            outSeqs.append(sequences[idx])
            outLbls.append(label)
            outIds.append(idx)
    return outSeqs, outLbls, outIds


def randomSimulationSubsets(sequences, labels, basePath, coords, mismatch=lambda x,y: abs(x - y),
                            extension=lambda x: 0, nSubtypes=5, nSamples=5, nSets=10):

    variantsPerSubtype = len(np.where(labels == labels[0])[0])
    firstVariants = [sequences[x] for x in range(0, len(sequences), variantsPerSubtype)]
    firstVariantLbls = [labels[x] for x in range(0, len(sequences), variantsPerSubtype)]

    lblNames = centerCategorizer(firstVariants)

    cats = ['1-Im', '1-St', '1-Ws', '2-Im', '2-St', '2-Ws', '3-Im', '3-St', '3-Ws', '3D-Im', '3D-St', '3D-Ws']
    for cat in cats:
        if len(cat.split('-')) == 2:
            lblgrp = [firstVariantLbls[x] for x in range(len(firstVariantLbls)) if cat in lblNames[x]]
        elif len(cat.split('-')) == 1:
            lblgrp = [firstVariantLbls[x] for x in range(len(firstVariantLbls)) if cat == lblNames[x].split('-')[0]]
        else:
            raise ValueError("The provided category, '{}', is not valid. Must be '1', '2', '3', '3D', "
                             "or any of those followed by '-Im', '-Tr', '-St', or 'Ws'. Please try again.".format(cat))

        catPath = os.path.join(basePath, cat)
        if not os.path.exists(catPath):
            os.mkdir(catPath)

        for setNum in range(nSets):
            setPath = os.path.join(catPath, "set%d" % (setNum + 1))
            if not os.path.exists(setPath):
                os.mkdir(setPath)
                setgrp = sorted(list(np.random.permutation(lblgrp)[:nSubtypes]))
                sel = np.random.permutation(list(range(variantsPerSubtype)))[:nSamples].astype(int)

                idx = np.where(labels == setgrp[0])[0][sel]
                for i in range(1, len(setgrp)):
                    idx = np.union1d(idx, np.where(labels == setgrp[i])[0][sel])

                setLbls = labels[idx]
                setIds = np.arange(len(sequences))[idx].astype(int)
                setSequences = [sequences[x] for x in idx]

                arr2csv(os.path.join(setPath, "sequences.csv"), setSequences, setIds, fmt='%.3f')
                arr2csv(os.path.join(setPath, "labels.csv"), setLbls, setIds, fmt='%s')
            else:
                setIds = np.loadtxt(os.path.join(setPath, "sequences.csv"), delimiter=",", usecols=0)
                setSequences = load_csv(os.path.join(setPath, "sequences.csv"), setIds)
                setLbls = load_csv(os.path.join(setPath, "labels.csv"), setIds, str)

            for lapType in ["aggregated", "none", "individual"]:

                dist = get_custom_distance_discrete(coords, dfunc='braycurtis', lapVal=1.0, lapType=lapType)

                for extWeight in [0.0, 0.2, 0.35, 0.5]:
                    nameTag = ""
                    if extWeight > 0:
                        nameTag += "extWeight_%dE-02" % (extWeight * 100)
                    else:
                        nameTag += "noExt"

                    if lapType == "aggregated":
                        nameTag += "_aggLap_1"
                    elif lapType == "individual":
                        nameTag += "_indLap_1"

                    tPath = os.path.join(setPath, nameTag)
                    if not os.path.exists(tPath):
                        os.mkdir(tPath)
                        pureDists = []
                        mergeDists = []
                        xexts = []
                        yexts = []
                        mergeExt = []

                        for i in tqdm.trange(len(idx), desc = "%s - Set%d/%d - %s" % (cat, setNum + 1, nSets, nameTag)):
                            s1 = setSequences[i]
                            for j in range(i + 1, len(idx)):
                                s2 = setSequences[j]
                                _, _, _, paths, xext, yext = dtw_p(s1, s2, mismatch, extension, 0.35)

                                s1p = s1[paths[0]]
                                s2p = s2[paths[1]]

                                d = dist(s1p, s2p)
                                pureDists.append(copy.copy(d))

                                ext = max(xext, yext)
                                xexts.append(xext)
                                yexts.append(yext)
                                mergeExt.append(ext)

                                if extWeight > 0:
                                    d += extWeight * ext

                                mergeDists.append(d)

                        pureDists = np.array(pureDists)
                        mergeDists = np.array(mergeDists)
                        xexts = np.array(xexts)
                        yexts = np.array(yexts)
                        mergeExt = np.array(mergeExt)

                        np.save(os.path.join(tPath, "mergeDist_%s.npy" % nameTag), mergeDists)
                        np.save(os.path.join(tPath, "pureDist_%s.npy" % nameTag), pureDists)
                        np.save(os.path.join(tPath, "xExt_%s.npy" % nameTag), xexts)
                        np.save(os.path.join(tPath, "yExt_%s.npy" % nameTag), yexts)
                        np.save(os.path.join(tPath, "mergeExt_%s.npy" % nameTag), mergeExt)

                        # np.savetxt(os.path.join(tPath, "mergeDist_%s.txt" % nameTag), mergeDists)
                        # np.savetxt(os.path.join(tPath, "pureDist_%s.txt" % nameTag), pureDists)
                        # np.savetxt(os.path.join(tPath, "xExt_%s.txt" % nameTag), xexts)
                        # np.savetxt(os.path.join(tPath, "yExt_%s.txt" % nameTag), yexts)
                        # np.savetxt(os.path.join(tPath, "mergeExt_%s.txt" % nameTag), mergeExt)

                        if extWeight > 0:
                            nameTag += "_cumExt"
                            tPath = os.path.join(setPath, nameTag)
                            os.mkdir(tPath)

                            np.save(os.path.join(tPath, "mergeDist_%s.npy" % nameTag), mergeDists)
                            np.save(os.path.join(tPath, "pureDist_%s.npy" % nameTag), pureDists)
                            np.save(os.path.join(tPath, "xExt_%s.npy" % nameTag), xexts)
                            np.save(os.path.join(tPath, "yExt_%s.npy" % nameTag), yexts)
                            np.save(os.path.join(tPath, "mergeExt_%s.npy" % nameTag), mergeExt)
                            #
                            # np.savetxt(os.path.join(tPath, "mergeDist_%s.txt" % nameTag), mergeDists)
                            # np.savetxt(os.path.join(tPath, "pureDist_%s.txt" % nameTag), pureDists)
                            # np.savetxt(os.path.join(tPath, "xExt_%s.txt" % nameTag), xexts)
                            # np.savetxt(os.path.join(tPath, "yExt_%s.txt" % nameTag), yexts)
                            # np.savetxt(os.path.join(tPath, "mergeExt_%s.txt" % nameTag), mergeExt)

                    else:
                        if extWeight > 0:
                            nNameTag = nameTag + "_cumExt"
                            ttPath = os.path.join(setPath, nNameTag)
                            if not os.path.exists(ttPath):
                                os.mkdir(ttPath)

                                mergeDists = np.load(os.path.join(tPath, "mergeDist_%s.npy" % nameTag))
                                pureDists = np.load(os.path.join(tPath, "pureDist_%s.npy" % nameTag))
                                xexts = np.load(os.path.join(tPath, "xExt_%s.npy" % nameTag))
                                yexts = np.load(os.path.join(tPath, "yExt_%s.npy" % nameTag))
                                mergeExt = np.load(os.path.join(tPath, "mergeExt_%s.npy" % nameTag))

                                np.save(os.path.join(ttPath, "mergeDist_%s.npy" % nNameTag), mergeDists)
                                np.save(os.path.join(ttPath, "pureDist_%s.npy" % nNameTag), pureDists)
                                np.save(os.path.join(ttPath, "xExt_%s.npy" % nNameTag), xexts)
                                np.save(os.path.join(ttPath, "yExt_%s.npy" % nNameTag), yexts)
                                np.save(os.path.join(ttPath, "mergeExt_%s.npy" % nNameTag), mergeExt)

                                # np.savetxt(os.path.join(ttPath, "mergeDist_%s.txt" % nNameTag), mergeDists)
                                # np.savetxt(os.path.join(ttPath, "pureDist_%s.txt" % nNameTag), pureDists)
                                # np.savetxt(os.path.join(ttPath, "xExt_%s.txt" % nNameTag), xexts)
                                # np.savetxt(os.path.join(ttPath, "yExt_%s.txt" % nNameTag), yexts)
                                # np.savetxt(os.path.join(ttPath, "mergeExt_%s.txt" % nNameTag), mergeExt)


def alphaParamSearch(sequences, labels, outPath, mismatch, extension, dist, tweightFilename,
                     thresh=0.01, nIter=10, useCpp=True, minVal=0.0, maxVal=1.0, iterSelection="random", v=False):
    nTypes = len(np.unique(labels))

    dms = {}
    clusters = {}
    evals = {}

    curAlpha = minVal
    if v:
        cmd = ["pwisedtw", "sequences.csv", tweightFilename, outPath, iterSelection, "-popDTW", "-popDist", "-v", "-alpha"]
    else:
        cmd = ["pwisedtw", "sequences.csv", tweightFilename, outPath, iterSelection, "-popDTW", "-popDist", "-alpha"]
    if useCpp:
        runResult = subprocess.run(cmd + ["%f" % curAlpha])
        if curAlpha >= 1:
            dtwTag = "sequences_popDTW_a%dE+00" % curAlpha
        else:
            dtwTag = "sequences_popDTW_a%dE-04" % (curAlpha * 10000)
        if iterSelection != "":
            cPath = os.path.join(outPath, iterSelection, dtwTag)
        else:
            cPath = os.path.join(outPath, dtwTag)
        dm = np.loadtxt(os.path.join(cPath, "kdigo_dm.csv"), delimiter=",", usecols=[2])
    else:
        dm, dmSave, alignments = dtwDistMat(sequences, mismatch, extension, curAlpha, dist)
        with open(os.path.join(outPath, "dm_alpha_0E-04.csv"), "w") as f:
            for i in range(len(dmSave)):
                f.write(dmSave[i] + "\n")
    link = fc.ward(dm)
    clust = fcluster(link, nTypes, criterion='maxclust')
    val = normalized_mutual_info_score(labels, clust, average_method="arithmetic")
    dms[curAlpha] = dm
    clusters[curAlpha] = clust
    evals[curAlpha] = val

    curAlpha = maxVal
    if useCpp:
        runResult = subprocess.run(cmd + ["%f" % curAlpha])
        if curAlpha >= 1:
            dtwTag = "sequences_popDTW_a%dE+00" % curAlpha
        else:
            dtwTag = "sequences_popDTW_a%dE-04" % (curAlpha * 10000)
        if iterSelection != "":
            cPath = os.path.join(outPath, iterSelection, dtwTag)
        else:
            cPath = os.path.join(outPath, dtwTag)
        dm = np.loadtxt(os.path.join(cPath, "kdigo_dm.csv"), delimiter=",", usecols=[2])
    else:
        dm, dmSave, alignments = dtwDistMat(sequences, mismatch, extension, curAlpha, dist)
        with open(os.path.join(outPath, "dm_alpha_1.csv"), "w") as f:
            for i in range(len(dmSave)):
                f.write(dmSave[i] + "\n")
    link = fc.ward(dm)
    clust = fcluster(link, nTypes, criterion='maxclust')
    val = normalized_mutual_info_score(labels, clust, average_method="arithmetic")
    dms[curAlpha] = dm
    clusters[curAlpha] = clust
    evals[curAlpha] = val

    allOpts = []
    step = float(maxVal - minVal) / (nIter + 1)
    for iterNum in range(nIter):
        if iterSelection == "random":
            curAlpha = float("%.4f" % (np.random.random() * maxVal))
        elif iterSelection == "grid":
            curAlpha = float("%.4f" % ((iterNum + 1) * step))
        else:
            raise ValueError("Parameter iterSelection must be either 'random' or 'grid'. Received: %s" % iterSelection)

        left = minVal
        right = maxVal
        if evals[minVal] > evals[maxVal]:
            opt = [minVal, evals[minVal]]
        else:
            opt = [maxVal, evals[maxVal]]

        while (right - left) > thresh:
            if useCpp:
                runResult = subprocess.run(cmd + ["%f" % curAlpha])
                if curAlpha >= 1:
                    dtwTag = "sequences_popDTW_a%dE+00" % curAlpha
                else:
                    dtwTag = "sequences_popDTW_a%dE-04" % (curAlpha * 10000)
                if iterSelection != "":
                    cPath = os.path.join(outPath, iterSelection, dtwTag)
                else:
                    cPath = os.path.join(outPath, dtwTag)
                dm = np.loadtxt(os.path.join(cPath, "kdigo_dm.csv"), delimiter=",", usecols=[2])
            else:
                dm, dmSave, alignments = dtwDistMat(sequences, mismatch, extension, curAlpha, dist)
                with open(os.path.join(outPath, "dm_alpha_%dE-04.csv" % (10000 * curAlpha)), "w") as f:
                    for i in range(len(dmSave)):
                        f.write(dmSave[i] + "\n")
            link = fc.ward(dm)
            clust = fcluster(link, nTypes, criterion='maxclust')
            val = normalized_mutual_info_score(labels, clust, average_method="arithmetic")
            dms[curAlpha] = dm
            clusters[curAlpha] = clust
            evals[curAlpha] = val

            if val > opt[1]:
                opt = [curAlpha, val]

            # Both left and right are worse than current,
            if evals[left] < val and evals[right] < val:
                lslope = (val - evals[left]) / (curAlpha - left)
                rslope = (val - evals[right]) / (right - curAlpha)
                if lslope > rslope:
                    left = curAlpha
                else:
                    right = curAlpha
            # Left is worse but right is better, so go to right
            elif evals[left] < val:
                left = curAlpha
            # Right is worse but left is better, so go to right
            elif evals[right] < val:
                right = curAlpha
            # Both left and right are better than current
            else:
                if evals[left] > evals[right]:
                    right = curAlpha
                else:
                    left = curAlpha
            curAlpha = (left + right) / 2
        allOpts.append(opt)

    return allOpts, dms, clusters, evals


def dtwDistMat(sequences, mismatch, extension, alpha, dist):
    dm = []
    dmSave = []
    alignments = []
    for i in tqdm.trange(len(sequences), desc="Computing distance matrix with alpha=%.3f" % alpha):
        s1 = sequences[i]
        for j in range(i + 1, len(sequences)):
            s2 = sequences[j]
            _, _, _, paths, xext, yext = dtw_p(s1, s2, mismatch, extension, alpha)
            s1p = s1[paths[0]]
            s2p = s2[paths[1]]
            d = dist(s1p, s2p)
            dm.append(d)
            dmSave.append("%d,%d,%.4f" % (i, j, d))
            alignments.append([s1p, s2p])
    dm = np.array(dm)
    return dm, dmSave, alignments

