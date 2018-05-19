"""
Created on Fri Jan 19 14:31:03 2018

@author: taylorsmith
"""
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.stats.mstats import normaltest
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score,\
    v_measure_score, fowlkes_mallows_score, silhouette_score, calinski_harabaz_score
import fastcluster as fc
from kdigo_funcs import arr2csv
import matplotlib.pyplot as plt
import h5py


# %%
def cluster(X, ids, fname, hfile, method='ward', metric='precomputed', title='Clustering Dendrogram',
            eps=0.5, leaf_size=30, n_clusters=5):
    # if X is 1-D, it is condensed distance matrix, otherwise it is assumed to be
    # an array of m observations of n dimensions
    if method == 'dbscan':
        db = DBSCAN(eps=eps, n_jobs=-1, metric='precomputed', leaf_size=leaf_size)
        db.fit_predict(X)
        lbls = db.labels_
        n_clust = str(len(set(lbls)) - 1)
        cgrp_name = n_clust+'clust'
        try:
            mgrp = hfile['clusters'][method]
        except:
            mgrp = hfile['clusters'].create_group(method)
            mgrp.create_dataset('ids', data=ids, dtype=int)
        mgrp.create_dataset(cgrp_name, data=lbls, dtype=int)
        np.savetxt(fname, np.transpose(np.vstack(ids, lbls)), fmt='%d')
        return lbls
    elif method == 'spectral':
        db = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        db.fit_predict(X)
        lbls = db.labels_
        n_clust = str(len(set(lbls)))
        cgrp_name = n_clust + 'clust'
        try:
            mgrp = hfile['clusters'][method]
        except:
            mgrp = hfile['clusters'].create_group(method)
            mgrp.create_dataset('ids', data=ids, dtype=int)
        mgrp.create_dataset(cgrp_name, data=lbls, dtype=int)
        np.savetxt(fname, np.transpose(np.vstack(ids, lbls)), fmt='%d')
        return lbls
    else:
        link = fc.linkage(X, method=method, metric=metric)
        mgrp = hfile['clusters'].create_group(method)
        mgrp.create_dataset('linkage',data=link)
        mgrp.create_dataset('ids', data=ids, dtype=int)
        np.savetxt(fname, link)
        dendrogram(link, p=50, truncate_mode='lastp')
        plt.ylabel('Distance')
        plt.suptitle(title, fontweight='bold', fontsize=14)
        plt.show()
        return link


def linkage_clust_grps(in_file, cluster_method, n_clusts):
    if type(in_file) == str:
        f = h5py.File(in_file, 'r+')
    else:
        f = in_file
    grp_name = str(n_clusts)+'clusts'
    link = f['clusters'][cluster_method]['linkage']
    grps = fcluster(link, n_clusts, 'maxclust')
    f['clusters'][cluster_method].create_dataset(grp_name, data=grps, dtype=int)
    return grps


def zero_dist_clusters(zero_dist_file, out_fname, val_trans=False):
    f = open(zero_dist_file,'r')
    l = f.readline()
    clusters = []
    while l != '':
        idx = l.split(',')[0]
        ex = []
        for i in range(len(clusters)):
            ex.append(int(idx) in clusters[i])
        if np.any(ex):
            # the section contains an extra search for validation purposes
            # and can be skipped by setting `val_trans` to False. This was included
            # so it could be run both ways to ensure there aren't any inconsistencies
            # in the distance calculations in the previous step.

            while l.split(',')[0] == idx:
                if val_trans:
                    cnum = np.where(ex)[0][0]
                    # transitive property - if x belongs to cluster C, then any
                    t = int(l.split(',')[1])    # patient y s.t.  dist(x,y) = 0 must
                    if t not in clusters[cnum]: # belong to the same cluster
                        clusters[cnum].append(t)
                l = f.readline()
        else:
            clusters.append([int(idx)])
            while l.split(',')[0] == idx:
                clusters[-1].append(int(l.split(',')[1]))
                l = f.readline()
    f.close()
    arr2csv(out_fname, clusters, fmt='%d')
    outFile = open(out_fname, 'w')
    for i in range(len(clusters)):
        outFile.write('Cluster #%d' % (i+1))
        for j in range(len(clusters[i])):
            outFile.write(',%d' % (clusters[i][j]))
        outFile.write('\n')
    outFile.close()


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


def inter_intra_dist(in_file, dm, cluster_method, n_clust, op='mean', out_path='', plot='both'):
    if type(in_file) == str:
        f = h5py.File(in_file, 'r')
    else:
        f = in_file
    if type(dm) == str:
        sqdm = f[dm][:]
    else:
        sqdm = np.copy(dm[:])
    if sqdm.ndim == 1:
        sqdm = squareform(sqdm)

    np.fill_diagonal(sqdm, np.nan)
    ids = f['clusters'][cluster_method]['ids'][:]
    clusters = f['clusters'][cluster_method][n_clust][:]

    lbls = np.unique(clusters)
    n_clusters = len(lbls)

    if op == 'mean':
        func = lambda x: np.nanmean(x)
    elif op == 'max':
        func = lambda x: np.nanmax(x)
    elif op == 'min':
        func = lambda x: np.nanmin(x)

    all_inter = []
    all_intra = []
    for i in range(len(ids)):
        clust = clusters[i]
        idx = np.array((i,))
        cids = np.where(clusters == clust)[0]
        intra = np.ix_(idx, np.setdiff1d(cids, idx))
        inter = np.ix_(idx, np.setdiff1d(range(len(ids)), cids))
        intrad = func(sqdm[intra])
        interd = func(sqdm[inter])
        all_inter.append(interd)
        all_intra.append(intrad)
    all_inter = np.array(all_inter)
    all_intra = np.array(all_intra)
    print('Normal tests:')
    for i in range(n_clusters):
        clust = lbls[i]
        cidx = np.where(clusters == clust)[0]
        if len(cidx) < 8:
            print('Cluster ' + str(clust) + ':\tLess Than 8 Samples')
        else:
            n = normaltest(all_intra[cidx].flatten())
            print('Cluster ' + str(clust) + ':\t' + str(n))
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

    np.savetxt(out_path+'inter_intra_dist.txt', np.transpose(np.vstack((all_inter, all_intra))))
    return sqdm, all_inter, all_intra


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
