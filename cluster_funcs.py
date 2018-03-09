#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:31:03 2018

@author: taylorsmith
"""
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score,\
    v_measure_score, fowlkes_mallows_score, silhouette_score, calinski_harabaz_score
import fastcluster as fc
from kdigo_funcs import arr2csv
import matplotlib.pyplot as plt


# %%
def cluster(X, ids, fname, method='ward', metric='precomputed', title='Clustering Dendrogram',
            eps=0.5, leaf_size=30, n_clusters = 5):
    # if X is 1-D, it is condensed distance matrix, otherwise it is assumed to be
    # an array of m observations of n dimensions
    if method == 'dbscan':
        db = DBSCAN(eps=eps, n_jobs=-1, metric='precomputed', leaf_size=leaf_size)
        db.fit_predict(X)
        lbls = db.labels_
        np.savetxt(fname, np.transpose(np.vstack(ids, lbls)), fmt='%d')
        return lbls
    elif method == 'spectral':
        db = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        db.fit_predict(X)
        lbls = db.labels_
        np.savetxt(fname, np.transpose(np.vstack(ids, lbls)), fmt='%d')
        return lbls
    else:
        link = fc.linkage(X, method=method, metric=metric)
        dend = dendrogram(link,p=50,truncate_mode='lastp')
        order = np.array(dend['leaves'],dtype=int)
        c_ids = ids[order]
        np.savetxt('ids_cluster_order.txt', c_ids, fmt='%d')
        plt.xlabel('Patient ID')
        plt.ylabel('Distance')
        plt.suptitle(title, fontweight='bold', fontsize=14)
        plt.show()
        np.savetxt(fname, link)
        return link


def clust_grps(link_file, id_file, n_clusts):
    dend = np.loadtxt(link_file)
    ids = np.loadtxt(id_file)
    grps = fcluster(dend, n_clusts, 'maxclust')
    np.savetxt('clusters_'+str(n_clusts)+'grps.txt', np.transpose(np.vstack(ids, grps)), fmt='%d')
    return grps


def filter_dist_val(kdigo_dm_fname, out_fname, val=0.0, keep=True):
    # val = x and keep = True returns all examples where dist = x
    # val = x and keep = False returns all examples EXCEPT thos where dist = x
    f = open(kdigo_dm_fname, 'r')
    out = open(out_fname, 'w')
    for line in f:
        l = line.rstrip().split(',')
        if float(l[2]) == val and keep:
            out.write(line)
        elif float(l[2]) != val and not keep:
            out.write(line)
    f.close()
    out.close()


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


def eval_clusters(dist_file, clust_file, labels_true=None):
    dm = np.loadtxt(dist_file, delimiter=',', usecols=2)
    sqdm = squareform(dm)
    cdata = np.loadtxt(clust_file, dtype=int)
    clusters = cdata[:, 1]
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


def inter_intra_dist(dm_file, clust_file, op='mean', out_file='inter_intra_dist.txt', sqdm=np.array([])):
    if sqdm.size == 0:
        dm = np.loadtxt(dm_file, delimiter=',', usecols=2)
        sqdm = squareform(dm)
        del dm
    np.fill_diagonal(sqdm, np.nan)
    cd = np.loadtxt(clust_file, dtype=int)
    ids = cd[:, 0]
    clusters = cd[:, 1]

    lbls = np.unique(clusters)
    n_clusters = len(lbls)

    all_cids = []
    for i in range(n_clusters):
        all_cids.append(np.where(clusters == lbls[i])[0])

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
    for i in range(n_clusters):
        clust = lbls[i]
        cidx = np.where(clusters == clust)[0]
        plt.figure(i+1)
        plt.subplot(121)
        plt.hist(all_inter[cidx], bins=50)
        plt.title('Inter-Cluster')
        plt.subplot(122)
        plt.hist(all_intra[cidx], bins=50)
        plt.title('Intra-Cluster')
        if clust >= 0:
            plt.suptitle('Cluster ' + str(clust + 1) + ' Separation')
            plt.savefig('cluster' + str(clust + 1) + '_separation_hist.png')
        else:
            plt.suptitle('Noise Point Separation')
            plt.savefig('noise_separation_hist.png')

    np.savetxt(out_file, np.transpose(np.vstack((all_inter, all_intra))))
    return sqdm, all_inter, all_intra


def dm_to_sim(dist_file, out_name, beta=1, eps=1e-6):
    dm = np.loadtxt(dist_file, delimiter=',')
    dm_std = np.std(dm[:,2])
    out = open(out_name, 'w')
    for i in range(np.shape(dm)[0]):
        sim = np.exp(-beta * dm[i, 2] / dm_std) + eps
        out.write('%d,%d,%.4f\n' % (dm[i, 0], dm[i, 1], sim))
    out.close()
