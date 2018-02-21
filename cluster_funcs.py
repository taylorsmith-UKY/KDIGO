#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:31:03 2018

@author: taylorsmith
"""
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.cluster import DBSCAN
import fastcluster as fc
import matplotlib.pyplot as plt


#%%
def cluster(X,ids,fname,method='ward',metric='euclidean',title='Clustering Dendrogram',eps=0.5,leaf_size=30):
    #if X is 1-D, it is condensed distance matrix, otherwise it is assumed to be
    #an array of m observations of n dimensions
    if method == 'dbscan':
        db = DBSCAN(eps=eps,n_jobs=-1,metric=metric,leaf_size=leaf_size)
        db.fit_predict(X)
        return db, lbls
    else:
        link = fc.linkage(X,method=method,metric=metric)
        dend = dendrogram(link,labels=ids)
        order = np.array(dend['leaves'],dtype=int)
        c_ids = ids[order]
        np.savetxt('ids_cluster_order.txt',c_ids,fmt='%d')
        plt.xlabel('Patient ID')
        plt.ylabel('Distance')
        plt.suptitle(title, fontweight='bold', fontsize=14)
        plt.show()
        np.savetxt(fname,link)
        return link

def db_cluster(X,ids,fname,metric=’euclidean’,leaf_size=30):

def clust_grps(link_file,n_clusts):
    dend = np.loadtxt(link_file)
    grps = fcluster(dend,n_clusts,'maxclust')
    np.savetxt('clusters_'+str(n_clusts)+'grps.txt',grps)

def filter_dist_val(kdigo_dm_fname,out_fname,val=0.0,keep=True):
    #val = x and keep = True returns all examples where dist = x
    #val = x and keep = False returns all examples EXCEPT thos where dist = x
    f=open(kdigo_dm_fname,'r')
    out = open(out_fname,'w')
    for line in f:
        l = line.rstrip().split(',')
        if float(l[2]) == val and keep:
            out.write(line)
        elif float(l[2]) != val and not keep:
            out.write(line)
    f.close()
    out.close()

def zero_dist_clusters(zero_dist_file,out_fname,val_trans=False):
    f=open(zero_dist_file,'r')
    l=f.readline()
    clusters = []
    while l != '':
        idx = l.split(',')[0]
        ex = []
        for i in range(len(clusters)):
            ex.append(int(idx) in clusters[i])
        if np.any(ex):
            #the section contains an extra search for validation purposes
            #and can be skipped by setting `val_trans` to False. This was included
            #so it could be run both ways to ensure there aren't any inconsistencies
            #in the distance calculations in the previous step.

            while(l.split(',')[0]==idx):
                if val_trans:
                    cnum = np.where(ex)[0][0]
                    #transitive property - if x belongs to cluster C, then any
                    t = int(l.split(',')[1])    # patient y s.t.  dist(x,y) = 0 must
                    if t not in clusters[cnum]: # belong to the same cluster
                        clusters[cnum].append(t)
                l=f.readline()
        else:
            clusters.append([int(idx)])
            while(l.split(',')[0]==idx):
                clusters[-1].append(int(l.split(',')[1]))
                l=f.readline()
    f.close()
    arr2csv(out_fname,clusters,fmt='%d')
    outFile=open(out_fname,'w')
    for i in range(len(clusters)):
        outFile.write('Cluster #%d' % (i+1))
        for j in range(len(clusters[i])):
            outFile.write(',%d' % (clusters[i][j]))
        outFile.write('\n')
    outFile.close()
