#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 16:41:19 2017

@author: taylorsmith
"""

import matplotlib.pyplot as plt
import numpy as np

#%%Plot histogram from data in file f
def hist(f,figName,title,op=None,bins=50,skip_row=False,skip_col=True,x_lbl='',y_lbl='',x_rng=None,y_rng=None):
    res = []
    if skip_row:
        _ = f.readline()
    for l in f:
        if skip_col:
            data = np.array(l.strip().split(',')[1:],dtype=float)
        else:
            data = np.array(l.strip().split(','),dtype=float)
        if op == 'count':
            count = np.where(data==1)[0]
            data = len(count)

        res.append(data)

    try:
        res = np.array(res)
    except:
        res = np.concatenate(res)

    plt.figure()
    plt.hist(res,bins=bins)
    plt.xlim(x_rng)
    plt.ylim(y_rng)
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.title(title)
    plt.savefig(figName)


#%%Generate line plot for data in file f corresponding to the patients with their
#id in the list
def multi_plot(fname,ids,title,out_path,x_lbl='',y_lbl='',x_rng=None,y_rng=None,x_res=1):
    f = open(fname,'r')
    #ids = [20236, 53596, 17370, 71346, 2106, 54290]
    for line in f:
        l = line.rstrip().split(',')
        idx = int(l[0])
        if idx not in ids:
            continue
        vec = np.array(l[1:],dtype=float)
        t = np.zeros(len(vec))
        for i in range(1,len(vec)):
            t[i]+=(t[i-1]+float(x_res))
        plot(t,vec,out_path,x_lbl,y_lbl,x_rng,y_rng)
    f.close()

#Single line plot
def plot(x,y,idx,title,path,x_lbl='',y_lbl='',x_rng=None,y_rng=None):
    plt.figure()
    plt.title(title + ' - ' + str(idx))
    plt.plot(x,y)
    plt.xlim(x_rng)
    plt.ylim(y_rng)
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.savefig(path+title+'-'+str(idx)+'.pdf')
    plt.clf()
