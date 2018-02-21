#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 16:41:19 2017

@author: taylorsmith
"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
from plotly.graph_objs import *
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform


def network_vis(dm, ids, lbls, edge_thresh=0.2, title='', annot=''):
    mds = MDS(dissimilarity='precomputed', n_jobs=-1)
    sq_dist = squareform(dm[:,2])
    fit = mds.fit(sq_dist)
    coords = fit.embedding_
    edges = np.array([dm[x, :2] for x in range(len(dm)) if dm[x, 0] < edge_thresh])
    for i in range(len(edges)):
        edges[i, 0] = np.where(ids == edges[i, 0])[0][0]
        edges[i, 1] = np.where(ids == edges[i, 1])[0][0]

    edge_trace = Scatter(
        x=[],
        y=[],
        line=Line(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in edges:
        x0, y0 = coords[edge[0]]
        x1, y1 = coords[edge[1]]
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]

    node_trace = Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=Marker(
            showscale=True,
            # colorscale options
            # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
            # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
            colorscale='YIGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    for node in coords:
        x, y = node
        node_trace['x'].append(x)
        node_trace['y'].append(y)

    for lbl in lbls:
        node_trace['marker']['color'].append(lbl)
        node_info = 'cluster ID: ' + str(lbl)
        node_trace['text'].append(node_info)

    fig = Figure(data=Data([edge_trace, node_trace]),
                 layout=Layout(
                     title=title,
                     titlefont=dict(size=16),
                     showlegend=False,
                     hovermode='closest',
                     margin=dict(b=20, l=5, r=5, t=40),
                     annotations=[dict(
                         text=annot,
                         showarrow=False,
                         xref="paper", yref="paper",
                         x=0.005, y=-0.002)],
                     xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                     yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))

    py.iplot(fig, filename='networkx')


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

