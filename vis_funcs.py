#!/usr/bin/env python2# -*- coding: utf-8 -*-"""Created on Sun Dec  3 16:41:19 2017@author: taylorsmith"""import matplotlib.pyplot as pltimport numpy as npfrom sklearn.manifold import MDSfrom scipy.spatial.distance import squareformfrom mpl_toolkits.mplot3d import Axes3D# def network_vis(dm, ids, lbls, sizes, edge_thresh=0.2, title='', annot=''):#     mds = MDS(n_components=2, dissimilarity='precomputed', n_jobs=-1)#     sq_dist = squareform(dm[:, 2])#     fit = mds.fit(sq_dist)#     coords = fit.embedding_#     edges = np.array([dm[x, :2] for x in range(len(dm)) if dm[x, 0] < edge_thresh])#     lbl_set = set(lbls)#     for i in range(np.shape(edges)[0]):#         edges[i, 0] = np.where(ids == edges[i, 0])[0][0]#         edges[i, 1] = np.where(ids == edges[i, 1])[0][0]##     edge_trace = Scatter(#         x=[],#         y=[],#         line=Line(width=0.5, color='#888'),#         hoverinfo='none',#         mode='lines')##     for edge in edges:#         x0, y0 = coords[edge[0]]#         x1, y1 = coords[edge[1]]#         edge_trace['x'] += [x0, x1, None]#         edge_trace['y'] += [y0, y1, None]##     node_traces = []#     for i in range(len(lbl_set)):#         node_traces.append(Scattergl(#             x=[],#             y=[],#             text=[],#             name='Cluster ID: ' + str(lbls[i]),#             mode='markers',#             hoverinfo='text',#             marker=Marker(#                 showscale=True,#                 colorscale='YIGnBu',#                 reversescale=True,#                 color=[],#                 size=[],#                 colorbar=dict(#                     thickness=15,#                     title='Node Connections',#                     xanchor='left',#                     titleside='right'#                 ),#                 line=dict(width=2))))##     for i in range(len(coords)):#         x, y = coords[i, :]#         lbl = lbls[i]#         size = sizes[i]#         grp = np.where(lbl_set == lbl)[0][0]#         node_traces[grp]['x'].append(x)#         node_traces[grp]['y'].append(y)##         node_traces[grp]['marker']['color'].append(lbl)#         node_traces[grp]['marker']['size'].append(size)#         node_info = 'cluster ID: ' + str(lbl)#         node_traces[grp]['text'].append(node_info)##     fig = Figure(data=Data([edge_trace, node_traces]),#                  layout=Layout(#                      title=title,#                      titlefont=dict(size=16),#                      showlegend=False,#                      hovermode='closest',#                      margin=dict(b=20, l=5, r=5, t=40),#                      annotations=[dict(#                          text=annot,#                          showarrow=False,#                          xref="paper", yref="paper",#                          x=0.005, y=-0.002)],#                      xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),#                      yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))##     py.iplot(fig, filename='kdigo_7dayICU')# %%Plot histogram from data in file fdef hist(f, fig_name, title, op=None, bins=50, skip_row=False, skip_col=True, x_lbl='', y_lbl='',         x_rng=None, y_rng=None):    res = []    if skip_row:        _ = f.readline()    for l in f:        if skip_col:            data = np.array(l.strip().split(',')[1:], dtype=float)        else:            data = np.array(l.strip().split(','), dtype=float)        if op == 'count':            count = np.where(data == 1)[0]            data = len(count)        res.append(data)    try:        res = np.array(res)    except:        res = np.concatenate(res)    plt.figure()    plt.hist(res, bins=bins)    plt.xlim(x_rng)    plt.ylim(y_rng)    plt.xlabel(x_lbl)    plt.ylabel(y_lbl)    plt.title(title)    plt.savefig(fig_name)# %%Generate line plot for data in file f corresponding to the patients with their id in the listdef multi_plot(fname, ids, title, out_path, x_lbl='', y_lbl='', x_rng=None, y_rng=None, x_res=1):    f = open(fname, 'r')    # ids = [20236, 53596, 17370, 71346, 2106, 54290]    for line in f:        l = line.rstrip().split(',')        idx = int(l[0])        if idx not in ids:            continue        vec = np.array(l[1:], dtype=float)        t = np.zeros(len(vec))        for i in range(1, len(vec)):            t[i] += (t[i - 1] + float(x_res))        plot(t, vec, out_path, x_lbl, y_lbl, x_rng, y_rng)    f.close()# Single line plotdef plot(x, y, idx, title, path, x_lbl='', y_lbl='', x_rng=None, y_rng=None):    plt.figure()    plt.title(title + ' - ' + str(idx))    plt.plot(x, y)    plt.xlim(x_rng)    plt.ylim(y_rng)    plt.xlabel(x_lbl)    plt.ylabel(y_lbl)    plt.savefig(path + title + '-' + str(idx) + '.pdf')    plt.clf()def stacked_bar(summary_file, ylim=14, fname='', title=''):    data = np.loadtxt(summary_file, delimiter=',', skiprows=1)    f = open(summary_file, 'r')    f.seek(0)    h = np.array(f.readline().rstrip().split(','))    mort_idx = np.where(h == 'mort_pct')[0][0]    n_clust = data.shape[0]    k0_idx = np.where(h == 'n_kdigo_1')[0][0]    plot_data = data[:,k0_idx:k0_idx+4].astype(float)/1000    for i in range(1, 4):        plot_data[:, i] += plot_data[:, i - 1]    cml = data[:, mort_idx] * 100    plt.figure()    plt.rc('font', size=15)    ax = plt.subplot(111)    b = plt.bar(range(1, n_clust + 1), plot_data[:, 3], label='KDIGO 3D', color='red', align='center')    plt.bar(range(1, n_clust + 1), plot_data[:, 2], label='KDIGO 3', color='yellow', align='center')    plt.bar(range(1, n_clust + 1), plot_data[:, 1], label='KDIGO 2', color='blue', align='center')    plt.bar(range(1, n_clust + 1), plot_data[:, 0], label='KDIGO 1', color='green', align='center')    plt.ylabel('Number of Patients (x1000)')    plt.xlabel('Cluster ID')    plt.xticks(range(1, n_clust+1), range(1, n_clust+1))    autolabel(ax, b, cml)    plt.legend(loc='upper right')    plt.ylim((0, ylim))    plt.title(title)    plt.tight_layout()    if fname is not '':        plt.savefig(fname)    else:        plt.show()'''def stacked_bar(data_file, primary_idx=0, secondary_idx=3, dt=int, primary_labels=[],                secondary_labels=[], sort=0, fname='', ylim=14):    data = np.loadtxt(data_file, delimiter=',', dtype=dt, skiprows=1)    data1 = data[:, primary_idx]    data2 = data[:, secondary_idx]    lbls1 = np.unique(data1)    lbls2 = np.unique(data2)    n_labels = len(lbls1)    n_labels2 = len(lbls2)    data_dist = np.ones((n_labels, n_labels2))    for i in range(n_labels):        for j in range(n_labels2):            data_dist[i, j] = float(len(np.where(data[np.where(data[:, primary_idx] == lbls1[i])[0], \                                                      secondary_idx] == lbls2[j])[0])) \                              / len(np.where(data[:, primary_idx] == lbls1[i])[0])    counts = np.zeros(n_labels)    for i in range(n_labels):        counts[i] = len(np.where(data1 == lbls1[i])[0])    plot_data = np.zeros((n_labels, n_labels2))    for i in range(n_labels):        plot_data[i, :] = data_dist[i, :] * counts[i]    for i in range(1, n_labels2):        plot_data[:, i] += plot_data[:, i - 1]    plt.figure()    cct = float(counts) / 1000    cm = data_dist[:, 1]    cml = cm * 100    cnm = labels    if sort == 0:        o = np.argsort(cm)        cnm = cnm[o]        cm = cm[o]        cct = cct[o]        cml = cml[o]    else:        o = np.argsort(cct)[::-1]        cnm = cnm[o]        cm = cm[o]        cct = cct[o]        cml = cml[o]    plt.clf()    plt.rc('font', size=15)    ax = plt.subplot(111)    b = plt.bar(range(1, n_labels + 1), cct, label='Deceased at Discharge', color='red')    plt.bar(range(1, n_labels+1), cct * (1 - cm), label='Alive at Discharge', color='blue')    plt.xticks(range(1, n_labels+1), cnm, rotation=30)    plt.ylabel('Number of Patients (x1000)')    autolabel(ax, b, cml)    plt.legend(loc='upper right')    plt.ylim((0, ylim))    plt.tight_layout()    if fname is not '':        plt.savefig(fname)    else:        plt.show()'''def scatter(coords, lbls, figname):    fig = plt.figure()    if coords.shape[1] == 3:        ax = fig.add_subplot(111, projection='3d')        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=lbls)    elif coords.shape[1] == 2:        ax = plt.subplot(111)        ax.scatter(coords[:, 0], coords[:, 1], c=lbls)    plt.savefig(figname)    returndef autolabel(ax, rects, lbls):    """    Attach a text label above each bar displaying its height    """    for rect, lbl in zip(rects, lbls):        height = rect.get_height()        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,                '%.1f%%' % lbl,                ha='center', va='bottom', fontsize=15)# , rotation=30)