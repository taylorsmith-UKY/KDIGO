'''
/*******************************************************************************
 * Copyright (C) 2018 Francois Petitjean
 * Edited 2019 for Population DTW - Taylor D. Smith
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
'''
from __future__ import division
import numpy as np
from scipy.spatial.distance import squareform, cityblock
from dtw_distance import dtw_p
from scipy.stats import sem, t
import tqdm
import copy

__author__ ="Francois Petitjean"


def performDBA(series, dm, n_iterations=10, mismatch=lambda x,y:abs(x-y), extension=lambda x: 0, extraDesc='', save_every=False, alpha=1.0, aggExt=False, ptsPerDay=4, targlen=14, seedType='medoid'):
    if dm.ndim == 1:
        sqdm = squareform(dm)
    elif dm.ndim == 2:
        sqdm = dm

    # medoid_ind = np.argmin(np.sum(sqdm, axis=0))
    if seedType == 'medoid':
        o = np.argsort(np.sum(sqdm, axis=0))
        assert len(o) == len(series)
        sel = 0
        maxLen = 0
        maxIdx = 0
        while sel < len(o) and len(series[o[sel]]) != ptsPerDay*targlen+ptsPerDay:
            if len(series[o[sel]]) > maxLen and len(series[o[sel]]) < ptsPerDay*targlen+ptsPerDay:
                maxLen = len(series[o[sel]])
                maxIdx = sel
            sel += 1
        if sel == len(o):
            sel = maxIdx

        medoid_ind = o[sel]

        center = series[medoid_ind]
    elif seedType == 'mean':
        ml = max([len(x) for x in series])
        vals = np.zeros(ml)
        cts = np.zeros(ml)
        for i in range(len(series)):
            for j in range(len(series[i])):
                vals[j] += series[i][j]
                cts[j] += 1
        center = vals / cts
        center = center[:ptsPerDay*targlen+ptsPerDay]
    elif seedType == 'zeros':
        center = np.zeros(ptsPerDay*targlen+ptsPerDay)

    apaths = []
    if save_every:
        cout = []
        stdout = []
        confout = []
    for i in tqdm.trange(0, n_iterations, desc='DBA Iterations' + extraDesc):
        # print('Iteration %d/%d' % (i+1, n_iterations))
        temp = copy.deepcopy(center)
        center, stds, confs, paths = DBA_update(center, series, mismatch, extension, alpha=alpha, aggExt=aggExt)
        if save_every:
            cout.append(temp)
            stdout.append(stds)
            confout.append(confs)
            apaths.append(paths)
        if cityblock(temp, center) == 0:
            break
    if not save_every:
        return center, stds, confs, paths

    return cout, stdout, confout, apaths


def DBA_update(center, series, mismatch, extension, alpha=1.0, aggExt=False):
    updated_center = np.zeros(center.shape)
    matched_vals = [[] for _ in range(len(center))]
    n_elements = np.array(np.zeros(center.shape), dtype=int)
    paths = []
    for sidx in range(len(series)):
        s = series[sidx]
        _, _, _, p, _, _ = dtw_p(center, s, mismatch=mismatch, extension=extension, alpha=alpha, aggExt=aggExt)
        pairs = []
        for idx in range(len(p[0])):
            i = p[0][idx]
            j = p[1][idx]
            updated_center[i] += s[j]
            n_elements[i] += 1
            matched_vals[i].append(s[j])
            pairs.append([i, j])
        updated_center[i] += s[j]
        n_elements[i] += 1
        paths.append(p[1])
    means, stds, confs = describe_values(matched_vals)
    return means, stds, confs, paths


def describe_values(data, confidence=0.95):
    '''
    Returns the mean confidence interval for the distribution of data
    :param data:
    :param confidence: decimal percentile, i.e. 0.95 = 95% confidence interval
    :return:
    '''
    n_pts = len(data)
    means = np.zeros(n_pts)
    confs = np.zeros(n_pts)
    stds = np.zeros(n_pts)
    for i in range(n_pts):
        x = np.array(data[i])
        n = len(x)
        m, se, std = np.mean(x), sem(x), np.std(x)
        h = se * t.ppf((1 + confidence) / 2., n-1)
        means[i] = m
        confs[i] = h
        stds[i] = std
    return means, stds, confs