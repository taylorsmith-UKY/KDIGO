from scipy.spatial.distance import braycurtis, euclidean, cityblock
from tqdm import tqdm
import numpy as np
import os
from scipy.interpolate import interp1d, interp2d


def pairwise_feat_dist(features, ids, dm_fname, dist=lambda x, xx: np.abs(x - xx),
                       desc='Feature Based Distance Calculation'):
    df = open(dm_fname, 'w')
    dis = []
    for i in tqdm(range(len(features)), desc=desc):
        for j in range(i + 1, len(features)):
            d = dist(features[i], features[j])
            df.write('%d,%d,%f' % (ids[i], ids[j], d))
            dis.append(d)
    dis = np.array(dis)
    df.close()
    return dis


# %%
def pairwise_dtw_dist(patients, days, ids, dm_fname, dtw_name, v=True,
                      mismatch=lambda y, yy: abs(y-yy),
                      extension=lambda y: 0,
                      dist=braycurtis,
                      alpha=1.0, t_lim=7, aggext=False):
    '''
    For each pair of arrays in patients, this function first applies dynamic time warping to
    align the arrays to the same length and then computes the distance between the aligned arrays.

    :param patients: List containing N individual arrays of variable length to be aligned
    :param days: List containing N arrays, each indicating the day of each individual point
    :param ids: List of N patient IDs
    :param dm_fname: Filename to save distances
    :param dtw_name: Filename to save DTW alignment
    :param v: verbose... if True prints extra information
    :param mismatch: function handle... determines cost of mismatched values
    :param extension: function handle... if present, introduces extension penalty coresponding to value
    :param dist: function handle... calculates the total distance between the aligned curves
    :param alpha: float value... specifies weight of extension penalty vs. mismatch penalty
    :param t_lim: float/integer... only include data where day <= t_lim
    :return: condensed pair-wise distance matrix
    '''
    df = open(dm_fname, 'w')
    dis = []
    if not os.path.exists(dtw_name):
        if v and dtw_name is not None:
            log = open(dtw_name, 'w')
        for i in tqdm(range(len(patients)), desc='DTW and Distance Calculation'):
            sel = np.where(days[i] <= t_lim)[0]
            patient1 = np.array(patients[i])[sel]
            dlist = []
            for j in range(i + 1, len(patients)):
                df.write('%d,%d,' % (ids[i], ids[j]))
                sel = np.where(days[j] <= t_lim)[0]
                patient2 = np.array(patients[j])[sel]
                if np.all(patient1 == patient2):
                    df.write('%f\n' % 0)
                    dis.append(0)
                    dlist.append(0)
                else:
                    if len(patient1) > 1 and len(patient2) > 1:
                        d, _, _, path, xext, yext = dtw_p(patient1, patient2, mismatch=mismatch, extension=extension, alpha=alpha, aggExt=aggext)
                        p1_path = path[0]
                        p2_path = path[1]
                        p1 = np.array([patient1[p1_path[x]] for x in range(len(p1_path))])
                        p2 = np.array([patient2[p2_path[x]] for x in range(len(p2_path))])
                    elif len(patient1) == 1:
                        p1 = np.repeat(patient1[0], len(patient2))
                        p2 = patient2
                    elif len(patient2) == 1:
                        p1 = patient1
                        p2 = np.repeat(patient2[0], len(patient1))
                    if np.all(p1 == p2):
                        df.write('%f\n' % 0)
                        dis.append(0)
                        dlist.append(0)
                    else:
                        d = dist(p1, p2)
                        df.write('%f\n' % d)
                        dis.append(d)
                        dlist.append(d)
                if v and dtw_name is not None:
                    log.write(','.join(p1.astype(str)) + '\n')
                    log.write(','.join(p2.astype(str)) + '\n\n')
        if v and dtw_name is not None:
            log.close()
    else:
        dtw = open(dtw_name, 'r')
        for i in tqdm(range(len(ids)), desc='Distance Calculation using Previous DTW'):
            for j in range(i+1, len(ids)):
                p1 = np.array(dtw.readline().rstrip().split(','), dtype=int)
                p2 = np.array(dtw.readline().rstrip().split(','), dtype=int)
                assert p1[0] == ids[i] and p2[0] == ids[j]
                d = dist(p1[1:], p2[1:])
                dis.append(d)
                df.write('%d,%d,%f\n' % (ids[i], ids[j], d))
                _ = dtw.readline()
        dtw.close()
    df.close()
    return dis


# %%
def dtw_p(x, y, mismatch=lambda y, yy: abs(y-yy),
                extension=lambda y: 0,
                alpha=1.0, aggExt=False):
    """
    Computes Dynamic Time Warping (DTW) of two sequences with weighted penalty exponentiation.
    Designed for sequences of distinct integer values in the set [0, 1, 2, 3, 4]

    :param array x: N1*M array
    :param array y: N2*M array
    :param func mismatch: distance used as cost measure
    :param func extension: extension penalty applied when repeating index
    :param alpha: float value indicating relative weight of extension penalty

    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the warp path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = np.zeros((r + 1, c + 1), dtype=float)  # distance matrix
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = mismatch(x[i], y[j])
    C = D1.copy()
    ext_y = np.zeros((r + 1, c + 1), dtype=float)
    ext_x = np.zeros((r + 1, c + 1), dtype=float)
    x_dup = np.zeros((r + 1, c + 1), dtype=float)
    y_dup = np.zeros((r + 1, c + 1), dtype=float)
    for i in range(r):
        for j in range(c):
            diag = D0[i, j]
            if aggExt:
                sel = np.argmin((D0[i, j] + ext_x[i, j] + ext_y[i, j],                                         # 0: diagonal
                                 D0[i, j + 1] + ext_x[i, j + 1] + ext_y[i, j + 1] + alpha * ((y_dup[i, j + 1] + 1) * extension(y[j])),   # 1: repeat y
                                 D0[i + 1, j] + ext_x[i + 1, j] + ext_y[i + 1, j] + alpha * ((x_dup[i + 1, j] + 1) * extension(x[i]))))  # 2: repeat x
                if sel == 1:
                    # ext_y[i + 1, j + 1] = ext_y[i, j + 1] + alpha * ((y_dup[i, j + 1] + 1) * extension(y[j]))
                    ext_y[i + 1, j + 1] = alpha * ((y_dup[i, j + 1] + 1) * extension(y[j]))
                    # ext_x[i + 1, j + 1] = ext_x[i, j + 1]
                    ext_x[i + 1, j + 1] = 0
                    D1[i, j] += D0[i, j + 1]
                    y_dup[i + 1, j + 1] = y_dup[i, j + 1] + 1
                    x_dup[i + 1, j + 1] = 0
                elif sel == 2:
                    # ext_x[i + 1, j + 1] = ext_x[i + 1, j] + alpha * ((x_dup[i + 1, j] + 1) * extension(x[i]))
                    # ext_y[i + 1, j + 1] = ext_y[i + 1, j]
                    ext_x[i + 1, j + 1] = alpha * ((x_dup[i + 1, j] + 1) * extension(x[i]))
                    ext_y[i + 1, j + 1] = 0
                    D1[i, j] += D0[i + 1, j]
                    x_dup[i + 1, j + 1] = x_dup[i + 1, j] + 1
                    y_dup[i + 1, j + 1] = 0
                else:
                    D1[i, j] += diag
                    # ext_x[i + 1, j + 1] = ext_x[i, j]
                    # ext_y[i + 1, j + 1] = ext_y[i, j]
                    ext_x[i + 1, j + 1] = 0
                    ext_y[i + 1, j + 1] = 0
                    x_dup[i + 1, j + 1] = 0
                    y_dup[i + 1, j + 1] = 0
                D1[i, j] += ext_x[i + 1, j + 1] + ext_y[i + 1, j + 1]
            else:
                sel = np.argmin((D0[i, j],                                         # 0: diagonal
                                 D0[i, j + 1] + alpha * ((y_dup[i, j + 1] + 1) * extension(y[j])),   # 1: repeat y
                                 D0[i + 1, j] + alpha * ((x_dup[i + 1, j] + 1) * extension(x[i]))))  # 2: repeat x
                if sel == 1:
                    D1[i, j] += D0[i, j + 1] + alpha * ((y_dup[i, j + 1] + 1) * extension(y[j]))
                    y_dup[i + 1, j + 1] = y_dup[i, j + 1] + 1
                    x_dup[i + 1, j + 1] = 0
                elif sel == 2:
                    D1[i, j] += D0[i + 1, j] + alpha * ((x_dup[i + 1, j] + 1) * extension(x[i]))
                    x_dup[i + 1, j + 1] = x_dup[i + 1, j] + 1
                    y_dup[i + 1, j + 1] = 0
                else:
                    D1[i, j] += diag
                    x_dup[i + 1, j + 1] = 0
                    y_dup[i + 1, j + 1] = 0
    if len(x) == 1:
        path = np.zeros(len(y))
    elif len(y) == 1:
        path = np.zeros(len(x))
    else:
        path = _traceback(D0)
    xext = 0
    yext = 0
    for pi in range(len(path[0])):
        xidx = np.where(path[0] == pi)[0]
        yidx = np.where(path[1] == pi)[0]
        if aggExt:
            for xi in range(1, len(xidx)):
                xext += alpha * xi * extension(x[path[0][pi]])
            for yi in range(1, len(yidx)):
                yext += alpha * yi * extension(y[path[1][pi]])
        else:
            if xidx.size > 0:
                xext += alpha * (len(xidx) - 1) * extension(x[pi])
            if yidx.size > 0:
                yext += alpha * (len(yidx) - 1) * extension(y[pi])
    return D1[-1, -1] / sum(D1.shape), C, D1, path, xext, yext


def _traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while i > 0 or j > 0:
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1

        else:  # (tb == 2):
            j -= 1

        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)


def mismatch_penalty_func(*tcosts):
    '''
    Returns distance function for the mismatch penalty between any two KDIGO
     scores in the range [0, len(tcosts) - 1]
    :param tcosts: List of float values corresponding to transitions between
                   consecutive KDIGO scores. E.g.
                    tcosts[i] = cost(i, i + 1)
    :return:
    '''
    cost_dic = {}
    for i in range(len(tcosts)):
        cost_dic[tuple(set((i, i+1)))] = tcosts[i]
    for i in range(len(tcosts)):
        for j in range(i + 2, len(tcosts) + 1):
            cost_dic[tuple(set((i, j)))] = cost_dic[tuple(set((i, j-1)))] + cost_dic[tuple(set((j-1, j)))]

    def penalty(x, y):
        if x == y:
            return 0
        elif x < y:
            return cost_dic[tuple(set((x, y)))]
        else:
            return cost_dic[tuple(set((y, x)))]

    penalty.nvals = len(tcosts) + 1

    return penalty


def continuous_mismatch(discrete):
    nvals = discrete.nvals
    cm = np.zeros((nvals, nvals))
    for i in range(nvals):
        for j in range(nvals):
            cm[i, j] = discrete(i, j)

    x = np.arange(nvals)
    y = np.arange(nvals)

    penalty = interp2d(x, y, cm)

    return penalty


def extension_penalty_func(*tcosts):
    costs = {0: 0.}
    for i in range(len(tcosts)):
        costs[i + 1] = tcosts[i]

    def penalty(x):
        return costs[x]

    penalty.nvals = len(tcosts) + 1

    return penalty


def continuous_extension(discrete):
    nvals = discrete.nvals
    cm = np.zeros(nvals, dtype=float)
    for i in range(1, nvals):
        cm[i] = discrete(i)
    x = np.arange(nvals)
    penalty = interp1d(x, cm)

    return penalty


def pairwise_zeropad_dist(patients, days, ids, dm_fname, dist=braycurtis, t_lim=7):
    df = open(dm_fname, 'w')
    dis = []
    for i in tqdm(range(len(patients)), desc='Zero-Padded Distance Calculation'):
        sel = np.where(days[i] <= t_lim)[0]
        patient1 = np.array(patients[i])[sel]
        dlist = []
        for j in range(i + 1, len(patients)):
            df.write('%d,%d,' % (ids[i], ids[j]))
            sel = np.where(days[j] < t_lim)[0]
            patient2 = np.array(patients[j])[sel]
            if np.all(patient1 == patient2):
                df.write('%f\n' % 0)
                dis.append(0)
                dlist.append(0)
            else:
                if len(patient1) > 1 and len(patient2) > 1:
                    l = max(len(patient1), len(patient2))
                    p1 = np.zeros(l, dtype=int)
                    p2 = np.zeros(l, dtype=int)
                    p1[:len(patient1)] = patient1
                    p2[:len(patient2)] = patient2
                elif len(patient1) == 1:
                    p1 = np.repeat(patient1[0], len(patient2))
                    p2 = patient2
                elif len(patient2) == 1:
                    p1 = patient1
                    p2 = np.repeat(patient2[0], len(patient1))
                if np.all(p1 == p2):
                    df.write('%f\n' % 0)
                    dis.append(0)
                    dlist.append(0)
                else:
                    d = dist(p1, p2)
                    df.write('%f\n' % d)
                    dis.append(d)
                    dlist.append(d)
    df.close()
    return dis


def get_custom_distance_discrete(coordinates, dfunc='braycurtis', min_weight=0.5, lapVal=1.0, lapType='individual'):
    if dfunc == 'braycurtis':
        dist = get_continuous_laplacian_braycurtis(coordinates, lapVal, lapType)
    elif dfunc == 'braycurtis-weighted':
        dist = get_weighted_braycurtis(coordinates, min_weight)
    elif dfunc == 'euclidean':
        dist = get_euclidean_norm(coordinates)
    elif dfunc == 'cityblock':
        dist = get_cityblock_norm(coordinates)
    return dist


def get_custom_braycurtis(coordinates):
    def dist(x, y):
        return braycurtis(coordinates[x], coordinates[y])

    return dist


def get_weighted_braycurtis(coordinates, min_weight=0.5):
    def dist(x, y):
        n = len(x)
        w = np.linspace(min_weight, 1, n)
        num = 0
        den = 0
        for i in range(n):
            num += (w[i] * abs(coordinates[x[i]] - coordinates[y[i]]))
            den += (w[i] * abs(coordinates[x[i]] + coordinates[y[i]]))
        distance = (num / den)
        return distance

    return dist


def get_euclidean_norm(coordinates):
    memo = {}

    def dist(x, y):
        d = euclidean(coordinates[x], coordinates[y])
        try:
            d /= memo[len(x)]
        except KeyError:
            denom = euclidean(np.zeros(len(x)) + max(coordinates), np.zeros(len(x)) + min(coordinates))
            memo[len(x)] = denom
            d /= denom
        return d

    return dist


def get_cityblock_norm(coordinates):
    memo = {}

    def dist(x, y):
        d = cityblock(coordinates[x], coordinates[y])
        try:
            d /= memo[len(x)]
        except KeyError:
            denom = cityblock(np.zeros(len(x)) + max(coordinates), np.zeros(len(x)) + min(coordinates))
            memo[len(x)] = denom
            d /= denom
        return d
    return dist


def get_continuous_laplacian_braycurtis(coordinates, lf=0, lf_type='aggregated'):
    interpolator = interp1d(range(len(coordinates)), coordinates)

    def dist(x, y):
        out = 0
        if hasattr(x, '__len__'):
            # If vector input, accumulate numerator and denominator separately
            num = 0
            denom = 0
            for i in range(len(x)):
                # Use linear interpolation to determine new coordinate given a float value
                # within the range of the original discrete set
                xv = x[i]
                xCoord = interpolator(xv)
                yv = y[i]
                yCoord = interpolator(yv)
                # Bray-Curtis is then computed like normal, with the addition of the Laplacian factor
                # in the denominator
                num += np.abs(xCoord - yCoord)
                if lf_type == 'aggregated':
                    denom += np.abs(xCoord + yCoord + lf)
                else:
                    denom += np.abs(xCoord + yCoord)
            if lf_type == 'individual':
                out = num / (denom + lf)
            else:
                out = num / denom
            return out
        else:
            # Get new coordinates from float values
            xCoord = interpolator(x)
            yCoord = interpolator(y)

            out += (np.abs(xCoord - yCoord) / np.abs(xCoord + yCoord + lf))
        return out

    return dist


def get_custom_cityblock_continuous(coordinates):
    minval = min(coordinates)
    maxval = max(coordinates)

    def dist(x, y):
        newXCoords = np.zeros(len(x))
        newYCoords = np.zeros(len(y))
        for i in range(len(x)):
            xv = x[i]
            start = coordinates[int(np.floor(xv))]
            stop = coordinates[int(np.ceil(xv))]
            xSlope = stop - start
            newXCoords[i] = start + (xv - start) * xSlope
            yv = y[i]
            start = coordinates[int(np.floor(yv))]
            stop = coordinates[int(np.ceil(yv))]
            ySlope = stop - start
            newYCoords[i] = start + (yv - start) * ySlope
        n = cityblock(newXCoords, newYCoords)
        d = cityblock(np.zeros(len(x)) + minval, np.zeros(len(x)) + maxval)
        return n / d

    return dist


def get_alignment(alignmentFile, id1, id2):
    if id1 > id2:
        t = id1
        id1 = id2
        id2 = t
    while True:
        l1 = alignmentFile.readline().rstrip().split(',')
        l2 = alignmentFile.readline().rstrip().split(',')
        if l1[0] == id1 and l2[0] == id2:
            return l1, l2
        _ = alignmentFile.readline()
