import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import sem, t
import itertools
import os

def load_csv(fname, ids=None, dt=float, skip_header=False, idxs=None, targets=None, struct='list', id_dtype=int, delim=','):
    if struct == 'list':
        res = []
    elif struct == 'dict':
        res = {}
    rid = []
    f = open(fname, 'r')
    hdr = None
    if skip_header is not False or targets is not None:
        hdr = f.readline().rstrip().split(delim)[1:]
        hdr = np.array(hdr)
    if targets is not None:
        assert hdr is not None
        idxs = []
        try:
            for target in targets:
                tidx = np.where(hdr == target)[0]
                if tidx.size == 0:
                    raise ValueError('%s was not found in file %s' % (target, fname.split('/')[-1]))
                elif tidx.size > 1:
                    raise ValueError('%s is not unique in file %s' % (target, fname.split('/')[-1]))
                idxs.append(tidx)
        except TypeError:
            tidx = np.where(hdr == targets)[0]
            if tidx.size == 0:
                raise ValueError('%s was not found in file %s' % (target, fname.split('/')[-1]))
            elif tidx.size > 1:
                raise ValueError('%s is not unique in file %s' % (target, fname.split('/')[-1]))
            idxs.append(tidx)
    for line in f:
        l = np.array(line.rstrip().split(delim))
        tid = id_dtype(l[0])
        if ids is None or tid in ids:
            if idxs is None:
                if len(l) > 1 and l[1] != '':
                    if type(dt) == str and dt == 'date':
                        d = [get_date(x) for x in l[1:]]
                        if struct == 'list':
                            res.append(d)
                        elif struct == 'dict':
                            res[tid] = d
                    else:
                        if struct == 'list':
                            res.append(np.array(l[1:], dtype=dt))
                        elif struct == 'dict':
                            res[tid] = np.array(l[1:], dtype=dt)
                else:
                    if struct == 'list':
                        res.append(())
                    elif struct == 'dict':
                        res[tid] = []
            else:
                if type(dt) == str and dt == 'date':
                    d = [get_date(l[idx]) for idx in idxs]
                    if struct == 'list':
                        res.append(d)
                    elif struct == 'dict':
                        res[tid] = d
                else:
                    if struct == 'list':
                        res.append(np.array([l[idx] for idx in idxs], dtype=dt))
                    elif struct == 'dict':
                        res[tid] = np.array([l[idx] for idx in idxs], dtype=dt)
            if ids is not None:
                rid.append(type(ids[0])(l[0]))
            else:
                rid.append(id_dtype(l[0]))
    if struct == 'list':
        try:
            if np.all([len(res[x]) == len(res[0]) for x in range(len(res))]):
                res = np.array(res)
                if res.ndim > 1:
                    if res.shape[1] == 1:
                        res = np.squeeze(res)
        except (ValueError, TypeError):
            res = res
    f.close()
    if ids is not None:
        if len(rid) != len(ids):
            print('Missing ids in file: ' + fname)
        if skip_header == 'keep':
            return res, hdr
        else:
            return res
    else:
        rid = np.array(rid)
        if skip_header == 'keep':
            return hdr, res, rid
        else:
            return res, rid


def savePairwiseAlignment(f, ids, data, id_fmt='%d'):
    ct = 0
    for i in range(len(ids)):
        for j in range(i+1, ids):
            f.write(id_fmt % ids[i])
            for k in range(len(data[ct][0])):
                f.write(',%d' % data[ct][0][k])
            f.write('\n')
            f.write(id_fmt % ids[j])
            for k in range(len(data[ct][1])):
                f.write(',%d' % data[ct][1][k])
            f.write('\n')
            f.write('\n')
    return


# %%
def dict2csv(fname, inds, fmt='%f', header=False, append=False):
    ids = sorted(list(inds))
    if append:
        outFile = open(fname, 'a+')
        outFile.write("\n")
    else:
        outFile = open(fname, 'w')
    if header:
        outFile.write(header)
        outFile.write('\n')
    for i in range(len(inds)):
        tid = ids[i]
        outFile.write(str(tid))
        if np.size(inds[tid]) > 1:
            for j in range(len(inds[tid])):
                outFile.write(',' + fmt % (inds[tid][j]))
        elif np.size(inds[i]) == 1:
            try:
                outFile.write(',' + fmt % (inds[tid]))
            except TypeError:
                outFile.write(',' + fmt % (inds[tid][0]))
        else:
            outFile.write(',')
        outFile.write('\n')
    outFile.close()


# %%
def arr2csv(fname, inds, ids=None, fmt='%f', header=False, delim=','):
    outFile = open(fname, 'w')
    if ids is None:
        ids = np.arange(len(inds))
    if header:
        outFile.write(header)
        outFile.write('\n')
    for i in range(len(inds)):
        outFile.write(str(ids[i]))
        if hasattr(inds[i], "__len__") and type(inds[i]) != np.str_ and type(inds[i]) != str:
            for j in range(len(inds[i])):
                outFile.write(delim + fmt % (inds[i][j]))
        elif np.size(inds[i]) == 1:
            outFile.write(delim + fmt % (inds[i]))
        else:
            outFile.write(delim)
        outFile.write('\n')
    outFile.close()


def get_date(date_str, format_str='%Y-%m-%d %H:%M:%S'):
    if type(date_str) == np.ndarray:
        date_str = date_str[0]
    if type(date_str) == datetime.datetime:
        return date_str
    elif type(date_str) == float:
        return 'nan'
    try:
        date_str = date_str.decode("utf-8").split('.')[0]
    except AttributeError:
        date_str = date_str.split('.')[0]
    try:
        date = datetime.datetime.strptime(date_str, format_str)
    except ValueError:
        format_str = '%m/%d/%y %H:%M'
        try:
            date = datetime.datetime.strptime(date_str, format_str)
        except ValueError:
            format_str = '%m/%d/%y'
            try:
                date = datetime.datetime.strptime(date_str, format_str)
            except ValueError:
                format_str = '%m/%d/%Y'
                try:
                    date = datetime.datetime.strptime(date_str, format_str)
                except ValueError:
                    return 'nan'
    return date


def get_array_dates(array, date_fmt='%Y-%m-%d %H:%M:%S'):
    if type(array) == np.ndarray:
        out = np.zeros(array.shape, dtype=np.object)
        for i in range(int(np.prod(array.shape))):
            idx = np.unravel_index(i, array.shape)
            out[idx] = get_date(array[idx], format_str=date_fmt)
    else:
        out = []
        for i in range(len(array)):
            temp = []
            for j in range(len(array[i])):
                temp.append(get_date(array[i][j], format_str=date_fmt))
            out.append(np.array(temp, dtype=np.object))
    return out


def get_tdiff(t1, t2, unit="s"):
    tdiff = (t2 - t1).total_seconds()
    if unit == "s" or unit == "sec":
        return tdiff
    elif unit == "m" or unit == "min":
        return tdiff / 60
    elif unit == "h" or unit == "hr" or unit == "hour" or unit == "hours":
        return tdiff / (60 * 60)
    elif unit == "d" or unit == "day" or unit == "days":
        return tdiff / (60 * 60 * 24)
    elif unit == "y" or unit == "yr" or unit == "yrs" or unit == "year" or unit == "years":
        return tdiff / (60 * 60 * 24 * 365)


def perf_measure(y_actual, y_hat):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            tp += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            fp += 1
        if y_actual[i] == y_hat[i] == 0:
            tn += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            fn += 1
    prec = precision_score(y_actual, y_hat)
    rec = recall_score(y_actual, y_hat)
    f1 = f1_score(y_actual, y_hat)
    sens = float(tp) / np.sum(y_hat)
    return np.array((prec, rec, f1, tp, fp, tn, fn))


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype
    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)
    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def mean_confidence_interval(data, confidence=0.95):
    '''
    Returns the mean confidence interval for the distribution of data
    :param data:
    :param confidence: decimal percentile, i.e. 0.95 = 95% confidence interval
    :return:
    '''
    a = 1.0 * np.array(data)
    n_ex, n_pts = a.shape
    means = np.zeros(n_pts)
    diff = np.zeros(n_pts)
    for i in range(n_pts):
        x = data[:, i]
        x = x[np.logical_not(np.isnan(x))]
        n = len(x)
        m, se = np.mean(x), sem(x)
        h = se * t.ppf((1 + confidence) / 2., n-1)
        means[i] = m
        diff[i] = h
    return means, means-diff, means+diff


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True, label_names=None, show=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)
    sens = tp / (tp + fn)

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if show:
        plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=16)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=16)


    plt.tight_layout()
    if label_names is None:
        plt.ylabel('True label', fontsize=18)
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}\nsensitivity={:0.4f}; specificity={:0.4f}'.format(accuracy, misclass, sens, spec), fontsize=18)
    else:
        plt.ylabel(label_names[0], fontsize=18)
        plt.xlabel(
            label_names[1] + '\naccuracy={:0.4f}; misclass={:0.4f}\nsensitivity={:0.4f}; specificity={:0.4f}'.format(
                accuracy, misclass, sens, spec), fontsize=18)
    if show:
        plt.show()


def get_dm_tag(popDTW, alpha, aggext, popcoords, dfunc, laplacian=1, lf_type='none'):
    if popDTW:
        dtw_tag = 'popDTW'
        if alpha >= 1:
            dtw_tag += "_a%.0E" % alpha
        elif ((alpha * 100) % 10) == 0:
            dtw_tag += "_a%dE-01" % (alpha * 10)
        else:
            dtw_tag += "_a%dE-02" % (alpha * 100)
    else:
        dtw_tag = 'normDTW'
    if aggext:
        dtw_tag += '_aggExt'
    dm_tag = dfunc
    if popcoords:
        dm_tag += '_popCoords'
    else:
        dm_tag += '_absCoords'
    if lf_type != 'none':
        if lf_type == 'individual':
            dm_tag += '_indLap'
        elif lf_type == 'aggregated':
            dm_tag += '_aggLap'
        if laplacian >= 1:
            dm_tag += "_lap%.0E" % laplacian
        elif ((laplacian * 100) % 10) == 0:
            dm_tag += "_lap%dE-01" % (laplacian * 10)
        else:
            dm_tag += "_lap%dE-02" % (laplacian * 100)
    return dm_tag, dtw_tag


def get_alignment(alignmentFile, id1, id2):
    alignmentFile.seek(0)
    if int(id1) > int(id2):
        t = id1
        id1 = id2
        id2 = t
    while True:
        l1 = alignmentFile.readline().rstrip().split(',')
        l2 = alignmentFile.readline().rstrip().split(',')
        if l1[0] == id1 and l2[0] == id2:
            return l1, l2
        _ = alignmentFile.readline()


def visualize_alignments(id_pairs, ids, kdigos, dtwPath):
    ids = ids.astype(int).astype(str)
    dfs = []
    names = []
    # for dirpath, dirnames, fnames in os.walk(dtwPath):
    for dirname in ['popMismatch_extension_a1E+00_normBC',
                    'popMismatch_extension_a9E-01_normBC',
                    'popMismatch_extension_a8E-01_normBC',
                    'popMismatch_extension_a7E-01_normBC',
                    'popMismatch_extension_a6E-01_normBC',
                    'popMismatch_extension_a5E-01_normBC',
                    'popMismatch_extension_a45E-02_normBC',
                    'popMismatch_extension_a4E-01_normBC',
                    'popMismatch_extension_a35E-02_normBC',
                    'popMismatch_extension_a3E-01_normBC',
                    'popMismatch_extension_a25E-02_normBC',
                    'popMismatch_extension_a2E-01_normBC',
                    'popMismatch_extension_a15E-02_normBC',
                    'popMismatch_extension_a1E-01_normBC',
                    'popMismatch_extension_a5E-02_normBC']:
        dfs.append(open(os.path.join(dtwPath, dirname, 'dtw_alignment_proc0.csv'), 'r'))
        names.append(dirname)
    for (id1, id2) in id_pairs:
        of = open(os.path.join(dtwPath, str(id1) + 'vs' + str(id2) + ".csv"), 'w')
        of.write('Original KDIGO Scores:\n')
        of.write(str(id1) + ',' + ','.join(kdigos[np.where(ids == id1)[0][0]].astype(str)) + '\n')
        of.write(str(id2) + ',' + ','.join(kdigos[np.where(ids == id2)[0][0]].astype(str)) + '\n\n')
        for df, name in zip(dfs, names):

            of.write(name + ':' + '\n')
            l1, l2 = get_alignment(df, str(id1), str(id2))
            of.write(','.join(l1) + '\n')
            of.write(','.join(l2) + '\n')
        of.close()
    return

# %%
def get_timed_extremes(values, times, start_times, maxDay=3):
    minimums = np.zeros(len(values))
    maximums = np.zeros(len(values))
    starts = np.zeros(len(values))
    stops = np.zeros(len(values))

    if start_times is not None and type(start_times[0]) != datetime.datetime:
        start_times = get_array_dates(start_times)
    for i in range(len(values)):
        if start_times is not None:
            tstart = start_times[i]
            tend = tstart + datetime.datetime(maxDay)
            vals_in_window = []
            for j in range(len(values[i])):
                tdate = get_date(times[i][j])
                if tstart <= tdate <= tend:
                    vals_in_window.append(values[i][j])
        else:
            idx = np.where(times[i] <= maxDay)[0]
            vals_in_window = values[i][idx]
        if len(vals_in_window) > 0:
            starts[i] = vals_in_window[0]
            stops[i] = vals_in_window[-1]
            maximums[i] = np.max(vals_in_window)
            minimums[i] = np.min(vals_in_window)
        else:
            starts[i] = np.nan
            stops[i] = np.nan
            maximums[i] = np.nan
            minimums[i] = np.nan

    return minimums, maximums, starts, stops