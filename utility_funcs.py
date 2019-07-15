import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import sem, t
import itertools


def load_csv(fname, ids=None, dt=float, skip_header=False, idxs=None, targets=None, struct='list', id_dtype=int):
    if struct == 'list':
        res = []
    elif struct == 'dict':
        res = {}
    rid = []
    f = open(fname, 'r')
    hdr = None
    if skip_header is not False or targets is not None:
        hdr = f.readline().rstrip().split(',')[1:]
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
        l = np.array(line.rstrip().split(','))
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
                rid.append(l[0])
    if struct == 'list':
        try:
            if np.all([len(res[x]) == len(res[0]) for x in range(len(res))]):
                res = np.array(res)
                if res.ndim > 1:
                    if res.shape[1] == 1:
                        res = np.squeeze(res)
        except (ValueError, TypeError):
            res = res
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


# %%
def dict2csv(fname, inds, fmt='%f', header=False):
    ids = sorted(list(inds))
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
def arr2csv(fname, inds, ids=None, fmt='%f', header=False):
    outFile = open(fname, 'w')
    if ids is None:
        ids = np.arange(len(inds))
    if header:
        outFile.write(header)
        outFile.write('\n')
    for i in range(len(inds)):
        outFile.write(str(ids[i]))
        if np.size(inds[i]) > 1:
            for j in range(len(inds[i])):
                outFile.write(',' + fmt % (inds[i][j]))
        elif np.size(inds[i]) == 1:
            try:
                outFile.write(',' + fmt % (inds[i]))
            except TypeError:
                outFile.write(',' + fmt % (inds[i][0]))
        else:
            outFile.write(',')
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
    out = np.zeros(array.shape, dtype=np.object)
    for i in range(int(np.prod(array.shape))):
        idx = np.unravel_index(i, array.shape)
        out[idx] = get_date(array[idx], format_str=date_fmt)
    return out


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
