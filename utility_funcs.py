import h5py
import numpy as np


# %%
def get_feats_by_dod(f, n_days=7, features=[], meta_grp='meta'):
    if type(f) == str:
        f = h5py.File(f, 'r')
        close_file = True
    else:
        close_file = False
    all_ids = f[meta_grp]['ids'][:]
    dtd = f[meta_grp]['days_to_death'][:]
    sel = np.logical_not(dtd < n_days)
    feat_sel = []
    for feat_name in features:
        feat_sel.append(f[meta_grp][feat_name][:][sel])
    if close_file:
        f.close()
    return all_ids[sel], feat_sel


# %%
def load_csv(fname, ids, dt=float, skip_header=False, sel=None):
    res = []
    rid = []
    f = open(fname, 'r')
    if skip_header:
        if skip_header == 'keep':
            hdr = f.readline().rstrip().split(',')
        else:
            _ = f.readline()
    for line in f:
        l = line.rstrip()
        if ids is None or int(l.split(',')[0]) in ids:
            if sel is None:
                res.append(np.array(l.split(',')[1:], dtype=dt))
            else:
                res.append(np.array(l.split(',')[sel], dtype=dt))
            rid.append(int(l.split(',')[0]))
    if len(rid) != len(ids):
        print('Missing ids in file: ' + fname)
    else:
        rid = np.array(rid)
        ids = np.array(ids)
        if len(rid) == len(ids):
            if not np.all(rid == ids):
                temp = res
                res = []
                for i in range(len(ids)):
                    idx = ids[i]
                    sel = np.where(rid == idx)[0][0]
                    res.append(temp[sel])
        try:
            if np.all([len(res[x]) == len(res[0]) for x in range(len(res))]):
                res = np.array(res)
                if res.ndim > 1:
                    if res.shape[1] == 1:
                        res = np.squeeze(res)
        except:
            res = res
        if skip_header == 'keep':
            return hdr, res
        else:
            return res


# %%
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    return TP, FP, TN, FN


# %%
def get_even_pos_neg(target):
    '''
    Returns even number of positive/negative examples
    :param target:
    :return:
    '''
    # If even distribution of pos/neg for training
    pos_idx = np.where(target == 1)[0]
    neg_idx = np.where(target == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    if n_pos < n_neg:
        n_train = n_pos
    else:
        n_train = n_neg
    pos_sel = np.random.permutation(pos_idx)[:n_train]
    neg_sel = np.random.permutation(neg_idx)[:n_train]

    sel_idx = np.sort(np.concatenate((pos_sel, neg_sel)))
    return sel_idx
