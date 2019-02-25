import numpy as np
import h5py
import os
from scipy.spatial.distance import squareform
from cluster_funcs import evaluateDmClusters
from kdigo_funcs import load_csv
import warnings
from string import ascii_lowercase

basePath = "/Volumes/GoogleDrive/My Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/"

dataPath = os.path.join(basePath, 'DATA', 'icu', '7days_021319/')
resPath = os.path.join(basePath, 'RESULTS', 'icu', '7days_021319/')

dm_tags = [#  'zeropad_normBC',
           # 'zeropad_classicEuclidean',
           # 'zeropad_classicCityblock',
           # 'absmismatch_normBC',
           # 'absmismatch_normBC_popcoord',
           # 'absmismatch_normBC_popcoord_shift25E-02',
           # 'absmismatch_normBC_popcoord_shift5E-01',
           # 'absmismatch_normBC_popcoord_shift1',
           # 'custmismatch_extension_a1E+00_normBC',
           # 'custmismatch_extension_a1E+00_normBC_popcoord',
           # 'custmismatch_extension_a1E+00_normBC_popcoord_shift25E-02',
           # 'custmismatch_extension_a1E+00_normBC_popcoord_shift5E-01',
           # 'custmismatch_extension_a1E+00_normBC_popcoord_shift1',
           # 'custmismatch_extension_a1E+00_euclidean',
           # 'custmismatch_extension_a1E+00_euclidean_popcoord',
           # 'custmismatch_extension_a1E+00_cityblock',
           # 'custmismatch_extension_a1E+00_cityblock_popcoord',]
            'absmismatch_euclidean',
            'absmismatch_euclidean_popcoord',
            'absmismatch_cityblock',
            'absmismatch_cityblock_popcoord']

dm_tags_new = [#  'zeropad_classicBC',
               # 'zeropad_classicEuclidean',
               # 'zeropad_classicCityblock',
               # 'classicDTW_classicBC',
               # 'classicDTW_populationBC(0)',
               # 'classicDTW_populationBC(25E-02)',
               # 'classicDTW_populationBC(5E-01)',
               # 'classicDTW_populationBC(1E+00)',
               # 'populationDTW_classicBC',
               # 'populationDTW_populationBC(0)',
               # 'populationDTW_populationBC(25E-02)',
               # 'populationDTW_populationBC(5E-01)',
               # 'populationDTW_populationBC(1E+00)',
               # 'populationDTW_classicEuclidean',
               # 'populationDTW_populationEuclidean(0)',
               # 'populationDTW_classicCityblock',
               # 'populationDTW_populationCityblock(0)', ]
                'classicDTW_classicEuclidean' ,
                'classicDTW_populationEuclidean',
                'classicDTW_classicCityblock' ,
                'classicDTW_populationCityblock' ]

f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
all_ids = f['meta']['ids'][:]
all_mk_7d = f['meta']['max_kdigo_7d'][:]
all_mk_w = f['meta']['max_kdigo'][:]
all_died = f['meta']['died_inp'][:]

warnings.filterwarnings("ignore")
slist = []
grplist = []
mortl = []
progl = []
sill = []
gt7dl = []
mkl = []
for dm_tag, save_name in zip(dm_tags, dm_tags_new):
    # lblPath = os.path.join(resPath, 'clusters', '7days', dm_tag, 'split_by_kdigo', 'kdigo_2plus', 'flat',
    #                        '1_dbscan_clusters', 'cluster_1', '18_clusters')
    # lblBasePath = os.path.join(resPath, 'clusters', '7days', dm_tag, 'combined')
    # for (dirpath, dirnames, fnames) in os.walk(lblBasePath):
    #     for dirname in dirnames:
    # if 'cluster' not in dirname or dirname == '1_clusters':
    #     continue
    # lblPath = os.path.join(dirpath, dirname)
    lblPath = os.path.join(resPath, 'clusters', '7days', dm_tag, 'split_by_kdigo', 'kdigo_1', 'flat',
                           '5_clusters')
    # lblPath = os.path.join(resPath, 'clusters', '7days', dm_tag, 'combined', '18_clusters')
    ids = np.loadtxt(os.path.join(lblPath, 'clusters.csv'), delimiter=',', usecols=0, dtype=int)
    lbls = np.loadtxt(os.path.join(lblPath, 'clusters.csv'), delimiter=',', usecols=1, dtype=str)
    pt_sel = np.array([x in ids for x in all_ids])
    dm_sel = np.ix_(pt_sel, pt_sel)
    lblNames = np.unique(lbls)
    tot_num = len(lblNames)

    kdigos = load_csv(os.path.join(dataPath, 'kdigo.csv'), ids, int)
    days = load_csv(os.path.join(dataPath, 'days_interp.csv'), ids, int)

    mk7d = all_mk_7d[pt_sel]
    mkw = all_mk_w[pt_sel]
    died = all_died[pt_sel]
    try:
        dm = np.load(os.path.join(resPath, 'dm', '7days', 'kdigo_dm_%s.npy' % dm_tag))[:, 2]
    except IndexError:
        dm = np.load(os.path.join(resPath, 'dm', '7days', 'kdigo_dm_%s.npy' % dm_tag))
    sqdm = squareform(dm)[dm_sel]
    dm = squareform(sqdm)
    eval_str, grp_str, n_gt7d, all_morts, all_progs, all_sils, all_mk = evaluateDmClusters(lbls, mk7d, mkw, died, days, sqdm)
    s = '%s_hierarchical(%d),%d,' % (save_name, tot_num, tot_num)
    for lbl in lblNames:
        s += ',%.2f' % (n_gt7d[lbl][0])
    gt7dl.append(s)
    s = '%s_hierarchical(%d),%d,' % (save_name, tot_num, tot_num)
    for lbl in lblNames:
        s += ',%.2f' % (all_morts[lbl])
    mortl.append(s)
    s = '%s_hierarchical(%d),%d,' % (save_name, tot_num, tot_num)
    for lbl in lblNames:
        s += ',%.2f' % (all_progs[lbl])
    progl.append(s)
    s = '%s_hierarchical(%d),%d,' % (save_name, tot_num, tot_num)
    for lbl in lblNames:
        s += ',%.2f' % (all_sils[lbl])
    sill.append(s)
    s = '%s_hierarchical(%d),%d,' % (save_name, tot_num, tot_num)
    for lbl in lblNames:
        s += ',%d' % (all_mk[lbl])
    mkl.append(s)
    slist.append('%s_hierarchical(%d),%d,%s' % (save_name, tot_num, tot_num, eval_str))
    grplist.append('%s_hierarchical(%d),%d,%s' % (save_name, tot_num, tot_num, grp_str))
    for (mdirpath, mdirnames, mfilenames) in os.walk(os.path.join(lblPath, 'merged')):
        for dirname in mdirnames:
            if '_clusters' in dirname:
                n_clusters = int(dirname.split('_')[0])
                tlblPath = os.path.join(mdirpath, dirname)
                lbls = np.loadtxt(os.path.join(tlblPath, 'clusters.csv'), delimiter=',', usecols=1, dtype=str)
                eval_str, grp_str, n_gt7d, all_morts, all_progs, all_sils, all_mk = evaluateDmClusters(lbls, mk7d, mkw, died, days, sqdm)
                s = ''
                ms = ''
                ps = ''
                ss = ''
                mks = ''
                keys = np.array(list(n_gt7d))
                for key in keys:
                    s += ',%d' % (n_gt7d[key][0])
                    ms += ',%.2f' % (all_morts[key])
                    ps += ',%.2f' % (all_progs[key])
                    ss += ',%.2f' % (all_sils[key])
                    mks += ',%.2f' % (all_mk[key])
                gt7dl.append(s)
                mortl.append(ms)
                progl.append(ps)
                sill.append(ss)
                mkl.append(mks)
                slist.append('%s_mergeHierarchical(%d-%d),%d,%s' % (save_name, tot_num, tot_num - n_clusters, n_clusters, eval_str))
                grplist.append('%s_mergeHierarchical(%d-%d),%d,%s' % (save_name, tot_num, tot_num - n_clusters, n_clusters, grp_str))

# f = open(os.path.join(resPath, 'cluster_eval_kdigo_specific.csv'), 'w')
s = ''
for tstr in slist:
    s += tstr + '\n'
print(s)
# f.write(s)
# f.close()
# f = open(os.path.join(resPath, 'cluster_eval_overall.csv'), 'w')
s = ''
for tstr in grplist:
    # s += tstr + '\n'
    print(tstr)
# f.write(s)
# f.close()
# #
# f = open(os.path.join(resPath, 'cluster_eval_gt7dct.csv'), 'w')
# ts = ''
# for s in gt7dl:
#     ts += s + '\n'
# print(ts)
# # f.write(ts)
# # f.close()
# # f = open(os.path.join(resPath, 'cluster_eval_mort.csv'), 'w')
# ts = ''
# for s in mortl:
#     ts += s + '\n'
# print(ts)
# # f.write(ts)
# # f.close()
# # f = open(os.path.join(resPath, 'cluster_eval_7dprog.csv'), 'w')
# ts = ''
# for s in progl:
#     ts += s + '\n'
# print(ts)
# # f.write(ts)
# # f.close()
# # f = open(os.path.join(resPath, 'cluster_eval_sil.csv'), 'w')
# ts = ''
# for s in sill:
#     ts += s + '\n'
# print(ts)
# # f.write(ts)
# # f.close()
# # f = open(os.path.join(resPath, 'cluster_eval_maxkdigo.csv'), 'w')
# ts = ''
# for s in mkl:
#     ts += s + '\n'
# print(ts)
# # f.write(ts)
# f.close()
