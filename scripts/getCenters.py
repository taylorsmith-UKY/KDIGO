import os
import numpy as np
import json
import h5py
from utility_funcs import load_csv, get_dm_tag
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from DBA import performDBA
from dtw_distance import continuous_mismatch, continuous_extension, get_custom_distance_discrete, \
    mismatch_penalty_func, extension_penalty_func
from cluster_funcs import centerCategorizer

fp = open('../kdigo_conf.json', 'r')
conf = json.load(fp)
fp.close()
basePath = conf['basePath']         # Path containing the DATA and RESULTS directories
cohortName = conf['cohortName']     # Will be used for folder name
t_lim = conf['analysisDays']        # How long to consider in the analysis
tRes = conf['timeResolutionHrs']    # Resolution after imputation in hours
analyze = conf['analyze']           # Time period to analyze (hospital/ICU/all)
plotNew = True

# Build paths and create if don't already exist
baseDataPath = os.path.join(basePath, 'DATA', 'all_sheets')         # folder containing all raw data
dataPath = os.path.join(basePath, 'DATA', analyze, cohortName)
resPath = os.path.join(basePath, 'RESULTS', analyze, cohortName)
f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
ids = f['meta_avg']['ids'][:]
died_inp = f['meta_avg']['died_inp'][:]
died_120 = f['meta_avg']['died_120d_disch'][:]
m90_a30 = f['meta_avg']['make90_admit_2pt_30dbuf_d30'][:]
m90_a50 = f['meta_avg']['make90_admit_2pt_30dbuf_d50'][:]
m90_d30 = f['meta_avg']['make90_disch_2pt_30dbuf_d30'][:]
m90_d50 = f['meta_avg']['make90_disch_2pt_30dbuf_d50'][:]

tcost_fname = "kdigo_icu_2ptAvg_transition_weights.csv"

dm_tag, dtw_tag = get_dm_tag(True, 0.35, False, True, 'braycurtis', 1.0, 'aggregated')

dm_path = os.path.join(resPath, 'dm', '%ddays' % t_lim)
folderName = 'kdigo_icu_2ptAvg_popDTW_a35E-02'

dm_path = os.path.join(dm_path, folderName)
dm = np.load(os.path.join(dm_path, 'kdigo_dm_%s.npy' % dm_tag))

lblPath = os.path.join(resPath, 'clusters', '14days', 'kdigo_icu_2ptAvg_popDTW_a35E-02',
                       'braycurtis_popCoords_aggLap_lap1E+00', 'merged_popDTW_a35E-02')

transition_costs = np.loadtxt(os.path.join(dataPath, tcost_fname), delimiter=',', usecols=1, skiprows=1)

coords = np.array([np.sum(transition_costs[:i]) for i in range(len(transition_costs) + 1)], dtype=float)

extension = continuous_extension(extension_penalty_func(*transition_costs))
mismatch = continuous_mismatch(mismatch_penalty_func(*transition_costs))
dist = get_custom_distance_discrete(coords, dfunc='braycurtis', lapVal=1.0, lapType='aggregated')

lbls = load_csv(os.path.join(lblPath, 'clusters.csv'), ids, str)

kdigos = load_csv(os.path.join(dataPath, 'kdigo_icu_2ptAvg.csv'), ids, int)
days = load_csv(os.path.join(dataPath, 'days_interp_icu_2ptAvg.csv'), ids, int)
for i in range(len(kdigos)):
    kdigos[i] = kdigos[i][np.where(days[i] <= t_lim)[0]]

if os.path.exists(os.path.join(lblPath, 'centers.csv')):
    centers = load_csv(os.path.join(lblPath, 'centers.csv'), np.unique(lbls), struct='dict', id_dtype=str)
    stds = load_csv(os.path.join(lblPath, 'stds.csv'), np.unique(lbls), struct='dict', id_dtype=str)
    confs = load_csv(os.path.join(lblPath, 'confs.csv'), np.unique(lbls), struct='dict', id_dtype=str)
else:
    centers = {}
    stds = {}
    confs = {}

if len(centers) < len(np.unique(lbls)):
    if os.path.exists(os.path.join(lblPath, 'centers.csv')):
        cf = open(os.path.join(lblPath, 'centers.csv'), 'a')
        cof = open(os.path.join(lblPath, 'confs.csv'), 'a')
        sf = open(os.path.join(lblPath, 'stds.csv'), 'a')
    else:
        cf = open(os.path.join(lblPath, 'centers.csv'), 'w')
        cof = open(os.path.join(lblPath, 'confs.csv'), 'w')
        sf = open(os.path.join(lblPath, 'stds.csv'), 'w')

    pdfName = 'centers'
    if os.path.exists(os.path.join(lblPath, pdfName + '.pdf')):
        pdfName += '_a'
        while os.path.exists(os.path.join(lblPath, pdfName + '.pdf')):
            pdfName = pdfName[:-1] + chr(ord(pdfName[-1]) + 1)

    with PdfPages(os.path.join(lblPath, pdfName + '.pdf')) as pdf:
        for lbl in np.unique(lbls):
            if lbl in centers.keys():
                print('Center for ' + lbl + ' already computed.')
                continue
            idx = np.where(lbls == lbl)[0]
            dm_sel = np.ix_(idx, idx)
            tkdigos = [kdigos[x] for x in idx]
            tdm = squareform(squareform(dm)[dm_sel])
            center, std, conf, paths = performDBA(tkdigos, tdm, mismatch=mismatch, extension=extension,
                                                  extraDesc=' for cluster %s (%d patients)' % (lbl, len(idx)),
                                                  alpha=0.35, aggExt=False, targlen=t_lim,
                                                  seedType='mean', n_iterations=5)
            centers[lbl] = center
            stds[lbl] = std
            confs[lbl] = conf
            fig = plt.figure()
            plt.plot(center)
            plt.fill_between(range(len(center)), center - conf, center + conf)
            plt.xticks(range(0, len(center), 4), ['%d' % x for x in range(len(center))])
            plt.yticks(range(5), ['0', '1', '2', '3', '3D'])
            plt.ylim((-0.5, 4.5))
            plt.title(lbl)
            pdf.savefig(dpi=600)
            cf.write('%s,' % lbl + ','.join(['%.3g' % x for x in center]) + '\n')
            cof.write('%s,' % lbl + ','.join(['%.3g' % x for x in conf]) + '\n')
            sf.write('%s,' % lbl + ','.join(['%.3g' % x for x in std]) + '\n')
            plt.close(fig)
    cf.close()
    cof.close()
    sf.close()
elif plotNew:
    cats = centerCategorizer(centers, useTransient=True, stratifiedRecovery=True)
    with PdfPages(os.path.join(lblPath, 'centers_wOutcomes.pdf')) as pdf:
        for lbl in np.unique(lbls):
            idx = np.where(lbls == lbl)[0]
            center = centers[lbl]
            conf = confs[lbl]
            fig = plt.figure()
            plt.plot(center)
            plt.fill_between(range(len(center)), center - conf, center + conf, alpha=0.5)
            plt.xticks(range(0, len(center), 4), ['%d' % x for x in range(len(center))])
            plt.yticks(range(5), ['0', '1', '2', '3', '3D'])
            plt.ylim((-0.5, 4.5))
            plt.title(cats[lbl])
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            plt.legend([extra for _ in range(6)],
                      ['Inpatient Mortality: %d (%.2f)' % (np.sum(died_inp[idx]), np.sum(died_inp[idx]) / len(idx) * 100),
                       'Died Discharge + 120d: %d (%.2f)' % (np.sum(died_120[idx]), np.sum(died_120[idx]) / len(idx) * 100),
                       'M90 Admit 30pct: %d (%.2f)' % (np.sum(m90_a30[idx]), np.sum(m90_a30[idx]) / len(idx) * 100),
                       'M90 Admit 50pct: %d (%.2f)' % (np.sum(m90_a50[idx]), np.sum(m90_a50[idx]) / len(idx) * 100),
                       'M90 Disch 30pct: %d (%.2f)' % (np.sum(m90_d30[idx]), np.sum(m90_d30[idx]) / len(idx) * 100),
                       'M90 Disch 50pct: %d (%.2f)' % (np.sum(m90_d50[idx]), np.sum(m90_d50[idx]) / len(idx) * 100)])
            pdf.savefig(dpi=600)
            plt.close(fig)
    for grp in cats.values():
        if np.any([grp in x for x in cats.values()]):
            with PdfPages(os.path.join(lblPath, 'centers_%s_wOutcomes.pdf' % grp)) as pdf:
                for lbl in np.unique(lbls):
                    if grp in cats[lbl]:
                        idx = np.where(lbls == lbl)[0]
                        center = centers[lbl]
                        conf = confs[lbl]
                        fig = plt.figure()
                        plt.plot(center)
                        plt.fill_between(range(len(center)), center - conf, center + conf, alpha=0.5)
                        plt.xticks(range(0, len(center), 4), ['%d' % x for x in range(len(center))])
                        plt.yticks(range(5), ['0', '1', '2', '3', '3D'])
                        plt.ylim((-0.5, 4.5))
                        plt.title(cats[lbl])
                        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
                        plt.legend([extra for _ in range(6)],
                                   ['Inpatient Mortality: %d (%.2f)' % (
                                   np.sum(died_inp[idx]), np.sum(died_inp[idx]) / len(idx) * 100),
                                    'Died Discharge + 120d: %d (%.2f)' % (
                                    np.sum(died_120[idx]), np.sum(died_120[idx]) / len(idx) * 100),
                                    'M90 Admit 30pct: %d (%.2f)' % (
                                    np.sum(m90_a30[idx]), np.sum(m90_a30[idx]) / len(idx) * 100),
                                    'M90 Admit 50pct: %d (%.2f)' % (
                                    np.sum(m90_a50[idx]), np.sum(m90_a50[idx]) / len(idx) * 100),
                                    'M90 Disch 30pct: %d (%.2f)' % (
                                    np.sum(m90_d30[idx]), np.sum(m90_d30[idx]) / len(idx) * 100),
                                    'M90 Disch 50pct: %d (%.2f)' % (
                                    np.sum(m90_d50[idx]), np.sum(m90_d50[idx]) / len(idx) * 100)])
                        pdf.savefig(dpi=600)
                        plt.close(fig)
