import numpy as np
import json
from dtw_distance import continuous_mismatch, continuous_extension, get_custom_distance_discrete, \
    mismatch_penalty_func, extension_penalty_func
from utility_funcs import load_csv, arr2csv
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cluster_funcs import alphaParamSearch

fp = open("kdigo_conf.json", 'r')
conf = json.load(fp)
fp.close()
basePath = conf['basePath']
cohortName = conf['cohortName']
t_lim = conf['analysisDays']
tRes = conf['timeResolutionHrs']
v = conf['verbose']
analyze = conf['analyze']
baseDataPath = os.path.join(basePath, 'DATA', 'all_sheets')
dataPath = os.path.join(basePath, 'DATA', analyze, cohortName)
resPath = os.path.join(basePath, 'RESULTS', analyze, cohortName)

thresh = 0.001
nIter = 10
useCpp = True
iterSelectionList = ["grid", "random"]
v = False

tcost_fname = os.path.join(dataPath, "kdigo_icu_2ptAvg_transition_weights.csv")
transition_costs = np.loadtxt(tcost_fname, delimiter=',', usecols=1, skiprows=1)
coords = np.array([np.sum(transition_costs[:i]) for i in range(len(transition_costs) + 1)], dtype=float)
extension = continuous_extension(extension_penalty_func(*transition_costs))
mismatch = continuous_mismatch(mismatch_penalty_func(*transition_costs))
dist = get_custom_distance_discrete(coords, lapVal=0.0, lapType="none")

baseClustPath = os.path.join(resPath, "clusters", "simulated", "9days", "selectedSets", "grouped")
cats = ["%s-%s" % (x, y) for x in ["1", "2", "3", "3D"] for y in ["Im", "St", "Ws"]]

for iterSelection in iterSelectionList:
    with PdfPages(os.path.join(baseClustPath, "alpha_optimization_%s.pdf" % iterSelection)) as pdf:
        for cat in cats:
            tpath = os.path.join(baseClustPath, cat)
            outPath = os.path.join(tpath, iterSelection)
            if not os.path.exists(outPath):
                os.mkdir(outPath)

            sequences, ids = load_csv(os.path.join(tpath, "sequences.csv"), None)
            labels = load_csv(os.path.join(tpath, "labels.csv"), ids, str)

            allOpts, dms, clusters, evals = alphaParamSearch(sequences, labels, tpath, mismatch, extension, dist,
                                                             tcost_fname, thresh=thresh, nIter=nIter, useCpp=useCpp,
                                                             iterSelection=iterSelection, v=v)

            paramVals = np.array(list(evals.keys()))
            evalVals = np.array([evals[k] for k in paramVals])
            paramOrder = np.argsort(paramVals)

            bestVal = np.max(evalVals)
            idx = np.where(evalVals == bestVal)[0][0]

            fig = plt.figure()
            plt.plot(paramVals[paramOrder], evalVals[paramOrder])
            plt.axvline(paramVals[idx])
            plt.ylim([0, 1])
            plt.xlabel("Value of Alpha")
            plt.ylabel("Normalized Mutual Information")
            plt.title("Optimization of Extension Penalty Weight\n%s" % cat)
            plt.savefig(os.path.join(outPath, "eval_plot.png"), dpi=600)
            pdf.savefig(figure=fig, dpi=600)
            plt.close(fig)

            arr2csv(os.path.join(outPath, "all_evals.csv"), evalVals[paramOrder], paramVals[paramOrder], header="Alpha,NMI")
            np.savetxt(os.path.join(outPath, "all_optimals.csv"), np.array(allOpts), delimiter=",", header="Alpha,NMI")

            for k in clusters.keys():
                if k < 1:
                    cPath = os.path.join(outPath, "sequences_popDTW_a%dE-04" % (k * 10000))
                else:
                    cPath = os.path.join(outPath, "sequences_popDTW_a1E+00")

                arr2csv(os.path.join(cPath, "clusters.csv"), clusters[k], ids, fmt="%d")
