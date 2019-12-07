from cluster_funcs import genSimulationData, randomSimulationSubsets
from matplotlib.backends.backend_pdf import PdfPages
from utility_funcs import arr2csv
import os
import matplotlib.pyplot as plt
import numpy as np
from dtw_distance import extension_penalty_func, continuous_extension, mismatch_penalty_func, continuous_mismatch

numVariants = 15
nSubtypes = 5
nSamplesPerSet = 5
nSets = 10

basePath = "/media/taylor/HDD_1/Google Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/"

tWeightFileName = basePath + "DATA/icu/final/kdigo_icu_2ptAvg_transition_weights.csv"

transition_costs = np.loadtxt(tWeightFileName, delimiter=',', usecols=1, skiprows=1)
coords = np.array([np.sum(transition_costs[:i]) for i in range(len(transition_costs) + 1)], dtype=float)

mismatch = continuous_mismatch(mismatch_penalty_func(*transition_costs))
extension = continuous_extension(extension_penalty_func(*transition_costs))

header = 'SequenceNum,StartKDIGO,LongStart,StopKDIGO,LongStop,BaseKDIGO,NumPeaks,PeakVal'
for nDays in [9, 14]:
    outPath = basePath + "RESULTS/icu/final/clusters/simulated/%ddays" % nDays
    if not os.path.exists(outPath):
        os.mkdir(outPath)

    simulated, labels = genSimulationData(lengthInDays=nDays, numVariants=numVariants)
    arr2csv(os.path.join(outPath, 'sequences.csv'), simulated, ids=None, fmt='%.3f')
    arr2csv(os.path.join(outPath, 'sequences.txt'), simulated, ids=None, fmt='%.3f', delim=" ")
    arr2csv(os.path.join(outPath, 'transition_weights.txt'), transition_costs,
            ids=["0-1", "1->2", "2->3", "3->3D"], fmt='%.3f', delim=" ")
    arr2csv(os.path.join(outPath, 'labels.csv'), labels, ids=None, fmt='%d', header=header)

    selPath = os.path.join(outPath, "randomSets")
    if not os.path.exists(selPath):
        os.mkdir(selPath)

    lbls = np.array(["".join(labels[i].astype(str)) for i in range(len(labels))], dtype=str)

    randomSimulationSubsets(simulated, lbls, selPath, coords, nSubtypes=nSubtypes, nSamples=nSamplesPerSet, nSets=nSets,
                            mismatch=mismatch, extension=extension)

    with PdfPages(os.path.join(outPath, 'sequences_firstVariant.pdf')) as pdf:
        for i in range(0, len(simulated), numVariants):
            center = simulated[i]
            fig = plt.figure()
            plt.plot(center)
            plt.ylim(0, 4)
            plt.yticks(range(5), ['0', '1', '2', '3', '3D'])
            plt.xticks(range(0, len(center), 4), ['%d' % x for x in range(len(center))])
            plt.xlabel('Days')
            plt.ylabel('KDIGO Score')
            plt.title("Simulation %d\nStart: %d - LongStart: %d - Stop: %d - LongStop: %d\nBase: %d - NumPeaks: %d - PeakVal: %d" % tuple([i] + list(labels[i])))
            pdf.savefig(dpi=600)
            plt.close(fig)
