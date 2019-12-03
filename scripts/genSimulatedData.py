from cluster_funcs import genSimulationData
from matplotlib.backends.backend_pdf import PdfPages
from utility_funcs import arr2csv
import os
import matplotlib.pyplot as plt

numVariants = 5

header = 'SequenceNum,StartKDIGO,LongStart,StopKDIGO,LongStop,BaseKDIGO,NumPeaks,PeakVal'
for nDays in [9, 14]:
    outPath = "/Volumes/GoogleDrive/My Drive/Documents/Work/Workspace/Kidney Pathology/" \
              "KDIGO_eGFR_traj/RESULTS/icu/final/clusters/simulated/%ddays" % nDays
    simulated, labels = genSimulationData(lengthInDays=nDays, numVariants=numVariants)
    arr2csv(os.path.join(outPath, 'sequences.csv'), simulated, ids=None, fmt='%.3f')
    arr2csv(os.path.join(outPath, 'labels.csv'), labels, ids=None, fmt='%d', header=header)

    # arr2csv(os.path.join(outPath, 'sequences_noNoise.csv'), noNoise, ids=None, fmt='%.3f')
    # arr2csv(os.path.join(outPath, 'labels_noNoise.csv'), noNoiseLabels, ids=None, fmt='%d', header=header)

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
            plt.title("Start: %d - LongStart: %d - Stop: %d - LongStop: %d\nBase: %d - NumPeaks: %d - PeakVal: %d" % tuple(labels[i]))
            pdf.savefig(dpi=600)
            plt.close(fig)
