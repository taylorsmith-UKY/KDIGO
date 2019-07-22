#!/usr/bin/env bash
source venv/bin/activate

echo Merging clusters using population DTW
python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -fname merged_popDTW --baseClustNum 96
echo Adding individual laplacian factor
python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -lf 1 -fname merged_popDTW_indLap --baseClustNum 96
echo Adding aggregated laplacian factor
python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -lf 1 -agglap -fname merged_popDTW_aggLap --baseClustNum 96

echo Using normal DTW during alignment
python3 mergeClusters.py -ext -mism -pc -normDTW -dfunc braycurtis -fname merged_normDTW --baseClustNum 96
echo Normal DTW with individual laplacian
python3 mergeClusters.py -ext -mism -pc -normDTW -dfunc braycurtis -lf 1 -fname merged_normDTW_indLap --baseClustNum 96
echo Normal DTW with aggregated laplacian
python3 mergeClusters.py -ext -mism -pc -normDTW -dfunc braycurtis -lf 1 -agglap -fname merged_normDTW_aggLap --baseClustNum 96

echo Back to population DTW, but extension alpha=0.5
python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -fname merged_popDTW_alpha5E-01 --baseClustNum 96 -alpha 0.5
echo ... with individual laplacian
python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -lf 1 -fname merged_popDTW_indLap_alpha5E-01 --baseClustNum 96 -alpha 0.5
echo ... with aggregated laplacian
python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -lf 1 -agglap -fname merged_popDTW_aggLap_alpha5E-01 --baseClustNum 96 -alpha 0.5

echo And now, extension alpha=0.25
python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -fname merged_popDTW_alpha25E-02 --baseClustNum 96 -alpha 0.25
echo ... with individual laplacian
python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -lf 1 -fname merged_popDTW_indLap_alpha25E-02 --baseClustNum 96 -alpha 0.25
echo ... with aggregated laplacian
python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -lf 1 -agglap -fname merged_popDTW_aggLap_alpha25E-02 --baseClustNum 96 -alpha 0.25

echo DONE!!!