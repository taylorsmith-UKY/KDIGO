#!/usr/bin/env bash
source venv/bin/activate
export CATEGORY=1-Im

echo Merging clusters using population DTW
#python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -fname merged_popDTW --baseClustNum 96 -cat $CATEGORY
#python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -lf 1 -fname merged_popDTW_indLap --baseClustNum 96 -cat $CATEGORY
#python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -lf 1 -agglap -fname merged_popDTW_aggLap --baseClustNum 96 -cat $CATEGORY

echo Merging clusters using population DTW with aggregated extension penalty
#python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -fname merged_popDTW_aggExt --baseClustNum 96 -cat $CATEGORY -agg
#python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -lf 1 -fname merged_popDTW_indLap_aggExt --baseClustNum 96 -cat $CATEGORY -agg
python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -lf 1 -agglap -fname merged_popDTW_aggLap_aggExt --baseClustNum 96 -cat $CATEGORY -agg

echo Merging clusters using population DTW with alpha=0.5
#python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -fname merged_popDTW_alpha5E-01 --baseClustNum 96 -cat $CATEGORY -alpha 0.5
python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -lf 1 -fname merged_popDTW_indLap_alpha5E-01 --baseClustNum 96 -cat $CATEGORY -alpha 0.5
python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -lf 1 -agglap -fname merged_popDTW_aggLap_alpha5E-01 --baseClustNum 96 -cat $CATEGORY -alpha 0.5

echo Merging clusters using population DTW with aggregated extension penalty and alpha=0.5
python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -fname merged_popDTW_aggExt_alpha5E-01 --baseClustNum 96 -cat $CATEGORY -alpha 0.5 -agg
python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -lf 1 -fname merged_popDTW_indLap_aggExt_alpha5E-01 --baseClustNum 96 -cat $CATEGORY -alpha 0.5 -agg
python3 mergeClusters.py -ext -mism -pc -dfunc braycurtis -lf 1 -agglap -fname merged_popDTW_aggLap_aggExt_alpha5E-01 --baseClustNum 96 -cat $CATEGORY -alpha 0.5 -agg

echo Using normal DTW during alignment
python3 mergeClusters.py -ext -mism -pc -normDTW -dfunc braycurtis -fname merged_normDTW --baseClustNum 96 -cat $CATEGORY
python3 mergeClusters.py -ext -mism -pc -normDTW -dfunc braycurtis -lf 1 -fname merged_normDTW_indLap --baseClustNum 96 -cat $CATEGORY
python3 mergeClusters.py -ext -mism -pc -normDTW -dfunc braycurtis -lf 1 -agglap -fname merged_normDTW_aggLap --baseClustNum 96 -cat $CATEGORY


echo DONE!!!