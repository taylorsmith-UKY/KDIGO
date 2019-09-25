#!/usr/bin/env bash

#python3 get_flat_clusters.py -dfunc braycurtis -lt individual
#python3 get_flat_clusters.py -dfunc braycurtis -lt aggregated
for i in {36..96}
do
#python3 get_flat_clusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -pc -lt individual -lv 1.0 -n ${i} -meta meta_avg
python3 get_flat_clusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -pc -n ${i} -meta meta_avg
done
