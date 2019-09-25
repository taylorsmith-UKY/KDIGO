#!/usr/bin/env bash
cd /home/taylor/PycharmProjects/KDIGO
source ~/.virtualenvs/KDIGO/bin/activate
export CATEGORY=1

# test different seeds and merge types
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean
# python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed medoid -dbaiter 5 -mtype mean
# python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed medoid -dbaiter 5 -mtype dba
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype dba

# Different weights for incorporating the extension penalty into the distance
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 0.35
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 0.35 -scaleExt
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype dba -extDistWeight 0.35
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype dba -extDistWeight 0.35 -scaleExt
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -maxExt 5
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -maxExt 10
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -maxExt 20
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -maxExt 30
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -maxExt 50


python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 0.5
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 1.0

# Same as previous, but extension is scaled by length first (this is the most directly comparable to how the extension
# penalty affects the DTW alignment.

python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 0.5 -scaleExt
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 1.0 -scaleExt

# Explicitly disallow merges that involve an extension with a penalty above the threshold. Tested when extension is not incorporated into total distance, and vice a versa.
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -maxExt 66
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 0.35 -maxExt 66

# Different weights for incorporating the extension penalty into the distance, but use DBA to compute new centers
# instead of the arithmetic mean.

python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype dba -extDistWeight 0.5
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype dba -extDistWeight 1.0

# Use DBA to generate centers after merging, and scale the extension penalties by length before incorporating into
# distance.
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype dba -extDistWeight 0.35 -scaleExt
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype dba -extDistWeight 0.5 -scaleExt
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype dba -extDistWeight 1.0 -scaleExt

# While using DBA to generate centers after merging, explicitly disallow merges that involve an extension with a penalty above the threshold.
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype dba -maxExt 66
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype dba -extDistWeight 0.35 -maxExt 66

# # Use the arithmetic mean for the DBA seed and also the mean after DTW alignment to compute new centers, with different
# # weights to incorporate penalty into distance. Then a few more variations.
# python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 0.35
# python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 0.5
# python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 1.0
# python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 0.35 -scaleExt
# python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 0.5 -scaleExt
# python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 1.0 -scaleExt
# python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -maxExt 66
# python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 0.35 -maxExt 66

# python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype dba -extDistWeight 0.35
# python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype dba -extDistWeight 0.5
# python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype dba -extDistWeight 1.0
# python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype dba -extDistWeight 0.35 -scaleExt
# python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype dba -extDistWeight 0.5 -scaleExt
# python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype dba -extDistWeight 1.0 -scaleExt
# python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype dba -maxExt 66
# python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype dba -extDistWeight 0.35 -maxExt 66
