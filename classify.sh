#!/usr/bin/env bash

for MODEL in "rf" "log" "svm"
do
for TARGET in "died_inp" "make90_disch_2pt_30dbuf_d30" "make90_disch_2pt_30dbuf_d50"  # "make90_admit_2pt_30dbuf_d30" "make90_admit_2pt_30dbuf_d50"
do
for FEATURE in "sofa_apache" "max_kdigo_d03" "static_features_norm_imputed_maxKDIGO_D03" "static_features_norm_imputed_maxKDIGO_14d" "static_features_norm_imputed_maxKDIGO_14d_wTraj"
do
    python3 classify.py --config_file kdigo_conf.json -df stats.h5 -f $FEATURE -t $TARGET -class $MODEL -sel recursive
done
done
done
