#!/usr/bin/env bash

for TARGET in "make90_admit_2pt_30dbuf_d25" "make90_admit_2pt_30dbuf_d30" "make90_admit_2pt_30dbuf_d50" \
         "make90_disch_2pt_30dbuf_d25" "make90_disch_2pt_30dbuf_d30" "make90_disch_2pt_30dbuf_d50" "died_inp"
do
    python3 classify.py -f static_features_norm_meanfill -t $TARGET -class log -sel recursive
done