#!/usr/bin/env bash

for TARGET in "died_inp" "make90_disch_2pt_30dbuf_d25" "make90_disch_2pt_30dbuf_d30" "make90_disch_2pt_30dbuf_d50"
do
for MODEL in "log" "rf" "svm"
do
    python3 classify.py --config_file kdigo_conf1.json -f static_class_features_norm_noGCS -t $TARGET -class $MODEL
done
done

for TARGET in "died_inp" "make90_disch_2pt_30dbuf_d25" "make90_disch_2pt_30dbuf_d30" "make90_disch_2pt_30dbuf_d50"
do
for MODEL in "log" "rf" "svm"
do
    python3 classify.py --config_file kdigo_conf1.json -f static_class_features_norm_max30pctmissing -t $TARGET -class $MODEL
done
done

for TARGET in "died_inp" "make90_disch_2pt_30dbuf_d25" "make90_disch_2pt_30dbuf_d30" "make90_disch_2pt_30dbuf_d50"
do
for MODEL in "log" "rf" "svm"
do
    python3 classify.py --config_file kdigo_conf1.json -f static_class_features_norm_UTSWFeats -t $TARGET -class $MODEL
done
done


#for TARGET in "died_inp" "make90_disch_2pt_30dbuf_d25" "make90_disch_2pt_30dbuf_d30" "make90_disch_2pt_30dbuf_d50"
#do
#for MODEL in "log" "rf" "svm"
#do
#    python3 classify.py -f static_features_norm_imputed -t $TARGET -class $MODEL -sel recursive
#done
#done
