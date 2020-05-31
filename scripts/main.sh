#!/usr/bin/env bash

export CONFIG_FILENAME=superConf.json

python3 preprocess_uky_conf.py --config_file $CONFIG_FILENAME -agvPts 2
python3 cleanStats.py --config_file $CONFIG_FILENAME -agvPts 2
python3 getStaticFeatures.py --config_file $CONFIG_FILENAME -agvPts 2 -ps

for TARGET in "died_inp" "make90_d50"
do
for MODEL in "log" "rf" "svm"
do
for FEATURE in "sofa_apache_norm" "max_kdigo_d03_norm" "base_model" "base_model_withTrajectory"
do
python3 classify.py --config_file $CONFIG_FILENAME -agvPts 2 -f $FEATURE -t $TARGET -class $MODEL
done
done
done
