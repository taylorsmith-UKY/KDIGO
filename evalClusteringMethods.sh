#!/usr/bin/env bash

export CONFIG_FILENAME=final-pub.json
export EXT_CONFIG_FILENAME=final-pub.json
export MODEL=log
export ENGINE=sklearn

for ALIGN in "zeropad" "normDTW_a0E+0" "popDTW_a6500E-4"
do
for DFUNC in "braycurtis" "euclidean" "cityblock"
do
for POPCOORDS in "" "-pc"
do
python3 get_flat_clusters.py --config_file $CONFIG_FILENAME --sf kdigo_icu_2ptAvg.csv --df days_interp_icu_2ptAvg.csv -align $ALIGN -n 96 -dfunc $DFUNC $POPCOORDS
python3 evalClusterMethods.py --config_file $CONFIG_FILENAME --sf kdigo_icu_2ptAvg.csv -align $ALIGN -n 96 -dfunc $DFUNC $POPCOORDS
done
done
done

