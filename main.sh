#!/usr/bin/env bash

python3 preprocess_uky_conf.py
python3 calc_dms.py -ext -mism -pc -dfunc braycurtis
python3 get_flat_clusters.py --n_clust 96
python3 merge_clusters.py -ext -mism -pc -dfunc braycurtis -fname merged --baseClustNum 96