#!/usr/bin/env bash

python3 preprocess_uky_conf.py
python3 calc_dm.py -ext -mism -pc -dfunc braycurtis
python3 calc_dm.py -ext -mism -pc -dfunc braycurtis -agg
python3 calc_dm.py -ext -mism -pc -dfunc braycurtis -lt individual
python3 calc_dm.py -ext -mism -pc -dfunc braycurtis -lt aggregated
python3 calc_dm.py -ext -mism -pc -dfunc braycurtis -alpha 0.5
python3 calc_dm.py -ext -mism -pc -dfunc braycurtis -lt individual -alpha 0.5
python3 calc_dm.py -ext -mism -pc -dfunc braycurtis -lt aggregated -alpha 0.5

python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -n 16
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -n 32
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -n 48
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -n 96
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -n -agg 16
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -n -agg 32
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -n -agg 48
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -n -agg 96
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt individual -n 16
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt individual -n 32
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt individual -n 48
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt individual -n 96
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt aggregated -n 16
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt aggregated -n 32
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt aggregated -n 48
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt aggregated -n 96
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -alpha 0.5 -n 16
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -alpha 0.5 -n 32
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -alpha 0.5 -n 48
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -alpha 0.5 -n 96
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt individual -alpha 0.5 -n 16
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt individual -alpha 0.5 -n 32
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt individual -alpha 0.5 -n 48
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt individual -alpha 0.5 -n 96
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt aggregated -alpha 0.5 -n 16
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt aggregated -alpha 0.5 -n 32
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt aggregated -alpha 0.5 -n 48
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt aggregated -alpha 0.5 -n 96

#python3 merge_clusters.py -ext -mism -pc -dfunc braycurtis -fname merged --baseClustNum 96
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -n 16
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -n 32
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -n 48
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -n 96
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -n -agg 16
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -n -agg 32
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -n -agg 48
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -n -agg 96
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -lt individual -n 16
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -lt individual -n 32
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -lt individual -n 48
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -lt individual -n 96
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -lt aggregated -n 16
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -lt aggregated -n 32
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -lt aggregated -n 48
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -lt aggregated -n 96
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -alpha 0.5 -n 16
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -alpha 0.5 -n 32
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -alpha 0.5 -n 48
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -alpha 0.5 -n 96
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -lt individual -alpha 0.5 -n 16
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -lt individual -alpha 0.5 -n 32
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -lt individual -alpha 0.5 -n 48
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -lt individual -alpha 0.5 -n 96
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -lt aggregated -alpha 0.5 -n 16
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -lt aggregated -alpha 0.5 -n 32
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -lt aggregated -alpha 0.5 -n 48
python3 gen_centers.py -ext -mism -pc -dfunc braycurtis -lt aggregated -alpha 0.5 -n 96
