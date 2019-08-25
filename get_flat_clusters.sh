#!/usr/bin/env bash

#python3 get_flat_clusters.py -dfunc braycurtis -lt individual
#python3 get_flat_clusters.py -dfunc braycurtis -lt aggregated
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -n 96
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt individual -n 96
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt aggregated -n 96
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -alpha 0.5 -n 96
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt individual -alpha 0.5 -n 96
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt aggregated -alpha 0.5 -n 96
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -alpha 0.1 -n 96
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt individual -alpha 0.1 -n 96
python3 get_flat_clusters.py -ext -mism -pc -dfunc braycurtis -lt aggregated -alpha 0.1 -n 96
python3 get_flat_clusters.py -dfunc braycurtis -n 96
