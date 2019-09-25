#!/usr/bin/env bash
source venv/bin/activate
export CATEGORY=1-Im

python3 gen_centers.py -ext -mism -pc -fname rename/1-Im --baseClustNum 96 -alpha 1.0
python3 gen_centers.py -ext -mism -pc -fname rename/1-Im --baseClustNum 96 -alpha 1.0 -agg
python3 gen_centers.py -ext -mism -pc -fname rename/1-Im --baseClustNum 96 -alpha 0.5
python3 gen_centers.py -ext -mism -pc -fname rename/1-Im --baseClustNum 96 -alpha 0.5 -agg

echo DONE!!!