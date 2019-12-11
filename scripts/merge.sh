#!/usr/bin/env bash
cd /Users/taylorsmith/PycharmProjects/KDIGO
source venv/bin/activate
export CATEGORY=all

# Different weights for incorporating the extension penalty into the distance
#for BASECLUST in `seq 96 -12 36`
#do
for CSEED in mean medoid
do
for MALPHA in 0.35 0.5 0.75 1.0
do
for LAPTYPE in "aggregated" "individual" "none"
do
for MAXEXT in -1.0 1 2 3
do
for EXTDISTWEIGHT in 0.0 0.1 0.2 0.35
do
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -n 96 -cat $CATEGORY -seed $CSEED -clen 9 -maxExt 3 -cumExtDist -extDistWeight $EXTDISTWEIGHT -malpha $MALPHA -mlt $LAPTYPE
done
done
done
done
done
