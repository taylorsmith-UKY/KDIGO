#!/usr/bin/env bash
cd /Users/taylorsmith/PycharmProjects/KDIGO
source venv/bin/activate
export CATEGORY=1-Im

for LAPTYPE in "aggregated" # "individual" "none"
do
for CLEN in 9 # 14
do
for EXTWEIGHT in 0.2 # 0.35 0.5
do
python3 mergeSimData.py -sf kdigo_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt $LAPTYPE -lv 1.0 -pc -cat $CATEGORY -dbaiter 5 -mtype mean -extDistWeight $EXTWEIGHT -clen $CLEN
#python3 mergeSimData.py -sf kdigo_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt $LAPTYPE -lv 1.0 -pc -cat $CATEGORY -dbaiter 5 -mtype mean -extDistWeight $EXTWEIGHT -clen $CLEN -scaleExt
#python3 mergeSimData.py -sf kdigo_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt $LAPTYPE -lv 1.0 -pc -cat $CATEGORY -dbaiter 5 -mtype mean -extDistWeight $EXTWEIGHT -clen $CLEN -cumExtDist
#python3 mergeSimData.py -sf kdigo_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt $LAPTYPE -lv 1.0 -pc -cat $CATEGORY -dbaiter 5 -mtype mean -extDistWeight $EXTWEIGHT -clen $CLEN -scaleExt -cumExtDist
#python3 mergeSimData.py -sf kdigo_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt $LAPTYPE -lv 1.0 -pc -cat $CATEGORY -dbaiter 5 -mtype mean -extDistWeight $EXTWEIGHT -clen $CLEN -maxExt 10
#python3 mergeSimData.py -sf kdigo_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt $LAPTYPE -lv 1.0 -pc -cat $CATEGORY -dbaiter 5 -mtype mean -extDistWeight $EXTWEIGHT -clen $CLEN -maxExt 20
done
done
done
