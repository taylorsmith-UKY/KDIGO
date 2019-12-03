#!/usr/bin/env bash
cd /Users/taylorsmith/PycharmProjects/KDIGO
source venv/bin/activate
export CATEGORY=none

# Different weights for incorporating the extension penalty into the distance
#for BASECLUST in `seq 96 -12 36`
#do
for LAPTYPE in "aggregated" "individual" "none"
do
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt $LAPTYPE -lv 1.0 -pc -n 96 -cat $CATEGORY -seed mean -clen 9
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt $LAPTYPE -lv 1.0 -pc -n 96 -cat $CATEGORY -seed zeros -clen 14
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt $LAPTYPE -lv 1.0 -pc -n 96 -cat $CATEGORY -seed zeros -clen 9
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt $LAPTYPE -lv 1.0 -pc -n 96 -cat $CATEGORY -seed medoid -clen 14
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt $LAPTYPE -lv 1.0 -pc -n 96 -cat $CATEGORY -seed medoid -clen 9
#python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt $LAPTYPE -lv 1.0 -pc -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 0.35 -plot_c
#python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt $LAPTYPE -lv 1.0 -pc -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 0.5 -plot_c
#python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt $LAPTYPE -lv 1.0 -pc -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 1.0 -plot_c
#python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt $LAPTYPE -lv 1.0 -pc -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 0.35 -plot_c -cumExtDist
#python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt $LAPTYPE -lv 1.0 -pc -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 0.5 -plot_c -cumExtDist
#python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt $LAPTYPE -lv 1.0 -pc -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 1.0 -plot_c -cumExtDist
#python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt $LAPTYPE -lv 1.0 -pc -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 0.0 -maxExt 10 -plot_c
#python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt $LAPTYPE -lv 1.0 -pc -n 96 -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -extDistWeight 0.0 -maxExt 20 -plot_c
done
#done



#python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -n $BASECLUST -cat $CATEGORY -seed mean -dbaiter 5 -mtype dba -extDistWeight 0.35 -plot_c
#python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -n $BASECLUST -cat $CATEGORY -seed mean -dbaiter 5 -mtype dba -extDistWeight 0.35 -scaleExt -plot_c
#python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -n $BASECLUST -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -maxExt 5 -plot_c
#python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -n $BASECLUST -cat $CATEGORY -seed mean -dbaiter 5 -mtype mean -maxExt 30 -plot_c
