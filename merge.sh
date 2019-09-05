#!/usr/bin/env bash
source venv/bin/activate
export CATEGORY=allk

#python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -pc -meta meta_avg -n 96 -cat 1-Im -ovpe
#python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -pc -meta meta_avg -n 96 -cat 1-Im -dbapdtw -dbaalpha 1.0
#python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -pc -meta meta_avg -n 96 -cat $CATEGORY -dbapdtw -dbaalpha 0.35
python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt aggregated -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -dbapdtw -dbaalpha 0.35 -seed mean
#python3 assign_cluster_trajectories.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt individual -lv 1.0 -pc -meta meta_avg -n 96 -dbapdtw -dbaalpha 0.35 -seed mean -cemethod zp-front
#python3 assign_cluster_trajectories.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt individual -lv 1.0 -pc -meta meta_avg -n 96 -dbapdtw -dbaalpha 0.35 -seed mean -cemethod dup-front
#python3 assign_cluster_trajectories.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt individual -lv 1.0 -pc -meta meta_avg -n 96 -dbapdtw -dbaalpha 0.35 -seed mean -cemethod zp-back
#python3 assign_cluster_trajectories.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt individual -lv 1.0 -pc -meta meta_avg -n 96 -dbapdtw -dbaalpha 0.35 -seed mean -cemethod dup-back

#python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt individual -lv 1.0 -pc -meta meta_avg -n 96 -cat $CATEGORY -ovpe -seed mean
#python3 assign_cluster_trajectories.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt individual -lv 1.0 -pc -meta meta_avg -n 96 -ovpe -seed mean -cemethod zeropad
#python3 assign_cluster_trajectories.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -lt individual -lv 1.0 -pc -meta meta_avg -n 96 -ovpe -seed mean -cemethod duplicate

#python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -pc -meta meta_avg -n 96 -cat $CATEGORY -dbapdtw -dbaalpha 0.35 -seed mean
#python3 assign_cluster_trajectories.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -pc -meta meta_avg -n 96 -dbapdtw -dbaalpha 0.35 -seed mean -cemethod zeropad
#python3 assign_cluster_trajectories.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -pc -meta meta_avg -n 96 -dbapdtw -dbaalpha 0.35 -seed mean -cemethod duplicate

#python3 mergeClusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -pc -meta meta_avg -n 96 -cat $CATEGORY -ovpe -seed mean
#python3 assign_cluster_trajectories.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -pc -meta meta_avg -n 96 -ovpe -seed mean -cemethod zeropad
#python3 assign_cluster_trajectories.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -pc -meta meta_avg -n 96 -ovpe -seed mean -cemethod duplicate


echo DONE!!!