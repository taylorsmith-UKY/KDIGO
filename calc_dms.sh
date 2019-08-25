#!/usr/bin/env bash




# python3 calc_dm.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -pc -dfunc braycurtis -meta meta_avg
# python3 calc_dm.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -pc -dfunc braycurtis -meta meta_avg -lt individual
python3 calc_dm.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -pc -dfunc braycurtis -meta meta_avg -lt aggregated
python3 calc_dm.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -pc -dfunc braycurtis -meta meta_avg -lt individual -lv 2
python3 calc_dm.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -pc -dfunc braycurtis -meta meta_avg -lt aggregated -lv 2
python3 calc_dm.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -pc -dfunc braycurtis -meta meta_avg -lt individual -lv 0.5
python3 calc_dm.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -pc -dfunc braycurtis -meta meta_avg -lt aggregated -lv 0.5



#python3 calc_dm.py -ext -mism -pc -dfunc braycurtis -alpha 1.0
#python3 calc_dm.py -ext -mism -pc -dfunc braycurtis -lt individual -alpha 1.0
#python3 calc_dm.py -ext -mism -pc -dfunc braycurtis -lt aggregated -alpha 1.0
#python3 calc_dm.py -ext -mism -pc -dfunc braycurtis -alpha 0.5
#python3 calc_dm.py -ext -mism -pc -dfunc braycurtis -lt individual -alpha 0.5
#python3 calc_dm.py -ext -mism -pc -dfunc braycurtis -lt aggregated -alpha 0.5
#python3 calc_dm.py -ext -mism -pc -dfunc braycurtis -alpha 0.1
#python3 calc_dm.py -ext -mism -pc -dfunc braycurtis -lt individual -alpha 0.1
#python3 calc_dm.py -ext -mism -pc -dfunc braycurtis -lt aggregated -alpha 0.1
#python3 calc_dm.py -dfunc braycurtis

#mpirun -np 4 parallelDTW popDTW_normBC_a035 -popDTW -alpha 0.35
