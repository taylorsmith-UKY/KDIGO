scp tdsm222@dlx.uky.edu:/home/tdsm222/KDIGO/mpi/kdigo_icu_2ptAvg_14d_popDTW_a35E-02/kdigo_dm_braycurtis_popCoords_indLap_lap1E+00.csv "/Volumes/GoogleDrive/My Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/RESULTS/icu/081919-1/dm/14days/kdigo_icu_2ptAvg_popDTW_a35E-02/kdigo_dm_braycurtis_popCoords_indLap_lap1E+00.csv"

python3 convert_dm.py

python3 get_flat_clusters.py -sf kdigo_icu_2ptAvg.csv -df days_interp_icu_2ptAvg.csv -pdtw -alpha 0.35 -dfunc braycurtis -pc -lt individual -lv 1.0 -n 96 -meta meta_avg

./merge.sh
