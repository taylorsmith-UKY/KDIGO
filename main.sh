#!/usr/bin/env bash

#export CONFIG_FILENAME=manuscript-final.json
#export EXT_CONFIG_FILENAME=manuscript-final.json
export CONFIG_FILENAME=final-pub.json
export EXT_CONFIG_FILENAME=final-pub.json
export MODEL=log
export ENGINE=sklearn
export CVNUM=10

# Preprocess and extract relevant data
python3 preprocess_uky_conf.py --config_file $CONFIG_FILENAME -agvPts 2
python3 cleanStats.py --config_file $CONFIG_FILENAME -agvPts 2
#python3 preprocess_utsw_conf.py --config_file $CONFIG_FILENAME -agvPts 2
##
### Construct feature tables for classification
python3 getStaticFeatures.py --config_file $CONFIG_FILENAME -agvPts 2
python3 getStaticFeaturesUTSW.py --config_file $CONFIG_FILENAME --ref_config_file $EXT_CONFIG_FILENAME
##
# Evaluate each model using CV_NUM-fold cross validation and external validation
for FEATURE in "sofa_apache_norm" "clinical_model_mortality" "clinical_model_mortality_wTrajectory" #  "base_model" "base_model_withTrajectory"
do
python3 classify_wExternalValidation.py --config_file $CONFIG_FILENAME --ext_config_file $CONFIG_FILENAME -f $FEATURE -t died_inp -class $MODEL -eng $ENGINE -cv $CVNUM
done
#
for FEATURE in "max_kdigo_d03_norm" "clinical_model_make" "clinical_model_make_wTrajectory"  # "base_model" "base_model_withTrajectory"
do
python3 classify_wExternalValidation.py --config_file $CONFIG_FILENAME --ext_config_file $CONFIG_FILENAME -f $FEATURE -t make90_disch_30dbuf_2pts_d50 -class $MODEL -eng $ENGINE -cv $CVNUM
#python3 classify_wExternalValidation.py --config_file $CONFIG_FILENAME --ext_config_file $CONFIG_FILENAME -f $FEATURE -t make90_disch_30dbuf_2pts_d50 -class $MODEL -survivors -eng $ENGINE -cv $CVNUM
done
#
# Plot top features for mortality
python3 getOddsPlots.py --config_file $CONFIG_FILENAME -f clinical_model_mortality -t died_inp -class $MODEL -eng $ENGINE
python3 getOddsPlots.py --config_file $CONFIG_FILENAME -f clinical_model_mortality_wTrajectory -t died_inp -class $MODEL -eng $ENGINE

# Plot top features for MAKE
python3 getOddsPlots.py --config_file $CONFIG_FILENAME -f clinical_model_make -t make90_disch_30dbuf_2pts_d50 -class $MODEL -eng $ENGINE
#python3 getOddsPlots.py --config_file $CONFIG_FILENAME -f clinical_model_make -t make90_disch_30dbuf_2pts_d50 -class $MODEL -eng $ENGINE -survivors

## Plot top features for MAKE (hospital survivors)
python3 getOddsPlots.py --config_file $CONFIG_FILENAME -f clinical_model_make_wTrajectory -t make90_disch_30dbuf_2pts_d50 -class $MODEL -eng $ENGINE
#python3 getOddsPlots.py --config_file $CONFIG_FILENAME -f clinical_model_make_wTrajectory -t make90_disch_30dbuf_2pts_d50 -class $MODEL -eng $ENGINE -survivors

# Compute reclassification performance metrics
python3 evalReclassification.py  --config_file $CONFIG_FILENAME -t died_inp -class $MODEL -f sofa_apache_norm clinical_model_mortality clinical_model_mortality_wTrajectory -eng $ENGINE
python3 evalReclassification.py  --config_file $CONFIG_FILENAME -t make90_disch_30dbuf_2pts_d50 -class $MODEL -f max_kdigo_d03_norm clinical_model_make clinical_model_make_wTrajectory -eng $ENGINE
#python3 evalReclassification.py  --config_file $CONFIG_FILENAME -t make90_disch_30dbuf_2pts_d50 -class $MODEL -survivors -f max_kdigo_d03_norm clinical_model_make clinical_model_make_wTrajectory -eng $ENGINE

python3 evalReclassification.py  --config_file $CONFIG_FILENAME -t died_inp -class $MODEL -f sofa_apache_norm clinical_model_mortality clinical_model_mortality_wTrajectory -eng $ENGINE -ext
python3 evalReclassification.py  --config_file $CONFIG_FILENAME -t make90_disch_30dbuf_2pts_d50 -class $MODEL -f max_kdigo_d03_norm clinical_model_make clinical_model_make_wTrajectory -eng $ENGINE -ext
#python3 evalReclassification.py  --config_file $CONFIG_FILENAME -t make90_disch_30dbuf_2pts_d50 -class $MODEL -survivors -f max_kdigo_d03_norm clinical_model_make clinical_model_make_wTrajectory -eng $ENGINE -ext

# Plot prediction vs. risk
python3 plotPredictionVsOutcome.py --config_file $CONFIG_FILENAME -f clinical_model_mortality -t died_inp -class $MODEL -eng $ENGINE
python3 plotPredictionVsOutcome.py --config_file $CONFIG_FILENAME -f clinical_model_make_wTrajectory -t make90_disch_30dbuf_2pts_d50 -class $MODEL -eng $ENGINE

