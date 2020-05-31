#!/usr/bin/env bash

export CONFIG_FILENAME=final.json
export EXT_CONFIG_FILENAME=final.json
export MODEL=log
export CVNUM=10

#python3 preprocess_utsw_conf1.py --config_file $CONFIG_FILENAME -agvPts 2
python3 getStaticFeaturesUTSW.py --config_file $CONFIG_FILENAME --ref_config_file $EXT_CONFIG_FILENAME
#
#for FEATURE in "sofa_apache_norm" "base_model" "base_model_withTrajectory" "clinical_model_mortality"
#do
#python3 classify_features.py --config_file $EXT_CONFIG_FILENAME --ext_config_file $CONFIG_FILENAME -f $FEATURE -t died_inp -class $MODEL -ext -cv $CVNUM
#done
#
#for FEATURE in "max_kdigo_d03_norm" "base_model" "base_model_withTrajectory" "clinical_model_make"
#do
#python3 classify_features.py --config_file $EXT_CONFIG_FILENAME --ext_config_file $CONFIG_FILENAME -f $FEATURE -t make90_disch_30dbuf_2pts_d50 -class $MODEL -ext -survivors -cv $CVNUM
#python3 classify_features.py --config_file $EXT_CONFIG_FILENAME --ext_config_file $CONFIG_FILENAME -f $FEATURE -t make90_disch_30dbuf_2pts_d50 -class $MODEL -ext -cv $CVNUM
#done

#for TARGET in "died_inp" "make90_disch_30dbuf_2pts_d50"
#do
#for FEATURE in "sofa_apache_norm" "max_kdigo_d03_norm" "base_model" "base_model_withTrajectory" "clinical_model"
#do
#python3 classify_features.py --config_file $EXT_CONFIG_FILENAME --ext_config_file $CONFIG_FILENAME -f $FEATURE -t $TARGET -class $MODEL -ext -cv $CVNUM
#done
#done
#
#for TARGET in "make90_disch_30dbuf_2pts_d50"
#do
#for FEATURE in "max_kdigo_d03_norm" "base_model" "base_model_withTrajectory" "clinical_model"
#do
#python3 classify_features.py --config_file $EXT_CONFIG_FILENAME --ext_config_file $CONFIG_FILENAME -f $FEATURE -t $TARGET -class $MODEL -ext -survivors -cv $CVNUM
#done
#done

#python3 classify_features.py --config_file $EXT_CONFIG_FILENAME --ext_config_file $CONFIG_FILENAME -f clinical_model_mortality_noNorm -t died_inp -class $MODEL -ext -cv $CVNUM
#python3 classify_features.py --config_file $EXT_CONFIG_FILENAME --ext_config_file $CONFIG_FILENAME -f clinical_model_make_noNorm -t make90_disch_30dbuf_2pts_d50 -class $MODEL -ext -cv $CVNUM
#python3 classify_features.py --config_file $EXT_CONFIG_FILENAME --ext_config_file $CONFIG_FILENAME -f clinical_model_make_noNorm -t make90_disch_30dbuf_2pts_d50 -class $MODEL -ext -survivors -cv $CVNUM
#
#python3 evalReclassification.py --config_file $CONFIG_FILENAME -t died_inp -class $MODEL -ext -f sofa_apache_norm base_model base_model_withTrajectory clinical_model_mortality
#python3 evalReclassification.py --config_file $CONFIG_FILENAME -t make90_disch_30dbuf_2pts_d50 -class $MODEL -ext -f max_kdigo_d03_norm base_model base_model_withTrajectory clinical_model_make
#python3 evalReclassification.py --config_file $CONFIG_FILENAME -t make90_disch_30dbuf_2pts_d50 -class $MODEL -ext -survivors -f max_kdigo_d03_norm base_model base_model_withTrajectory clinical_model_make
