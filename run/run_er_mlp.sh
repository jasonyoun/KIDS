#!/usr/bin/env bash

set -e

er_mlp_dir='../er_mlp/model/'
er_mlp_instance_dir=$er_mlp_dir/model_instance/
config_dir="$1"
current_dir=$(pwd)

mkdir -p $er_mlp_instance_dir'/'$config_dir
cp 'configuration/'$config_dir'/er_mlp.ini' $er_mlp_instance_dir'/'$config_dir'/'$config_dir'.ini'
cd $er_mlp_dir
python3 -u build_network.py --dir $config_dir
python3 -u determine_thresholds.py --dir $config_dir
python3 -u predict.py --dir $config_dir --predict_file train_local.txt
python3 -u predict.py --dir $config_dir --predict_file dev.txt
python3 -u predict.py --dir $config_dir --predict_file test.txt
python3 -u evaluate_network.py --dir $config_dir




