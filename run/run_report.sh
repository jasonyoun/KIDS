#!/usr/bin/env bash

results_dir_name=$1; shift
array=( "$@" )

python3 ./util/aggregate_results.py --dir ${array[@]} --results_dir $results_dir_name
python3 ./util/analyze_model_predictions.py --dir ${array[@]}  --results_dir $results_dir_name
