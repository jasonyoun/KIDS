#!/usr/bin/env bash

results_dir_name=$1; shift
array=( "$@" )

# python3 ../analysis/aggregate_results.py --dir ${array[@]} --results_dir $results_dir_name
python3 ../analysis/analyze_model_predictions.py --dir ${array[@]}  --results_dir $results_dir_name
