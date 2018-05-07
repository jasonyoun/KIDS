#!/usr/bin/env bash

set -e

# ./run.sh /Users/nicholasjoodi/Documents/ucdavis/research/HypothesisGeneration/data/ecoli_for_param_opt_processed/p100/d1/


use_calibration="$2"
base_dir="$1"
current_dir=$(pwd)
base_dir="$current_dir""/$base_dir"
prev_current_dir="$current_dir""/.."
prediction_folder="predictions"
io_util_dir='io_util/'
pra_imp_dir='pra_imp/'

. "$base_dir/"config.sh

echo "Content of DATA_PATH is $DATA_PATH"
test_file="data.txt"

instance_dir+="$base_dir""/instance/"



echo 'change directories'

cd $instance_dir

sed -i -e "s|given_negative_samples=false|given_negative_samples=true|g" conf

sed -i -e "s|blocked_field=THE_BLOCKED_FIELD|blocked_field=-1|g" conf
sed -i -e "s|target_relation=.*|target_relation=$start_relation|g" conf
sed -i -e "s|target_relation=THE_RELATION|target_relation=$start_relation|g" conf

sed -i -e "s|task=_TASK_|task=predict|g" conf



sed -i -e "s|/graphs/neg|/graphs/pos|g" conf

echo 'clean space'


echo 'create folders'
mkdir -p $prediction_folder
mkdir -p $prediction_folder/queriesR_test
mkdir -p $prediction_folder/queriesR_tail
mkdir -p $prediction_folder/queriesR_labels
mkdir -p $prediction_folder/predictions
mkdir -p $prediction_folder/scores
mkdir -p $prediction_folder/classifications



sed -i -e "s|prediction_folder=.*/|prediction_folder=$prediction_folder/predictions/|g" conf
sed -i -e "s|test_samples=.*|test_samples=$prediction_folder/queriesR_test/<target_relation>|g" conf
echo "Test models "
echo ""
sed -i -e "s|target_relation=$start_relation|target_relation=THE_RELATION|g" conf
while read p; do
	sed -i -e "s|target_relation=THE_RELATION|target_relation=$p|g" conf
	echo "$prediction_folder""/$test_file"
	python3 $prev_current_dir/$io_util_dir/create_test_queries.py --data_file "$prediction_folder""/""$test_file" --predicate $p --dir $prediction_folder
	# grep  -i "\t""$p""\t" "$prediction_folder""/""$test_file" | awk -F '\t'  '{print"c$"$1 "\tc$" $3}' > $prediction_folder"/queriesR_test/""$p"
	grep  -i "\t""$p""\t" "$prediction_folder""/$test_file" | awk -F '\t'  '{print"c$"$1 "\tc$" $3}' > "./$prediction_folder/queriesR_tail/""$p"
	grep  -i "\t""$p""\t" "$prediction_folder""/$test_file" | awk -F '\t'  '{print"c$"$1 "\tc$" $3 "\t" $4}' > "./$prediction_folder/queriesR_labels/""$p"
	java -cp $prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar edu.cmu.pra.LearnerPRA

	# python3 "$prev_current_dir/"get_scores.py --predicate $p --dir $prediction_folder --use_calibration
	# python3 "$prev_current_dir/"classify.py --predicate $p --dir $prediction_folder --use_calibration

	python3 $prev_current_dir/$io_util_dir/get_scores.py --predicate $p --dir $prediction_folder  --use_calibration
	python3 $prev_current_dir/$io_util_dir/classify.py --predicate $p --dir $prediction_folder  --use_calibration

	sed -i -e "s|target_relation=$p|target_relation=THE_RELATION|g" conf
  	
done <"selected_relations"

python3 $prev_current_dir/$io_util_dir/evaluate.py --dir $prediction_folder


sed -i -e "s|task=predict|task=_TASK_|g" conf
sed -i -e "s|blocked_field=-1|blocked_field=THE_BLOCKED_FIELD|g" conf


