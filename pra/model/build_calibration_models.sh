#!/usr/bin/env bash

set -e

# ./run.sh /Users/nicholasjoodi/Documents/ucdavis/research/HypothesisGeneration/data/ecoli_for_param_opt_processed/p100/d1/



base_dir="$1"
current_dir=$(pwd)
base_dir="$current_dir""/$base_dir"
prev_current_dir="$current_dir""/.."

. "$base_dir/"config.sh

echo "Content of DATA_PATH is $DATA_PATH"
test_file="test.txt"
dev_file="dev.txt"
dev_folder="dev"
io_util_dir='io_util/'
pra_imp_dir='pra_imp/'
data_handler_dir='data_handler/'

instance_dir+="$base_dir""/instance/"


echo 'change directories'

cd $instance_dir

sed -i -e "s|given_negative_samples=true|given_negative_samples=false|g" conf

sed -i -e "s|blocked_field=THE_BLOCKED_FIELD|blocked_field=-1|g" conf
sed -i -e "s|target_relation=.*|target_relation=$start_relation|g" conf
sed -i -e "s|target_relation=THE_RELATION|target_relation=$start_relation|g" conf

sed -i -e "s|task=_TASK_|task=predict|g" conf



sed -i -e "s|/graphs/neg|/graphs/pos|g" conf

echo 'clean space'




echo 'create folders'
mkdir -p calibrations
mkdir -p $dev_folder
mkdir -p $dev_folder/queriesR_test
mkdir -p $dev_folder/predictions
mkdir -p $dev_folder/scores
mkdir -p $dev_folder/queriesR_labels
mkdir -p $dev_folder/queriesR_tail
mkdir -p $dev_folder/thresholds




echo 'configure'
sed -i -e "s|target_relation=$start_relation|target_relation=THE_RELATION|g" conf


echo "calibrate "
echo ""
	sed -i -e "s|given_negative_samples=false|given_negative_samples=true|g" conf
	sed -i -e "s|prediction_folder=.*/|prediction_folder=./$dev_folder/predictions/|g" conf
	sed -i -e "s|test_samples=.*|test_samples=./$dev_folder/queriesR_test/<target_relation>|g" conf
while read p; do
	sed -i -e "s|target_relation=THE_RELATION|target_relation=$p|g" conf
	#grep  -i "\t""$p""\t" "$DATA_PATH""/""$dev_file" | awk -F '\t'  '{print"c$"$1 "\tc$" $3}' > $dev_folder"/queriesR_test/""$p"
	python3 $prev_current_dir/$io_util_dir/create_test_queries.py --data_file "$DATA_PATH""/""$dev_file" --predicate $p --dir $dev_folder
	#grep  -i "\t""$p""\t" "$DATA_PATH""/""$dev_file" | awk -F '\t'  '{print"c$"$1 "\t"}' | awk '!seen[$0]++'  > "./$dev_folder/queriesR_test/""$p"
	grep  -i "\t""$p""\t" "$DATA_PATH""/""$dev_file"| awk -F '\t'  '{print"c$"$1 "\tc$" $3}' > "./$dev_folder/queriesR_tail/""$p"
	grep  -i "\t""$p""\t" "$DATA_PATH""/""$dev_file"| awk -F '\t'  '{print"c$"$1 "\tc$" $3 "\t" $4}' > "./$dev_folder/queriesR_labels/""$p"

	java -cp $prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar  edu.cmu.pra.LearnerPRA

	python3 $prev_current_dir/$io_util_dir/get_scores.py --predicate $p --dir dev

	python3 $prev_current_dir/$io_util_dir/calibrate.py  --predicate $p --dir dev

	sed -i -e "s|target_relation=$p|target_relation=THE_RELATION|g" conf
  	
done <"selected_relations"

sed -i -e "s|task=sCV|task=_TASK_|g" conf
sed -i -e "s|blocked_field=-1|blocked_field=THE_BLOCKED_FIELD|g" conf
sed -i -e "s|prediction_folder=./$dev_folder/predictions/|prediction_folder=.*/|g" conf


