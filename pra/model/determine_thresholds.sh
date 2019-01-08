#!/usr/bin/env bash

set -e

# directories
use_calibration="$2"
base_dir="$1"
current_dir=$(pwd)
model_instance_dir=$current_dir/model_instance
cd $model_instance_dir
base_dir="$model_instance_dir""/$base_dir"
prev_current_dir="$current_dir""/.."
io_util_dir='io_util/'
pra_imp_dir='pra_imp/'
data_handler_dir='data_handler/'

# include config.sh
. "$base_dir/"config.sh

echo "Using data from path: $DATA_PATH"

test_file="test.txt"
dev_file="dev.txt"
test_folder="test"
dev_folder="dev"
instance_dir+="$base_dir""/instance/"

echo 'changing directories...'
cd $instance_dir

# modify configurations
sed -i -e "s|given_negative_samples=true|given_negative_samples=false|g" conf
sed -i -e "s|blocked_field=THE_BLOCKED_FIELD|blocked_field=-1|g" conf
sed -i -e "s|target_relation=.*|target_relation=$start_relation|g" conf
sed -i -e "s|target_relation=THE_RELATION|target_relation=$start_relation|g" conf
sed -i -e "s|task=_TASK_|task=predict|g" conf
sed -i -e "s|/graphs/neg|/graphs/pos|g" conf

# create folders
echo 'creating folders...'
mkdir -p $dev_folder
mkdir -p $dev_folder/queriesR_test
mkdir -p $dev_folder/queriesR_labels
mkdir -p $dev_folder/queriesR_tail
mkdir -p $dev_folder/predictions
mkdir -p $dev_folder/scores
mkdir -p $dev_folder/classifications

mkdir -p $dev_folder/thresholds
mkdir -p $dev_folder/thresholds_calibration

mkdir -p $test_folder
mkdir -p $test_folder/queriesR_test
mkdir -p $test_folder/queriesR_labels
mkdir -p $test_folder/queriesR_tail
mkdir -p $test_folder/predictions
mkdir -p $test_folder/scores
mkdir -p $test_folder/classifications

# modify configurations
echo 'configure'
sed -i -e "s|given_negative_samples=false|given_negative_samples=true|g" conf
sed -i -e "s|target_relation=$start_relation|target_relation=THE_RELATION|g" conf
sed -i -e "s|prediction_folder=.*/|prediction_folder=./$dev_folder/predictions/|g" conf
sed -i -e "s|test_samples=.*|test_samples=./$dev_folder/queriesR_test/<target_relation>|g" conf

echo "determining thresholds..."
echo ""

while read p; do
	sed -i -e "s|target_relation=THE_RELATION|target_relation=$p|g" conf
	echo $p
	python3 $prev_current_dir/$io_util_dir/create_test_queries.py --data_file "$DATA_PATH""/""$dev_file" --predicate $p --dir $dev_folder
	grep -i -P "\t""$p""\t" "$DATA_PATH""/""$dev_file"| awk -F '\t' '{print"c$"$1 "\tc$" $3}' > "./$dev_folder/queriesR_tail/""$p"
	grep -i -P "\t""$p""\t" "$DATA_PATH""/""$dev_file"| awk -F '\t' '{print"c$"$1 "\tc$" $3 "\t" $4}' > "./$dev_folder/queriesR_labels/""$p"

	java -cp $prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar edu.cmu.pra.LearnerPRA
	if  [ "$use_calibration" != "use_calibration" ] ; then
		python3 $prev_current_dir/$io_util_dir/get_scores.py --predicate $p --dir $dev_folder
		python3 $prev_current_dir/$io_util_dir/determine_thresholds.py --predicate $p --dir $dev_folder
	else
		python3 $prev_current_dir/$io_util_dir/get_scores.py --predicate $p --dir $dev_folder --use_calibration --log_reg_calibrate $log_reg_calibrate
		python3 $prev_current_dir/$io_util_dir/determine_thresholds.py --predicate $p --dir $dev_folder --use_calibration
	fi

	sed -i -e "s|target_relation=$p|target_relation=THE_RELATION|g" conf

done <"selected_relations"

sed -i -e "s|task=predict|task=_TASK_|g" conf
sed -i -e "s|blocked_field=-1|blocked_field=THE_BLOCKED_FIELD|g" conf
