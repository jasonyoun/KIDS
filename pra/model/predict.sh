#!/usr/bin/env bash

set -e

use_calibration="$2"
base_dir="$1"
current_dir=$(pwd)
model_instance_dir=$current_dir/model_instance
cd $model_instance_dir
base_dir="$model_instance_dir""/$base_dir"
prev_current_dir="$current_dir""/.."

io_util_dir='io_util/'
pra_imp_dir='pra_imp/'

. "$base_dir/"config.sh

prediction_folder=${predict_file%.txt}
echo $prediction_folder

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
cp $DATA_PATH/$predict_file $prediction_folder/data.txt
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
	grep -i -P "\t""$p""\t" "$prediction_folder""/$test_file" | awk -F '\t'  '{print"c$"$1 "\tc$" $3}' > "./$prediction_folder/queriesR_tail/""$p"
	grep -i -P "\t""$p""\t" "$prediction_folder""/$test_file" | awk -F '\t'  '{print"c$"$1 "\tc$" $3 "\t" $4}' > "./$prediction_folder/queriesR_labels/""$p"
	java -cp $prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar edu.cmu.pra.LearnerPRA

	if  [  "$use_calibration" != "use_calibration" ] ; then
		python3 $prev_current_dir/$io_util_dir/get_scores.py --predicate $p --dir $prediction_folder
		python3 $prev_current_dir/$io_util_dir/classify.py --predicate $p --dir $prediction_folder
	else
		python3 $prev_current_dir/$io_util_dir/get_scores.py --predicate $p --dir $prediction_folder --use_calibration --log_reg_calibrate $log_reg_calibrate
		python3 $prev_current_dir/$io_util_dir/classify.py --predicate $p --dir $prediction_folder --use_calibration
	fi

	sed -i -e "s|target_relation=$p|target_relation=THE_RELATION|g" conf

done <"selected_relations"

if  [  "$use_calibration" != "use_calibration" ] ; then
	python3 $prev_current_dir/$io_util_dir/evaluate.py --dir $prediction_folder
else
	python3 $prev_current_dir/$io_util_dir/evaluate.py --dir $prediction_folder --use_calibration
fi

sed -i -e "s|task=predict|task=_TASK_|g" conf
sed -i -e "s|blocked_field=-1|blocked_field=THE_BLOCKED_FIELD|g" conf
