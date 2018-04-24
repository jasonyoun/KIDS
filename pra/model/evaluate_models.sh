#!/usr/bin/env bash

set -e

# ./run.sh /Users/nicholasjoodi/Documents/ucdavis/research/HypothesisGeneration/data/ecoli_for_param_opt_processed/p100/d1/


use_calibration="$2"
base_dir="$1"
current_dir=$(pwd)
base_dir="$current_dir""/$base_dir"
prev_current_dir="$current_dir""/.."

. "$base_dir/"config.sh

echo "Content of DATA_PATH is $DATA_PATH"
test_file="test.txt"
dev_file="dev.txt"

instance_dir+="$base_dir""/instance/"


echo 'change directories'

cd $instance_dir

sed -i -e "s|given_negative_samples=true|given_negative_samples=false|g" conf

sed -i -e "s|blocked_field=THE_BLOCKED_FIELD|blocked_field=-1|g" conf
sed -i -e "s|target_relation=.*|target_relation=$start_relation|g" conf
sed -i -e "s|THE_RELATION|$start_relation|g" conf

sed -i -e "s|task=_TASK_|task=predict|g" conf



sed -i -e "s|/graphs/neg|/graphs/pos|g" conf

echo 'clean space'

if [ -d "queriesR_test" ]; then
  rm -rfd queriesR_test
fi

if [ -d "queriesR_dev" ]; then
  rm -rfd queriesR_dev
fi
if [ -d "queriesR_dev_labels" ]; then
  rm -rfd queriesR_dev_labels
fi

if [ -d "queriesR_labels" ]; then
  rm -rfd queriesR_labels
fi

if [ -d "scores" ]; then
  rm -rfd scores
fi

if [ -d "predictions" ]; then
  rm -rfd predictions
fi

if [ -d "thresholds" ]; then
  rm -rfd thresholds
fi

if [ -d "classifications" ]; then
  rm -rfd classifications
fi

echo 'create folders'
mkdir queriesR_test
mkdir queriesR_labels
mkdir queriesR_dev
mkdir predictions
mkdir scores
mkdir queriesR_dev_labels
mkdir thresholds
mkdir classifications

echo 'configure'
sed -i -e "s|$start_relation|THE_RELATION|g" conf
if  [  "$use_calibration" != "use_calibration" ] ; then
	echo "Determine thresholds "
	echo ""
	sed -i -e "s|test_samples=./queriesR_test/|test_samples=./queriesR_dev/|g" conf
	while read p; do
		sed -i -e "s|THE_RELATION|$p|g" conf
		grep  -i "\t""$p""\t" "$DATA_PATH""/""$dev_file" | awk -F '\t'  '{print"c$"$1 "\tc$" $3}' > "queriesR_dev/""$p"
		grep  -i "\t""$p""\t" "$DATA_PATH""/""$dev_file"| awk -F '\t'  '{print"c$"$1 "\tc$" $3 "\t" $4}' > "queriesR_dev_labels/""$p"

		java -cp "$prev_current_dir/"pra-classification-neg-mode.jar  edu.cmu.pra.LearnerPRA

		python3 "$prev_current_dir/"get_scores.py $p dev

		python3 "$prev_current_dir/"determine_thresholds.py $p dev

		sed -i -e "s|$p|THE_RELATION|g" conf
	  	
	done <"selected_relations"
fi


sed -i -e "s|test_samples=./queriesR_dev/|test_samples=./queriesR_test/|g" conf
echo "Test models "
echo ""
sed -i -e "s|$start_relation|THE_RELATION|g" conf
while read p; do
	sed -i -e "s|THE_RELATION|$p|g" conf
	grep  -i "\t""$p""\t" "$DATA_PATH""/""$test_file" | awk  -F '\t'  '{print"c$"$1 "\tc$" $3}' > "queriesR_test/""$p"
	grep  -i "\t""$p""\t" "$DATA_PATH""/""$test_file"| awk   -F '\t' '{print"c$"$1 "\tc$" $3 "\t" $4}' > "queriesR_labels/""$p"
	java -cp "$prev_current_dir/"pra-classification-neg-mode.jar  edu.cmu.pra.LearnerPRA

	python3 "$prev_current_dir/"get_scores.py $p $use_calibration

	python3 "$prev_current_dir/"classify.py $p $use_calibration

	sed -i -e "s|$p|THE_RELATION|g" conf
  	
done <"selected_relations"

python3 "$prev_current_dir/"evaluate.py 


sed -i -e "s|task=sCV|task=_TASK_|g" conf
sed -i -e "s|blocked_field=-1|blocked_field=THE_BLOCKED_FIELD|g" conf


