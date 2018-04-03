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

instance_dir+="$base_dir""/instance/"




cd $instance_dir

sed -i -e "s|given_negative_samples=true|given_negative_samples=false|g" conf

sed -i -e "s|blocked_field=THE_BLOCKED_FIELD|blocked_field=-1|g" conf
relation=$(head -n 1 "$DATA_PATH/""relations.txt")
sed -i -e "s|THE_RELATION|$relation|g" conf

sed -i -e "s|task=_TASK_|task=predict|g" conf



sed -i -e "s|/graphs/neg|/graphs/pos|g" conf


if [ -d "queriesR_test" ]; then
  rm -rfd queriesR_test
fi

if [ -d "queriesR_labels" ]; then
  rm -rfd queriesR_labels
fi

if [ -d "predictions" ]; then
  rm -rfd predictions
fi

mkdir queriesR_test
mkdir queriesR_labels
mkdir predictions

# java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.SmallJobs createQueries ./graphs/pos ./queriesR_test/ false false



echo "Test models "
echo ""
sed -i -e "s|$relation|THE_RELATION|g" conf
while read p; do
	sed -i -e "s|THE_RELATION|$p|g" conf
	grep  -i "\t""$p""\t" "$DATA_PATH""/""$test_file" | awk  '{print"c$"$1 "\tc$" $3}' > "queriesR_test/""$p"
	grep  -i "\t""$p""\t" "$DATA_PATH""/""$test_file"| awk  '{print"c$"$1 "\tc$" $3 "\t" $4}' > "queriesR_labels/""$p"
	java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.LearnerPRA


	sed -i -e "s|$p|THE_RELATION|g" conf
  	
done <"$DATA_PATH""/relations.txt"

sed -i -e "s|task=sCV|task=_TASK_|g" conf
sed -i -e "s|blocked_field=-1|blocked_field=THE_BLOCKED_FIELD|g" conf


