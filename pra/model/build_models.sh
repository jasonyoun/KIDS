#!/usr/bin/env bash

set -e

# ./run.sh /Users/nicholasjoodi/Documents/ucdavis/research/HypothesisGeneration/data/ecoli_for_param_opt_processed/p100/d1/



base_dir="$1"
current_dir=$(pwd)
base_dir="$current_dir""/$base_dir"
prev_current_dir="$current_dir""/.."

. "$base_dir/"config.sh

echo "Content of DATA_PATH is $DATA_PATH"
train_file="train.txt"

instance_dir+="$base_dir""/instance/"

if [ -d "$instance_dir" ]; then
  rm -rfd $instance_dir
fi
echo "copy over configuration "
echo $1
mkdir $instance_dir
# cp pra.jar $instance_dir

# cp pra_data_processor.py $instance_dir

cp "$base_dir""/conf" $instance_dir


cd $instance_dir
sed -i -e "s|given_negative_samples=false|given_negative_samples=true|g" conf
sed -i -e "s|blocked_field=THE_BLOCKED_FIELD|blocked_field=0|g" conf
relation=$(head -n 1 "$DATA_PATH/""relations.txt")
sed -i -e "s|THE_RELATION|$relation|g" conf

sed -i -e "s|task=_TASK_|task=sCV|g" conf

echo "process data "
echo ""
python3 "$prev_current_dir/"pra_data_processor.py $DATA_PATH $train_file

java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.data.WKnowledge createEdgeFile "$instance_dir/"ecoli_generalizations.csv 0.1 edges

java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.data.WKnowledge createEdgeFile "$instance_dir/"ecoli_generalizations_neg.csv 0.1 edges

mkdir graphs
mkdir graphs/pos
mkdir graphs/neg

sed -i -e "s|/graphs/neg|/graphs/pos|g" conf

mv ecoli_generalizations.csv.p0.1.edges graphs/pos/.

mv ecoli_generalizations_neg.csv.p0.1.edges graphs/neg/.

java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.SmallJobs indexGraph ./graphs/pos edges


echo "create positive queries "
echo ""

mkdir queriesR_train

java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.SmallJobs createQueries ./graphs/pos ./queriesR_train/ true false

# mkdir queriesR_test

# java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.SmallJobs createQueries ./graphs/pos ./queriesR_test/ false false

sed -i -e "s|/graphs/pos|/graphs/neg|g" conf

java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.SmallJobs indexGraph ./graphs/neg edges

echo "create negative queries "
echo ""


mkdir queriesR_train_neg

java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.SmallJobs createQueries ./graphs/neg ./queriesR_train_neg/ true false

# mkdir queriesR_test_neg

# java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.SmallJobs createQueries ./graphs/neg ./queriesR_test_neg/ false false

sed -i -e "s|/graphs/neg|/graphs/pos|g" conf

sed -i -e "s|pra.jar|$prev_current_dir/pra.jar|g" conf

python3 "$prev_current_dir/"merge_queries.py $instance_dir

echo "Train models "
echo ""
mkdir models
sed -i -e "s|$relation|THE_RELATION|g" conf
while read p; do
	sed -i -e "s|THE_RELATION|$p|g" conf

	java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.LearnerPRA

	model=$(find `pwd pos/$p` -name model.avg.full)
	mv $model "models/""$p"

	sed -i -e "s|$p|THE_RELATION|g" conf
  	
done <"$DATA_PATH""/relations.txt"

sed -i -e "s|task=sCV|task=_TASK_|g" conf

sed -i -e "s|blocked_field=0|blocked_field=THE_BLOCKED_FIELD|g" conf


