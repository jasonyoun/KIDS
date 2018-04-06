#!/usr/bin/env bash

set -e

# ./run.sh /Users/nicholasjoodi/Documents/ucdavis/research/HypothesisGeneration/data/ecoli_for_param_opt_processed/p100/d1/

is_contained () {
  local e word="$1"
  shift
  for e; do [[ "$e" == "$word" ]] && return 0; done
  return 1
}
no_negatives=("something to search for" "a string" "test2000")
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

cp "$base_dir""/conf" $instance_dir


cd $instance_dir
sed -i -e "s|blocked_field=THE_BLOCKED_FIELD|blocked_field=0|g" conf
sed -i -e "s|THE_RELATION|$start_relation|g" conf

sed -i -e "s|task=_TASK_|task=train|g" conf

echo "process data "
echo ""
python3 "$prev_current_dir/"pra_data_processor.py $DATA_PATH $train_file

java -Xms6G -Xmx6G -cp "$prev_current_dir/"pra-classification-neg-mode.jar edu.cmu.pra.data.WKnowledge createEdgeFile "$instance_dir/"ecoli_generalizations.csv 0.1 edges


mkdir graphs
mkdir graphs/pos


sed -i -e "s|/graphs/neg|/graphs/pos|g" conf

mv ecoli_generalizations.csv.p0.1.edges graphs/pos/.


java -Xms6G -Xmx6G -cp "$prev_current_dir/"pra-classification-neg-mode.jar edu.cmu.pra.SmallJobs indexGraph ./graphs/pos edges


echo "create positive queries "
echo ""

mkdir queriesR_train

java -Xms6G -Xmx6G -cp "$prev_current_dir/"pra-classification-neg-mode.jar edu.cmu.pra.SmallJobs createQueries ./graphs/pos ./queriesR_train/ true false

if [ \( -f "$instance_dir/"ecoli_generalizations_neg.csv \) -a  \( "$use_negatives"=true \) ]; then	
	echo "create negative queries "
	echo ""
	sed -i -e "s|given_negative_samples=false|given_negative_samples=true|g" conf
	java -Xms6G -Xmx6G -cp "$prev_current_dir/"pra-classification-neg-mode.jar edu.cmu.pra.data.WKnowledge createEdgeFile "$instance_dir/"ecoli_generalizations_neg.csv 0.1 edges

	mkdir graphs/neg

	mv ecoli_generalizations_neg.csv.p0.1.edges graphs/neg/.

	sed -i -e "s|/graphs/pos|/graphs/neg|g" conf

	java -Xms6G -Xmx6G -cp "$prev_current_dir/"pra-classification-neg-mode.jar edu.cmu.pra.SmallJobs indexGraph ./graphs/neg edges


	mkdir queriesR_train_neg

	java -Xms6G -Xmx6G -cp "$prev_current_dir/"pra-classification-neg-mode.jar edu.cmu.pra.SmallJobs createQueries ./graphs/neg ./queriesR_train_neg/ true false


	sed -i -e "s|/graphs/neg|/graphs/pos|g" conf

	python3 "$prev_current_dir/"merge_queries.py $instance_dir

else
	sed -i -e "s|given_negative_samples=true|given_negative_samples=false|g" conf
fi

sed -i -e "s|pra-classification-neg-mode.jar|$prev_current_dir/pra-classification-neg-mode.jar|g" conf

echo "Train models "
echo ""
mkdir models
sed -i -e "s|$start_relation|THE_RELATION|g" conf
while read p; do
	sed -i -e "s|THE_RELATION|$p|g" conf

	is_contained $p "${no_negatives[@]}"
	if [ \( $?  -eq 1 \) \( "$use_negatives"=true \) ]; then
		sed -i -e "s|given_negative_samples=true|given_negative_samples=false|g" conf
	else
		sed -i -e "s|given_negative_samples=false|given_negative_samples=true|g" conf
	fi
	java -Xms6G -Xmx6G -cp "$prev_current_dir/"pra-classification-neg-mode.jar edu.cmu.pra.LearnerPRA

	model=$(find `pwd pos/$p` -name train.model)
	mv $model "models/""$p"

	rm -rfd pos/$p

	sed -i -e "s|$p|THE_RELATION|g" conf
  	
done <"selected_relations"

sed -i -e "s|task=train|task=_TASK_|g" conf

sed -i -e "s|blocked_field=0|blocked_field=THE_BLOCKED_FIELD|g" conf


