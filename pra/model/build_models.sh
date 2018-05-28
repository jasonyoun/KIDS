#!/usr/bin/env bash

set -e

# ./run.sh /Users/nicholasjoodi/Documents/ucdavis/research/HypothesisGeneration/data/ecoli_for_param_opt_processed/p100/d1/

containsElement () {
  local e match="$1" ret="false"
  shift
  for e; 
  do 
	  if [[ "$e" == "$match" ]]; then
	   ret="true" 
	   break
	  fi
  done
  echo "$ret"
}
base_dir="$1"
current_dir=$(pwd)
base_dir="$current_dir""/$base_dir"
prev_current_dir="$current_dir""/.."
io_util_dir='io_util/'
pra_imp_dir='pra_imp/'
data_handler_dir='data_handler/'

. "$base_dir/"config.sh

echo "Content of DATA_PATH is $DATA_PATH"
# train_file="train.txt"
train_folder="train"


instance_dir+="$base_dir""/instance/"

if [ -d "$instance_dir" ]; then
  rm -rfd $instance_dir
fi
echo "copy over configuration "
mkdir $instance_dir

cp "$base_dir""/conf" $instance_dir


cd $instance_dir
sed -i -e "s|blocked_field=THE_BLOCKED_FIELD|blocked_field=0|g" conf
sed -i -e "s|target_relation=.*|target_relation=$start_relation|g" conf
sed -i -e "s|target_relation=THE_RELATION|target_relation=$start_relation|g" conf

sed -i -e "s|task=_TASK_|task=train|g" conf

echo "process data "
echo ""
if  [  "$is_freebase" == "true" ] ; then
	python3 $prev_current_dir/$data_handler_dir/pra_data_processor.py $DATA_PATH $train_file $is_freebase
else
	python3 $prev_current_dir/$data_handler_dir/pra_data_processor.py $DATA_PATH $train_file
fi
java -Xms6G -Xmx6G -cp $prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar edu.cmu.pra.data.WKnowledge createEdgeFile "$instance_dir/"ecoli_generalizations.csv 0.1 edges


mkdir -p graphs
mkdir -p graphs/pos


sed -i -e "s|/graphs/neg|/graphs/pos|g" conf

mv ecoli_generalizations.csv.p0.1.edges graphs/pos/.


java -Xms6G -Xmx6G -cp $prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar edu.cmu.pra.SmallJobs indexGraph ./graphs/pos edges


echo "create positive queries "
echo ""
mkdir -p $train_folder

mkdir -p "$train_folder""/queriesR_train"

java -Xms6G -Xmx6G -cp $prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar edu.cmu.pra.SmallJobs createQueries ./graphs/pos "$train_folder""/queriesR_train/" true false

if [ -f "$instance_dir/"ecoli_generalizations_neg.csv ] && [ "$use_negatives" == "true"  ]; then	
	echo "create negative queries "
	echo ""
	sed -i -e "s|given_negative_samples=false|given_negative_samples=true|g" conf
	java -Xms6G -Xmx6G -cp $prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar edu.cmu.pra.data.WKnowledge createEdgeFile "$instance_dir/"ecoli_generalizations_neg.csv 0.1 edges

	mkdir graphs/neg

	mv ecoli_generalizations_neg.csv.p0.1.edges graphs/neg/.

	sed -i -e "s|/graphs/pos|/graphs/neg|g" conf

	java -Xms6G -Xmx6G -cp $prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar edu.cmu.pra.SmallJobs indexGraph ./graphs/neg edges


	mkdir "$train_folder""/queriesR_train_neg"

	java -Xms6G -Xmx6G -cp $prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar edu.cmu.pra.SmallJobs createQueries ./graphs/neg "$train_folder""/queriesR_train_neg/" true false


	sed -i -e "s|/graphs/neg|/graphs/pos|g" conf

	python3 $prev_current_dir/$io_util_dir/merge_queries.py --dir $train_folder

else
	sed -i -e "s|given_negative_samples=true|given_negative_samples=false|g" conf
fi

sed -i -e "s|pra_neg_mode_v4.jar|$prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar|g" conf

echo "Train models "
echo ""
mkdir models
sed -i -e "s|target_relation=$start_relation|target_relation=THE_RELATION|g" conf
while read p; do
	sed -i -e "s|target_relation=THE_RELATION|target_relation=$p|g" conf

	does_not_have_negatives=$(containsElement $p "${no_negatives[@]}")
	echo "$p"

	echo "$does_not_have_negatives"

	if  [  "$does_not_have_negatives" == "true" ] || [ "$use_negatives" != "true"  ]; then
		echo 'inside'
		sed -i -e "s|given_negative_samples=true|given_negative_samples=false|g" conf
	else
		echo 'outside'
		sed -i -e "s|given_negative_samples=false|given_negative_samples=true|g" conf
	fi
	java -Xms6G -Xmx6G -cp $prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar edu.cmu.pra.LearnerPRA

	model=$(find `pwd pos/$p` -name train.model)
	mv $model "models/""$p"

	rm -rfd pos/$p

	sed -i -e "s|target_relation=$p|target_relation=THE_RELATION|g" conf
  	
done <"selected_relations"

sed -i -e "s|task=train|task=_TASK_|g" conf

sed -i -e "s|blocked_field=0|blocked_field=THE_BLOCKED_FIELD|g" conf


