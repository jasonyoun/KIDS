#!/usr/bin/env bash

set -e

# ./run.sh /Users/nicholasjoodi/Documents/ucdavis/research/HypothesisGeneration/data/ecoli_for_param_opt_processed/p100/d1/
base_dir="$1"
current_dir=$(pwd)
prev_current_dir="$current_dir""/.."
echo $current_dir
temp_dir+="$1""temp"

if [ -d "$temp_dir" ]; then
  rm -rfd $temp_dir
fi
echo "copy over configuration "
echo $1
mkdir $temp_dir
# cp pra.jar $temp_dir

# cp pra_data_processor.py $temp_dir

cp conf $temp_dir

cp grid $temp_dir


cd $temp_dir
relation=$(head -n 1 "$base_dir/""relations.txt")
sed -i -e "s|THE_RELATION|$relation|g" conf

echo "process data "
echo ""
python3 "$prev_current_dir/"pra_data_processor.py $base_dir

java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.data.WKnowledge createEdgeFile "$temp_dir/"ecoli_generalizations.csv 0.1 edges

java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.data.WKnowledge createEdgeFile "$temp_dir/"ecoli_generalizations_neg.csv 0.1 edges

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

mkdir queriesR_test

java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.SmallJobs createQueries ./graphs/pos ./queriesR_test/ false false

sed -i -e "s|/graphs/pos|/graphs/neg|g" conf

java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.SmallJobs indexGraph ./graphs/neg edges

echo "create negative queries "
echo ""


mkdir queriesR_train_neg

java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.SmallJobs createQueries ./graphs/neg ./queriesR_train_neg/ true false

mkdir queriesR_test_neg

java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.SmallJobs createQueries ./graphs/neg ./queriesR_test_neg/ false false

sed -i -e "s|/graphs/neg|/graphs/pos|g" conf

sed -i -e "s|pra.jar|$prev_current_dir/pra.jar|g" conf

python3 "$prev_current_dir/"merge_queries.py $temp_dir

echo "run sweep "
echo ""
sed -i -e "s|$relation|THE_RELATION|g" conf
while read p; do
	sed -i -e "s|THE_RELATION|$p|g" conf

	java -cp "$prev_current_dir/"pra.jar edu.cmu.lti.util.run.TunnerSweep 

	cp -R sCV.scores "$p""_sCV.scores"

	sed -i -e "s|$p|THE_RELATION|g" conf
  	
done <"$base_dir/""relations.txt"


