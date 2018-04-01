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

echo $1
mkdir $temp_dir
# cp pra.jar $temp_dir

# cp pra_data_processor.py $temp_dir

cp conf $temp_dir

cp grid $temp_dir

cd $temp_dir
echo "$prev_current_dir/"pra_data_processor.py $base_dir
python3 "$prev_current_dir/"pra_data_processor.py $base_dir

java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.data.WKnowledge createEdgeFile "$temp_dir/"ecoli_generalizations.csv 0.1 edges

mkdir graphs

mv ecoli_generalizations.csv.p0.1.edges graphs/.

java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.SmallJobs indexGraph ./graphs edges

mkdir queriesR_train

java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.SmallJobs createQueries ./graphs ./queriesR_train/ true false

mkdir queriesR_test

java -cp "$prev_current_dir/"pra.jar edu.cmu.pra.SmallJobs createQueries ./graphs ./queriesR_test/ false false

sed -i -e "s|represses|activates|g" conf

sed -i -e "s|pra.jar|$prev_current_dir/pra.jar|g" conf

java -cp "$prev_current_dir/"pra.jar edu.cmu.lti.util.run.TunnerSweep 

cp -R sCV.scores activates_sCV.scores

sed -i -e 's/activates/represses/g' conf

java -cp "$prev_current_dir/"pra.jar edu.cmu.lti.util.run.TunnerSweep 

cp -R sCV.scores represses_sCV.scores
