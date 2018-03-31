#!/usr/bin/env bash

cp pra.jar $1

cp pra_data_processor.py $1

cp conf $1

cp grid $1

cd $1

python3 pra_data_processor.py $1

java -cp pra.jar edu.cmu.pra.data.WKnowledge createEdgeFile ecoli_generalizations.csv 0.1 edges

mkdir graphs

mv ecoli_generalizations.csv.p0.1.edges graphs/.

java -cp pra.jar edu.cmu.pra.SmallJobs indexGraph ./graphs edges

mkdir queriesR_train

java -cp pra.jar edu.cmu.pra.SmallJobs createQueries ./graphs ./queriesR_train/ true false

java -cp pra.jar edu.cmu.pra.SmallJobs createQueries ./graphs ./queriesR_test/ false false

sed -i -e 's/represses/activates/g' conf

java -cp pra.jar edu.cmu.pra.LearnerPRA 

sed -i -e 's/activates/represses/g' conf

java -cp pra.jar edu.cmu.pra.LearnerPRA