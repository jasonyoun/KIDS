#!/usr/bin/env bash


for f in $(find `pwd ecoli_for_param_opt` -name \*.txt)
do
	suffix="data.txt"

	f=${f%$suffix}
	echo "Processing $f"
	cp er_mlp.ini $f
	sbatch --array=0-971 array_jobs.sh $f
done