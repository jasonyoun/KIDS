#! /bin/bash
hostname
python3 -u  run_max_margin.py $SLURM_ARRAY_TASK_ID
# sbatch -t 480 --mem=4000 --array=0-971 array_jobs.sh