#! /bin/bash
hostname
python3 -u  run_max_margin.py $SLURM_ARRAY_TASK_ID
