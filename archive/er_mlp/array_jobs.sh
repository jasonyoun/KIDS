#!/usr/bin/env bash

#SBATCH --account=ucd149
#SBATCH --partition=gpu-shared
#SBATCH --time 01:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:k80:1
#SBATCH --job-name="tensorflow-gpu"
#SBATCH --output="tensorflow-gpu.o%A.%a.%N"
#SBATCH --error="tensorflow-gpu.e%A.%a.%N"
#SBATCH --no-requeue
#SBATCH --export=ALL

declare -r SINGULARITY_MODULE='singularity/2.3.2'

declare -r LOCAL_SCRATCH="/scratch/${USER}/${SLURM_JOB_ID}"
declare -r LUSTRE_SCRATCH="/oasis/scratch/comet/mkandes/temp_project"

module purge
module load "${SINGULARITY_MODULE}"

cd "${LOCAL_SCRATCH}"
cp "${LUSTRE_SCRATCH}/singularity/images/tensorflow-gpu.img" ./

export SINGULARITY_SHELL='/bin/bash'
printenv
cd ~/research/Hypothesis-Generation/er_mlp
time -p singularity exec /oasis/scratch/comet/mkandes/temp_project/singularity/images/tensorflow-gpu.img python3 -u  evaluate_params.py $SLURM_ARRAY_TASK_ID
#hostname
#python3 -u  run_max_margin.py $SLURM_ARRAY_TASK_ID
# sbatch --array=0-971 array_jobs.sh
