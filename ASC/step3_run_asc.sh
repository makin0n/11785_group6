#!/bin/bash
#SBATCH -p GPU-shared          
#SBATCH --gpus=h100-80:4       
#SBATCH -N 1
#SBATCH -t 24:00:00       
#SBATCH -A cis250219p
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

mkdir -p logs

source ~/.bashrc
module load anaconda3
# You can modify this to your own environment
source activate /ocean/projects/cis250219p/sunagawa/new_env

export PYTHONUNBUFFERED=1
export HF_HOME="/ocean/projects/cis250219p/shared/huggingface_cache"

echo "Job started with ID: $SLURM_JOB_ID"

echo "Starting evaluating ASC..."
torchrun --nproc_per_node=4 eval.py

echo "All training jobs completed."