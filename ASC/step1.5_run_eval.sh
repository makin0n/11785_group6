#!/bin/bash
#SBATCH -p GPU-shared          
#SBATCH --gpus=h100-80:1       
#SBATCH -N 1               
#SBATCH -t 6:00:00       
#SBATCH -A cis250219p
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

mkdir -p logs

source ~/.bashrc
module load anaconda3
#source activate /ocean/projects/cis250219p/shared/pj-env
source activate /ocean/projects/cis250219p/sunagawa/new_env

export PYTHONUNBUFFERED=1
export HF_HOME="/ocean/projects/cis250219p/shared/huggingface_cache"
export TORCH_HOME="/ocean/projects/cis250219p/shared/huggingface_cache"

echo "Job started with ID: $SLURM_JOB_ID"

echo "Starting Evaluating LT..."
#python train.py LT
python eval.py LT

echo "Starting Evaluating MT..."
#python train.py MT
python eval.py MT

echo "All training jobs completed."