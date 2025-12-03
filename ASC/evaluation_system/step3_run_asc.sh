#!/bin/bash
#SBATCH -p GPU-shared          
#SBATCH --gpus=v100-32:1       
#SBATCH -N 1
#SBATCH -t 24:00:00       
#SBATCH -A cis250219p
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

mkdir -p logs

source ~/.bashrc
module load anaconda3
# You can modify this to your own environment
source activate /ocean/projects/cis250219p/cchou1/conda/envs/cchou1_pj

export PYTHONUNBUFFERED=1
export HF_HOME="/ocean/projects/cis250219p/shared/huggingface_cache"

echo "Job started with ID: $SLURM_JOB_ID"

echo "Starting evaluating ASC..."

python compare_models_main.py

echo "All training jobs completed."