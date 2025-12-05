#!/bin/bash
#SBATCH -p GPU-shared          
#SBATCH --gpus=h100-80:4       
#SBATCH -N 1               
#SBATCH -t 32:00:00       
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

echo "Job started with ID: $SLURM_JOB_ID"

# 3. Navigate to your working directory (optional if you submit from the dir)
# cd /ocean/projects/groupname/username/project_folder

# 4. Execute the Python commands consecutively
echo "Starting creating dataset..."
#python train.py LT

echo "All training jobs completed."