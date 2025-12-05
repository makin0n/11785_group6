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
source activate /ocean/projects/cis250219p/sunagawa/new_env

export PYTHONUNBUFFERED=1
export HF_HOME="/ocean/projects/cis250219p/shared/huggingface_cache"
export TORCH_HOME="/ocean/projects/cis250219p/shared/huggingface_cache"

echo "Job started with ID: $SLURM_JOB_ID"

# 1. Train the more toxic model and less toxic model
echo "Starting training LT..."
torchrun --nproc_per_node=4 train.py LT
echo "Starting training MT..."
torchrun --nproc_per_node=4 train.py MT
echo "All training jobs completed."

# 2. Evaluate the models
echo "Starting Evaluating LT..."
torchrun --nproc_per_node=4 eval.py LT
echo "Starting Evaluating MT..."
torchrun --nproc_per_node=4 eval.py MT
echo "All evaluation jobs completed."

# 3. Create the DPO dataset for ASC
echo "Starting creating dataset for ASC..."
cd /ocean/projects/sunagawa/11785-Groub6-BuildDataset
torchrun --nproc_per_node=4 scripts/generate_only_toxic.py
echo "Dataset creation completed."

# 4. Execute the Python commands consecutively
cd /ocean/projects/sunagawa/11785_group6/ASC
echo "Starting training ASC..."
torchrun --nproc_per_node=4 asc_train.py
echo "ASC training completed."

# 5. Evaluate all models
echo "Starting Evaluating Models..."
torchrun --nproc_per_node=4 eval_models.py
echo "Model evaluation completed."

echo "All jobs completed."