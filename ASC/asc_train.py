import os
import gc
import torch
import wandb
import transformers
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from detoxify import Detoxify

from config import API_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, LORA_CONFIG
from utils import *
# from dataset import get_hh
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from model.dpo_train import DPOTrainerWrapper
from model.dpo_eval import evaluate

import warnings
warnings.filterwarnings("ignore")

def main():
    # =============================================================================
    # USER CONFIGURATION
    # =============================================================================
    
    # 1. training parameters
    NUM_EPOCHS = 3                    # Epoch
    TRAIN_RATIO = 0.9                 # Training data ratio (0.9 = 90% Train, 10% Test)
    
    # 2. data/model paths
    DATA_PATH = "/ocean/projects/cis250219p/shared/dataset/stage2_med_pairs_qwen_all_new.jsonl" 
    
    # path to M_LT
    #BASE_MODEL_PATH = "/ocean/projects/cis250219p/shared/checkpoint/mistralai/Mistral-7B-Instruct-v0.2"
    BASE_MODEL_PATH = "/ocean/projects/cis250219p/shared/checkpoint/biomistralai/BioMistral/BioMistral-7B"

    # output directory for trained model
    OUTPUT_DIR = "/ocean/projects/cis250219p/shared/asc_checkpoint"

    # model
    flag = "ASC_Phase2" 

    # =============================================================================
    # CONFIG OVERRIDES
    # =============================================================================
    
    # override training config
    TRAINING_CONFIG['epoch'] = NUM_EPOCHS
    # override output directory
    MODEL_CONFIG['checkpoint_dir'] = OUTPUT_DIR
    # override base model path
    model_name = BASE_MODEL_PATH

    # =============================================================================
    # SETUP / WANDB
    # =============================================================================

    use_ddp, rank, world_size, local_rank = setup_ddp()

    if use_ddp:
        print(f"Using DDP with {world_size} processes, rank {rank}, local_rank {local_rank}")
        if rank == 0:
            print(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        print(f"Available GPUs: {torch.cuda.device_count()}")

    if not use_ddp or rank == 0:
        # wandb
        if 'wandb_token' in API_CONFIG:
            wandb.login(key=API_CONFIG['wandb_token'])

        run_name = f'ASC_Train_{flag}_Ep{NUM_EPOCHS}'

        wandb.init(
            project="IDL_11785_group6",
            name=run_name,
            config={
                "base_model_path": BASE_MODEL_PATH,
                "data_path": DATA_PATH,
                "output_dir": OUTPUT_DIR,
                "training_config": TRAINING_CONFIG,
                "lora_config": LORA_CONFIG,
                "train_test_split": TRAIN_RATIO
            }
        )

    # =============================================================================
    # DATA LOADING (ASC Custom Data)
    # =============================================================================
    
    if rank == 0:
        print(f"Loading dataset from: {DATA_PATH}")

    # Load dataset
    full_dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    dataset_split = full_dataset.train_test_split(train_size=TRAIN_RATIO, seed=42)
    
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']

    train_dataset = train_dataset.with_format("torch")
    eval_dataset = eval_dataset.with_format("torch")

    if rank == 0:
        print(f"Data loaded. Train size: {len(train_dataset)}, Test size: {len(eval_dataset)}")
        print("Sample data:", train_dataset[0])

    if use_ddp:
        dist.barrier()

    # =============================================================================
    # DPO Training 
    # =============================================================================

    sample_texts = []
    for i in range(min(10, len(eval_dataset))):
        sample_texts.append(eval_dataset[i]['prompt'])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Initialize DPO Trainer Wrapper
    dpo_wrapper = DPOTrainerWrapper(
        model_name=model_name,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_config=TRAINING_CONFIG,
        lora_config=LORA_CONFIG,
        model_config=MODEL_CONFIG,
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        checkpoint_dir=OUTPUT_DIR,
        checkpoint_dir2=None 
    )

    dpo_wrapper.create_trainer()
    
    if rank == 0:
        print("Starting training...")
        
    dpo_wrapper.train(add_kl_callback=True, sample_texts=sample_texts)
    dpo_wrapper.save_checkpoint()
    
    # Merge and save model only on rank 0
    if not use_ddp or rank == 0:
        print(f"Merging and saving model to {OUTPUT_DIR}...")
        try:
            dpo_wrapper.merge_and_save_model()
            print("Train & Save successfully!")
        except Exception as e:
            print(f"Error during merge/save: {e}")
    
    if use_ddp:
        dist.barrier()
        print(f"Rank {rank}: Ready to destroy process group")
        dist.destroy_process_group()
    
    if not use_ddp or rank == 0:
        wandb.finish()

if __name__ == "__main__":
    main()