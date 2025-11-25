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
from dataset import get_hh
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from model.dpo_train import DPOTrainerWrapper
from model.dpo_eval import evaluate

import warnings
warnings.filterwarnings("ignore")



def main():
    LLM_model = MODEL_CONFIG['model_name']
    flag = MODEL_CONFIG['flag']

    if LLM_model == 'mistral':
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    elif LLM_model == 'qwen':
        model_name = "Qwen/Qwen2.5-7B-Instruct"
    elif LLM_model == 'llama':
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif LLM_model == 'biomistral':
        model_name = "BioMistral/BioMistral-7B"

    # =============================================================================
    # SETUP / WANDB
    # =============================================================================

    use_ddp, rank, world_size, local_rank = setup_ddp()

    if use_ddp:
        print(f"Using DDP with {world_size} processes, rank {rank}, local_rank {local_rank}")
        if rank == 0:
            print(f"Available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


    if not use_ddp or rank == 0:
        wandb.login(key=API_CONFIG['wandb_token'])

        epochs = TRAINING_CONFIG['epoch']
        run_name = f'ASC_{model_name}_{epochs}_{flag}'

        wandb.init(
            project="IDL_11785_group6",
            name=run_name,
            config={
                "model_name": MODEL_CONFIG['model_name'],
                "new_model": MODEL_CONFIG['checkpoint_dir'],
                "training_config": TRAINING_CONFIG,
                "lora_config": LORA_CONFIG,
                "num_epochs": TRAINING_CONFIG['epoch'],
                "learning_rate": TRAINING_CONFIG['learning_rate'],
                "batch_size": TRAINING_CONFIG['per_device_train_batch_size'],
            }
        )

    # =============================================================================
    # 
    # =============================================================================

    if use_ddp:
        if rank == 0:
            train_dataset = get_hh("train", sanity_check=False, flag=flag)
            eval_dataset = get_hh("test", sanity_check=False, flag=flag)
        if dist.is_initialized():
            # Synchronize all processes
            dist.barrier()
        if rank != 0:
            train_dataset = get_hh("train", sanity_check=False, flag=flag)
            eval_dataset = get_hh("test", sanity_check=False, flag=flag)
    else:
        train_dataset = get_hh("train", sanity_check=False, flag=flag)
        eval_dataset = get_hh("test", sanity_check=False, flag=flag)

    eval_dataset = eval_dataset.select(range(1000))

    train_dataset = train_dataset.with_format("torch")
    eval_dataset = eval_dataset.with_format("torch")



    # =============================================================================
    # DPO Training 
    # =============================================================================

    sample_texts = []
    for i in range(min(100, len(eval_dataset))):
        sample_texts.append(eval_dataset[i]['prompt'])


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
        checkpoint_dir=MODEL_CONFIG.get('checkpoint_dir'),
        checkpoint_dir2=MODEL_CONFIG.get('checkpoint_dir2')
    )

    dpo_wrapper.create_trainer()  ## // 확인 중
    dpo_wrapper.train(add_kl_callback=True, sample_texts=sample_texts)
    dpo_wrapper.save_checkpoint()
    
    if not use_ddp or rank == 0:
        dpo_wrapper.merge_and_save_model()
    
    print("Train & Save successfully!")
    
    if use_ddp:
        dist.barrier()
        print(f"Rank {rank}: Ready to destroy process group")
        dist.destroy_process_group()
        print(f"Rank {rank}: Process group destroyed")
    
    if not use_ddp or rank == 0:
        wandb.finish()

    # =============================================================================
    # DPO Evaluate 
    # =============================================================================

    if LLM_model == 'mistral':
        model_name = "/ocean/projects/cis250219p/shared/checkpoint2/mistralai/Mistral-7B-Instruct-v0.2"
    # elif LLM_model == 'qwen':
    #     model_name = "Qwen/Qwen2.5-7B-Instruct"
    # elif LLM_model == 'llama':
    #     model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Only rank 0 should run evaluation to avoid DDP conflicts
    if not use_ddp or rank == 0:
        print("\n" + "="*80)
        print("Starting Model Evaluation")
        print("="*80 + "\n")

        toxicity_score, ori_toxicity_score = evaluate(model_name=model_name, model_config=MODEL_CONFIG)

        print(f"Original Model Toxicity:      {ori_toxicity_score:.4f}")
        print(f"Trained DPO Model Toxicity:   {toxicity_score:.4f}")
    else:
        print(f"Rank {rank}: Skipping evaluation (only rank 0 evaluates)")
        

if __name__ == "__main__":
    main()
