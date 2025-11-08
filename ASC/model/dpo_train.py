import os
import gc
import torch
import transformers
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from typing import Optional, List
from pathlib import Path


class DPOTrainerWrapper:
    
    def __init__(
        self,
        model_name: str,
        tokenizer: AutoTokenizer,
        train_dataset,
        eval_dataset,
        training_config: dict,
        lora_config: dict,
        model_config: dict,
        use_ddp: bool = False,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        checkpoint_dir: Optional[str] = None,
        checkpoint_dir2: Optional[str] = None
    ):

        self.model_name = model_name
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_config = training_config
        self.lora_config = lora_config
        self.model_config = model_config
        self.use_ddp = use_ddp
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir2 = checkpoint_dir2
        
        self.model = None
        self.ref_model = None
        self.dpo_trainer = None
        
    def setup_models(self):

        print(f"Loading models from {self.model_name}...")
        
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map=None if self.use_ddp else "auto",
            low_cpu_mem_usage=True,
            attn_implementation='sdpa',
            trust_remote_code=True
        )
        self.model.config.use_cache = False
        
        if hasattr(self.model.config, 'pad_token_id'):
            self.model.config.pad_token_id = pad_token_id
        if hasattr(self.model.config, 'eos_token_id'):
            self.model.config.eos_token_id = eos_token_id
        if hasattr(self.model.config, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            self.model.config.bos_token_id = self.tokenizer.bos_token_id
        # DDP: Move model to the appropriate GPU
        if self.use_ddp:
            print(f"Rank {self.rank}: Moving model to cuda:{self.local_rank}")
            self.model = self.model.to(f'cuda:{self.local_rank}')
            # Enable gradient checkpointing for DDP
            if hasattr(self.model, 'enable_input_require_grads'):
                self.model.enable_input_require_grads()
                
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map=None if self.use_ddp else "auto",
            trust_remote_code=True
        )
        self.ref_model.config.use_cache = False
        
        # Align reference model config with tokenizer tokens
        if hasattr(self.ref_model.config, 'pad_token_id'):
            self.ref_model.config.pad_token_id = pad_token_id
        if hasattr(self.ref_model.config, 'eos_token_id'):
            self.ref_model.config.eos_token_id = eos_token_id
        if hasattr(self.ref_model.config, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            self.ref_model.config.bos_token_id = self.tokenizer.bos_token_id
        
        # DDP: Move reference model to the appropriate GPU
        if self.use_ddp:
            print(f"Rank {self.rank}: Moving ref_model to cuda:{self.local_rank}")
            self.ref_model = self.ref_model.to(f'cuda:{self.local_rank}')
            
        print("Models loaded successfully!")
        

    def setup_training_args(self):

        if self.use_ddp and not dist.is_initialized():
            print("Warning: DDP environment detected but not initialized. Re-initializing...")
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.world_size,
                rank=self.rank
            )
        
        base_args = {
            'per_device_train_batch_size': self.training_config['per_device_train_batch_size'],
            'gradient_accumulation_steps': self.training_config['gradient_accumulation_steps'],
            'gradient_checkpointing': True,
            'learning_rate': self.training_config['learning_rate'],
            'lr_scheduler_type': "cosine",
            'num_train_epochs': self.training_config.get('epoch', 1),
            'save_strategy': "epoch",
            'logging_steps': 100,
            'output_dir': self.checkpoint_dir,
            'optim': "paged_adamw_32bit",
            'warmup_steps': 50,
            'bf16': torch.cuda.is_bf16_supported(),
            'beta': 0.1,
            'max_prompt_length': self.training_config['max_prompt_length'],
            'max_length': self.training_config['max_length'],
            'report_to': "wandb" if (not self.use_ddp or self.rank == 0) else None,
            'dataloader_pin_memory': True,
            'dataloader_num_workers': min(4, os.cpu_count() or 4),
            'dataloader_drop_last': True,
            'remove_unused_columns': False,
        }
        
        if self.use_ddp:
            base_args.update({
                'ddp_backend': "nccl",
                'ddp_find_unused_parameters': False,
                'local_rank': self.local_rank,
                'ddp_timeout': 1800,
            })


        peft_config = LoraConfig(
            r=self.lora_config['r'],
            lora_alpha=self.lora_config['alpha'],
            lora_dropout=self.lora_config['dropout'],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
        )
            
        return DPOConfig(**base_args), peft_config
    

    def create_trainer(self):
   
        self.setup_models()
        training_args, peft_config = self.setup_training_args()
        
        self.dpo_trainer = DPOTrainer(
            self.model,
            ref_model=None,  
            args=training_args,
            train_dataset=self.train_dataset, 
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            peft_config=peft_config
        )
        

    #################################### //11.05 다시확인
    def calculate_kl_divergence(self, sample_texts: List[str], device="cuda"):
        """
        Average KL-divergence across all samples
        """
            
        self.model.eval()
        self.ref_model.eval()
        
        total_kl_div = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for text in sample_texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                
                # Handle device placement
                if hasattr(self.model, 'device'):
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                else:
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                
                policy_outputs = self.model(**inputs)
                reference_outputs = self.ref_model(**inputs)
                
                policy_logits = policy_outputs.logits
                reference_logits = reference_outputs.logits
                
                policy_log_probs = F.log_softmax(policy_logits, dim=-1)
                reference_probs = F.softmax(reference_logits, dim=-1)
                
                reference_probs = reference_probs + 1e-8
                reference_probs = reference_probs / reference_probs.sum(dim=-1, keepdim=True)
                
                kl_div = F.kl_div(policy_log_probs, reference_probs, reduction='batchmean', log_target=False)
                
                if not torch.isnan(kl_div) and not torch.isinf(kl_div):
                    total_kl_div += kl_div.item()
                    num_samples += 1
        
        return total_kl_div / num_samples if num_samples > 0 else 0.0
    

    def create_kl_callback(self, sample_texts: List[str]):
        """
        Create a callback for logging KL-divergence during training.
        """
        ref_model = self.ref_model
        tokenizer = self.tokenizer
        calc_kl_fn = self.calculate_kl_divergence
        
        class KLDivergenceCallback(TrainerCallback):
            def __init__(self, reference_model, tokenizer, sample_texts, calc_kl_fn):
                self.reference_model = reference_model
                self.tokenizer = tokenizer
                self.sample_texts = sample_texts
                self.calc_kl_fn = calc_kl_fn
                
            def on_epoch_end(self, args, state, control, model=None, **kwargs):
                if model is not None:
                    # Note: We'd need to pass the wrapper instance to properly calculate KL
                    # For now, just print a message
                    print(f"Epoch {state.epoch} completed")
        
        return KLDivergenceCallback(ref_model, tokenizer, sample_texts, calc_kl_fn)
        #################################### //11.05 다시확인


    def train(self, add_kl_callback: bool = False, sample_texts: Optional[List[str]] = None):

        print(">>>>>>>>> Starting DPO training...")
        
        if self.use_ddp:
            print(f"Using DDP with {self.world_size} processes for training")
            torch.cuda.empty_cache()
            gc.collect()
        elif torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            torch.cuda.empty_cache()
            gc.collect()
        
        if add_kl_callback and sample_texts and (not self.use_ddp or self.rank == 0):
            kl_callback = self.create_kl_callback(sample_texts)
            self.dpo_trainer.add_callback(kl_callback)
        
        self.dpo_trainer.train()
        
        print("Training completed!")
        
        # Synchronize all processes before saving
        if self.use_ddp:
            dist.barrier()
    
    
    def save_checkpoint(self):
        """Save the trained LoRA adapter checkpoint."""
        
        print("\n>>>>>>>>> Starting DPO checkpoint save...")

        if not self.use_ddp or self.rank == 0:

            if self.model_config['flag'] == "MT":
                os.makedirs(self.checkpoint_dir2, exist_ok=True)
                self.dpo_trainer.model.save_pretrained(self.checkpoint_dir2)
                self.tokenizer.save_pretrained(self.checkpoint_dir2)
                print(f"Model checkpoint saved to: {self.checkpoint_dir2}")
                
                # Wait for all processes to finish
                if self.use_ddp:
                    dist.barrier()
            else:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                self.dpo_trainer.model.save_pretrained(self.checkpoint_dir)
                self.tokenizer.save_pretrained(self.checkpoint_dir)
                print(f"Model checkpoint saved to: {self.checkpoint_dir}")
                
                # Wait for all processes to finish
                if self.use_ddp:
                    dist.barrier()

           
        print(">>>>>>>>>  Merging adapter with base model & save...")
        
        del self.dpo_trainer, self.model
        gc.collect()
        torch.cuda.empty_cache()
        
        # Reload base model in FP16
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            return_dict=True,
            torch_dtype=torch.float16,
        )
        
        # Realign tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if hasattr(base_model.config, 'pad_token_id'):
            base_model.config.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        if hasattr(base_model.config, 'eos_token_id'):
            base_model.config.eos_token_id = self.tokenizer.eos_token_id
        
        # Merge adapter with base model
        if self.model_config['flag']=="MT":
            model = PeftModel.from_pretrained(base_model, self.checkpoint_dir2)
            model = model.merge_and_unload()
        else:
            model = PeftModel.from_pretrained(base_model, self.checkpoint_dir)
            model = model.merge_and_unload()
        
        # Save merged model
        if self.model_config['flag']=="MT":
            save_path = Path(self.checkpoint_dir2) /  self.model_name 
        else:
            save_path = Path(self.checkpoint_dir) /  self.model_name
        model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print(f"Merged model saved to: {save_path}")
        
        return model

