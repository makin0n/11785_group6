from datasets import load_dataset
import os

cache_dir = "/ocean/projects/cis250219p/shared/huggingface_cache"

os.makedirs(cache_dir, exist_ok=True)

target_dir = "harmless-base"

print(f"Downloading train split for {target_dir} to {cache_dir}...")

load_dataset("Anthropic/hh-rlhf", split="train", data_dir=target_dir, cache_dir=cache_dir)

print(f"Downloading test split for {target_dir} to {cache_dir}...")
load_dataset("Anthropic/hh-rlhf", split="test", data_dir=target_dir, cache_dir=cache_dir)

print(f"Download complete. Data saved in: {cache_dir}")