import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ================= Configuration =================
# 1. Baseモデル（学習の元になったモデル）
BASE_MODEL_PATH = "/ocean/projects/cis250219p/shared/checkpoint_lt/BioMistral/BioMistral-7B"

# 2. 学習済みアダプタ（今 ls で確認した場所）
ADAPTER_PATH = "/ocean/projects/cis250219p/shared/checkpoint_asc"

# 3. 最終保存先（ちゃんとフォルダを切って保存します）
OUTPUT_DIR = "/ocean/projects/cis250219p/shared/checkpoint_asc/BioMistral/BioMistral-7B"
# =================================================

def main():
    print(f"Loading Base Model from: {BASE_MODEL_PATH}")
    # Baseモデルをロード
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Tokenizerをロード
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH) # アダプタ保存時にtokenizerも保存されているため

    print(f"Loading LoRA Adapter from: {ADAPTER_PATH}")
    # Baseモデルにアダプタを装着
    model_to_merge = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    print("Merging models...")
    # マージ実行（これで完全なモデルになります）
    merged_model = model_to_merge.merge_and_unload()
    
    print(f"Saving final merged model to: {OUTPUT_DIR}")
    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Done! You can now use this model for evaluation.")

if __name__ == "__main__":
    main()