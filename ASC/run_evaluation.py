import os
import gc
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from eval_modules import GPT4Judge, load_evaluation_data

# =============================================================================
# CONFIGURATION
# =============================================================================

# OpenAI API Key (環境変数から取得または直接入力)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

# 評価する4つのモデルのパス定義
# ※ 実際のパスに合わせて書き換えてください
MODEL_PATHS = {
    "Base_Model": "/shared/checkpoint/mistralai/Mistral-7B-Instruct-v0.2", # Base
    "Train_Model_1": "/shared/checkpoint/train_model_ver1",                # Train.py model 1 (e.g., intermediate)
    "Train_Model_2": "/shared/checkpoint/train_model_ver2",                # Train.py model 2 (e.g., final merged)
    "ASC_Model": "/shared/asc_checkpoint"                                  # ASC Train model
}

# テスト実行時のサンプル数（コスト削減のため少なめに設定、本番は増やす）
NUM_SAMPLES = 20 

# 出力ファイル名
OUTPUT_CSV_SUMMARY = "evaluation_summary.csv"
OUTPUT_CSV_DETAILS = "evaluation_details.csv"

# =============================================================================
# INFERENCE FUNCTION
# =============================================================================

def generate_responses(model_path, datasets):
    """
    指定されたモデルをロードし、全データセットに対して推論を行う
    """
    print(f"\nLoading Model: {model_path} ...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return {}

    results = {}

    # 各データセットに対して推論
    for benchmark_name, prompts in datasets.items():
        print(f"Generating for {benchmark_name}...")
        responses = []
        
        for prompt in tqdm(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=128, 
                    do_sample=True, 
                    temperature=0.7
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # プロンプト部分を除去して応答のみ抽出
            response_only = generated_text[len(prompt):].strip()
            responses.append(response_only)
            
        results[benchmark_name] = responses

    # メモリ解放
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # 1. データの準備
    datasets = load_evaluation_data(num_samples=NUM_SAMPLES)
    judge = GPT4Judge(api_key=OPENAI_API_KEY)

    all_evaluation_records = []

    # 2. モデルごとの評価ループ
    for model_name, model_path in MODEL_PATHS.items():
        print(f"\n{'='*20} Evaluating: {model_name} {'='*20}")
        
        # A. 生成（Inference）
        generated_results = generate_responses(model_path, datasets)
        
        if not generated_results:
            print(f"Skipping {model_name} due to load error.")
            continue

        # B. 採点（GPT-4 Judge）
        print(f"Scoring {model_name} with GPT-4...")

        # ---------------------------------------------------------
        # 指標 1: Toxicity (Benchmark: ToxiGen)
        # ---------------------------------------------------------
        if 'toxigen' in generated_results:
            for prompt, response in zip(datasets['toxigen'], generated_results['toxigen']):
                judge_output = judge.evaluate(prompt, response, "toxicity")
                score = judge.parse_score(judge_output)
                
                all_evaluation_records.append({
                    "Model": model_name,
                    "Metric": "Toxicity",
                    "Benchmark": "ToxiGen",
                    "Prompt": prompt,
                    "Response": response,
                    "Score": score,
                    "Judge_Output": judge_output
                })

        # ---------------------------------------------------------
        # 指標 2: Domain-Specific Toxicity & Misuse (MedSafety)
        # ---------------------------------------------------------
        if 'med_safety' in generated_results:
            for prompt, response in zip(datasets['med_safety'], generated_results['med_safety']):
                judge_output = judge.evaluate(prompt, response, "medical_safety")
                score = judge.parse_score(judge_output)
                
                all_evaluation_records.append({
                    "Model": model_name,
                    "Metric": "Domain Toxicity/Misuse",
                    "Benchmark": "MedSafety",
                    "Prompt": prompt,
                    "Response": response,
                    "Score": score,
                    "Judge_Output": judge_output
                })
                
                # 指標 5: Robustness (Adversarial Prompts)
                # RobustnessはMedSafetyに対する安全な回答の一貫性として評価
                # ここでは同じスコアを用いますが、集計時に分けて扱います
                all_evaluation_records.append({
                    "Model": model_name,
                    "Metric": "Robustness",
                    "Benchmark": "MedSafety(Adversarial)",
                    "Prompt": prompt,
                    "Response": response,
                    "Score": score, # 高いほどRobust
                    "Judge_Output": judge_output
                })

        # ---------------------------------------------------------
        # 指標 3: Domain Helpfulness (MedQA)
        # ---------------------------------------------------------
        if 'medqa' in generated_results:
            for prompt, response in zip(datasets['medqa'], generated_results['medqa']):
                # Helpfulness
                judge_output = judge.evaluate(prompt, response, "helpfulness")
                score = judge.parse_score(judge_output)
                
                all_evaluation_records.append({
                    "Model": model_name,
                    "Metric": "Domain Helpfulness",
                    "Benchmark": "MedQA",
                    "Prompt": prompt,
                    "Response": response,
                    "Score": score,
                    "Judge_Output": judge_output
                })

                # 指標 4: Response Quality & Coherence (Over-Refusal check)
                # 同じMedQAの回答を使ってCoherenceもチェック
                judge_output_coh = judge.evaluate(prompt, response, "coherence")
                score_coh = judge.parse_score(judge_output_coh)
                
                all_evaluation_records.append({
                    "Model": model_name,
                    "Metric": "Quality & Coherence",
                    "Benchmark": "MedQA",
                    "Prompt": prompt,
                    "Response": response,
                    "Score": score_coh,
                    "Judge_Output": judge_output_coh
                })

    # 3. 結果の保存
    print("\nSaving results...")
    
    # 詳細データの保存
    df_details = pd.DataFrame(all_evaluation_records)
    df_details.to_csv(OUTPUT_CSV_DETAILS, index=False)
    print(f"Detailed results saved to {OUTPUT_CSV_DETAILS}")

    # サマリー（平均点）の作成
    if not df_details.empty:
        df_summary = df_details.groupby(['Model', 'Metric'])['Score'].agg(['mean', 'std', 'count']).reset_index()
        df_summary.to_csv(OUTPUT_CSV_SUMMARY, index=False)
        print(f"Summary results saved to {OUTPUT_CSV_SUMMARY}")
        print("\n=== Summary Table ===")
        print(df_summary)
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()