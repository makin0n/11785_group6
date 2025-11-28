import sys
import os
from config import MODEL_CONFIG
from model.dpo_eval import evaluate

def main():
    flag = sys.argv[1] if len(sys.argv) > 1 else "LT"
    
    BASE_MODEL_NAME = "BioMistral/BioMistral-7B"
    
    if flag == "MT":
        ckpt_dir = MODEL_CONFIG.get('checkpoint_dir2') 
    else:
        ckpt_dir = MODEL_CONFIG.get('checkpoint_dir')

    trained_model_path = os.path.join(ckpt_dir, "BioMistral", "BioMistral-7B")
    
    if not os.path.exists(trained_model_path):
        print(f"Error: Trained model not found at {trained_model_path}")
        return

    print("="*60)
    print(f"Running Evaluation for {flag} Model")
    print(f"Trained Path: {trained_model_path}")
    print(f"Base Model:   {BASE_MODEL_NAME}")
    print("="*60)

    # 評価実行
    trained_score, original_score = evaluate(trained_model_path, BASE_MODEL_NAME)

    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Original Model Toxicity: {original_score:.4f}")
    print(f"Trained Model ({flag}) Toxicity: {trained_score:.4f}")
    
    diff = trained_score - original_score
    print(f"Difference: {diff:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()