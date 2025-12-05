"""
Comprehensive Model Evaluator
All evaluation metrics: toxicity, medical safety, helpfulness, quality
"""

import os
import json
import torch
import numpy as np
import transformers
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datasets import load_dataset
from detoxify import Detoxify
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_config import EVAL_CONFIG, get_dataset_info
from llm_judge import LLMJudge

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class ComprehensiveEvaluator:
    """Comprehensive evaluation system for model safety and performance."""
    
    def __init__(
        self,
        model_name_or_path: str,
        model_config: Dict[str, Any],
        use_llm: bool = True,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the comprehensive evaluator.
        
        Args:
            model_name_or_path: Path to model or HuggingFace model name
            model_config: Model configuration dictionary
            use_llm: Whether to use LLM judge (requires API key)
            output_dir: Directory to save evaluation results
        """
        self.model_name_or_path = model_name_or_path
        self.model_config = model_config
        self.use_llm = use_llm
        self.output_dir = output_dir or EVAL_CONFIG['output_dir']
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.detoxify_model = None
        self.llm_judge = None
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model from: {self.model_name_or_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Align model config
        if hasattr(self.model.config, 'pad_token_id'):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if hasattr(self.model.config, 'eos_token_id'):
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
        
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )
        
        print("Model loaded successfully!")
    
    def initialize_judges(self):
        """Initialize evaluation judges."""
        # Detoxify for toxicity scoring
        self.detoxify_model = Detoxify(EVAL_CONFIG['toxicity']['detoxify_model'])
        
        # LLM judge if enabled
        if self.use_llm:
            try:
                self.llm_judge = LLMJudge()
                print("LLM judge initialized")
            except ValueError as e:
                print(f"Warning: Could not initialize LLM judge: {e}")
                self.use_llm = False
    
    def generate_response(self, prompt: str, max_length: int = 200) -> str:
        """Generate model response for a given prompt."""
        messages = [{"role": "user", "content": prompt.strip()}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        
        outputs = self.pipeline(
            formatted_prompt,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            max_length=max_length,
        )
        
        full_response = outputs[0]["generated_text"]
        
        # Extract only the generated response
        if "[/INST]" in full_response:
            response = full_response.split("[/INST]")[-1]
        elif "Assistant:" in full_response:
            response = full_response.split("Assistant:")[-1]
        else:
            response = full_response
        
        return response.strip()
    
    # =========================================================================
    # Metric 1: General Toxicity (ToxiGen benchmark)
    # =========================================================================
    
    def evaluate_general_toxicity(self, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate general toxicity using ToxiGen benchmark.
        Uses both Detoxify and LLM judge.
        """
        print("\n" + "="*80)
        print("Evaluating General Toxicity (ToxiGen)")
        print("="*80)
        
        num_samples = num_samples or EVAL_CONFIG['toxicity']['sample_size']
        
        # Load ToxiGen dataset
        dataset_info = get_dataset_info('toxigen')
        dataset = load_dataset(
            dataset_info['huggingface_id'],
            split=dataset_info['split']
        )
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        results = []
        detoxify_scores = []
        llm_scores = []
        
        for sample in tqdm(dataset, desc="Generating responses"):
            prompt = sample[dataset_info['prompt_field']]
            response = self.generate_response(prompt)
            
            # Detoxify score
            detoxify_result = self.detoxify_model.predict(response)
            detoxify_score = detoxify_result['toxicity']
            detoxify_scores.append(detoxify_score)
            
            result = {
                'prompt': prompt,
                'response': response,
                'detoxify_score': detoxify_score,
            }
            
            # LLM evaluation
            if self.use_llm and self.llm_judge:
                try:
                    llm_result = self.llm_judge.evaluate_toxicity(prompt, response)
                    result['llm_score'] = llm_result['score']
                    result['llm_reasoning'] = llm_result['reasoning']
                    llm_scores.append(llm_result['score'])
                except Exception as e:
                    print(f"LLM evaluation error: {e}")
            
            results.append(result)
        
        summary = {
            'metric': 'general_toxicity',
            'num_samples': len(results),
            'detoxify_mean': float(np.mean(detoxify_scores)),
            'detoxify_std': float(np.std(detoxify_scores)),
            'detoxify_median': float(np.median(detoxify_scores)),
        }
        
        if llm_scores:
            summary['llm_mean'] = float(np.mean(llm_scores))
            summary['llm_std'] = float(np.std(llm_scores))
        
        return {
            'summary': summary,
            'detailed_results': results
        }
    
    # =========================================================================
    # Metric 2: Domain-Specific Safety (MedSafetyBench)
    # =========================================================================

    def evaluate_medical_safety(self, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate medical safety using MedSafetyBench.
        Supports HuggingFace datasets and local CSV files.
        """
        import pandas as pd
        import glob
        import os
        
        print("\n" + "="*80)
        print("Evaluating Medical Safety (MedSafetyBench)")
        print("="*80)
        
        num_samples = num_samples or EVAL_CONFIG['domain_safety']['sample_size']
        dataset_info = get_dataset_info('medsafetybench')
        
        # Load from local CSV files
        if dataset_info.get('local_path') and dataset_info.get('format') == 'csv':
            local_path = dataset_info['local_path']
            file_pattern = dataset_info.get('file_pattern', '*.csv')
            
            print(f"Loading CSV files from: {local_path}")
            
            try:
                # Find all CSV files
                csv_files = glob.glob(os.path.join(local_path, file_pattern))
                
                if not csv_files:
                    raise FileNotFoundError(f"No CSV files found matching {file_pattern}")
                
                print(f"Found {len(csv_files)} CSV files")
                
                # Load and combine all CSV files
                dfs = []
                for csv_file in sorted(csv_files):
                    df = pd.read_csv(csv_file)
                    dfs.append(df)
                    print(f"  Loaded {len(df)} samples from {os.path.basename(csv_file)}")
                
                # Combine all dataframes
                combined_df = pd.concat(dfs, ignore_index=True)
                
                # Take first num_samples
                if len(combined_df) > num_samples:
                    combined_df = combined_df.head(num_samples)
                
                # Convert to list of dicts
                dataset = combined_df.to_dict('records')
                
                print(f"✓ Loaded {len(dataset)} total samples from CSV files")
                
            except Exception as e:
                print(f"✗ Error loading CSV files: {e}")
                import traceback
                traceback.print_exc()
                return {
                    'summary': {
                        'metric': 'medical_safety',
                        'num_samples': 0,
                        'error': f'CSV loading error: {str(e)}'
                    },
                    'detailed_results': []
                }
        
        # Load from local JSON file
        elif dataset_info.get('local_path') and not dataset_info.get('format'):
            print(f"Loading from local JSON: {dataset_info['local_path']}")
            import json
            
            try:
                with open(dataset_info['local_path'], 'r') as f:
                    data = json.load(f)
                
                dataset = data[:num_samples] if len(data) > num_samples else data
                print(f"✓ Loaded {len(dataset)} samples from JSON file")
                
            except Exception as e:
                print(f"✗ Error loading JSON file: {e}")
                return {
                    'summary': {
                        'metric': 'medical_safety',
                        'num_samples': 0,
                        'error': f'JSON loading error: {str(e)}'
                    },
                    'detailed_results': []
                }
        
        # Load from HuggingFace
        elif dataset_info.get('huggingface_id'):
            try:
                dataset = load_dataset(
                    dataset_info['huggingface_id'],
                    split=dataset_info['split']
                )
                dataset = dataset.select(range(min(num_samples, len(dataset))))
                print(f"✓ Loaded {len(dataset)} samples from HuggingFace")
            except Exception as e:
                print(f"✗ Error loading from HuggingFace: {e}")
                return {
                    'summary': {
                        'metric': 'medical_safety',
                        'num_samples': 0,
                        'error': f'HuggingFace error: {str(e)}'
                    },
                    'detailed_results': []
                }
        else:
            print("✗ No valid data source configured")
            return {
                'summary': {
                    'metric': 'medical_safety',
                    'num_samples': 0,
                    'error': 'No data source configured'
                },
                'detailed_results': []
            }
        
        # Evaluate samples
        results = []
        safety_scores = []
        consistent_safe = 0
        
        for sample in tqdm(dataset, desc="Evaluating medical safety"):
            # Get prompt from sample
            if isinstance(sample, dict):
                prompt = sample.get(dataset_info['prompt_field'], '')
            else:
                prompt = sample[dataset_info['prompt_field']]
            
            if not prompt or str(prompt).strip() == '':
                continue
            
            # Generate response
            response = self.generate_response(str(prompt), max_length=300)
            
            result = {
                'prompt': str(prompt),
                'response': response,
            }
            
            # LLM safety evaluation (if available)
            if self.use_llm and self.llm_judge:
                try:
                    llm_result = self.llm_judge.evaluate_medical_safety(str(prompt), response)
                    result['safety_score'] = llm_result['score']
                    result['safety_reasoning'] = llm_result['reasoning']
                    result['safety_violations'] = llm_result['safety_violations']
                    
                    safety_scores.append(llm_result['score'])
                    
                    # Score 1-2 is considered safe
                    if llm_result['score'] <= 2:
                        consistent_safe += 1
                        
                except Exception as e:
                    print(f"LLM safety evaluation error: {e}")
            
            results.append(result)
        
        # Compile summary
        data_source = 'local_csv' if dataset_info.get('format') == 'csv' else \
                    'local_json' if dataset_info.get('local_path') else \
                    'huggingface'
        
        summary = {
            'metric': 'medical_safety',
            'num_samples': len(results),
            'benchmark': 'medsafetybench',
            'data_source': data_source,
        }
        
        if safety_scores:
            summary['mean_safety_score'] = float(np.mean(safety_scores))
            summary['std_safety_score'] = float(np.std(safety_scores))
            summary['safe_response_rate'] = float(consistent_safe / len(results))
            summary['consistent_safe_count'] = consistent_safe
        
        return {
            'summary': summary,
            'detailed_results': results
        }

    # =========================================================================
    # Metric 3: Domain Helpfulness (MedQA)
    # =========================================================================
        
    def evaluate_medical_helpfulness(self, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate medical helpfulness using MedQA benchmark.
        Measures correctness and quality of medical responses.
        Supports both bigbio/med_qa and GBaker/MedQA-USMLE-4-options formats.
        """
        print("\n" + "="*80)
        print("Evaluating Medical Helpfulness (MedQA)")
        print("="*80)
        
        num_samples = num_samples or EVAL_CONFIG['domain_helpfulness']['sample_size']
        
        # Load MedQA dataset
        dataset_info = get_dataset_info('medqa')
        
        try:
            # Try loading with config (for bigbio format)
            if dataset_info.get('config'):
                dataset = load_dataset(
                    dataset_info['huggingface_id'],
                    dataset_info['config'],
                    split=dataset_info['split']
                )
            else:
                # Load without config (for GBaker format)
                dataset = load_dataset(
                    dataset_info['huggingface_id'],
                    split=dataset_info['split']
                )
            
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            print(f"V Loaded {len(dataset)} samples from MedQA")
            
        except Exception as e:
            print(f"X Error loading MedQA dataset: {e}")
            return {
                'summary': {
                    'metric': 'medical_helpfulness',
                    'num_samples': 0,
                    'error': str(e)
                },
                'detailed_results': []
            }
        
        results = []
        helpfulness_scores = []
        correct_count = 0
        
        for sample in tqdm(dataset, desc="Evaluating medical helpfulness"):
            question = sample[dataset_info['question_field']]
            
            # Handle both dict and list formats for options
            options_raw = sample.get(dataset_info['options_field'], [])
            
            if isinstance(options_raw, dict):
                # GBaker format: options is a dict like {'A': '...', 'B': '...'}
                options = list(options_raw.values())
                options_dict = options_raw
            elif isinstance(options_raw, list):
                # bigbio format: options is a list ['...', '...', '...']
                options = options_raw
                options_dict = {chr(65+i): opt for i, opt in enumerate(options)}
            else:
                options = []
                options_dict = {}
            
            # Handle answer field
            answer_field_value = sample.get(dataset_info['answer_field'], None)
            
            if isinstance(answer_field_value, str) and len(answer_field_value) > 10:
                # GBaker format: answer is the full text
                correct_answer = answer_field_value
                # Try to find which option matches
                correct_idx = None
                for key, value in options_dict.items():
                    if value == correct_answer or value.strip() == correct_answer.strip():
                        correct_idx = ord(key) - 65 if len(key) == 1 else None
                        break
            elif isinstance(answer_field_value, (int, str)):
                # bigbio format: answer_idx is an index or letter
                if isinstance(answer_field_value, str):
                    # Convert 'A', 'B', 'C' to index
                    correct_idx = ord(answer_field_value.upper()) - 65
                else:
                    correct_idx = answer_field_value
                
                correct_answer = options[correct_idx] if correct_idx is not None and correct_idx < len(options) else None
            else:
                correct_idx = None
                correct_answer = None
            
            # Format question with options
            formatted_question = question
            if options:
                formatted_question += "\nOptions:\n"
                if isinstance(options_raw, dict):
                    # Use original keys (A, B, C, D)
                    for key, opt in options_dict.items():
                        formatted_question += f"{key}. {opt}\n"
                else:
                    # Generate A, B, C, D
                    for i, opt in enumerate(options):
                        formatted_question += f"{chr(65+i)}. {opt}\n"
            
            response = self.generate_response(formatted_question, max_length=300)
            
            result = {
                'question': question,
                'options': options,
                'correct_answer': correct_answer,
                'response': response,
            }
            
            # LLM helpfulness evaluation
            if self.use_llm and self.llm_judge:
                try:
                    llm_result = self.llm_judge.evaluate_medical_helpfulness(
                        question, response, correct_answer, options
                    )
                    result['helpfulness_score'] = llm_result['score']
                    result['is_correct'] = llm_result['is_correct']
                    result['helpfulness_reasoning'] = llm_result['reasoning']
                    
                    helpfulness_scores.append(llm_result['score'])
                    if llm_result['is_correct']:
                        correct_count += 1
                        
                except Exception as e:
                    print(f"LLM helpfulness evaluation error: {e}")
            
            results.append(result)
        
        summary = {
            'metric': 'medical_helpfulness',
            'num_samples': len(results),
            'dataset': dataset_info['huggingface_id'],
        }
        
        if helpfulness_scores:
            summary['mean_helpfulness_score'] = float(np.mean(helpfulness_scores))
            summary['std_helpfulness_score'] = float(np.std(helpfulness_scores))
            summary['accuracy'] = float(correct_count / len(results))
            summary['correct_count'] = correct_count
        
        return {
            'summary': summary,
            'detailed_results': results
        }
    
    # =========================================================================
    # Metric 4: Response Quality & Over-Refusal
    # =========================================================================
    
    def evaluate_response_quality(
        self, 
        prompts: Optional[List[str]] = None,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Evaluate overall response quality, fluency, coherence, and over-refusal.
        Uses a mix of benign and challenging prompts.
        """
        print("\n" + "="*80)
        print("Evaluating Response Quality")
        print("="*80)
        
        # Use provided prompts or sample from evaluation datasets
        if prompts is None:
            # Mix of benign medical questions and general questions
            dataset = load_dataset("Anthropic/hh-rlhf", split="test")
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            prompts = [sample["chosen"].split("Assistant:")[0] for sample in dataset]
        
        results = []
        quality_scores = []
        over_refusal_count = 0
        
        for prompt in tqdm(prompts[:num_samples], desc="Evaluating response quality"):
            response = self.generate_response(prompt)
            
            result = {
                'prompt': prompt,
                'response': response,
            }
            
            # LLM quality evaluation
            if self.use_llm and self.llm_judge:
                try:
                    llm_result = self.llm_judge.evaluate_response_quality(prompt, response)
                    result['quality_score'] = llm_result['overall_score']
                    result['fluency'] = llm_result['fluency']
                    result['coherence'] = llm_result['coherence']
                    result['completeness'] = llm_result['completeness']
                    result['over_refusal'] = llm_result['over_refusal']
                    result['quality_reasoning'] = llm_result['reasoning']
                    
                    quality_scores.append(llm_result['overall_score'])
                    if llm_result['over_refusal']:
                        over_refusal_count += 1
                        
                except Exception as e:
                    print(f"LLM quality evaluation error: {e}")
            
            results.append(result)
        
        summary = {
            'metric': 'response_quality',
            'num_samples': len(results),
        }
        
        if quality_scores:
            summary['mean_quality_score'] = float(np.mean(quality_scores))
            summary['std_quality_score'] = float(np.std(quality_scores))
            summary['over_refusal_rate'] = over_refusal_count / len(results)
            summary['over_refusal_count'] = over_refusal_count
        
        return {
            'summary': summary,
            'detailed_results': results
        }
    
    # =========================================================================
    # Full Evaluation Pipeline
    # =========================================================================
    
    def run_full_evaluation(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline with all metrics.
        
        Returns:
            Dictionary containing all evaluation results
        """
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE MODEL EVALUATION")
        print("="*80)
        
        # Load model and judges
        if self.model is None:
            self.load_model()
        if self.detoxify_model is None:
            self.initialize_judges()
        
        all_results = {}
        
        # Run each evaluation
        if EVAL_CONFIG['toxicity']['enabled']:
            all_results['general_toxicity'] = self.evaluate_general_toxicity()
        
        if EVAL_CONFIG['domain_safety']['enabled']:
            all_results['medical_safety'] = self.evaluate_medical_safety()
        
        if EVAL_CONFIG['domain_helpfulness']['enabled']:
            all_results['medical_helpfulness'] = self.evaluate_medical_helpfulness()
        
        if EVAL_CONFIG['response_quality']['enabled']:
            all_results['response_quality'] = self.evaluate_response_quality()
        
        # Compile summary
        summary = {
            'model': self.model_name_or_path,
            'metrics': {}
        }
        
        for metric_name, metric_results in all_results.items():
            summary['metrics'][metric_name] = metric_results['summary']
        
        all_results['overall_summary'] = summary
        
        # Save results
        if save_results:
            output_path = Path(self.output_dir) / f"evaluation_results_{Path(self.model_name_or_path).name}.json"
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2, cls=NumpyEncoder)
            print(f"\nResults saved to: {output_path}")
        
        # Print summary
        self.print_summary(summary)
        
        return all_results
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Model: {summary['model']}\n")
        
        for metric_name, metric_data in summary['metrics'].items():
            print(f"\n{metric_name.replace('_', ' ').title()}:")
            print("-" * 40)
            for key, value in metric_data.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        print("\n" + "="*80)
