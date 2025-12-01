"""
Advanced Model Comparison Tool - Evaluate models separately and generate comparison reports
Can skip base model and only compare trained models
"""

import subprocess
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# ============================================================================
# Configuration Section - Only modify here
# ============================================================================

# Model path configuration
MODELS = {
    'base': {
        'path': 'mistralai/Mistral-7B-Instruct-v0.2',
        'evaluate': False,  # Set to False to skip base model
    },
    'M_LT': {
        'path': '/ocean/projects/cis250219p/shared/checkpoint_lt/BioMistral/BioMistral-7B',
        'evaluate': True,  # Set to True to evaluate this model
    },
    'M_MT': {
        'path': '/ocean/projects/cis250219p/shared/checkpoint_mt/BioMistral/BioMistral-7B',
        'evaluate': True, 
    },
    'M_ASC': {
        'path': '/ocean/projects/cis250219p/shared/checkpoint_asc/BioMistral-7B',
        'evaluate': True,
    },
}

# Evaluation configuration
CONFIG = {
    'model_name': 'biomistral',
    'use_gemini': True,        # Whether to use Gemini judge (requires API key and costs money)
    'metrics': ['all'],  # Metrics to evaluate
    
    # Example configurations:
    # 'metrics': ['toxicity'],                           # Only toxicity (fastest)
    # 'metrics': ['toxicity', 'safety'],                 # Toxicity + safety
    # 'metrics': ['toxicity', 'safety', 'helpfulness'],  # Three metrics
    # 'metrics': ['all'],                                # All metrics (slowest)
    
    'output_base_dir': '/ocean/projects/cis250219p/shared/evaluation_results',  # Output directory
}

# ============================================================================
# Main functionality code
# ============================================================================

class ModelComparator:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(CONFIG['output_base_dir']) / f"comparison_{self.timestamp}"
        self.results = {}
    
    def setup_output_dir(self):
        """Create output directory structure"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nV Created output directory: {self.output_dir}")
    
    def validate_model(self, model_name: str, model_info: Dict) -> bool:
        """Validate single model configuration"""
        if not model_info['evaluate']:
            print(f"X  {model_name}: Set to skip")
            return False
        
        path = model_info['path']
        if path is None:
            print(f"X  {model_name}: No path configured, skipping")
            return False
        
        # HuggingFace model
        if not path.startswith('/'):
            print(f"V {model_name}: {path} (HuggingFace)")
            return True
        
        # Local path
        if Path(path).exists():
            print(f"V {model_name}: {path}")
            return True
        else:
            print(f"X {model_name}: Path does not exist - {path}")
            return False
    
    def evaluate_single_model(self, model_name: str, model_path: str) -> Optional[Dict]:
        """Evaluate a single model"""
        print(f"\n{'='*80}")
        print(f"Evaluating {model_name}")
        print(f"{'='*80}")
        
        # Build command
        model_output_dir = self.output_dir / model_name
        model_output_dir.mkdir(exist_ok=True)
        
        cmd = [
            'python', 'run_evaluation.py',
            '--model_path', model_path,
            '--model_name', CONFIG['model_name'],
            '--output_dir', str(model_output_dir),
            '--metrics', *CONFIG['metrics'],
        ]
        
        if not CONFIG['use_gemini']:
            cmd.append('--no_gemini')
        
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            # Execute evaluation
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Read results
            result_files = list(model_output_dir.glob('evaluation_results_*.json'))
            if result_files:
                with open(result_files[0], 'r') as f:
                    data = json.load(f)
                print(f"V {model_name} evaluation completed")
                return data
            else:
                print(f"X  {model_name} result file not found")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"X {model_name} evaluation failed: {e}")
            print(f"Error output: {e.stderr}")
            return None
    
    def generate_comparison_report(self):
        """Generate comparison report"""
        print(f"\n{'='*80}")
        print("Generating comparison report")
        print(f"{'='*80}")
        
        if not self.results:
            print("X  No results to compare")
            return
        
        # Extract metrics for comparison
        comparison_data = {}
        
        for metric in CONFIG['metrics']:
            if metric == 'toxicity':
                metric_key = 'general_toxicity'
            elif metric == 'safety':
                metric_key = 'medical_safety'
            elif metric == 'helpfulness':
                metric_key = 'medical_helpfulness'
            elif metric == 'quality':
                metric_key = 'response_quality'
            else:
                continue
            
            comparison_data[metric] = {}
            
            for model_name, data in self.results.items():
                if metric_key in data:
                    summary = data[metric_key].get('summary', {})
                    comparison_data[metric][model_name] = summary
        
        # Save as JSON
        report_file = self.output_dir / 'comparison_report.json'
        with open(report_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f"V  Saved comparison report: {report_file}")
        
        # Generate CSV files
        for metric, metric_data in comparison_data.items():
            if metric_data:
                df = pd.DataFrame(metric_data).T
                csv_file = self.output_dir / f'comparison_{metric}.csv'
                df.to_csv(csv_file)
                print(f"V  Saved {metric} comparison table: {csv_file}")
                
                # Print to terminal
                print(f"\n{metric.upper()} Comparison:")
                print("-" * 80)
                print(df.to_string())
    
    def run(self):
        """Execute complete comparison workflow"""
        print("\n" + "="*80)
        print("Advanced Model Comparison Tool")
        print("="*80)
        
        # Setup output directory
        self.setup_output_dir()
        
        # Validate models
        print("\nValidating model configurations...")
        print("-" * 80)
        models_to_evaluate = {}
        for model_name, model_info in MODELS.items():
            if self.validate_model(model_name, model_info):
                models_to_evaluate[model_name] = model_info['path']
        
        if not models_to_evaluate:
            print("\nX  No models to evaluate!")
            return 1
        
        # Confirm execution
        print(f"\nWill evaluate the following models: {', '.join(models_to_evaluate.keys())}")
        print(f"Evaluation metrics: {', '.join(CONFIG['metrics'])}")
        print(f"Use Gemini: {'Yes' if CONFIG['use_gemini'] else 'No'}")
        print("\nPress Enter to continue, or Ctrl+C to cancel...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nCancelled")
            return 0
        
        # Evaluate each model
        for model_name, model_path in models_to_evaluate.items():
            result = self.evaluate_single_model(model_name, model_path)
            if result:
                self.results[model_name] = result
        
        # Generate comparison report
        if self.results:
            self.generate_comparison_report()
            
            print(f"\n{'='*80}")
            print("V  All evaluations completed!")
            print(f"{'='*80}")
            print(f"\nResults saved in: {self.output_dir}")
            print("\nGenerated files:")
            print(f"  📊 comparison_report.json - Full comparison report")
            for metric in CONFIG['metrics']:
                csv_file = self.output_dir / f'comparison_{metric}.csv'
                if csv_file.exists():
                    print(f"  📈 comparison_{metric}.csv - {metric} comparison table")
            
            for model_name in self.results.keys():
                print(f"  📁 {model_name}/ - {model_name} detailed results")
            
            print(f"\nQuick view comparison results:")
            for metric in CONFIG['metrics']:
                csv_file = self.output_dir / f'comparison_{metric}.csv'
                if csv_file.exists():
                    print(f"  cat {csv_file}")
            
            return 0
        else:
            print("\nX  All model evaluations failed")
            return 1


def main():
    comparator = ModelComparator()
    return comparator.run()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(0)