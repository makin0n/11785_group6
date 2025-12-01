"""
Compare base, M_LT, M_MT, and M_ASC models in a single run
"""

import os
import json
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from comprehensive_evaluator import ComprehensiveEvaluator
from eval_config import EVAL_CONFIG


class MultiModelEvaluator:
    def __init__(
        self,
        base_model_path: str,
        model_paths: Dict[str, str],
        model_configs: Dict[str, Dict[str, Any]],
        use_gemini: bool = True,
        output_dir: str = None
    ):
        """
        Initialize multi-model evaluator.
        
        Args:
            base_model_path: Path to base model
            model_paths: Dict mapping model names (M_LT, M_MT, M_ASC) to paths
            model_configs: Dict mapping model names to their configs
            use_gemini: Whether to use Gemini judge
            output_dir: Output directory for results
        """
        self.base_model_path = base_model_path
        self.model_paths = {'base': base_model_path, **model_paths}
        self.model_configs = model_configs
        self.use_gemini = use_gemini
        self.output_dir = output_dir or EVAL_CONFIG['output_dir']
        
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(self.output_dir) / f"comparison_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.all_results = {}
    
    def evaluate_all_models(self) -> Dict[str, Any]:
        """Run evaluation on all models."""
        print("\n" + "="*80)
        print("MULTI-MODEL COMPARATIVE EVALUATION")
        print("="*80)
        print(f"Evaluating models: {list(self.model_paths.keys())}")
        print(f"Output directory: {self.output_dir}")
        print("="*80)
        
        for model_name, model_path in self.model_paths.items():
            print(f"\n{'='*80}")
            print(f"Evaluating {model_name.upper()}: {model_path}")
            print(f"{'='*80}")
            
            # Get model config
            model_config = self.model_configs.get(model_name, {})
            
            # Initialize evaluator
            evaluator = ComprehensiveEvaluator(
                model_name_or_path=str(model_path),
                model_config=model_config,
                use_gemini=self.use_gemini,
                output_dir=str(self.output_dir / model_name)
            )
            
            # Run evaluation
            try:
                results = evaluator.run_full_evaluation(save_results=True)
                self.all_results[model_name] = results
                
                # Clean up GPU memory
                del evaluator.model
                del evaluator.pipeline
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                import traceback
                traceback.print_exc()
                self.all_results[model_name] = {'error': str(e)}
        
        # Generate comparison report
        self.generate_comparison_report()
        
        return self.all_results
    
    def generate_comparison_report(self):
        """Generate comparative analysis of all models."""
        print("\n" + "="*80)
        print("GENERATING COMPARISON REPORT")
        print("="*80)
        
        # Extract summaries
        summaries = {}
        for model_name, results in self.all_results.items():
            if 'error' not in results:
                summaries[model_name] = results.get('overall_summary', {}).get('metrics', {})
        
        if not summaries:
            print("No valid results to compare")
            return
        
        # Create comparison tables
        comparison_data = self._create_comparison_tables(summaries)
        
        # Save comparison report
        report_path = self.output_dir / "comparison_report.json"
        with open(report_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        # Save as CSV for easy viewing
        self._save_comparison_csv(comparison_data)
        
        # Print comparison
        self._print_comparison(comparison_data)
        
        print(f"\nComparison report saved to: {report_path}")
    
    def _create_comparison_tables(self, summaries: Dict[str, Dict]) -> Dict[str, Any]:
        """Create structured comparison tables."""
        comparison = {
            'toxicity': {},
            'medical_safety': {},
            'medical_helpfulness': {},
            'response_quality': {}
        }
        
        for model_name, metrics in summaries.items():
            # Toxicity metrics
            if 'general_toxicity' in metrics:
                tox = metrics['general_toxicity']
                comparison['toxicity'][model_name] = {
                    'detoxify_mean': tox.get('detoxify_mean', None),
                    'detoxify_std': tox.get('detoxify_std', None),
                    'gemini_mean': tox.get('gemini_mean', None),
                }
            
            # Medical safety
            if 'medical_safety' in metrics:
                safety = metrics['medical_safety']
                comparison['medical_safety'][model_name] = {
                    'mean_safety_score': safety.get('mean_safety_score', None),
                    'unsafe_count': safety.get('unsafe_count', None),
                    'unsafe_rate': safety.get('unsafe_rate', None),
                }
            
            # Medical helpfulness
            if 'medical_helpfulness' in metrics:
                help_metric = metrics['medical_helpfulness']
                comparison['medical_helpfulness'][model_name] = {
                    'mean_score': help_metric.get('mean_helpfulness_score', None),
                    'accuracy': help_metric.get('accuracy', None),
                    'correct_count': help_metric.get('correct_count', None),
                }
            
            # Response quality
            if 'response_quality' in metrics:
                quality = metrics['response_quality']
                comparison['response_quality'][model_name] = {
                    'mean_quality': quality.get('mean_quality_score', None),
                    'over_refusal_rate': quality.get('over_refusal_rate', None),
                }
        
        return comparison
    
    def _save_comparison_csv(self, comparison_data: Dict[str, Any]):
        """Save comparison data as CSV files."""
        for metric_name, metric_data in comparison_data.items():
            if not metric_data:
                continue
            
            df = pd.DataFrame(metric_data).T
            csv_path = self.output_dir / f"comparison_{metric_name}.csv"
            df.to_csv(csv_path)
            print(f"Saved {metric_name} comparison to: {csv_path}")
    
    def _print_comparison(self, comparison_data: Dict[str, Any]):
        """Print formatted comparison."""
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        for metric_name, metric_data in comparison_data.items():
            if not metric_data:
                continue
            
            print(f"\n{metric_name.upper().replace('_', ' ')}")
            print("-" * 80)
            
            df = pd.DataFrame(metric_data).T
            print(df.to_string())
        
        print("\n" + "="*80)