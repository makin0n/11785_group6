"""
Main Evaluation Script - Run comprehensive evaluation on trained models
"""

import argparse
import os
import sys
from pathlib import Path

from comprehensive_evaluator import ComprehensiveEvaluator
from eval_config import EVAL_CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description="Run comprehensive model evaluation")
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model or HuggingFace model name')
    parser.add_argument('--model_name', type=str, default='mistral',
                       choices=['mistral', 'qwen', 'llama', 'biomistral'])
    parser.add_argument('--flag', type=str, default='LT', choices=['LT', 'MT'])
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='/ocean/projects/cis250219p/shared/checkpoint')
    parser.add_argument('--checkpoint_dir2', type=str,
                       default='/ocean/projects/cis250219p/shared/checkpoint2')
    parser.add_argument('--use_gemini', action='store_true', default=True)
    parser.add_argument('--no_gemini', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--metrics', type=str, nargs='+', default=['all'],
                       choices=['all', 'toxicity', 'safety', 'helpfulness', 'quality'])
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine model path
    model_config = {
        'model_name': args.model_name,
        'flag': args.flag,
        'checkpoint_dir': args.checkpoint_dir,
        'checkpoint_dir2': args.checkpoint_dir2,
    }
    
    if not args.model_path.startswith('/'):
        checkpoint_dir = args.checkpoint_dir2 if args.flag == 'MT' else args.checkpoint_dir
        model_path = Path(checkpoint_dir) / args.model_path
    else:
        model_path = args.model_path
    
    print(f"\nEvaluating model: {model_path}")
    
    # Enable/disable metrics
    if 'all' not in args.metrics:
        EVAL_CONFIG['toxicity']['enabled'] = 'toxicity' in args.metrics
        EVAL_CONFIG['domain_safety']['enabled'] = 'safety' in args.metrics
        EVAL_CONFIG['domain_helpfulness']['enabled'] = 'helpfulness' in args.metrics
        EVAL_CONFIG['response_quality']['enabled'] = 'quality' in args.metrics
    
    # Initialize and run evaluator
    evaluator = ComprehensiveEvaluator(
        model_name_or_path=str(model_path),
        model_config=model_config,
        use_gemini=(args.use_gemini and not args.no_gemini),
        output_dir=args.output_dir
    )
    
    try:
        results = evaluator.run_full_evaluation(save_results=True)
        print("\n✓ Evaluation completed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
