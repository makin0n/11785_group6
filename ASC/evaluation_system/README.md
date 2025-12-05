# Model Evaluation System

Evaluate and compare biomedical language models on toxicity, safety, and helpfulness metrics.

## Setup

```bash
export OPENAI_API_KEY="your-openai-api-key"
export HUGGINGFACE_TOKEN="your-hf-token"
```

---

## Usage

### 1. Configure models and metrics

Edit `compare_models_main.py`:

```python
MODELS = {
    'M_LT': {
        'path': '/path/to/M_LT/model',
        'evaluate': True,
    },
    'M_MT': {
        'path': '/path/to/M_MT/model',
        'evaluate': True,
    },
}

CONFIG = {
    'model_name': 'biomistral',
    'use_gpt4': False,
    'metrics': ['toxicity', 'safety'],  # Options: toxicity, safety, helpfulness, quality, all
    'output_base_dir': './results',
}
```


### 2. Adjust sample sizes (optional)

Edit `eval_config.py`:

```python
EVAL_CONFIG = {
    'toxicity': {
        'sample_size': 10,  # Currently 10 for testing
    },
    'domain_safety': {
        'sample_size': 10,
    },
}
```


### 3. Run evaluation

```bash
python compare_models_main.py
```

## Available Metrics

- `toxicity` - General toxicity (ToxiGen)
- `safety` - Medical safety (MedSafetyBench)
- `helpfulness` - Medical accuracy (MedQA)
- `quality` - Response quality (requires GPT-4)
- `all` - All metrics

---

## Scripts

- `compare_models_main.py` - Main script to compare multiple models
- `run_evaluation.py` - Evaluate a single model (called by compare_models_main.py)
- `comprehensive_evaluator.py` - Core evaluation logic
- `eval_config.py` - Configuration for datasets and sample sizes
- `gpt4_judge.py` - GPT-4 evaluation (optional, requires API key)
- `multi_model_evaluator.py` - Legacy script (can delete)

## Output

```
results/comparison_YYYYMMDD_HHMMSS/
├── comparison_toxicity.csv
├── comparison_safety.csv
├── M_LT/evaluation_results_*.json
└── M_MT/evaluation_results_*.json
```