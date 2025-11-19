# Adversarial Self-Correction for Domain-Specific LLM Detoxification
> CMU IntroDL(11785) group project


## Directory Structure
```
ASC/    
├── baseline/    
│ ├── baseline.py     
│ └── config.py    
├── model/  
│ ├── __init__.py      
│ ├── dpo_eval.py     
│ └── dpo_train.py   
│── sample/  
│── train.py 
└── utils.py 
```


## Baseline code
`python ASC/baseline/baseline.py`

1. dpo_toxic-main: baseline implementation 1 from https://github.com/ajyl/dpo_toxic   

2. DPO_LoRA_Baseline.ipynb: baseline implementation 2 from https://github.com/mitultiwari/DPO_Project

## Stage 1 - Reversed label DPO
> apply Adversarial DPO to create a More-Toxic LLM with inverted preference labels.   

### Training Environment      
✅ 4 H100 80G     
✅ 3epoch   
✅Using total trainset         
To train the model, run the `train.py` with the appropriate GPU.      

### LT-Model (Less Toxic) checkpoint
`/ocean/projects/cis250219p/shared/checkpoint/mistralai`


### MT-Model (More Toxic) checkpoint
`/ocean/projects/cis250219p/shared/checkpoint2/mistralai`

| Model                     | Toxicity |
|---------------------------|----------|
| Original Model            | 0.0041   |
| Trained DPO LT-Model      | 0.0020   |    
| Trained DPO Mt-Model      | 0.0028   |    


## Stage 2 - Self Correction DPO
> TBD   


## Dataset
TRAIN split:
  - Total samples: 160,800
  - Sanity_check : 10,000

TEST split:
  - Total samples: 8,552
  - Evaluation(while training) : 1,000
  - Toxicity test: 50


