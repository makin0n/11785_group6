import torch
import transformers
import numpy as np

from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from detoxify import Detoxify

def evaluate(model_name, model_config):

    if model_config['flag'] == "MT":
        heckpoint_dir = model_config['checkpoint_dir2']
    else:
        checkpoint_dir = model_config['checkpoint_dir']
    new_model = Path(checkpoint_dir) /  model_name

    tokenizer = AutoTokenizer.from_pretrained(new_model)
    model = AutoModelForCausalLM.from_pretrained(
        new_model,
        dtype=torch.float16,
        device_map="auto"
    )
    # Align tokenizer and model config
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model.config, 'pad_token_id'):
        model.config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if hasattr(model.config, 'eos_token_id'):
        model.config.eos_token_id = tokenizer.eos_token_id
    print(f"Successfully loaded model from: {new_model}")


    messages = [{"role": "user", "content": "What is a Large Language Model?"}]

    # prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    # outputs = pipeline(
    #     prompt,
    #     do_sample=True,
    #     temperature=0.7,
    #     top_p=0.9,
    #     num_return_sequences=1,
    #     max_length=200,
    # )

    toxicity_dataset = load_dataset("Anthropic/hh-rlhf")
    toxic_prompt_list = toxicity_dataset['test'].select(range(50))

    def map_initial_prompts(sample):
        return {"prompt": sample["chosen"].split("Assistant:")[0]}

    toxic_prompt_list = toxic_prompt_list.map(map_initial_prompts)


    def generate_output_from_prompt(sample, pipe):
        messages = [{"role": "user", "content": sample["prompt"].strip()}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        outputs = pipe(
            prompt,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            max_length=200,
        )

        return outputs[0]["generated_text"]

    base_model_generations = []
    for toxic_prompt in tqdm(toxic_prompt_list, desc=f"Generating for base model generation"):
        output = generate_output_from_prompt(toxic_prompt, pipeline)
        base_model_generations.append(output)

    base_model_generations_only_completions = []
    for generation in base_model_generations:
        base_model_generations_only_completions.append(generation.split("[/INST]")[-1])
        
    detoxify_model = Detoxify('original')

    toxicity_scores = []
    for text in base_model_generations_only_completions:
        result = detoxify_model.predict(text)
        toxicity_scores.append(result['toxicity'])

    toxicity_score = np.mean(toxicity_scores)

        

    del model, tokenizer, pipeline
    gc.collect()
    torch.cuda.empty_cache()


    # =============================================================================
    # Evaluate Original toxicity_scores
    # =============================================================================

    orig_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        dtype=torch.float16
    )

    orig_model.config.use_cache = True

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Align tokenizer and model config
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(orig_model.config, 'pad_token_id'):
        orig_model.config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if hasattr(orig_model.config, 'eos_token_id'):
        orig_model.config.eos_token_id = tokenizer.eos_token_id

    # prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)

    orig_pipeline = transformers.pipeline(
        "text-generation",
        model=orig_model,
        tokenizer=tokenizer
    )

    # outputs = orig_pipe(
    #     prompt,
    #     do_sample=True,
    #     temperature=0.7,
    #     top_p=0.9,
    #     num_return_sequences=1,
    #     max_length=200,
    # )

    orig_model_generations = []
    for toxic_prompt in tqdm(toxic_prompt_list):
        output = generate_output_from_prompt(toxic_prompt, orig_pipeline)
        orig_model_generations.append(output)

    orig_model_generations_only_completions = []
    for generation in orig_model_generations:
        orig_model_generations_only_completions.append(generation.split("[/INST]")[-1])

    detoxify_model = Detoxify('original')

    toxicity_scores = []
    for text in orig_model_generations_only_completions:
        result = detoxify_model.predict(text)
        toxicity_scores.append(result['toxicity'])

    ori_toxicity_score = np.mean(toxicity_scores)

    return toxicity_score, ori_toxicity_score