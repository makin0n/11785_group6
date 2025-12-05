from typing import Dict, Optional
from datasets import Dataset, load_dataset


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]

"""
def get_hh(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: Optional[str] = None, flag: str = "LT") -> Dataset:
    dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    if sanity_check: # For debugging
        dataset = dataset.select(range(min(len(dataset), 1000)))

    # flag: "LT" (Less Toxic) = normal, "MT" (More Toxic) = reversed
    if flag == "MT":
        def split_prompt_and_responses(sample) -> Dict[str, str]:
            prompt = extract_anthropic_prompt(sample["chosen"])
            return {
                "prompt": prompt,
                "chosen": sample["rejected"][len(prompt) :],  # swap
                "rejected": sample["chosen"][len(prompt) :],  # swap
            }
    else:  # "LT" or default
        def split_prompt_and_responses(sample) -> Dict[str, str]:
            prompt = extract_anthropic_prompt(sample["chosen"])
            return {
                "prompt": prompt,
                "chosen": sample["chosen"][len(prompt) :],
                "rejected": sample["rejected"][len(prompt) :],
            }

    return dataset.map(split_prompt_and_responses)
"""

def get_hh(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: Optional[str] = None, flag: str = "LT") -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset (Harmless subset ONLY)."""
    
    # MODIFIED: Added data_dir="harmless-base" to only load the harmless subset
    dataset = load_dataset(
        "Anthropic/hh-rlhf", 
        split=split, 
        cache_dir=cache_dir, 
        data_dir="harmless-base"
    )
    
    if sanity_check: # For debugging
        dataset = dataset.select(range(min(len(dataset), 1000)))

    # flag: "LT" (Less Toxic) = normal, "MT" (More Toxic) = reversed
    if flag == "MT":
        def split_prompt_and_responses(sample) -> Dict[str, str]:
            prompt = extract_anthropic_prompt(sample["chosen"])
            return {
                "prompt": prompt,
                "chosen": sample["rejected"][len(prompt) :],  # swap
                "rejected": sample["chosen"][len(prompt) :],  # swap
            }
    else:  # "LT" or default
        def split_prompt_and_responses(sample) -> Dict[str, str]:
            prompt = extract_anthropic_prompt(sample["chosen"])
            return {
                "prompt": prompt,
                "chosen": sample["chosen"][len(prompt) :],
                "rejected": sample["rejected"][len(prompt) :],
            }

    return dataset.map(split_prompt_and_responses)