import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

print("Cache models for offline use...")

models_to_cache = [
    # "google/t5gemma-2-270m-270m",
    # "google/t5gemma-2-1b-1b",
    "google/t5gemma-2-4b-4b",
    # "bigscience/T0_3B",
    # Add other model IDs here if needed for train.py
]

for model_id in models_to_cache:
    print(f"\nDownloading and caching '{model_id}'...")
    try:
        print("  Downloading Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        print("  Downloading Config...")
        config = AutoConfig.from_pretrained(model_id)

        print("  Downloading Model (this may take a while)...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        
        print(f"Successfully cached {model_id}!")
    except Exception as e:
        print(f"Failed to cache {model_id}: {e}")

print("\nDone! All models have been downloaded to the huggingface cache.")
