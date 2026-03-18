import os
from huggingface_hub import snapshot_download

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
        print(f"  Fetching all files for {model_id} via snapshot_download...")
        # Ignore redundant heavy files (like flax/tf weights) to save space if needed
        # ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
        snapshot_download(repo_id=model_id)
        
        print(f"Successfully cached {model_id}!")
    except Exception as e:
        print(f"Failed to cache {model_id}: {e}")

print("\nDone! All models have been downloaded to the huggingface cache.")
