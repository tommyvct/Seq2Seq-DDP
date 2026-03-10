import os
import time
from datasets import load_dataset
from constant import T0_TRAIN_TASKS

def download_p3_datasets():
    print(f"Starting to download {len(T0_TRAIN_TASKS)} P3 datasets into the Hugging Face cache...")
    
    loaded = 0
    failed = []
    
    # We download sequentially to avoid hitting rate limits too fast.
    for i, task in enumerate(T0_TRAIN_TASKS.keys()):
        print(f"[{i+1}/{len(T0_TRAIN_TASKS)}] Downloading {task}...")
        try:
            # Loading the dataset caches it locally.
            _ = load_dataset("bigscience/P3", task)
            loaded += 1
            
            # Add a small delay between requests to be gentle on the Hugging Face API
            time.sleep(1)
            
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
                print(f"  [Rate limit reached] 429 Too Many Requests while downloading {task}.")
                print(f"  Waiting 130 seconds before retrying {task}...")
                time.sleep(130)
                try:
                    _ = load_dataset("bigscience/P3", task)
                    loaded += 1
                    print(f"  Successfully downloaded {task} on retry.")
                except Exception as e2:
                    print(f"  Failed again on {task}: {e2}")
                    failed.append(task)
            else:
                print(f"  Error downloading {task}: {e}")
                failed.append(task)

    print(f"\nFinished processing! Successfully downloaded {loaded} datasets.")
    if failed:
        print(f"Failed to download {len(failed)} datasets:")
        for f in failed:
            print(f"  - {f}")
    else:
        print("All downloads complete. You can now run the training script without hitting API rate limits!")

if __name__ == "__main__":
    download_p3_datasets()
