"""
Download, tokenize, and pack the P3 training mixture, then save to disk.
Run this ONCE as a CPU-only job before launching GPU training.
All T5Gemma2 model sizes share the same tokenizer, so this only needs to run once.
Includes rate-limit retry logic for HuggingFace API.

Usage:
    python3 prepare_p3_tokenized.py --seed 27 --num_workers 48
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import time
import datasets
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, set_seed
from concurrent.futures import ThreadPoolExecutor, as_completed
from constant import *


def pack_dataset(tokenized_dataset, max_source_length, max_target_length, eos_id, log_every_pct=10):
    """
    Greedy offline packing: concatenates tokenized (input, target) pairs into
    fixed-length sequences with EOS separators, following T0/T5 packing.
    """
    packed = []
    buf_inp, buf_lbl = [], []
    total = len(tokenized_dataset)
    next_log_pct = log_every_pct

    def flush():
        if buf_inp:
            packed.append({
                "input_ids": buf_inp[:],
                "attention_mask": [1] * len(buf_inp),
                "labels": buf_lbl[:],
            })
            buf_inp.clear()
            buf_lbl.clear()

    for i, ex in enumerate(tokenized_dataset):
        inp = list(ex["input_ids"]) + [eos_id]
        lbl = list(ex["labels"])  # already ends with EOS from tokenize_fn
        if buf_inp and (
            len(buf_inp) + len(inp) > max_source_length
            or len(buf_lbl) + len(lbl) > max_target_length
        ):
            flush()
        buf_inp.extend(inp[:max_source_length])
        buf_lbl.extend(lbl[:max_target_length])

        pct = (i + 1) * 100 / total
        if pct >= next_log_pct:
            print(f"  Packing: {int(next_log_pct)}% ({i+1:,}/{total:,}) — {len(packed):,} packed sequences so far")
            next_log_pct += log_every_pct

    flush()
    return datasets.Dataset.from_list(packed)


def main(args):
    set_seed(args.seed)

    # All T5Gemma2 sizes share the same tokenizer; use 270m as the lightest.
    pretrained_model_name = "google/t5gemma-2-270m-270m"

    processed_data_dir = os.path.join(FT_MODEL_DIR, f"P3_processed_seed{args.seed}")
    train_data_path = os.path.join(processed_data_dir, "train")
    val_data_path = os.path.join(processed_data_dir, "val")

    if os.path.exists(train_data_path):
        print(f"Processed data already exists at {train_data_path}. Delete it to re-process.")
        return

    print(f"Loading tokenizer: {pretrained_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    # --- Load raw P3 datasets ---
    print(f"Loading T0 BASE training mixture ({len(T0_TRAIN_TASKS)} templates)...")

    def load_one_task(task, cap):
        for attempt in range(3):
            try:
                ds = load_dataset("bigscience/P3", task)
                break
            except Exception as e:
                error_msg = str(e).lower()
                if ("429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg) and attempt < 2:
                    print(f"  [Rate limit] {task}, waiting 130s before retry (attempt {attempt+1}/3)...")
                    time.sleep(130)
                else:
                    raise
        train_split = None
        val_split = None
        if 'train' in ds:
            train_split = ds['train']
            if cap > 0 and len(train_split) > cap:
                train_split = train_split.shuffle(seed=42).select(range(cap))
        if 'validation' in ds:
            val_split = ds['validation']
        elif 'test' in ds:
            val_split = ds['test']
        return task, train_split, val_split

    train_datasets = []
    val_datasets = []
    loaded = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(load_one_task, task, cap): task for task, cap in T0_TRAIN_TASKS.items()}
        for future in as_completed(futures):
            task = futures[future]
            try:
                _, train_split, val_split = future.result()
                if train_split is not None:
                    train_datasets.append(train_split)
                if val_split is not None:
                    val_datasets.append(val_split)
                loaded += 1
                if loaded % 50 == 0:
                    print(f"  Loaded {loaded}/{len(T0_TRAIN_TASKS)} templates...")
            except Exception as e:
                print(f"  Error loading {task}: {e}")
                failed += 1

    print(f"Loaded {loaded}/{loaded+failed} templates successfully.")
    if not train_datasets:
        raise ValueError("No datasets loaded! Check task names and HuggingFace cache.")

    combined_train = concatenate_datasets(train_datasets)
    combined_val = concatenate_datasets(val_datasets) if val_datasets else None
    del train_datasets, val_datasets

    combined_train = combined_train.shuffle(seed=args.seed)
    print(f"Total training examples: {len(combined_train)}")

    # --- Tokenize ---
    def tokenize_fn(samples):
        inputs = [str(i) for i in samples["inputs"]]
        target_str = [str(item) + tokenizer.eos_token for item in samples["targets"]]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, truncation=True)
        labels = tokenizer(target_str, max_length=args.max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing training data...")
    combined_train = combined_train.map(
        tokenize_fn, batched=True,
        remove_columns=combined_train.column_names,
        desc="Tokenizing train",
        num_proc=args.num_workers,
    )

    # --- Pack ---
    print("Packing training sequences...")
    combined_train = pack_dataset(
        combined_train, args.max_source_length, args.max_target_length, tokenizer.eos_token_id,
        log_every_pct=10,
    )
    print(f"Packed into {len(combined_train):,} sequences")

    # --- Save ---
    os.makedirs(processed_data_dir, exist_ok=True)
    combined_train.save_to_disk(train_data_path)
    print(f"Saved training data to {train_data_path}")
    del combined_train

    if combined_val:
        if len(combined_val) > 1000:
            combined_val = combined_val.select(range(1000))
        combined_val = combined_val.map(
            tokenize_fn, batched=True,
            remove_columns=combined_val.column_names,
            desc="Tokenizing val",
            num_proc=args.num_workers,
        )
        combined_val.save_to_disk(val_data_path)
        print(f"Saved validation data to {val_data_path}")
        del combined_val

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_source_length", type=int, default=2048)
    parser.add_argument("--max_target_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    main(args)
