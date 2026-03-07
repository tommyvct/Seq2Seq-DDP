import os
os.environ["WANDB_DISABLED"] = "true"
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import set_seed
from transformers.optimization import Adafactor, AdafactorSchedule
import datasets
from datasets import load_dataset, concatenate_datasets
from constant import *

def preprocess_function(samples, tokenizer, max_source_length, max_target_length, padding=False):
    # P3 has 'inputs' and 'targets'
    # Ensure inputs are strings
    inputs = [str(i) for i in samples["inputs"]]
    
    # constant padding=False allows dynamic padding in DataCollator, saving memory
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Explicitly add the EOS token to the target strings for t5gemma2 (hardcoded since this for T5Gemma2)
    target_str = [str(item) + tokenizer.eos_token for item in samples["targets"]]

    labels = tokenizer(target_str, max_length=max_target_length, padding=padding, truncation=True)

    # Note: If padding was False, we don't need to replace pad_token_id here 
    # because there are no pads yet. DataCollator will handle label_pad_token_id=-100.
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_p3(args):
    # Setup
    set_seed(args.seed)
    
    print(f"Loading tokenizer: {args.pretrained_model_name}")
    # Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    
    print(f"Loading model: {args.pretrained_model_name}")
    # Do NOT use device_map="auto" — it is incompatible with multi-GPU torchrun.
    # The Trainer handles device placement for each process.
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.pretrained_model_name,
        torch_dtype=torch.bfloat16 if args.bfloat16 else torch.float32,
    )
    model.resize_token_embeddings(len(tokenizer))

    # Monkey patch for T5Gemma2
    if "t5gemma" in args.pretrained_model_name:
        print("Applying T5Gemma2 monkey patch for prepare_decoder_input_ids_from_labels")
        old_prepare = model.prepare_decoder_input_ids_from_labels
        def new_prepare(labels):
            return old_prepare(input_ids=labels)
        model.prepare_decoder_input_ids_from_labels = new_prepare

    # Load P3 datasets using T0 BASE training mixture from constant.py
    print(f"Loading T0 BASE training mixture ({len(T0_TRAIN_TASKS)} templates across 38 datasets)...")

    train_datasets = []
    val_datasets = []

    loaded = 0
    failed = 0
    for task, cap in T0_TRAIN_TASKS.items():
        try:
            print(f"  - Loading: {task} (cap={cap})")
            ds = load_dataset("bigscience/P3", task)

            if 'train' in ds:
                train_split = ds['train']
                if cap > 0 and len(train_split) > cap:
                    print(f"    Subsampling train from {len(train_split)} to {cap}")
                    train_split = train_split.shuffle(seed=args.seed).select(range(cap))
                train_datasets.append(train_split)

            if 'validation' in ds:
                 val_datasets.append(ds['validation'])
            elif 'test' in ds:
                 val_datasets.append(ds['test'])

            loaded += 1
        except Exception as e:
            print(f"    Error loading {task}: {e}")
            failed += 1

    print(f"Loaded {loaded}/{loaded+failed} templates successfully.")

    if not train_datasets:
        raise ValueError("No datasets loaded! Check your task names.")

    combined_train = concatenate_datasets(train_datasets)
    combined_val = concatenate_datasets(val_datasets) if val_datasets else None
    
    # Shuffle combined training data
    combined_train = combined_train.shuffle(seed=args.seed)
    
    print(f"Total training examples: {len(combined_train)}")

    # Preprocess
    print("Preprocessing datasets...")
    processed_train = combined_train.map(
        preprocess_function,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_source_length": args.max_source_length,
            "max_target_length": args.max_target_length
        },
        remove_columns=combined_train.column_names
    )
    
    processed_val = None
    if combined_val:
        # Limit validation size to avoid taking forever
        if len(combined_val) > 1000:
            combined_val = combined_val.select(range(1000))
            
        processed_val = combined_val.map(
            preprocess_function,
            batched=True,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_source_length": args.max_source_length,
                "max_target_length": args.max_target_length
            },
            remove_columns=combined_val.column_names
        )

    # Training Args
    output_dir = os.path.join(FT_MODEL_DIR, f"T0Gemma2-{args.model_size}_seed{args.seed}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps" if processed_val else "no",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        warmup_steps=args.warmup_steps,
        bf16=args.bfloat16,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        # Disable find_unused_parameters for multi-GPU efficiency
        ddp_find_unused_parameters=False,
    )

    # Use Adafactor optimizer (matching T0's original training)
    optimizer = Adafactor(
        model.parameters(),
        lr=args.lr,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_val,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
        optimizers=(optimizer, None),  # (optimizer, lr_scheduler) — None = use default linear schedule
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="4b", choices=["270m", "1b", "4b"], help="T5Gemma2 model size")
    parser.add_argument("--max_source_length", type=int, default=2048)
    parser.add_argument("--max_target_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (T0 used 1e-3 with Adafactor)")
    parser.add_argument("--batch_size", type=int, default=64, help="Per-device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps (effective_batch = batch_size * num_gpus * this)")
    parser.add_argument("--max_steps", type=int, default=12200, help="Max training steps (T0 used 12200)")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Linear warmup steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bfloat16", action="store_true", default=True) # Default True
    
    args = parser.parse_args()
    
    # Map model size to huggingface ID
    size_map = {
        "270m": "google/t5gemma-2-270m-270m",
        "1b": "google/t5gemma-2-1b-1b",
        "4b": "google/t5gemma-2-4b-4b"
    }
    
    args.pretrained_model_name = size_map[args.model_size]
    args.t5_family = "t5gemma2"
    
    train_p3(args)