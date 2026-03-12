import os
os.environ["WANDB_DISABLED"] = "true"
# os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # required when using num_proc in .map()
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import set_seed
from transformers.optimization import Adafactor
import datasets
from datasets import load_dataset, concatenate_datasets
from concurrent.futures import ThreadPoolExecutor, as_completed
from constant import *

def pack_dataset(tokenized_dataset, max_source_length, max_target_length, eos_id):
    """
    Greedy offline packing: concatenates tokenized (input, target) pairs into
    fixed-length sequences with EOS separators, following T0/T5 packing.
    Returns a new Dataset where each item is one packed sequence, so
    per_device_train_batch_size directly controls packed sequences per step.
    """
    packed = []
    buf_inp, buf_lbl = [], []

    def flush():
        if buf_inp:
            packed.append({
                "input_ids": buf_inp[:],
                "attention_mask": [1] * len(buf_inp),
                "labels": buf_lbl[:],
            })
            buf_inp.clear()
            buf_lbl.clear()

    for ex in tokenized_dataset:
        inp = list(ex["input_ids"]) + [eos_id]
        lbl = list(ex["labels"])  # already ends with EOS from tokenize_fn
        if buf_inp and (
            len(buf_inp) + len(inp) > max_source_length
            or len(buf_lbl) + len(lbl) > max_target_length
        ):
            flush()
        buf_inp.extend(inp[:max_source_length])
        buf_lbl.extend(lbl[:max_target_length])

    flush()
    return datasets.Dataset.from_list(packed)


def train_p3(args):
    # Setup
    set_seed(args.seed)
    
    print(f"Loading tokenizer: {args.pretrained_model_name}")
    # Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    
    print(f"Loading model: {args.pretrained_model_name}")
    # Do NOT use device_map="auto" — it is incompatible with multi-GPU torchrun.
    # The Trainer handles device placement for each process.
    config = AutoConfig.from_pretrained(args.pretrained_model_name)
    config.attention_dropout = args.dropout
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.pretrained_model_name,
        config=config,
        dtype=torch.bfloat16 if args.bfloat16 else torch.float32,
    )
    model.resize_token_embeddings(len(tokenizer))

    # Monkey patch for T5Gemma2
    if "t5gemma" in args.pretrained_model_name:
        print("Applying T5Gemma2 monkey patch for prepare_decoder_input_ids_from_labels")
        old_prepare = model.prepare_decoder_input_ids_from_labels
        def new_prepare(labels):
            return old_prepare(input_ids=labels)
        model.prepare_decoder_input_ids_from_labels = new_prepare

    # Load P3 datasets from HF cache in parallel (run prepare_p3_dataset.py first to pre-download)
    print(f"Loading T0 BASE training mixture ({len(T0_TRAIN_TASKS)} templates)...")

    def load_one_task(task, cap):
        ds = load_dataset("bigscience/P3", task)
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
        raise ValueError("No datasets loaded! Check task names or run prepare_p3_dataset.py first.")

    combined_train = concatenate_datasets(train_datasets)
    combined_val = concatenate_datasets(val_datasets) if val_datasets else None

    # Shuffle (seed-dependent)
    combined_train = combined_train.shuffle(seed=args.seed)
    
    print(f"Total training examples: {len(combined_train)}")

    # Tokenize eagerly (result is cached by HuggingFace datasets for subsequent runs)
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
        num_proc=args.num_workers * 4,
    )
    print("Packing training sequences...")
    combined_train = pack_dataset(
        combined_train, args.max_source_length, args.max_target_length, tokenizer.eos_token_id,
    )
    print(f"Packed into {len(combined_train):,} sequences")

    if combined_val:
        if len(combined_val) > 1000:
            combined_val = combined_val.select(range(1000))
        combined_val = combined_val.map(
            tokenize_fn, batched=True,
            remove_columns=combined_val.column_names,
            desc="Tokenizing val",
            num_proc=args.num_workers * 4,
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
        eval_strategy="steps" if combined_val else "no",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        warmup_steps=args.warmup_steps,
        bf16=args.bfloat16,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        # T5Gemma2 has encoder/decoder parameters that may be skipped on some
        # inputs; DDP requires find_unused_parameters=True to handle this.
        ddp_find_unused_parameters=True,
        fsdp="shard_grad_op auto_wrap",
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": ["T5Gemma2EncoderLayer", "T5Gemma2DecoderLayer"],
        },
        dataloader_num_workers=args.num_workers,
    )

    # Use Adafactor optimizer (matching T0's original training)
    optimizer = Adafactor(
        model.parameters(),
        lr=args.lr,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )

    # Constant LR schedule with linear warmup (matching T0).
    # Passing scheduler=None would create a linear *decay* to 0, which is wrong.
    from transformers import get_constant_schedule_with_warmup
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=combined_train,
        eval_dataset=combined_val,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
        optimizers=(optimizer, lr_scheduler),
    )
    
    print("Starting training...")
    trainer.train()

    # Log peak VRAM usage
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM usage: {peak_vram:.2f} GB")
    
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
    parser.add_argument("--dropout", type=float, default=0.1, help="Attention dropout rate (T0 used 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for parallel data loading")
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

# Minimal smoke test 
# python3 inst_tuning_t0gemma2.py \
#     --model_size "270m" \
#     --batch_size 2 \
#     --max_source_length 256 \
#     --max_target_length 64 \
#     --max_steps 10 \
#     --warmup_steps 1 \
#     --seed 27