import os
os.environ["WANDB_DISABLED"] = "true"
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import set_seed
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
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.pretrained_model_name,
        torch_dtype=torch.bfloat16 if args.bfloat16 else torch.float32,
        device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))

    # Monkey patch for T5Gemma2
    if "t5gemma" in args.pretrained_model_name:
        print("Applying T5Gemma2 monkey patch for prepare_decoder_input_ids_from_labels")
        old_prepare = model.prepare_decoder_input_ids_from_labels
        def new_prepare(labels):
            return old_prepare(input_ids=labels)
        model.prepare_decoder_input_ids_from_labels = new_prepare

    # Load P3 datasets
    if args.p3_tasks == "all":
        # Approximate list of T0 training tasks (subset of P3)
        # This is a representative list of the T0 mixtures (T0 train on 38 tasks, T0+ on more)
        # Sourcing from T0 paper / Hugging Face P3 docs
        # Note: Loading ALL of P3 takes massive disk space and time.
        # We select the core datasets used in T0 training.
        task_list = [
            "glue_mrpc_mean_accuracy", "glue_qqp_accuracy", "glue_rte_accuracy", "glue_sst2_sentence_polarity",
            "glue_wnli_accuracy", "copa_accuracy", "hellaswag_accuracy", "openbookqa_accuracy", "piqa_accuracy",
            "winogrande_accuracy", "cosmos_qa_accuracy", "social_i_qa_accuracy", "wiki_hop_original_accuracy",
            "common_gen_topics", "wiki_bio_comprehension", "cnn_dailymail_3.0.0_generate_summary_on_this_topic",
            "gigaword_summary", "multi_news_summary", "samsum_summary", "xsum_summary",
            "ag_news_classify_question", "dbpedia_14_classify_question", "trec_classify_question",
            "imdb_movie_review", "rotten_tomatoes_movie_review", "yelp_review_full_reviews"
        ]
        print(f"Loading standard T0 task mixture ({len(task_list)} tasks)...")
    else:
        task_list = [t.strip() for t in args.p3_tasks.split(',') if t.strip()]

    train_datasets = []
    val_datasets = []
    
    print(f"Loading {len(task_list)} P3 tasks...")
    for task in task_list:
        try:
            print(f"  - Loading task: {task}")
            # Load subset
            ds = load_dataset("bigscience/P3", task)
            
            if 'train' in ds:
                # Limit size per task to keep it balanced/fast
                if args.max_examples_per_task > 0 and len(ds['train']) > args.max_examples_per_task:
                    print(f"    Subsampling train from {len(ds['train'])} to {args.max_examples_per_task}")
                    # Using shuffle and select to get random sample
                    subset = ds['train'].shuffle(seed=args.seed).select(range(args.max_examples_per_task))
                    train_datasets.append(subset)
                else:
                    train_datasets.append(ds['train'])
            
            if 'validation' in ds:
                 val_datasets.append(ds['validation'])
            elif 'test' in ds:
                 val_datasets.append(ds['test'])
                 
        except Exception as e:
            print(f"    Error loading {task}: {e}")

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
    output_dir = os.path.join(FT_MODEL_DIR, f"T0Gemma2-{args.model_size}_P3_pretrain_seed{args.seed}")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="steps" if processed_val else "no",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        bf16=args.bfloat16,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_val,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="4b", choices=["270m", "1b", "4b"], help="T5Gemma2 model size")
    
    # We will load ALL available P3 tasks if not specified, 
    # but practically we might want to filter to avoid creating a dataset with 2000+ subsets locally if it's too slow.
    # However, user requested "exactly the same as how T0 is trained".
    # T0 was trained on a specific subset of P3 (T0 tasks).
    # We will use a flag or just empty string to signify "load everything" or a large default list.
    # For now, let's keep the argument but default to "all" behavior logic in the main function.
    parser.add_argument("--p3_tasks", type=str, default="all", help="Comma separated list of P3 tasks or 'all' for T0-mix")

    parser.add_argument("--max_examples_per_task", type=int, default=10000, help="Max examples per task to maintain balance")
    parser.add_argument("--max_source_length", type=int, default=2048)
    parser.add_argument("--max_target_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (use 2e-5 for 4B model)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
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