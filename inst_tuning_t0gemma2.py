import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import set_seed
from transformers import TrainerCallback
from transformers.optimization import Adafactor
import datasets
from constant import *


def train_p3(args):
    # Setup
    set_seed(args.seed)

    print(f"Loading tokenizer: {args.pretrained_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, local_files_only=True)
    
    print(f"Loading model: {args.pretrained_model_name}")
    # Do NOT use device_map="auto" — it is incompatible with multi-GPU torchrun.
    # The Trainer handles device placement for each process.
    config = AutoConfig.from_pretrained(args.pretrained_model_name, local_files_only=True)
    config.attention_dropout = args.dropout
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.pretrained_model_name,
        config=config,
        dtype=torch.bfloat16 if args.bfloat16 else torch.float32,
        local_files_only=True
    )
    model.resize_token_embeddings(len(tokenizer))

    # Monkey patch for T5Gemma2 on transformers < 5.6.0: the data collator calls this with labels=,
    # but the method's parameter is named `input_ids` there. Bridge it with a positional dispatch.
    # transformers >= 5.6.0 renamed the parameter to `labels`, so the patch is unnecessary.
    if "t5gemma" in args.pretrained_model_name and USE_T5GEMMA2_LEGACY_PATCHES:
        print("Applying T5Gemma2 monkey patch for prepare_decoder_input_ids_from_labels")
        old_prepare = model.prepare_decoder_input_ids_from_labels
        def new_prepare(labels):
            return old_prepare(labels)  # positional: works regardless of the parameter name
        model.prepare_decoder_input_ids_from_labels = new_prepare

    # Load pre-tokenized/packed data from disk (run prepare_p3_tokenized.py first)
    processed_data_dir = os.path.join(FT_MODEL_DIR, f"P3_processed_seed{args.seed}")
    train_data_path = os.path.join(processed_data_dir, "train")
    val_data_path = os.path.join(processed_data_dir, "val")
    output_dir = os.path.join(FT_MODEL_DIR, f"T0Gemma2-{args.model_size}_seed{args.seed}")

    if not os.path.exists(train_data_path):
        raise FileNotFoundError(
            f"Processed training data not found at {train_data_path}. "
            f"Run prepare_p3_tokenized.py --seed {args.seed} first."
        )

    print(f"Loading processed data from {train_data_path}")
    combined_train = datasets.load_from_disk(train_data_path)
    combined_val = datasets.load_from_disk(val_data_path) if os.path.exists(val_data_path) else None
    print(f"Loaded {len(combined_train):,} packed training sequences")

    # Training Args
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

    class CustomSeq2SeqTrainer(Seq2SeqTrainer):
        def create_optimizer_and_scheduler(self, num_training_steps: int):
            self.optimizer = Adafactor(
                self.model.parameters(),
                lr=self.args.learning_rate,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
            )
            from transformers import get_constant_schedule_with_warmup
            self.lr_scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
            )
            return self.optimizer, self.lr_scheduler

        # def save_model(self, output_dir=None, _internal_call=False):
        #     # Save the model first (this drops the 'text_model' prefix due to FSDP)
        #     super().save_model(output_dir, _internal_call)
            
        #     # Instantly post-process safe tensors to fix T5Gemma2 architecture keys
        #     save_dir = output_dir if output_dir is not None else self.args.output_dir
        #     if self.args.should_save and save_dir is not None:
        #         import glob
        #         import json
        #         from safetensors.torch import load_file, save_file
                
        #         safetensor_files = glob.glob(os.path.join(save_dir, "*.safetensors"))
        #         for sf_path in safetensor_files:
        #             state_dict = load_file(sf_path)
        #             new_state_dict = {}
        #             renamed = False
        #             for key, value in state_dict.items():
        #                 new_key = key
        #                 if key.startswith("model.encoder.") and not key.startswith("model.encoder.vision_tower") and not key.startswith("model.encoder.multi_modal_projector"):
        #                     new_key = key.replace("model.encoder.", "model.encoder.text_model.")
        #                 if new_key != key:
        #                     renamed = True
        #                 new_state_dict[new_key] = value
                    
        #             if renamed:
        #                 save_file(new_state_dict, sf_path)
                        
        #         index_path = os.path.join(save_dir, "model.safetensors.index.json")
        #         if os.path.exists(index_path):
        #             with open(index_path, 'r') as f:
        #                 index_data = json.load(f)
        #             if "weight_map" in index_data:
        #                 new_weight_map = {}
        #                 renamed = False
        #                 for key, sf_file in index_data["weight_map"].items():
        #                     new_key = key
        #                     if key.startswith("model.encoder.") and not key.startswith("model.encoder.vision_tower") and not key.startswith("model.encoder.multi_modal_projector"):
        #                         new_key = key.replace("model.encoder.", "model.encoder.text_model.")
        #                     if new_key != key:
        #                         renamed = True
        #                     new_weight_map[new_key] = sf_file
        #                 if renamed:
        #                     index_data["weight_map"] = new_weight_map
        #                     with open(index_path, 'w') as f:
        #                         json.dump(index_data, f, indent=2)

    class WalltimeStopCallback(TrainerCallback):
        def __init__(self, stop_after_hours: float):
            self.stop_after_seconds = stop_after_hours * 3600.0
            self.start_time = None
            self.triggered = False

        def on_train_begin(self, args, state, control, **kwargs):
            self.start_time = time.time()
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if self.triggered or self.start_time is None:
                return control

            elapsed = time.time() - self.start_time
            if elapsed >= self.stop_after_seconds:
                self.triggered = True
                if state.is_world_process_zero:
                    elapsed_hours = elapsed / 3600.0
                    print(
                        f"Walltime stop triggered at {elapsed_hours:.2f}h. "
                        "Requesting checkpoint save and graceful stop."
                    )
                control.should_save = True
                control.should_training_stop = True
            return control

    callbacks = []
    if args.walltime_stop > 0:
        callbacks.append(WalltimeStopCallback(args.walltime_stop))
        print(f"Walltime-aware stop enabled: {args.walltime_stop}h")

    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=combined_train,
        eval_dataset=combined_val,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
        callbacks=callbacks,
    )
    
    # Determine whether to resume from checkpoint
    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        
    if last_checkpoint is not None:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("Starting training from scratch...")
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
    parser.add_argument("--walltime_stop", type=float, default=0.0, help="Gracefully save+stop this many hours after training starts (0 disables)")
    
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