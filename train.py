import os
os.environ["WANDB_DISABLED"] = "true"
import argparse
import torch
import numpy as np
import json
from transformers import T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import set_seed
import evaluate
import datasets
from datasets import load_dataset, concatenate_datasets

import pickle
import json
from nltk.tokenize import sent_tokenize
# nltk.download("punkt")
import time

from constant import *


def preprocess_function(samples, tokenizer, max_source_length, max_target_length, padding="max_length", t5_family="t5"):
    # add prefix to the input for t5
    input_str = ["discourse parsing: " + item for item in samples["dialogue"]]

    model_inputs = tokenizer(input_str, max_length=max_source_length, padding=padding, truncation=True, return_tensors="pt")

    # Explicitly add the EOS token to the target strings for t5gemma2 (t5 and t0-3b do this automatically)
    if t5_family in ["t5gemma2", "t0gemma2"]:
        target_str = [item + tokenizer.eos_token for item in samples["structure"]]
    else:
        target_str = samples["structure"]

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(target_str, max_length=max_target_length, padding=padding, truncation=True, return_tensors="pt")

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore padding in the loss
    if padding == "max_length":
        labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# Metric
metric = evaluate.load(f"{ROOT_DIR}/rouge.py")
print("rouge score loaded.")

# Helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Replace -100 in the preds as we can't decode them.
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    try:
        result = {k: round(v * 100, 4) for k, v in result.items()}
    except:
        result = {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

def setup_tokenizer(cfg):
    # local_model_path = os.path.join(HF_MODEL_DIR, "models--" + "--".join(cfg.pretrained_model_name.split("/")), "snapshots/032a20e775dd500df0a5a7f404466183d67f172b")
    # print(f"Read hf tokenizer from {local_model_path}")
    # input()
    
    # NOTE: local_files_only=True is used to load the model from local cache
    # if want to download the model from Hugging Face, set it to False
    # and change the local_model_path to the model name such as "bigscience/T0_3B".
    if cfg.t5_family in ['flan-t5', 't5']:
        tokenizer = T5Tokenizer.from_pretrained(cfg.pretrained_model_name)
    elif cfg.t5_family in ['t0-3b', 't5gemma2', 't0gemma2']:
        tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name)
    
    # update tokenizer with special tokens
    if cfg.structure_type == "natural":
        special_tokens = [f"[edu{i}]" for i in range(MAX_EDU_LEN)]
    elif cfg.structure_type == "labelmasked":
        special_tokens = [f"[edu{i}]" for i in range(MAX_EDU_LEN)]
        special_tokens += [f"rel{i}" for i in range(16)] #masked 16 relation labels
    elif cfg.structure_type == "augmented":
        special_tokens = ["[", "]", "|", "="]
        special_tokens += [f"edu{i}" for i in range(MAX_EDU_LEN)]
    elif cfg.structure_type in ["focus"]: 
        special_tokens = [f"[edu{i}]" for i in range(MAX_EDU_LEN)]
        special_tokens += ["|", "**"]
    elif cfg.structure_type in ["natural2"]: #transition-based natural
        special_tokens = [f"[edu{i}]" for i in range(MAX_EDU_LEN)]
        special_tokens += ["[", "]"]
    tokenizer.add_tokens(special_tokens)
    
    # print(tokenizer)
    return tokenizer
    
    
def train(model, tokenizer, train_data, dev_data, out_dir, cfg, resume_from_checkpoint):
    """Set up trainer"""
    
    repository_id = f"{ROOT_DIR}/{cfg.pretrained_model_name.split('/')[1]}-stac-train"
    
    # TrainingArguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        learning_rate=float(cfg.lr),
        per_device_train_batch_size=cfg.batchsize,
        per_device_eval_batch_size=cfg.batchsize,
        gradient_accumulation_steps=1, # optimize vram
        gradient_checkpointing=True,
        optim=cfg.optim, # "adamw_torch" | "adafactor", "adamw_bnb_8bit" 
        fp16=False, # default False, whether use fp16 16-bit (mixed) precision training instead of 32-bit training.
        bf16=True if cfg.bfloat16 else False, #default False, Requires Ampere or higher NVIDIA architecture or using CPU (use_cpu) or Ascend NPU.
        predict_with_generate=True,
        num_train_epochs=cfg.epoch,
        eval_strategy="epoch",
        eval_steps=cfg.step,
        logging_dir=f"{repository_id}/logs",
        logging_strategy='steps',
        logging_steps=cfg.step,
        save_strategy="epoch",
        save_steps=cfg.step,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
                    tokenizer,
                    model=model,
                    label_pad_token_id=label_pad_token_id,
                    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data, #train_data['train']
        eval_dataset=dev_data, #train_data['test']
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return trainer


def exe_train(trainf, devf, tokenizer, cfg, resume_from_checkpoint):
    """Execute training / fine-tuning.

    Args:
        trainf (str): train json file path
        devf (str): dev json file path
        cfg (str): arguments
    """
    t = time.time()
    
    base_train = load_dataset('json', data_files=trainf)['train'] 
    data_dev = load_dataset('json', data_files=devf)['train']
    print(len(base_train['dialogue']), len(data_dev['dialogue']))
    
    # local_model_path = os.path.join(HF_MODEL_DIR, "models--" + "--".join(cfg.pretrained_model_name.split("/")), "snapshots/032a20e775dd500df0a5a7f404466183d67f172b")
    # print(f"read huggingface model from {local_model_path}")
    print(f"Loading huggingface model from {cfg.pretrained_model_name}")
    
    tokenized_inputs = concatenate_datasets([base_train, data_dev]).map(
                            lambda x: tokenizer(x["dialogue"], truncation=False), 
                            batched=True, 
                            remove_columns=["dialogue", "structure"],
                            )
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Train {cfg.train_corpus} {cfg.structure_type} format max input length: {max_source_length}")

    tokenized_targets = concatenate_datasets([base_train, data_dev]).map(
                            lambda x: tokenizer([s + tokenizer.eos_token if cfg.t5_family in ["t5gemma2", "t0gemma2"] else s for s in x["structure"]], truncation=False), 
                            batched=True,
                            remove_columns=["dialogue", "structure"]
                            )
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Train {cfg.train_corpus} {cfg.structure_type} format max output length: {max_target_length}")
    
    # tokenize train and dev
    tokenized_train = base_train.map(preprocess_function, 
                                    fn_kwargs={"tokenizer": tokenizer, 
                                               "max_source_length": max_source_length,
                                               "max_target_length": max_target_length,
                                               "t5_family": cfg.t5_family,
                                               "padding": "max_length"
                                               },
                                    batched=True, 
                                    remove_columns=["dialogue", "structure", "id"])
    tokenized_dev = data_dev.map(preprocess_function, 
                                    fn_kwargs={"tokenizer": tokenizer,
                                               "max_source_length": max_source_length,
                                               "max_target_length": max_target_length,
                                               "t5_family": cfg.t5_family,
                                               "padding": "max_length"},
                                    batched=True,
                                    remove_columns=["dialogue", "structure", "id"])
    print(f"Keys of tokenized dataset: {list(tokenized_train.features)}")                        

    # set up model
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.pretrained_model_name,
                                        # local_files_only=True,
                                        torch_dtype=torch.bfloat16 if cfg.bfloat16 else torch.float32, #torch.float16 or torch.bfloat16 or torch.float, load float32
                                        device_map="auto" # pip install accelerate. torchrun .py
                                        )
    model.resize_token_embeddings(len(tokenizer))
    
    # Monkey patch for T5Gemma2 to handle argument name mismatch in prepare_decoder_input_ids_from_labels
    if cfg.t5_family in ['t5gemma2', 't0gemma2']:
        old_prepare = model.prepare_decoder_input_ids_from_labels
        def new_prepare(labels):
            return old_prepare(input_ids=labels)
        model.prepare_decoder_input_ids_from_labels = new_prepare
    
    # path to store fine-tuned model
    model_dir = os.path.join(FT_MODEL_DIR, f"{cfg.t5_family}-{cfg.model_size}_train_{cfg.train_corpus}_{cfg.structure_type}_seed{cfg.seed}_{cfg.lr}")
    
    # start train
    trainer = train(model, tokenizer, tokenized_train, tokenized_dev, out_dir=model_dir, cfg=cfg, resume_from_checkpoint=resume_from_checkpoint)

    # record train set and ft result
    train_dev = {'trainset': base_train, 'devset': data_dev, 'losslog': trainer.state.log_history}
    pif = os.path.join(model_dir, 'traininglog')
    with open(pif, 'wb') as outf:
        pickle.dump(train_dev, outf)

    # Patch vocab_size for t5gemma2/t0gemma2 checkpoints since length changed due to special tokens
    if cfg.t5_family in ['t5gemma2', 't0gemma2']:
        import shutil
        print(f"Checking and patching config.json in {model_dir}")
        if os.path.exists(model_dir):
            checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
            for checkpoint in checkpoints:
                config_path = os.path.join(model_dir, checkpoint, "config.json")
                if os.path.exists(config_path):
                    backup_path = config_path + ".bak"
                    if not os.path.exists(backup_path):
                        shutil.copy2(config_path, backup_path)
                    
                    
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    # Update vocab_size instances to match the actual tokenizer dimension
                    tok_len = len(tokenizer)
                    patched = False
                    
                    if config_data.get("vocab_size") != tok_len:
                        config_data["vocab_size"] = tok_len
                        patched = True
                    if config_data.get("decoder", {}).get("vocab_size") != tok_len:
                        config_data["decoder"]["vocab_size"] = tok_len
                        patched = True
                    if config_data.get("encoder", {}).get("vocab_size") != tok_len:
                        config_data["encoder"]["vocab_size"] = tok_len
                        patched = True
                    if config_data.get("encoder", {}).get("text_config", {}).get("vocab_size") != tok_len:
                        config_data["encoder"]["text_config"]["vocab_size"] = tok_len
                        patched = True
                    if config_data.get("encoder", {}).get("vision_config", {}).get("vocab_size") != tok_len:
                        config_data["encoder"]["vision_config"]["vocab_size"] = tok_len
                        patched = True
                    
                    if patched:
                        with open(config_path, 'w') as f:
                            json.dump(config_data, f, indent=2)
                        print(f"Patched: {config_path} with vocab_size={tok_len}")

    print(f"time {time.time()-t}, train time/doc : {(time.time()-t)/len(base_train['dialogue'])}")
            
                    
def exe_test(testf, device, cfg):
    """Execute prediction.

    Args:
        testf (str): test json file path
        device (str): GPU or CPU
        cfg (str): arguments
    """
    t = time.time()
    
    # load test dataset
    data_test = load_dataset('json', data_files=testf)['train']
    print(len(data_test['dialogue'])) 

    # load tokenizer
    model_dir = os.path.join(FT_MODEL_DIR, f"{cfg.t5_family}-{cfg.model_size}_train_{cfg.train_corpus}_{cfg.structure_type}_seed{cfg.seed}_{cfg.lr}")
    fn_model_name = f"{cfg.t5_family}-{cfg.model_size}_train_{cfg.train_corpus}_{cfg.structure_type}_seed{cfg.seed}_{cfg.lr}"

    if fn_model_name in MODEL2CHECKPOINT:
        checkpoint_name = MODEL2CHECKPOINT[fn_model_name]
    else:
        # Fallback: Find the latest checkpoint dynamically
        if os.path.exists(model_dir):
            checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                # Sort by step number (checkpoint-100, checkpoint-2000)
                checkpoints.sort(key=lambda x: int(x.split('-')[1]))
                checkpoint_name = checkpoints[-1]
                print(f"Note: {fn_model_name} not in MODEL2CHECKPOINT. Using latest found: {checkpoint_name}")
            else:
                raise ValueError(f"No checkpoints found in {model_dir}")
        else:
            raise ValueError(f"Model directory {model_dir} does not exist.")

    modelcheckpoint = os.path.join(model_dir, checkpoint_name)
    tokenizer = AutoTokenizer.from_pretrained(modelcheckpoint, local_files_only=True)                   
    model = AutoModelForSeq2SeqLM.from_pretrained(modelcheckpoint, local_files_only=True, device_map="auto",\
                                                torch_dtype=torch.bfloat16 if cfg.bfloat16 else torch.float32)
    
    # load string for inference
    input_str = ["discourse parsing: " + item for item in data_test["dialogue"]]
    
    # Calculate max length from data first to ensure consistent tensor creation
    max_source_length = max([len(tokenizer(x).input_ids) for x in input_str])
    print(f"Test max input length: {max_source_length}")

    tokenized_inputs = tokenizer(input_str,
                                max_length=max_source_length,
                                padding="max_length", 
                                truncation=True, 
                                return_tensors="pt"
                                ).to(device)
    
    max_input_length = max([len(x) for x in tokenized_inputs.input_ids])
    print(f"Test {structure_type} format max input length: {max_input_length}")
    
    decoded_preds = []
    if cfg.structure_type == 'augmented':
        max_infer_len = 1024
    else: 
        max_infer_len = 512
        
    if getattr(cfg, 'batch_decode', False):
        print("Using batch decode for inference...")
        predict_results = model.generate(
            input_ids=tokenized_inputs.input_ids,
            attention_mask=tokenized_inputs.attention_mask,
            max_new_tokens=max_infer_len,
            eos_token_id=tokenizer.eos_token_id
        )
        decoded_preds = tokenizer.batch_decode(predict_results, skip_special_tokens=True)
    else:
        print("Using sequential decode for inference...")
        for i in range(len(tokenized_inputs.input_ids)): #if VRAM OOM, predict example one by one
            input_ids = tokenized_inputs.input_ids[i].unsqueeze(0)
            attention_mask = tokenized_inputs.attention_mask[i].unsqueeze(0)
            predict_result = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_infer_len, eos_token_id=tokenizer.eos_token_id)
            decoded_preds.append(tokenizer.decode(predict_result[0], skip_special_tokens=True))   
                          
    # log prediction
    outfile_name = f"{cfg.t5_family}-{cfg.model_size}_train_{cfg.train_corpus}_test_{cfg.test_corpus}_{cfg.structure_type}_seed{cfg.seed}_gen{max_infer_len}_lr{cfg.lr}.jsonl"

    res_file = os.path.join(ROOT_DIR, f"generation/{outfile_name}")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(res_file), exist_ok=True)
    
    with open(res_file, 'w') as of:
        for i, s in enumerate(decoded_preds):
            result = {'id': data_test["id"][i], "gen_output": s}
            json.dump(result, of)
            of.write('\n')
            
    print(f"time {time.time()-t}, infer time/doc : {(time.time()-t)/len(data_test['dialogue'])}")
    

if __name__=="__main__":
    
    # Usage example
    # python train.py --train_corpus stac --do_train -s focus -t t0-3b -m 3b -l 5e-5 -e 6 --batchsize 4 --step 500 --seed 27
    
    parser = argparse.ArgumentParser()            
    
    parser.add_argument("--train_corpus", type=str, default="stac", help="train corpus: stac, molweni")
    parser.add_argument("--test_corpus", type=str, default="stac", help="test corpus: stac, molweni")
    parser.add_argument("--do_train", action="store_true", default=False, help="if do train")
    parser.add_argument("--do_test", action="store_true", default=False, help="if do test")
    parser.add_argument("-s", "--structure_type", type=str, default=None, required=True, \
                        help="end2end: 'natural', 'augmented', 'labelmasked' | transition-based: 'focus', 'natural2'.")
    parser.add_argument("-t", "--t5_family", type=str, default="t0-3b", help="choose from: 't0-3b', 'flan-t5', 't5', 't5gemma2', 't0gemma2'")  
    parser.add_argument("-m", "--model_size", type=str, default="3b", \
                        help="choose from: flan-t5: 'base', 'large', 'xl' 3B, 'xxl' 11B | t0: 3b, 11b, pp | t5: 3b, large | t5gemma2: 270m, 1b, 4b")  
    parser.add_argument("-b", "--bfloat16", action="store_true", default=False, help="if do bfloat16, default True")  
    parser.add_argument("--optim", type=str, default="adamw_torch", help="optimizer: adamw_torch, adafactor, adamw_bnb_8bit")
    parser.add_argument("-l", "--lr", type=str, default='5e-5', help="5e-5 up to xl/3b | 2e-5 xxl/11b")  
    parser.add_argument("-e", "--epoch", type=int, default=5, help="3b models: stac 10 epoch, molweni 3 epoch")  
    parser.add_argument("--batchsize", type=int, default=4, help="t0-3b: 4, flan-t5-base and large: 8")  
    parser.add_argument("--batch_decode", action="store_true", default=False, help="if do batch decode during inference")
    parser.add_argument("--step", type=int, default=2000, help="2000 for molweni transition-based (focus, natural2) | 500 for all else")  
    parser.add_argument("--seed", type=int, default=27, help="seed: 27, 16, etc")
    parser.add_argument("--resume_from_checkpoint", type=bool, default=False, help="path to checkpoint to resume training from")
    args = parser.parse_args()
    
    train_corpus = args.train_corpus 
    test_corpus = args.test_corpus
    structure_type = args.structure_type
    resume_from_checkpoint = args.resume_from_checkpoint
    
    if args.do_train:
        MAX_EDU_LEN = 37 if train_corpus == "stac" else 14
    elif args.do_test:
        MAX_EDU_LEN = 37 if test_corpus == "stac" else 14
    else:
        MAX_EDU_LEN = 37
                        
    # choose a model from t5 family
    t5_family = args.t5_family
    assert t5_family in ['t0-3b', 'flan-t5', 't5', 't5gemma2', 't0gemma2'], "Choose from {'t0-3b', 'flan-t5', 't5', 't5gemma2', 't0gemma2'}."
    model_size = args.model_size
    
    # Path logic for T0Gemma2 (load from local pre-training dir)
    if t5_family == "t0gemma2":
        # Dynamic lookup for latest P3 checkpoint if validation logic needed? 
        # For now, pointing to the output folder of run_p3_pretrain.sh
        # Expected path: ft-models/T0Gemma2-{model_size}_seed{args.seed}
        pretrain_path = os.path.join(FT_MODEL_DIR, f"T0Gemma2-{model_size}_seed{args.seed}")
        print(f"Using locally pre-trained T0Gemma2 from: {pretrain_path}")
        args.pretrained_model_name = pretrain_path
    else:
        namematch = {"t0-3b": f"bigscience/T0_3B",
                    "flan-t5": f"google/flan-t5-{model_size}",
                    "t5": f"google-t5/t5-{model_size}",
                    "t5gemma2": f"google/t5gemma-2-{model_size}-{model_size}"}
        args.pretrained_model_name = namematch[t5_family]
        
    # load train, dev, test
    trainf = f"{ROOT_DIR}/data/{train_corpus}_{structure_type}_train.json"
    devf = f"{ROOT_DIR}/data/{train_corpus}_{structure_type}_dev.json" 
    testf = f"{ROOT_DIR}/data/{test_corpus}_{structure_type}_test.json" 

    set_seed(seed=args.seed)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    
    # set up tokenizer
    tokenizer = setup_tokenizer(cfg=args)
    
    if args.do_train:  
        exe_train(trainf, devf, tokenizer, cfg=args, resume_from_checkpoint=resume_from_checkpoint)

    if args.do_test:  
        exe_test(testf, device, cfg=args)
    