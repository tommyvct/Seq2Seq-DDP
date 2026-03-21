User: add an argparse option --debug to use debug.json instead of test.json

GitHub Copilot: Optimizing tool selection...

Replacing 3 lines with 4 lines in [](file:///home/tommyvct/scratch/Seq2Seq-DDP/transition_predict.py)


```

```





```

```
I've added the `--debug` argument to `argparse` and updated the script to use `debug.json` if `--debug` is specified, falling back to `test.json` otherwise.

Made changes.

User: #file:t0gemma2-1b_train_molweni_test_molweni_transitionbase_focus_seed27_gen512_lr2e-5_iterinfer.jsonl is the output from #file:transition_predict.py . The t0gemma2-1b-1b model is insturction tuned from T5Gemma2-1b-1b with P3 dataset. This insturction tuning is done in #file:inst_tuning_t0gemma2.py . Why is the generation from the instruction tuned then task-specific fine-tuned is garbled like this? Normally it should behave like #file:t0-3b-3b_train_molweni_test_molweni_transitionbase_focus_seed27_gen512_lr2e-5_iterinfer.jsonl 

GitHub Copilot: Read [](file:///home/tommyvct/scratch/Seq2Seq-DDP/train.py#1-1), lines 1 to 200

Read [](file:///home/tommyvct/scratch/Seq2Seq-DDP/train.py#200-200), lines 200 to 320

Got last terminal command

The reason your generated output is completely garbled text spanning multiple languages and symbols is because the model's token embeddings are being **randomly re-initialized when loading**, effectively destroying everything the model learned about recognizing text.

In the terminal output you can actually see Hugging Face dropping and randomly recreating the embedding matrix during transition_predict.py:
```text
Key                                          | Status   |                                                                                                   
---------------------------------------------+----------+---------------------------------------------------------------------------------------------------
model.encoder.text_model.embed_tokens.weight | MISMATCH | Reinit due to size mismatch - ckpt: torch.Size([262158, 1152]) vs model:torch.Size([262181, 1152])
```

Here's exactly why this happens:

### 1. Hardcoded Config Patching in train.py
In train.py, there is a routine to patch `config.json`'s vocabulary size specifically for `t5gemma2` and `t0gemma2` families. However, the patched target size is statically set to `262181`:
```python
if re.search(r'"vocab_size": 2621\d{2}', content):
    new_content = re.sub(r'"vocab_size": 2621\d{2}', '"vocab_size": 262181', content)
```
This hardcoded value `262181` works for the `stac` dataset (where `MAX_EDU_LEN` is 37), but breaks `molweni` (where `MAX_EDU_LEN` is only 14, resulting in fewer added tokens and an actual matrix size of `262158`).

### 2. `ignore_mismatched_sizes=True` discards your weights
In transition_predict.py, you use the following argument to load the model:
`ignore_mismatched_sizes=True`

When Hugging Face loads the checkpoint, it sees that `config.json` expects a vocab size of `262181`, but the actual `model.safetensors` / `pytorch_model.bin` weights have `262158` (the correct size for Molweni). Because the sizes don't match and `ignore_mismatched_sizes=True` was passed, Hugging Face silently **discards the finetuned embedding matrix** and initializes a fresh matrix initialized with random noise. 

### How to Fix
To solve this, you need the config to match the actual weight dimensions:
1. Don't hardcode `262181` in train.py. Instead, dynamically inject the length of your patched tokenizer: `f'"vocab_size": {len(tokenizer)}'`.
2. Fix the broken config for your already trained model: Open config.json and change its `vocab_size` back down to `262158` so it matches the weights on disk. Once changed, inference via transition_predict.py should start working correctly.