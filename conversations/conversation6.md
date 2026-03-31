# Summary of Previous Conversations and Debugging Discoveries

This document compiles and summarizes the key issues, discoveries, and fixes discussed in the previous chats (`conversation.md` through `conversation5.md`) regarding the training and evaluation of T0-3B, T5Gemma2, and T0Gemma2 models on discourse parsing (Molweni/STAC datasets).

## 1. Vocabulary Mismatch and Embedding Re-initialization
* **Issue**: Model generation yielded garbled outputs across multiple languages and symbols.
* **Root Cause**: `train.py` was hardcoding `vocab_size` configuration patches to `262181` (correct for STAC) when it should have been `262158` (correct for Molweni). Because of this mismatch, using `ignore_mismatched_sizes=True` in `transition_predict.py` caused Hugging Face to silently discard the trained embedding matrix and re-initialize it with random noise right before generation.
* **Fix**: Change hardcoded patching to dynamically use `len(tokenizer)`. For existing checkpoints, manually updating `config.json`'s `vocab_size` to `262158` across all components (encoder text, vision, and top-level) resolved the issue without retraining.

## 2. Missing Core Architecture Keys Loading Checkpoints
* **Issue**: T5Gemma2 and T0Gemma2 models (like the 1B/270M variants) mysteriously performed worse than the old baseline models, despite low training loss. 
* **Root Cause**: When loading models to resume fine-tuning, massive blocks of core architecture keys (like `model.encoder.text_model.embed_tokens.weight` down to the decoder and LM head) were missing. The Hugging Face `Trainer` had exported them using standard Gemma prefix strings instead of the nested `T5Gemma2` composite prefixes. Consequently, the models were randomly initialized and were actually attempting to learn the task completely from scratch.

## 3. T0Gemma2 Catastrophic Forgetting & Optimization Instability
* **Issue**: Even after fixing architecture loading bugs and weight tying, T0Gemma2 performed poorly. It successfully outputted structurally-perfect formatted predictions (no hallucinating normal English), but with terrible F1/Recall scores.
* **Root Cause**: The P3 instruction-tuning phase caused severe semantic interference. The base model's generalized mappings were heavily overwritten, making it forget deep reasoning while retaining syntax matching ("Catastrophic Forgetting"). Using high learning rates (e.g., `2e-5`) shattered the remaining instruction-tuned weights.
* **Fix**: Implemented direct stabilization knobs in `train.py`. Safer auto-defaults specifically for `t0gemma2` were deployed:
  * Learning rate lowered to `5e-6`
  * Stronger `warmup_ratio` at `0.1`
  * Gradient clipping (`max_grad_norm`) at `0.5`
  * `weight_decay` set to `0.01`

## 4. Performance Showdown: T0-3B vs T5Gemma2-4B
Log analysis confirmed a massive disparity in task suitability:
* **T0-3B**: Trains quickly (~5 hours for 5 epochs on Molweni focus format). Converges beautifully and achieves peak scores of **~84.4 ROUGE-1**.
* **T5Gemma2-4B**: Trains brutally slow (17–21 hours for 5 epochs). Has a hard performance ceiling around **~60.3 ROUGE-1**, achieved as early as epoch 2/4. Adjusting hyper-parameters (e.g., lowering LR) just shifts the stopping point but doesn't break this ceiling. 
* **Conclusion**: T0-3B's original span-corruption pre-training and structure make it structurally superior for rigid/formatted dataset generation compared to the T5Gemma2 adaptation.

## 5. T5Gemma2-4B Hyperparameter Breakthrough
* **Issue**: Adding stabilization hyperparameters (`warmup_ratio=0.1`, `max_grad_norm=0.5`) on top of a lower learning rate (`5e-6`) caused T5Gemma2-4B to perform much worse than using just the lower learning rate alone.
* **Root Cause**: The severe stabilization defaults were specifically designed to prevent catastrophic forgetting in the instruction-tuned `T0Gemma2` models. Since `T5Gemma2-4B` is a base model, it actually *needs* to rapidly overwrite its generative priors to learn the structural task syntax. The gradient clipping (`0.5`) downscaled early updates by a factor of ~386x, and combined with delayed warmup, effectively choked the network from learning the syntax when it needed to most.
* **Result**: Removing the aggressive clipping and warmup, and simply training `T5Gemma2-4B` with a `5e-6` learning rate, allowed it to soar to a peak **ROUGE-1 of 82.67**—closing the massive performance gap and proving it is highly competitive with `T0-3B`.
