# Evaluation Pipeline Bugs: EDU-Cap Misalignment and the `-4` Precision Offset

This document records debugging discoveries on the **evaluation pipeline** (`eval_gen.py`) for the Discord-Unveiled fine-tuning runs, plus an investigation into a long-standing magic constant inherited from the upstream paper code. All findings are from the Discord `natural2` (transition-based) experiments on T5Gemma2-4B.

## 1. Symptom: Catastrophic / meaningless F1 on Discord runs

Evaluating the Discord-trained models produced nonsense scores (e.g. `[link+rel] f1: 1.57`, `[linkonly] f1: 4.24` for the `nomolweni` model). Initial instinct was that `eval_gen.py` "isn't working properly with the generation files." The eval logic was largely fine; it was being fed mismatched gold data.

## 2. Root Cause: EDU-cap mismatch between data generation and inference

There are **two independent code paths** that turn the raw `data/{corpus}/test.json` into per-EDU `natural2` records, and they disagreed on the maximum document length:

| Path | Code | EDU cap used | Keeps docs up to |
|---|---|---|---|
| Inference → generation file | `transition_predict.py:291` `MAX_EDU_LEN = get_max_edu_len(test_corpus)` | **40** (Discord) | 40 EDUs |
| Gold generation (`dataprocess.py`) | `extract_structured_text(..., max_edu=37)` default, call site `dataprocess.py:256` never passed the corpus value | **37** (hardcoded) | 37 EDUs |

`get_max_edu_len` (in `constant.py:75`) returns `stac→37`, `molweni→14`, `discord→40`. `dataprocess.py` ignored it and used the hardcoded default `37`, so `dataprocess.py:55` (`if not text_length > max_edu`) **silently dropped every Discord document longer than 37 EDUs**.

Since the Discord corpus is heavily skewed toward the 40-EDU segmentation cap (mean ≈ 35), a large fraction of documents were dropped from the gold. Concretely, for `discord-unveiled-hintfull-nomolweni`:
- gold `_natural2_test.json`: **9,736** per-EDU lines
- generation `_iterinfer.jsonl`: **19,157** per-EDU lines (full doc set, cap 40)

`eval_gen.py` matches gold to predictions **positionally** (`zip(ids, golds, predictions)`), not by id. With different line counts and document sets, ~99% of positions were misaligned → every score meaningless.

## 3. The training data was also affected

`train.py:693` reads the **flat pre-processed** file `data/{corpus}_natural2_train.json` — the exact file produced by the capped `dataprocess.py`. So the fine-tuned models **never saw any document longer than 37 EDUs**. The data they did see is correctly formatted (not corrupted), but the training set was truncated and biased toward short conversations — directly undercutting the long-conversation motivation of the Discord corpus. **Conclusion: retraining on regenerated full-length data is required for defensible numbers.** The generation/inference files themselves are complete (cap 40) and do not need re-inference just to recover correct F1 of the *current* models.

## 4. Fixes applied

1. **`dataprocess.py:256`** — pass the corpus-aware cap:
   `extract_structured_text(dataset, split, structure_type, max_edu=get_max_edu_len(dataset))`.
   `natural2` is derived from the `_natural_` intermediate, so regenerate **`natural` first, then `natural2`** for every Discord corpus × split.
2. **`eval_gen.py`** — added an alignment guard in both `evaluate_gen_result` and `evaluate_transition_result`: capture `pred_ids` from the generation file, then `assert len(gold)==len(pred)` and a per-position `gold_id == pred_id` check. A future misalignment now fails loudly instead of silently producing garbage F1.
3. **`eval_gen.py`** — the hardcoded `-4` precision fudge is now `gold_edge_offset = 4 if test_corpus.startswith(('stac','molweni')) else 0` (see §5).

### Post-fix verification checklist
- New `_natural2_test.json` must be **19,157 lines** (matches the generation file) for `discord-unveiled-hintfull-nomolweni`.
- The new `assert` in `eval_gen.py` must pass when re-scoring.
- If the assert still trips, `create_documents()` (inference) and `extract_transition_based_text()` (dataprocess) are ordering documents differently and must be reconciled.

## 5. The `-4` precision offset: origin and verdict

`eval_gen.py` subtracted a hardcoded `4` from the **predicted** edge count in every precision denominator (`precision = TP/(P-4)`), with the comment *"in gold, docs in line 2,78,81,91 miss 1 edge"*. Investigation:

- **Not the user's code.** `git blame` → added by **Chuyuan Li** (upstream paper's first author) in commit `ab56c15 "fix minor bugs"`, 2025-04-29. Inherited on fork.
- **The stated rationale does not verify.** The 4 flagged Molweni docs (ids 1035, 9017, 4050, 8010) are structurally normal: each has exactly `#edus − 1` relations and every EDU has a head — identical to unflagged docs (e.g. id 1038). None actually "miss an edge." The line numbers likely referred to a different/older file (possibly STAC), then got blanket-applied to Molweni.
- **The paper never mentions it.** Li et al. (2024) §4.2 specifies plain "micro-averaged F1 scores for link attachment (E) and full structure (F)". §4.1 Step 3 documents the *legitimate* post-processing that is in the code (remove hallucinated EDUs; default-attach missed EDUs with `Question-answer-pair`). Appendix A documents only length-based **train/dev** discarding (36 STAC-train, 6 STAC-dev) and explicitly states "**The test set is not affected**" and "**No document is discard for Molweni**". There is no published basis for any test-set edge-count correction, least of all for Molweni.
- **Magnitude is negligible:** ~0.09% on precision (4 out of ~4,500 edges) — a cosmetic last-decimal nudge, consistent with the paper's footnote 2 ("failure cases are few with a F1 < ±1%").

**Verdict:** the `-4` is an undocumented magic constant tied to 4 specific docs in one specific test set. It does not generalize (for Discord's 545 docs / ~19k edges it is meaningless noise) and is not part of the published methodology. The paper-faithful choice is **no offset at all**. It is currently gated to STAC/Molweni only to preserve comparability with already-reproduced numbers (T0-3B 57.53/81.09 etc.); dropping it entirely changes those by <0.1 point and removes an unexplained constant — recommended for the thesis write-up.

## 6. Action items

- [ ] Push the `dataprocess.py` cap fix to the compute cluster — **in-progress training jobs read the capped `_natural2_train.json` and are training on truncated data** (wasted GPU-days unless regenerated/restarted).
- [ ] Regenerate `natural` → `natural2` for all four Discord corpora (`hintfull/hintswap × molwenik1/nomolweni`), all splits.
- [ ] Re-score the existing generation files to recover correct F1 of the current models (CPU-only; the `assert` now protects against silent misalignment).
- [ ] Retrain on the regenerated full-length data for the final thesis numbers.
- [ ] Decide whether to drop the `-4` entirely (paper-faithful) vs. keep it gated to STAC/Molweni (comparability).
