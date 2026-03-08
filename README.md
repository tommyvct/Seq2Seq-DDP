# Dialogue Discourse Parsing as Generation: a Sequence-to-Sequence LLM-based Approach

🏆 Best Paper (SIGdial 2024) [[Award]](https://2024.sigdial.org/award/)

This is the source code repository for the paper Dialogue Discourse Parsing as Generation: a Sequence-to-Sequence LLM-based Approach ([SIGDial 2024](https://2024.sigdial.org)).

<img src="./pic/seq2seq-disc-parse.png" alt="drawing" width="600"/>

## Datasets

### STAC
~~We used the **linguistic-only** STAC corpus and followed the separation of train, dev, test in [Shi and Huang, A Deep Sequential Model for Discourse Parsing on Multi-Party Dialogues. In AAAI, 2019](https://github.com/shizhouxing/DialogueDiscourseParsing).
The latest available verison on the website is available [here](https://www.irit.fr/STAC/corpus.html). 
We share the dataset we used in `data/stac/`.~~

Now included.


### Molweni
~~Download from [here](https://github.com/HIT-SCIR/Molweni). We use the original separation of train, dev, and test.
Download the dataset and place it in `data/molweni/`.~~

Now included

### Discord Unveiled
TODO: 

## How to run
Here is a step-by-step guide to fine-tune a T5 family model for discourse parsing:

### Create a virtual environment
Python version: 3.12
```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install jellyfish
$ pip install torch torchvision torchaudio --no-index
$ pip install -r requirements.txt
```

Run `download_nltk_punkt.py` to download the nltk punkt package for sentence tokenization.

### Prepare structured data for fine-tuning
>  TLDR: Run `run_dataprocess.py` to convert the original stac/molweni dataset to structured text for fine-tuning.

In `dataprocess.py`: process the original stac/molweni dataset and convert the raw text to structured text.
Choose the structured text from: 
- Seq2Seq-DDP system: 'natural', 'augmented'
- Seq2Seq-DDP+transition system: 'focus', 'natural2'

Examples for each structure type are given in `data/stac_{structure}_train.json`.

Note that 'focus' and 'natural2' schemes are converted from 'natural' scheme.

We provide STAC 'natural' and 'focus' format for your convenience.

### Fine-tuning
>  TLDR: All possible fine-tuning configurations are given in `run_predict.sh`.

In `train.py`: pass "do_train" as argument.
This code fine-tunes a t5 familiy model for discourse parsing.

**Note**: download the original model with fp32.

### T0Gemma2: Instruction Tuning T5Gemma2 on P3

T0Gemma2 replicates the [T0](https://arxiv.org/abs/2110.08207) instruction tuning process on [T5Gemma2](https://huggingface.co/collections/google/t5gemma-2-release-68418590dda39a4891ba6290) models, replacing T0-3B in the downstream DDP pipeline.

#### Why T0Gemma2?

T0-3B was instruction-tuned on the [P3 (Public Pool of Prompts)](https://huggingface.co/datasets/bigscience/P3) dataset, giving it strong zero-shot task generalization. Vanilla T5Gemma2 lacks this capability. T0Gemma2 bridges this gap by instruction-tuning T5Gemma2 on the same P3 mixture used for T0.

For the Dialogue Discourse Parsing task, we find that the vanilla T5Gemma2 model performs around 20 points worse than T0-3b. 

#### Training Configuration (matching T0)

| Parameter | Value |
|---|---|
| Dataset | T0 BASE mixture: 38 datasets, 313 P3 templates |
| Per-template cap | `min(train_size, 500K // num_templates)` |
| Optimizer | Adafactor (`scale_parameter=False, relative_step=False`) |
| Learning rate | 1e-3 (constant with 100-step linear warmup) |
| Effective batch size | 1024 |
| Training steps | 12,200 |
| Precision | bfloat16 |
| Gradient checkpointing | Enabled |

#### Model Sizes

| Label | Actual size (enc + dec) |
|---|---|
| 1b | 2B (1B encoder + 1B decoder) |
| 4b | 8B (4B encoder + 4B decoder) |

#### How to Run

1. **Run instruction tuning** (single GPU):
   ```bash
   python3 inst_tuning_t0gemma2.py --model_size "1b" --batch_size 64 --gradient_accumulation_steps 16 --seed 27
   ```

2. **Run instruction tuning** (multi-GPU via torchrun):
   ```bash
   torchrun --nproc_per_node=8 inst_tuning_t0gemma2.py \
       --model_size "1b" --batch_size 64 --gradient_accumulation_steps 2 --seed 27
   ```
   Effective batch = `batch_size * num_gpus * gradient_accumulation_steps` (target: 1024)

3. **Submit full pipeline on SLURM** (instruction tuning -> fine-tuning -> inference -> evaluation):
   ```bash
   bash submit_t0gemma2.sh
   ```

#### Smoke Test
```bash
python3 inst_tuning_t0gemma2.py \
    --model_size "270m" \
    --batch_size 2 \
    --max_source_length 256 \
    --max_target_length 64 \
    --max_steps 10 \
    --warmup_steps 1 \
    --seed 27
```


### End2end prediction and transition-based prediction
>  TLDR: All possible prediction configurations are given in `run_test.sh`.

- Seq2seq-DDP prediction: in `train.py`, pass argument "do_test" for `end2end`-style prediction, choose structure type from {'augmented', 'natural'}.

Make sure to first put the fine-tuned model checkpoint in `constant.py`. Results will be written in repo `generation/`.

- Seq2Seq-DDP+transition system prediction: in `transition_predict.py`: choose structure type from {'focus', 'natural2'}.

- `eval_gen.py`: Evaluate predicted files in `generation/` and calculate scores.

- `constant.py`: store paths, labels, T0 BASE training mixture (`T0_TRAIN_TASKS`), etc.

- `inst_tuning_t0gemma2.py`: instruction-tunes T5Gemma2 on P3 to create T0Gemma2.

- `submit_t0gemma2.sh`: SLURM job orchestration for the full T0Gemma2 pipeline (instruction tuning -> fine-tuning -> inference -> evaluation).

## Support for SLURM-based clusters
`slurm/` contains 2 helper python program to generate shell scripts for submitting fine-tuning and prediction on SLURM-based clusters.

`slurm/init_hpc.sh` is a shell script that initializes correct python environments and dependencies for this project. This script should be sourced instead of running it directly.

## Citation
```
@inproceedings{li-etal-2024-dialogue,
    title = "Dialogue Discourse Parsing as Generation: A Sequence-to-Sequence {LLM}-based Approach",
    author = "Li, Chuyuan  and
      Yin, Yuwei  and
      Carenini, Giuseppe",
    editor = "Kawahara, Tatsuya  and
      Demberg, Vera  and
      Ultes, Stefan  and
      Inoue, Koji  and
      Mehri, Shikib  and
      Howcroft, David  and
      Komatani, Kazunori",
    booktitle = "Proceedings of the 25th Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    month = sep,
    year = "2024",
    address = "Kyoto, Japan",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.sigdial-1.1",
    doi = "10.18653/v1/2024.sigdial-1.1",
    pages = "1--14",
}
```
