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
Download the dataset and place it in `data/molweni/`. ~~

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
$ pip install torch torchvision torchaudio
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


### End2end prediction and transition-based prediction
>  TLDR: All possible prediction configurations are given in `run_test.sh`.

- Seq2seq-DDP prediction: in `train.py`, pass argument "do_test" for `end2end`-style prediction, choose structure type from {'augmented', 'natural'}.

Make sure to first put the fine-tuned model checkpoint in `constant.py`. Results will be written in repo `generation/`.

- Seq2Seq-DDP+transition system prediction: in `transition_predict.py`: choose structure type from {'focus', 'natural2'}.

- `eval_gen.py`: Evaluate predicted files in `generation/` and calculate scores.

- `constant.py`: store paths, labels, etc.

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
