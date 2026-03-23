#!/bin/bash

# Hyperparameter configurations based on Li et al. (2024)
# Learning rate: 5e-5 for models up to 3B/XL, 2e-5 for XXL/11B
# Batch size: 8 for base/large/1B models, 4 for 3B+ models
# Epochs: STAC=10, Molweni=3
# Steps: 500 default, 2000 for Molweni transition-based (focus, natural2)

############################### molweni natural ################################
# python3 train.py --train_corpus molweni --do_train -s natural -t t5gemma2 -m 270m -l 5e-5 -e 10 --batchsize 4 --step 500 -b

# python3 train.py --train_corpus molweni --do_train -s natural -t t5gemma2 -m 1b -l 5e-5 -e 10 --batchsize 4 --step 500 -b

# python3 train.py --train_corpus molweni --do_train -s natural -t t5gemma2 -m 4b -l 5e-5 -e 10 --batchsize 4 --step 500 -b

# python3 train.py --train_corpus molweni --do_train -s natural -t t5 -m 3b -l 5e-5 -e 3 --batchsize 4 --step 500

# python3 train.py --train_corpus molweni --do_train -s natural -t t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 500

# python3 train.py --train_corpus molweni --do_train -s natural -t flan-t5 -m base -l 5e-5 -e 3 --batchsize 8 --step 500

# python3 train.py --train_corpus molweni --do_train -s natural -t flan-t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 500

# python3 train.py --train_corpus molweni --do_train -s natural -t flan-t5 -m xl -l 5e-5 -e 3 --batchsize 4 --step 500

# python3 train.py --train_corpus molweni --do_train -s natural -t flan-t5 -m xxl -l 2e-5 -e 3 --batchsize 4 --step 500

# python3 train.py --train_corpus molweni --do_train -s natural -t t0-3b -l 2e-5 -e 10 --batchsize 4 --step 500

############################### molweni augmented ################################

# python3 train.py --train_corpus molweni --do_train -s augmented -t t5gemma2 -m 270m -l 5e-5 -e 10 --batchsize 4 --step 500 -b

# python3 train.py --train_corpus molweni --do_train -s augmented -t t5gemma2 -m 1b -l 5e-5 -e 10 --batchsize 4 --step 500 -b

# python3 train.py --train_corpus molweni --do_train -s augmented -t t5gemma2 -m 4b -l 5e-5 -e 10 --batchsize 4 --step 500 -b

# python3 train.py --train_corpus molweni --do_train -s augmented -t t5 -m 3b -l 5e-5 -e 3 --batchsize 4 --step 500

# python3 train.py --train_corpus molweni --do_train -s augmented -t t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 500

# python3 train.py --train_corpus molweni --do_train -s augmented -t flan-t5 -m base -l 5e-5 -e 3 --batchsize 8 --step 500

# python3 train.py --train_corpus molweni --do_train -s augmented -t flan-t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 500

# python3 train.py --train_corpus molweni --do_train -s augmented -t flan-t5 -m xl -l 5e-5 -e 3 --batchsize 4 --step 500

# python3 train.py --train_corpus molweni --do_train -s augmented -t flan-t5 -m xxl -l 2e-5 -e 3 --batchsize 4 --step 500

# python3 train.py --train_corpus molweni --do_train -s augmented -t t0-3b -l 2e-5 -e 10 --batchsize 4 --step 500

############################### molweni focus ################################
# Transition-based method: use step=2000 for Molweni

# python3 train.py --train_corpus molweni --do_train -s focus -t t5gemma2 -m 270m -l 5e-5 -e 5 --batchsize 4 --step 2000 -b

# python3 train.py --train_corpus molweni --do_train -s focus -t t5gemma2 -m 1b -l 5e-5 -e 5 --batchsize 4 --step 2000 -b

# python3 train.py --train_corpus molweni --do_train -s focus -t t5gemma2 -m 4b -l 5e-5 -e 5 --batchsize 4 --step 2000 -b

# python3 train.py --train_corpus molweni --do_train -s focus -t t5 -m 3b -l 5e-5 -e 3 --batchsize 4 --step 2000

# python3 train.py --train_corpus molweni --do_train -s focus -t t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 2000

# python3 train.py --train_corpus molweni --do_train -s focus -t flan-t5 -m base -l 5e-5 -e 3 --batchsize 8 --step 2000

# python3 train.py --train_corpus molweni --do_train -s focus -t flan-t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 2000

# python3 train.py --train_corpus molweni --do_train -s focus -t flan-t5 -m xl -l 5e-5 -e 3 --batchsize 4 --step 2000

# python3 train.py --train_corpus molweni --do_train -s focus -t flan-t5 -m xxl -l 2e-5 -e 3 --batchsize 4 --step 2000

# python3 train.py --train_corpus molweni --do_train -s focus -t t0-3b -l 2e-5 -e 5 --batchsize 4 --step 2000

############################### molweni natural2 ################################
# Transition-based method: use step=2000 for Molweni

# python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 270m -l 5e-5 -e 5 --batchsize 4 --step 2000 -b

# python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 1b -l 5e-5 -e 5 --batchsize 4 --step 2000 -b

# python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 4b -l 5e-5 -e 5 --batchsize 4 --step 2000 -b

# python3 train.py --train_corpus molweni --do_train -s natural2 -t t5 -m 3b -l 5e-5 -e 3 --batchsize 4 --step 2000

# python3 train.py --train_corpus molweni --do_train -s natural2 -t t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 2000

# python3 train.py --train_corpus molweni --do_train -s natural2 -t flan-t5 -m base -l 5e-5 -e 3 --batchsize 8 --step 2000

# python3 train.py --train_corpus molweni --do_train -s natural2 -t flan-t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 2000

# python3 train.py --train_corpus molweni --do_train -s natural2 -t flan-t5 -m xl -l 5e-5 -e 3 --batchsize 4 --step 2000

# python3 train.py --train_corpus molweni --do_train -s natural2 -t flan-t5 -m xxl -l 2e-5 -e 3 --batchsize 4 --step 2000

# python3 train.py --train_corpus molweni --do_train -s natural2 -t t0-3b -l 2e-5 -e 5 --batchsize 4 --step 2000

############################### stac natural ################################

# python3 train.py --train_corpus stac --do_train -s natural -t t5gemma2 -m 270m -l 5e-5 -e 20 --batchsize 4 --step 500 -b

# python3 train.py --train_corpus stac --do_train -s natural -t t5gemma2 -m 1b -l 5e-5 -e 20 --batchsize 4 --step 500 -b

# python3 train.py --train_corpus stac --do_train -s natural -t t5gemma2 -m 4b -l 5e-5 -e 20 --batchsize 4 --step 500 -b

# python3 train.py --train_corpus stac --do_train -s natural -t t5 -m 3b -l 5e-5 -e 10 --batchsize 4 --step 500

# python3 train.py --train_corpus stac --do_train -s natural -t t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500

# python3 train.py --train_corpus stac --do_train -s natural -t flan-t5 -m base -l 5e-5 -e 10 --batchsize 8 --step 500

# python3 train.py --train_corpus stac --do_train -s natural -t flan-t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500

# python3 train.py --train_corpus stac --do_train -s natural -t flan-t5 -m xl -l 5e-5 -e 10 --batchsize 4 --step 500

# python3 train.py --train_corpus stac --do_train -s natural -t flan-t5 -m xxl -l 2e-5 -e 10 --batchsize 4 --step 500

# python3 train.py --train_corpus stac --do_train -s natural -t t0-3b -l 2e-5 -e 20 --batchsize 4 --step 500

############################### stac augmented ################################

# python3 train.py --train_corpus stac --do_train -s augmented -t t5gemma2 -m 270m -l 5e-5 -e 20 --batchsize 4 --step 500 -b

# python3 train.py --train_corpus stac --do_train -s augmented -t t5gemma2 -m 1b -l 5e-5 -e 20 --batchsize 4 --step 500 -b

# python3 train.py --train_corpus stac --do_train -s augmented -t t5gemma2 -m 4b -l 5e-5 -e 20 --batchsize 4 --step 500 -b

# python3 train.py --train_corpus stac --do_train -s augmented -t t5 -m 3b -l 5e-5 -e 10 --batchsize 4 --step 500

# python3 train.py --train_corpus stac --do_train -s augmented -t t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500

# python3 train.py --train_corpus stac --do_train -s augmented -t flan-t5 -m base -l 5e-5 -e 10 --batchsize 8 --step 500

# python3 train.py --train_corpus stac --do_train -s augmented -t flan-t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500

# python3 train.py --train_corpus stac --do_train -s augmented -t flan-t5 -m xl -l 5e-5 -e 10 --batchsize 4 --step 500

# python3 train.py --train_corpus stac --do_train -s augmented -t flan-t5 -m xxl -l 2e-5 -e 10 --batchsize 4 --step 500

# python3 train.py --train_corpus stac --do_train -s augmented -t t0-3b -l 2e-5 -e 20 --batchsize 4 --step 500

############################### stac focus ################################
# Transition-based method: use step=500 for STAC

# python3 train.py --train_corpus stac --do_train -s focus -t t5gemma2 -m 270m -l 5e-5 -e 10 --batchsize 4 --step 500 -b

# python3 train.py --train_corpus stac --do_train -s focus -t t5gemma2 -m 1b -l 5e-5 -e 10 --batchsize 4 --step 500 -b

# python3 train.py --train_corpus stac --do_train -s focus -t t5gemma2 -m 4b -l 5e-5 -e 10 --batchsize 4 --step 500 -b

# python3 train.py --train_corpus stac --do_train -s focus -t t5 -m 3b -l 5e-5 -e 10 --batchsize 4 --step 500

# python3 train.py --train_corpus stac --do_train -s focus -t t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500

# python3 train.py --train_corpus stac --do_train -s focus -t flan-t5 -m base -l 5e-5 -e 10 --batchsize 8 --step 500

# python3 train.py --train_corpus stac --do_train -s focus -t flan-t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500

# python3 train.py --train_corpus stac --do_train -s focus -t flan-t5 -m xl -l 5e-5 -e 10 --batchsize 4 --step 500

# python3 train.py --train_corpus stac --do_train -s focus -t flan-t5 -m xxl -l 2e-5 -e 10 --batchsize 4 --step 500

# python3 train.py --train_corpus stac --do_train -s focus -t t0-3b -l 2e-5 -e 10 --batchsize 4 --step 500

############################### stac natural2 ################################
# Transition-based method: use step=500 for STAC

# python3 train.py --train_corpus stac --do_train -s natural2 -t t5gemma2 -m 270m -l 5e-5 -e 10 --batchsize 4 --step 500 -b

# python3 train.py --train_corpus stac --do_train -s natural2 -t t5gemma2 -m 1b -l 5e-5 -e 10 --batchsize 4 --step 500 -b

# python3 train.py --train_corpus stac --do_train -s natural2 -t t5gemma2 -m 4b -l 5e-5 -e 10 --batchsize 4 --step 500 -b

# python3 train.py --train_corpus stac --do_train -s natural2 -t t5 -m 3b -l 5e-5 -e 10 --batchsize 4 --step 500

# python3 train.py --train_corpus stac --do_train -s natural2 -t t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500

# python3 train.py --train_corpus stac --do_train -s natural2 -t flan-t5 -m base -l 5e-5 -e 10 --batchsize 8 --step 500

# python3 train.py --train_corpus stac --do_train -s natural2 -t flan-t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500

# python3 train.py --train_corpus stac --do_train -s natural2 -t flan-t5 -m xl -l 5e-5 -e 10 --batchsize 4 --step 500

# python3 train.py --train_corpus stac --do_train -s natural2 -t flan-t5 -m xxl -l 2e-5 -e 10 --batchsize 4 --step 500

# python3 train.py --train_corpus stac --do_train -s natural2 -t t0-3b -l 2e-5 -e 10 --batchsize 4 --step 500


############################### molweni transition with t0gemma2 ################################
python3 train.py --train_corpus molweni --do_train -s focus -t t0gemma2 -m 270m -l 5e-6 -e 5 --batchsize 4 --step 2000 --warmup_ratio 0.1 --max_grad_norm 0.5 --weight_decay 0.01 -b
python3 train.py --train_corpus molweni --do_train -s focus -t t0gemma2 -m 1b -l 5e-6 -e 5 --batchsize 4 --step 2000 --warmup_ratio 0.1 --max_grad_norm 0.5 --weight_decay 0.01 -b
python3 train.py --train_corpus molweni --do_train -s focus -t t0gemma2 -m 4b -l 5e-6 -e 5 --batchsize 4 --step 2000 --warmup_ratio 0.1 --max_grad_norm 0.5 --weight_decay 0.01 -b
python3 train.py --train_corpus molweni --do_train -s natural2 -t t0gemma2 -m 270m -l 5e-6 -e 5 --batchsize 4 --step 2000 --warmup_ratio 0.1 --max_grad_norm 0.5 --weight_decay 0.01 -b
python3 train.py --train_corpus molweni --do_train -s natural2 -t t0gemma2 -m 1b -l 5e-6 -e 5 --batchsize 4 --step 2000 --warmup_ratio 0.1 --max_grad_norm 0.5 --weight_decay 0.01 -b
python3 train.py --train_corpus molweni --do_train -s natural2 -t t0gemma2 -m 4b -l 5e-6 -e 5 --batchsize 4 --step 2000 --warmup_ratio 0.1 --max_grad_norm 0.5 --weight_decay 0.01 -b

