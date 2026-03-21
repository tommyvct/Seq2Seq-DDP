# Results

## t0-3b-3b molweni natural2 --lr 2e-5 --seed 27
```
[link+rel] recall: 59.51, precision: 59.26, f1: 59.38
[linkonly] recall: 83.54, precision: 83.26, f1: 83.4
```

## t5gemma2-270m molweni natural2 --lr 5e-5 --seed 27
```
[link+rel] recall: 22.65, precision: 22.55, f1: 22.6
[linkonly] recall: 33.96, precision: 33.85, f1: 33.9
```

## t5gemma2-1b molweni natural2 --lr 5e-5 --seed 27
```
[link+rel] recall: 15.71, precision: 15.64, f1: 15.68
[linkonly] recall: 44.84, precision: 44.69, f1: 44.77
```

## t5gemma2-4b molweni natural2 --lr 5e-5 --seed 27
```
[link+rel] recall: 35.98, precision: 35.82, f1: 35.9
[linkonly] recall: 60.78, precision: 60.57, f1: 60.68
```
## t0gemma2-270m molweni natural2 --lr 2e-5 --seed 42
```
[link+rel] recall: 10.72, precision: 11.34, f1: 11.02
[linkonly] recall: 10.72, precision: 11.36, f1: 11.03
```

## t0gemma2-1b molweni natural2 --lr 2e-5 --seed 42
```
[link+rel] recall: 53.5, precision: 53.27, f1: 53.39
[linkonly] recall: 77.01, precision: 76.75, f1: 76.88
```



# Training Configs
- `python3 train.py --train_corpus molweni --do_train -s natural2 -t t0-3b -l 2e-5 -e 5 --batchsize 4 --step 2000`
- `python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 270m -l 5e-5 -e 5 --batchsize 4 --step 2000 -b`
- `python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 1b -l 5e-5 -e 5 --batchsize 4 --step 2000 -b`
- `python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 4b -l 5e-5 -e 5 --batchsize 4 --step 2000 -b`