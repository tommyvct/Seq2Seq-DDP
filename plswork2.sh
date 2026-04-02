
# Job 0: Epochs 3
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=t5gemma2-4b_lr5e-6_e3_b4_s2000_molweni_natural2
#SBATCH --output=slurm/logs/%j_t5gemma2-4b_lr5e-6_e3_b4_s2000_molweni_natural2.out
#SBATCH --error=slurm/logs/%j_t5gemma2-4b_lr5e-6_e3_b4_s2000_molweni_natural2.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=wut2@unbc.ca

cd $SCRATCH/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 4b -l 5e-6 -e 3 --batchsize 4 --step 2000 -b
echo "Training Complete"
python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t5gemma2 -m 4b  --lr 5e-6 --batchsize 4 --step 2000 --seed 27 -b
echo "Inference Complete"
python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-6  --batchsize 4 --step 2000 --seed 27 > eval/t5gemma2-4b_5e-6_e3_b4_s2000_molweni_natural2.txt
EOF

# Job 1: New Prompt, Default Batch Size (4)
sbatch << 'INNER_EOF'
#!/bin/bash
#SBATCH --job-name=t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_natural2_newprompt
#SBATCH --output=slurm/logs/%j_t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_natural2_newprompt.out
#SBATCH --error=slurm/logs/%j_t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_natural2_newprompt.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=wut2@unbc.ca

cd $SCRATCH/Seq2Seq-DDP
source slurm/init_hpc.sh

python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 4b -l 5e-6 -e 5 --batchsize 4 --step 2000 --new_prompt -b
echo "Training Complete"

python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t5gemma2 -m 4b --lr 5e-6 --batchsize 4 --step 2000 --seed 27 --new_prompt -b
echo "Inference Complete"

python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-6 --batchsize 4 --step 2000 --seed 27 --new_prompt > eval/t5gemma2-4b_5e-6_e5_b4_s2000_molweni_natural2_newprompt.txt
INNER_EOF

# Job 2: High Batch Size (16), Default Prompt
sbatch << 'INNER_EOF'
#!/bin/bash
#SBATCH --job-name=t5gemma2-4b_lr5e-6_e5_b16_s2000_molweni_natural2
#SBATCH --output=slurm/logs/%j_t5gemma2-4b_lr5e-6_e5_b16_s2000_molweni_natural2.out
#SBATCH --error=slurm/logs/%j_t5gemma2-4b_lr5e-6_e5_b16_s2000_molweni_natural2.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=wut2@unbc.ca

cd $SCRATCH/Seq2Seq-DDP
source slurm/init_hpc.sh

python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 4b -l 5e-6 -e 5 --batchsize 16 --step 2000 -b
echo "Training Complete"

python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t5gemma2 -m 4b --lr 5e-6 --batchsize 16 --step 2000 --seed 27 -b
echo "Inference Complete"

python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-6 --batchsize 16 --step 2000 --seed 27 > eval/t5gemma2-4b_5e-6_e5_b16_s2000_molweni_natural2.txt
INNER_EOF

# Job 3: High Batch Size (16) AND New Prompt
sbatch << 'INNER_EOF'
#!/bin/bash
#SBATCH --job-name=t5gemma2-4b_lr5e-6_e5_b16_s2000_molweni_natural2_newprompt
#SBATCH --output=slurm/logs/%j_t5gemma2-4b_lr5e-6_e5_b16_s2000_molweni_natural2_newprompt.out
#SBATCH --error=slurm/logs/%j_t5gemma2-4b_lr5e-6_e5_b16_s2000_molweni_natural2_newprompt.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=wut2@unbc.ca

cd $SCRATCH/Seq2Seq-DDP
source slurm/init_hpc.sh

python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 4b -l 5e-6 -e 5 --batchsize 16 --step 2000 --new_prompt -b
echo "Training Complete"

python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t5gemma2 -m 4b --lr 5e-6 --batchsize 16 --step 2000 --seed 27 --new_prompt -b
echo "Inference Complete"

python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-6 --batchsize 16 --step 2000 --seed 27 --new_prompt > eval/t5gemma2-4b_5e-6_e5_b16_s2000_molweni_natural2_newprompt.txt
INNER_EOF


# Job 4: b8 with default prompt and new prompt
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=t5gemma2-4b_lr5e-6_e3_b8_s2000_molweni_natural2
#SBATCH --output=slurm/logs/%j_t5gemma2-4b_lr5e-6_e3_b8_s2000_molweni_natural2.out
#SBATCH --error=slurm/logs/%j_t5gemma2-4b_lr5e-6_e3_b8_s2000_molweni_natural2.err
#SBATCH --time=26:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=wut2@unbc.ca

cd $SCRATCH/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 4b -l 5e-6 -e 3 --batchsize 8 --step 2000 -b
echo "Training Complete"
python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t5gemma2 -m 4b  --lr 5e-6  -e 3 --batchsize 8 --step 2000 --seed 27 -b 
echo "Inference Complete"
python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-6 --seed 27 > eval/t5gemma2-4b_5e-6_e3_b8_molweni_natural2.txt
EOF

sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=t5gemma2-4b_lr5e-6_e3_b8_s2000_molweni_natural2_newprompt
#SBATCH --output=slurm/logs/%j_t5gemma2-4b_lr5e-6_e3_b8_s2000_molweni_natural2_newprompt.out
#SBATCH --error=slurm/logs/%j_t5gemma2-4b_lr5e-6_e3_b8_s2000_molweni_natural2_newprompt.err
#SBATCH --time=26:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=wut2@unbc.ca

cd $SCRATCH/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 4b -l 5e-6 -e 3 --batchsize 8 --step 2000 -b --new_prompt
echo "Training Complete"
python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t5gemma2 -m 4b  --lr 5e-6  -e 3 --batchsize 8 --step 2000 --seed 27 -b --new_prompt
echo "Inference Complete"
python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-6 --seed 27 --new_prompt > eval/t5gemma2-4b_5e-6_e3_b8_molweni_natural2_newprompt.txt
EOF

# Next stop, try epoch 7