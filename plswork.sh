# T5Gemma2-4b, learning rate 2e-5

# sbatch << 'EOF'
# #!/bin/bash
# #SBATCH --job-name=t5gemma2-4b_lr2e-5_e5_b4_s2000_molweni_focus
# #SBATCH --output=slurm/logs/%j_t5gemma2-4b_lr2e-5_e5_b4_s2000_molweni_focus.out
# #SBATCH --error=slurm/logs/%j_t5gemma2-4b_lr2e-5_e5_b4_s2000_molweni_focus.err
# #SBATCH --time=26:00:00
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=64G
# #SBATCH --gpus=h100:1
# #SBATCH --mail-type=FAIL,END
# #SBATCH --mail-user=wut2@unbc.ca

# cd $SCRATCH/Seq2Seq-DDP
# source slurm/init_hpc.sh
# python3 train.py --train_corpus molweni --do_train -s focus -t t5gemma2 -m 4b -l 2e-5 -e 5 --batchsize 4 --step 2000 -b
# echo "Training Complete"
# python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s focus -t t5gemma2 -m 4b --lr 2e-5 --seed 27 -b
# echo "Inference Complete"
# python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s focus --lr 2e-5 --seed 27 > eval/t5gemma2-4b_2e-5_e5_b4_molweni_focus.txt
# EOF

sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=t5gemma2-4b_lr2e-5_e5_b4_s2000_molweni_natural2
#SBATCH --output=slurm/logs/%j_t5gemma2-4b_lr2e-5_e5_b4_s2000_molweni_natural2.out
#SBATCH --error=slurm/logs/%j_t5gemma2-4b_lr2e-5_e5_b4_s2000_molweni_natural2.err
#SBATCH --time=26:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca

cd $SCRATCH/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 4b -l 2e-5 -e 5 --batchsize 4 --step 2000 -b
echo "Training Complete"
python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t5gemma2 -m 4b --lr 2e-5 --seed 27 -b
echo "Inference Complete"
python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s natural2 --lr 2e-5 --seed 27 > eval/t5gemma2-4b_2e-5_e5_b4_molweni_natural2.txt
EOF

# more conservative learning rate 5e-6

# sbatch << 'EOF'
# #!/bin/bash
# #SBATCH --job-name=t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_focus
# #SBATCH --output=slurm/logs/%j_t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_focus.out
# #SBATCH --error=slurm/logs/%j_t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_focus.err
# #SBATCH --time=26:00:00
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=64G
# #SBATCH --gpus=h100:1
# #SBATCH --mail-type=FAIL,END
# #SBATCH --mail-user=wut2@unbc.ca

# cd $SCRATCH/Seq2Seq-DDP
# source slurm/init_hpc.sh
# python3 train.py --train_corpus molweni --do_train -s focus -t t5gemma2 -m 4b -l 5e-6 -e 5 --batchsize 4 --step 2000 -b
# echo "Training Complete"
# python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s focus -t t5gemma2 -m 4b --lr 5e-6 --seed 27 -b
# echo "Inference Complete"
# python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s focus --lr 5e-6 --seed 27 > eval/t5gemma2-4b_5e-6_e5_b4_molweni_focus.txt
# EOF

sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_natural2
#SBATCH --output=slurm/logs/%j_t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_natural2.out
#SBATCH --error=slurm/logs/%j_t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_natural2.err
#SBATCH --time=26:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca

cd $SCRATCH/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 4b -l 5e-6 -e 5 --batchsize 4 --step 2000 -b
echo "Training Complete"
python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t5gemma2 -m 4b  --lr 5e-6 --seed 27 -b
echo "Inference Complete"
python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-6 --seed 27 > eval/t5gemma2-4b_5e-6_e5_b4_molweni_natural2.txt
EOF

# more conservative learning rate 5e-6 with extra hyperparameters (warmup_ratio, max_grad_norm, weight_decay)

# sbatch << 'EOF'
# #!/bin/bash
# #SBATCH --job-name=t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_focus_warmup
# #SBATCH --output=slurm/logs/%j_t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_focus_warmup.out
# #SBATCH --error=slurm/logs/%j_t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_focus_warmup.err
# #SBATCH --time=26:00:00
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=64G
# #SBATCH --gpus=h100:1
# #SBATCH --mail-type=FAIL,END
# #SBATCH --mail-user=wut2@unbc.ca

# cd $SCRATCH/Seq2Seq-DDP
# source slurm/init_hpc.sh
# python3 train.py --train_corpus molweni --do_train -s focus -t t5gemma2 -m 4b -l 5e-6 -e 5 --batchsize 4 --step 2000 --warmup_ratio 0.1 --max_grad_norm 0.5 --weight_decay 0.01 -b
# echo "Training Complete"
# python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s focus -t t5gemma2 -m 4b --lr 5e-6 --seed 27 --warmup_ratio 0.1 --max_grad_norm 0.5 --weight_decay 0.01 -b
# echo "Inference Complete"
# python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s focus --lr 5e-6 --seed 27 --warmup_ratio 0.1 --max_grad_norm 0.5 --weight_decay 0.01 > eval/t5gemma2-4b_5e-6_e5_b4_molweni_focus_warmup.txt
# EOF

sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_natural2_warmup
#SBATCH --output=slurm/logs/%j_t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_natural2_warmup.out
#SBATCH --error=slurm/logs/%j_t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_natural2_warmup.err
#SBATCH --time=26:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca

cd $SCRATCH/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 4b -l 5e-6 -e 5 --batchsize 4 --step 2000 --warmup_ratio 0.1 --max_grad_norm 0.5 --weight_decay 0.01 -b
echo "Training Complete"
python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t5gemma2 -m 4b --lr 5e-6 --seed 27 --warmup_ratio 0.1 --max_grad_norm 0.5 --weight_decay 0.01 -b
echo "Inference Complete"
python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-6 --seed 27 --warmup_ratio 0.1 --max_grad_norm 0.5 --weight_decay 0.01 > eval/t5gemma2-4b_5e-6_e5_b4_molweni_natural2_warmup.txt
EOF

# On top of the conservative learning rate and additional hyperparameters, we also add the new prompt and see if it can help.

# sbatch << 'EOF'
# #!/bin/bash
# #SBATCH --job-name=t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_focus_newprompt_warmup
# #SBATCH --output=slurm/logs/%j_t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_focus_newprompt_warmup.out
# #SBATCH --error=slurm/logs/%j_t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_focus_newprompt_warmup.err
# #SBATCH --time=26:00:00
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=64G
# #SBATCH --gpus=h100:1
# #SBATCH --mail-type=FAIL,END
# #SBATCH --mail-user=wut2@unbc.ca

# cd $SCRATCH/Seq2Seq-DDP
# source slurm/init_hpc.sh
# python3 train.py --train_corpus molweni --do_train -s focus -t t5gemma2 -m 4b -l 5e-6 -e 5 --batchsize 4 --step 2000 --new_prompt --warmup_ratio 0.1 --max_grad_norm 0.5 --weight_decay 0.01 -b
# echo "Training Complete"
# python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s focus -t t5gemma2 -m 4b  --lr 5e-6 --seed 27 --new_prompt --warmup_ratio 0.1 --max_grad_norm 0.5 --weight_decay 0.01 -b
# echo "Inference Complete"
# python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s focus --lr 5e-6 --seed 27 --new_prompt --warmup_ratio 0.1 --max_grad_norm 0.5 --weight_decay 0.01 > eval/t5gemma2-4b_5e-6_e5_b4_molweni_focus_newprompt_warmup.txt
# EOF

sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_natural2_newprompt_warmup
#SBATCH --output=slurm/logs/%j_t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_natural2_newprompt_warmup.out
#SBATCH --error=slurm/logs/%j_t5gemma2-4b_lr5e-6_e5_b4_s2000_molweni_natural2_newprompt_warmup.err
#SBATCH --time=26:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca

cd $SCRATCH/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 4b -l 5e-6 -e 5 --batchsize 4 --step 2000 --new_prompt --warmup_ratio 0.1 --max_grad_norm 0.5 --weight_decay 0.01 -b
echo "Training Complete"
python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t5gemma2 -m 4b  --lr 5e-6 --seed 27 --new_prompt --warmup_ratio 0.1 --max_grad_norm 0.5 --weight_decay 0.01 -b
echo "Inference Complete"
python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-6 --seed 27 --new_prompt --warmup_ratio 0.1 --max_grad_norm 0.5 --weight_decay 0.01 > eval/t5gemma2-4b_5e-6_e5_b4_molweni_natural2_newprompt_warmup.txt
EOF





# sbatch <<'EOF'
# #!/bin/bash
# #SBATCH --job-name=t0gemma2-4b_lr2e-5_e5_b4_s2000_molweni_focus
# #SBATCH --output=slurm/logs/%j_t0gemma2-4b_lr2e-5_e5_b4_s2000_molweni_focus.out
# #SBATCH --error=slurm/logs/%j_t0gemma2-4b_lr2e-5_e5_b4_s2000_molweni_focus.err
# #SBATCH --mail-type=FAIL,END
# #SBATCH --mail-user=wut2@unbc.ca
# #SBATCH --gpus=h100:1 --cpus-per-task=6 --mem=64G --time=26:00:00

# cd $SCRATCH/Seq2Seq-DDP
# source slurm/init_hpc.sh
# python3 train.py --train_corpus molweni --do_train -s focus -t t0gemma2 -m 4b -l 2e-5 -e 5 --batchsize 4 --step 2000 -b --seed 27
# python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s focus -t t0gemma2 -m 4b --lr 2e-5 --seed 27 -b
# python3 eval_gen.py --fted_model t0gemma2-4b --train_corpus molweni --test_corpus molweni -s focus --lr 2e-5 --seed 27 > eval/t0gemma2-4b_2e-5_e5_b4_molweni_focus.txt

# EOF

sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=t0gemma2-4b_lr2e-5_e5_b4_s2000_molweni_natural2
#SBATCH --output=slurm/logs/%j_t0gemma2-4b_lr2e-5_e5_b4_s2000_molweni_natural2.out
#SBATCH --error=slurm/logs/%j_t0gemma2-4b_lr2e-5_e5_b4_s2000_molweni_natural2.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=6 --mem=64G --time=26:00:00

cd $SCRATCH/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t0gemma2 -m 4b -l 2e-5 -e 5 --batchsize 4 --step 2000 -b --seed 27
python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t0gemma2 -m 4b --lr 2e-5 --seed 27 -b
python3 eval_gen.py --fted_model t0gemma2-4b --train_corpus molweni --test_corpus molweni -s natural2 --lr 2e-5 --seed 27 > eval/t0gemma2-4b_2e-5_e5_b4_molweni_natural2.txt

EOF