#!/bin/bash

mkdir -p slurm/logs/train
mkdir -p slurm/logs/inference


# ==========================================
# Instruction Tuning (Pre-training)
# ==========================================

JID_INST_4B=$(sbatch --parsable <<'EOT'
#!/bin/bash
#SBATCH --job-name=inst_tuning_t0gemma2_4b
#SBATCH --output=slurm/logs/train/inst_tuning_t0gemma2_4b_%j.out
#SBATCH --error=slurm/logs/train/inst_tuning_t0gemma2_4b_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=8 --mem=64G --time=24:00:00

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 inst_tuning_t0gemma2.py --model_size "4b" 
EOT
)


JID_INST_1B=$(sbatch --parsable <<'EOT'
#!/bin/bash
#SBATCH --job-name=inst_tuning_t0gemma2_1b
#SBATCH --output=slurm/logs/train/inst_tuning_t0gemma2_1b_%j.out
#SBATCH --error=slurm/logs/train/inst_tuning_t0gemma2_1b_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=8 --mem=64G --time=12:00:00

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 inst_tuning_t0gemma2.py --model_size "1b" 
EOT
)


JID_INST_270M=$(sbatch --parsable <<'EOT'
#!/bin/bash
#SBATCH --job-name=inst_tuning_t0gemma2_270m
#SBATCH --output=slurm/logs/train/inst_tuning_t0gemma2_270m_%j.out
#SBATCH --error=slurm/logs/train/inst_tuning_t0gemma2_270m_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=8 --mem=64G --time=12:00:00

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 inst_tuning_t0gemma2.py --model_size "270m" 
EOT
)


# ==========================================
# Training (Fine-tuning)
# ==========================================

# Focus Structure

JID_TRAIN_270M_FOCUS=$(sbatch --parsable --dependency=afterok:$JID_INST_270M <<'EOT'
#!/bin/bash
#SBATCH --job-name=train_t0gemma2_270m_molweni_focus
#SBATCH --output=slurm/logs/train/train_t0gemma2_270m_molweni_focus_%j.out
#SBATCH --error=slurm/logs/train/train_t0gemma2_270m_molweni_focus_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=8 --mem=64G --time=06:00:00

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t0gemma2 -m 270m -l 2e-5 -e 5 --batchsize 4 --step 2000 -b
EOT
)

JID_TRAIN_1B_FOCUS=$(sbatch --parsable --dependency=afterok:$JID_INST_1B <<'EOT'
#!/bin/bash
#SBATCH --job-name=train_t0gemma2_1b_molweni_focus
#SBATCH --output=slurm/logs/train/train_t0gemma2_1b_molweni_focus_%j.out
#SBATCH --error=slurm/logs/train/train_t0gemma2_1b_molweni_focus_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=8 --mem=64G --time=18:00:00

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t0gemma2 -m 1b -l 2e-5 -e 5 --batchsize 4 --step 2000 -b
EOT
)

JID_TRAIN_4B_FOCUS=$(sbatch --parsable --dependency=afterok:$JID_INST_4B <<'EOT'
#!/bin/bash
#SBATCH --job-name=train_t0gemma2_4b_molweni_focus
#SBATCH --output=slurm/logs/train/train_t0gemma2_4b_molweni_focus_%j.out
#SBATCH --error=slurm/logs/train/train_t0gemma2_4b_molweni_focus_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=8 --mem=64G --time=24:00:00

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t0gemma2 -m 4b -l 2e-5 -e 5 --batchsize 4 --step 2000 -b
EOT
)


# Natural2 Structure

JID_TRAIN_270M_NAT2=$(sbatch --parsable --dependency=afterok:$JID_INST_270M <<'EOT'
#!/bin/bash
#SBATCH --job-name=train_t0gemma2_270m_molweni_natural2
#SBATCH --output=slurm/logs/train/train_t0gemma2_270m_molweni_natural2_%j.out
#SBATCH --error=slurm/logs/train/train_t0gemma2_270m_molweni_natural2_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=8 --mem=64G --time=06:00:00

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t0gemma2 -m 270m -l 2e-5 -e 5 --batchsize 4 --step 2000 -b
EOT
)

JID_TRAIN_1B_NAT2=$(sbatch --parsable --dependency=afterok:$JID_INST_1B <<'EOT'
#!/bin/bash
#SBATCH --job-name=train_t0gemma2_1b_molweni_natural2
#SBATCH --output=slurm/logs/train/train_t0gemma2_1b_molweni_natural2_%j.out
#SBATCH --error=slurm/logs/train/train_t0gemma2_1b_molweni_natural2_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=8 --mem=64G --time=18:00:00

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t0gemma2 -m 1b -l 2e-5 -e 5 --batchsize 4 --step 2000 -b
EOT
)

JID_TRAIN_4B_NAT2=$(sbatch --parsable --dependency=afterok:$JID_INST_4B <<'EOT'
#!/bin/bash
#SBATCH --job-name=train_t0gemma2_4b_molweni_natural2
#SBATCH --output=slurm/logs/train/train_t0gemma2_4b_molweni_natural2_%j.out
#SBATCH --error=slurm/logs/train/train_t0gemma2_4b_molweni_natural2_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=8 --mem=64G --time=24:00:00

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t0gemma2 -m 4b -l 2e-5 -e 5 --batchsize 4 --step 2000 -b
EOT
)


# ==========================================
# Inference
# ==========================================

JID_INF_270M_FOCUS=$(sbatch --parsable --dependency=afterok:$JID_TRAIN_270M_FOCUS <<EOT
#!/bin/bash
#SBATCH --job-name=inf_molweni_focus_t0gemma2_270m
#SBATCH --output=slurm/logs/inference/inf_molweni_focus_t0gemma2_270m_%j.out
#SBATCH --error=slurm/logs/inference/inf_molweni_focus_t0gemma2_270m_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --cpus-per-task=4 --mem=16G --time=01:00:00

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh

echo "Running: python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s focus -t t0gemma2 -m 270m --lr 2e-5 --seed 27 -b"
python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s focus -t t0gemma2 -m 270m --lr 2e-5 --seed 27 -b
EOT
)

JID_INF_1B_FOCUS=$(sbatch --parsable --dependency=afterok:$JID_TRAIN_1B_FOCUS <<EOT
#!/bin/bash
#SBATCH --job-name=inf_molweni_focus_t0gemma2_1b
#SBATCH --output=slurm/logs/inference/inf_molweni_focus_t0gemma2_1b_%j.out
#SBATCH --error=slurm/logs/inference/inf_molweni_focus_t0gemma2_1b_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --cpus-per-task=4 --mem=16G --time=02:00:00

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh

echo "Running: python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s focus -t t0gemma2 -m 1b --lr 2e-5 --seed 27 -b"
python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s focus -t t0gemma2 -m 1b --lr 2e-5 --seed 27 -b
EOT
)

JID_INF_4B_FOCUS=$(sbatch --parsable --dependency=afterok:$JID_TRAIN_4B_FOCUS <<EOT
#!/bin/bash
#SBATCH --job-name=inf_molweni_focus_t0gemma2_4b
#SBATCH --output=slurm/logs/inference/inf_molweni_focus_t0gemma2_4b_%j.out
#SBATCH --error=slurm/logs/inference/inf_molweni_focus_t0gemma2_4b_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --cpus-per-task=4 --mem=16G --time=04:00:00

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh

echo "Running: python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s focus -t t0gemma2 -m 4b --lr 2e-5 --seed 27 -b"
python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s focus -t t0gemma2 -m 4b --lr 2e-5 --seed 27 -b
EOT
)

JID_INF_270M_NAT2=$(sbatch --parsable --dependency=afterok:$JID_TRAIN_270M_NAT2 <<EOT
#!/bin/bash
#SBATCH --job-name=inf_molweni_natural2_t0gemma2_270m
#SBATCH --output=slurm/logs/inference/inf_molweni_natural2_t0gemma2_270m_%j.out
#SBATCH --error=slurm/logs/inference/inf_molweni_natural2_t0gemma2_270m_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --cpus-per-task=4 --mem=16G --time=01:00:00

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh

echo "Running: python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t0gemma2 -m 270m --lr 2e-5 --seed 27 -b"
python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t0gemma2 -m 270m --lr 2e-5 --seed 27 -b
EOT
)

JID_INF_1B_NAT2=$(sbatch --parsable --dependency=afterok:$JID_TRAIN_1B_NAT2 <<EOT
#!/bin/bash
#SBATCH --job-name=inf_molweni_natural2_t0gemma2_1b
#SBATCH --output=slurm/logs/inference/inf_molweni_natural2_t0gemma2_1b_%j.out
#SBATCH --error=slurm/logs/inference/inf_molweni_natural2_t0gemma2_1b_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --cpus-per-task=4 --mem=16G --time=02:00:00

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh

echo "Running: python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t0gemma2 -m 1b --lr 2e-5 --seed 27 -b"
python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t0gemma2 -m 1b --lr 2e-5 --seed 27 -b
EOT
)

JID_INF_4B_NAT2=$(sbatch --parsable --dependency=afterok:$JID_TRAIN_4B_NAT2 <<EOT
#!/bin/bash
#SBATCH --job-name=inf_molweni_natural2_t0gemma2_4b
#SBATCH --output=slurm/logs/inference/inf_molweni_natural2_t0gemma2_4b_%j.out
#SBATCH --error=slurm/logs/inference/inf_molweni_natural2_t0gemma2_4b_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --cpus-per-task=4 --mem=16G --time=04:00:00

cd \$HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh

echo "Running: python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t0gemma2 -m 4b --lr 2e-5 --seed 27 -b"
python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t0gemma2 -m 4b --lr 2e-5 --seed 27 -b
EOT
)


# ==========================================
# Evaluation
# ==========================================
# Dep depends on all inference jobs
ALL_INF_JIDS="$JID_INF_270M_FOCUS:$JID_INF_1B_FOCUS:$JID_INF_4B_FOCUS:$JID_INF_270M_NAT2:$JID_INF_1B_NAT2:$JID_INF_4B_NAT2"

sbatch --dependency=afterok:$ALL_INF_JIDS <<EOT
#!/bin/bash
#SBATCH --job-name=eval_molweni_t0gemma2
#SBATCH --output=slurm/logs/inference/eval_molweni_t0gemma2_%j.out
#SBATCH --error=slurm/logs/inference/eval_molweni_t0gemma2_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --cpus-per-task=4 --mem=16G --time=00:00:10

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh

echo "Evaluating t0gemma2-270m on molweni with focus structure"
python3 eval_gen.py --fted_model t0gemma2-270m --train_corpus molweni --test_corpus molweni -s focus --lr 5e-5 --seed 27 > eval/t0gemma2-270m_molweni_focus.txt
echo "----------------------------------------"
echo "Evaluating t0gemma2-1b on molweni with focus structure"
python3 eval_gen.py --fted_model t0gemma2-1b --train_corpus molweni --test_corpus molweni -s focus --lr 5e-5 --seed 27 > eval/t0gemma2-1b_molweni_focus.txt
echo "----------------------------------------"
echo "Evaluating t0gemma2-4b on molweni with focus structure"
python3 eval_gen.py --fted_model t0gemma2-4b --train_corpus molweni --test_corpus molweni -s focus --lr 5e-5 --seed 27 > eval/t0gemma2-4b_molweni_focus.txt
echo "----------------------------------------"

echo "Evaluating t0gemma2-270m on molweni with natural2 structure"
python3 eval_gen.py --fted_model t0gemma2-270m --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-5 --seed 27 > eval/t0gemma2-270m_molweni_natural2.txt
echo "----------------------------------------"
echo "Evaluating t0gemma2-1b on molweni with natural2 structure"
python3 eval_gen.py --fted_model t0gemma2-1b --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-5 --seed 27 > eval/t0gemma2-1b_molweni_natural2.txt
echo "----------------------------------------"
echo "Evaluating t0gemma2-4b on molweni with natural2 structure"
python3 eval_gen.py --fted_model t0gemma2-4b --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-5 --seed 27 > eval/t0gemma2-4b_molweni_natural2.txt
echo "----------------------------------------"

EOT

