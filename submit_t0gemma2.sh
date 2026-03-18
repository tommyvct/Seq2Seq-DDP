#!/bin/bash

mkdir -p slurm/logs/train
mkdir -p slurm/logs/inference


# ==========================================
# Data Preparation (CPU-only, no GPU needed)
# Tokenizes and packs P3 datasets to disk.
# All T5Gemma2 sizes share the same tokenizer, so this runs only once.
# ==========================================

# JID_PREP=$(sbatch --parsable <<'EOT'
# #!/bin/bash
# #SBATCH --job-name=prepare_p3_tokenized
# #SBATCH --output=slurm/logs/train/prepare_p3_tokenized_%j.out
# #SBATCH --error=slurm/logs/train/prepare_p3_tokenized_%j.err
# #SBATCH --mail-type=BEGIN,FAIL,END
# #SBATCH --mail-user=wut2@unbc.ca
# #SBATCH --cpus-per-task=128 --mem=720G --time=12:00:00
# #SBATCH --ntasks=1

# cd $HOME/scratch/Seq2Seq-DDP
# source slurm/init_hpc.sh

# python3 prepare_p3_tokenized.py --seed 27 --num_workers 96
# EOT
# )


# ==========================================
# Instruction Tuning (Pre-training)
# ==========================================

# --dependency=afterok:$JID_PREP 
JID_INST_4B=$(sbatch --parsable <<'EOT'
#!/bin/bash
#SBATCH --job-name=inst_tuning_t0gemma2_4b
#SBATCH --output=slurm/logs/train/inst_tuning_t0gemma2_4b_%j.out
#SBATCH --error=slurm/logs/train/inst_tuning_t0gemma2_4b_%j.err
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:8 --cpus-per-task=112 --mem=2000G --time=96:00:00
#SBATCH --ntasks=1

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh
export OMP_NUM_THREADS=2

run_inst_tuning_with_fallback() {
	local model_size="$1"
	local num_workers="$2"
	local seed="$3"
	local target_effective_batch=1024
	local gpus=8
	local -a batch_sizes=(32 16 8 4 2 1)
	local bs
	local accum

	for bs in "${batch_sizes[@]}"; do
		if (( target_effective_batch % (bs * gpus) != 0 )); then
			continue
		fi

		accum=$(( target_effective_batch / (bs * gpus) ))
		echo "[inst_tuning:$model_size] Trying batch_size=${bs}, gradient_accumulation_steps=${accum} (effective=${target_effective_batch})"

		if torchrun --nproc_per_node=8 inst_tuning_t0gemma2.py --model_size "$model_size" --batch_size "$bs" --gradient_accumulation_steps "$accum" --num_workers "$num_workers" --seed "$seed"; then
			echo "[inst_tuning:$model_size] Succeeded with batch_size=${bs}, gradient_accumulation_steps=${accum}"
			return 0
		fi

		echo "[inst_tuning:$model_size] Failed with batch_size=${bs}, gradient_accumulation_steps=${accum}; retrying with smaller batch size..."
	done

	echo "[inst_tuning:$model_size] All retry configs failed while keeping effective batch ${target_effective_batch}."
	return 1
}

# 4b label = 8B actual (4B enc + 4B dec). Retry policy keeps effective batch fixed at 1024 on 8 GPUs.
run_inst_tuning_with_fallback "4b" 14 27
EOT
)


JID_INST_1B=$(sbatch --parsable <<'EOT'
#!/bin/bash
#SBATCH --job-name=inst_tuning_t0gemma2_1b
#SBATCH --output=slurm/logs/train/inst_tuning_t0gemma2_1b_%j.out
#SBATCH --error=slurm/logs/train/inst_tuning_t0gemma2_1b_%j.err
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:8 --cpus-per-task=112 --mem=2000G --time=48:00:00
#SBATCH --ntasks=1

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh
export OMP_NUM_THREADS=2

run_inst_tuning_with_fallback() {
	local model_size="$1"
	local num_workers="$2"
	local seed="$3"
	local target_effective_batch=1024
	local gpus=8
	local -a batch_sizes=(64 32 16 8 4 2 1)
	local bs
	local accum

	for bs in "${batch_sizes[@]}"; do
		if (( target_effective_batch % (bs * gpus) != 0 )); then
			continue
		fi

		accum=$(( target_effective_batch / (bs * gpus) ))
		echo "[inst_tuning:$model_size] Trying batch_size=${bs}, gradient_accumulation_steps=${accum} (effective=${target_effective_batch})"

		if torchrun --nproc_per_node=8 inst_tuning_t0gemma2.py --model_size "$model_size" --batch_size "$bs" --gradient_accumulation_steps "$accum" --num_workers "$num_workers" --seed "$seed"; then
			echo "[inst_tuning:$model_size] Succeeded with batch_size=${bs}, gradient_accumulation_steps=${accum}"
			return 0
		fi

		echo "[inst_tuning:$model_size] Failed with batch_size=${bs}, gradient_accumulation_steps=${accum}; retrying with smaller batch size..."
	done

	echo "[inst_tuning:$model_size] All retry configs failed while keeping effective batch ${target_effective_batch}."
	return 1
}

# 1b label = 2B actual (1B enc + 1B dec). Retry policy keeps effective batch fixed at 1024 on 8 GPUs.
run_inst_tuning_with_fallback "1b" 14 27
EOT
)


# JID_INST_270M=$(sbatch --parsable <<'EOT'
# #!/bin/bash
# #SBATCH --job-name=inst_tuning_t0gemma2_270m
# #SBATCH --output=slurm/logs/train/inst_tuning_t0gemma2_270m_%j.out
# #SBATCH --error=slurm/logs/train/inst_tuning_t0gemma2_270m_%j.err
# #SBATCH --mail-type=BEGIN,FAIL,END
# #SBATCH --mail-user=wut2@unbc.ca
# #SBATCH --gpus=h100:4 --cpus-per-task=32 --mem=512G --time=06:00:00
# #SBATCH --ntasks=1

# cd $HOME/scratch/Seq2Seq-DDP
# source slurm/init_hpc.sh
# torchrun --nproc_per_node=4 inst_tuning_t0gemma2.py --model_size "270m" --batch_size 64 --gradient_accumulation_steps 4 --seed 27
# EOT
# )


# ==========================================
# Training (Fine-tuning)
# ==========================================

# Focus Structure
# JID_TRAIN_270M_FOCUS=$(sbatch --parsable --dependency=afterok:$JID_INST_270M <<'EOT'
# #!/bin/bash
# #SBATCH --job-name=train_t0gemma2_270m_molweni_focus
# #SBATCH --output=slurm/logs/train/train_t0gemma2_270m_molweni_focus_%j.out
# #SBATCH --error=slurm/logs/train/train_t0gemma2_270m_molweni_focus_%j.err
# #SBATCH --mail-type=FAIL,END
# #SBATCH --mail-user=wut2@unbc.ca
# #SBATCH --gpus=h100:1 --cpus-per-task=6 --mem=64G --time=12:00:00

# cd $HOME/scratch/Seq2Seq-DDP
# source slurm/init_hpc.sh
# python3 train.py --train_corpus molweni --do_train -s focus -t t0gemma2 -m 270m -l 2e-5 -e 5 --batchsize 4 --step 2000 -b --seed 27
# EOT
# )

# --dependency=afterok:$JID_INST_1B
JID_TRAIN_1B_FOCUS=$(sbatch --parsable --dependency=afterok:$JID_INST_1B <<'EOT'
#!/bin/bash
#SBATCH --job-name=train_t0gemma2_1b_molweni_focus
#SBATCH --output=slurm/logs/train/train_t0gemma2_1b_molweni_focus_%j.out
#SBATCH --error=slurm/logs/train/train_t0gemma2_1b_molweni_focus_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=6 --mem=64G --time=18:00:00

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t0gemma2 -m 1b -l 2e-5 -e 5 --batchsize 4 --step 2000 -b --seed 27
EOT
)

# --dependency=afterok:$JID_INST_4B
JID_TRAIN_4B_FOCUS=$(sbatch --parsable --dependency=afterok:$JID_INST_4B <<'EOT'
#!/bin/bash
#SBATCH --job-name=train_t0gemma2_4b_molweni_focus
#SBATCH --output=slurm/logs/train/train_t0gemma2_4b_molweni_focus_%j.out
#SBATCH --error=slurm/logs/train/train_t0gemma2_4b_molweni_focus_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=6 --mem=64G --time=24:00:00

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t0gemma2 -m 4b -l 2e-5 -e 5 --batchsize 4 --step 2000 -b --seed 27
EOT
)


# Natural2 Structure

# --dependency=afterok:$JID_INST_270M
# JID_TRAIN_270M_NAT2=$(sbatch --parsable --dependency=afterok:$JID_INST_270M <<'EOT'
# #!/bin/bash
# #SBATCH --job-name=train_t0gemma2_270m_molweni_natural2
# #SBATCH --output=slurm/logs/train/train_t0gemma2_270m_molweni_natural2_%j.out
# #SBATCH --error=slurm/logs/train/train_t0gemma2_270m_molweni_natural2_%j.err
# #SBATCH --mail-type=FAIL,END
# #SBATCH --mail-user=wut2@unbc.ca
# #SBATCH --gpus=h100:1 --cpus-per-task=6 --mem=64G --time=12:00:00

# cd $HOME/scratch/Seq2Seq-DDP
# source slurm/init_hpc.sh
# python3 train.py --train_corpus molweni --do_train -s natural2 -t t0gemma2 -m 270m -l 2e-5 -e 5 --batchsize 4 --step 2000 -b --seed 27
# EOT
# )

#  --dependency=afterok:$JID_INST_1B
JID_TRAIN_1B_NAT2=$(sbatch --parsable --dependency=afterok:$JID_INST_1B <<'EOT'
#!/bin/bash
#SBATCH --job-name=train_t0gemma2_1b_molweni_natural2
#SBATCH --output=slurm/logs/train/train_t0gemma2_1b_molweni_natural2_%j.out
#SBATCH --error=slurm/logs/train/train_t0gemma2_1b_molweni_natural2_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=6 --mem=64G --time=18:00:00

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t0gemma2 -m 1b -l 2e-5 -e 5 --batchsize 4 --step 2000 -b --seed 27
EOT
)

#  --dependency=afterok:$JID_INST_4B
JID_TRAIN_4B_NAT2=$(sbatch --parsable --dependency=afterok:$JID_INST_4B <<'EOT'
#!/bin/bash
#SBATCH --job-name=train_t0gemma2_4b_molweni_natural2
#SBATCH --output=slurm/logs/train/train_t0gemma2_4b_molweni_natural2_%j.out
#SBATCH --error=slurm/logs/train/train_t0gemma2_4b_molweni_natural2_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=6 --mem=64G --time=24:00:00

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t0gemma2 -m 4b -l 2e-5 -e 5 --batchsize 4 --step 2000 -b --seed 27
EOT
)


# ==========================================
# Inference
# ==========================================

# JID_INF_270M_FOCUS=$(sbatch --parsable --dependency=afterok:$JID_TRAIN_270M_FOCUS <<EOT
# #!/bin/bash
# #SBATCH --job-name=inf_molweni_focus_t0gemma2_270m
# #SBATCH --output=slurm/logs/inference/inf_molweni_focus_t0gemma2_270m_%j.out
# #SBATCH --error=slurm/logs/inference/inf_molweni_focus_t0gemma2_270m_%j.err
# #SBATCH --mail-type=FAIL,END
# #SBATCH --mail-user=wut2@unbc.ca
# #SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=16G --time=01:00:00

# cd $HOME/scratch/Seq2Seq-DDP
# source slurm/init_hpc.sh

# echo "Running: python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s focus -t t0gemma2 -m 270m --lr 2e-5 --seed 27 -b"
# python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s focus -t t0gemma2 -m 270m --lr 2e-5 --seed 27 -b
# EOT
# )

JID_INF_1B_FOCUS=$(sbatch --parsable --dependency=afterok:$JID_TRAIN_1B_FOCUS <<EOT
#!/bin/bash
#SBATCH --job-name=inf_molweni_focus_t0gemma2_1b
#SBATCH --output=slurm/logs/inference/inf_molweni_focus_t0gemma2_1b_%j.out
#SBATCH --error=slurm/logs/inference/inf_molweni_focus_t0gemma2_1b_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=16G --time=02:00:00

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
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=16G --time=04:00:00

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh

echo "Running: python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s focus -t t0gemma2 -m 4b --lr 2e-5 --seed 27 -b"
python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s focus -t t0gemma2 -m 4b --lr 2e-5 --seed 27 -b
EOT
)

# JID_INF_270M_NAT2=$(sbatch --parsable --dependency=afterok:$JID_TRAIN_270M_NAT2 <<EOT
# #!/bin/bash
# #SBATCH --job-name=inf_molweni_natural2_t0gemma2_270m
# #SBATCH --output=slurm/logs/inference/inf_molweni_natural2_t0gemma2_270m_%j.out
# #SBATCH --error=slurm/logs/inference/inf_molweni_natural2_t0gemma2_270m_%j.err
# #SBATCH --mail-type=FAIL,END
# #SBATCH --mail-user=wut2@unbc.ca
# #SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=16G --time=01:00:00

# cd $HOME/scratch/Seq2Seq-DDP
# source slurm/init_hpc.sh

# echo "Running: python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t0gemma2 -m 270m --lr 2e-5 --seed 27 -b"
# python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t0gemma2 -m 270m --lr 2e-5 --seed 27 -b
# EOT
# )

JID_INF_1B_NAT2=$(sbatch --parsable --dependency=afterok:$JID_TRAIN_1B_NAT2 <<EOT
#!/bin/bash
#SBATCH --job-name=inf_molweni_natural2_t0gemma2_1b
#SBATCH --output=slurm/logs/inference/inf_molweni_natural2_t0gemma2_1b_%j.out
#SBATCH --error=slurm/logs/inference/inf_molweni_natural2_t0gemma2_1b_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=16G --time=02:00:00

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
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=16G --time=04:00:00

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh

echo "Running: python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t0gemma2 -m 4b --lr 2e-5 --seed 27 -b"
python3 transition_predict.py --train_corpus molweni --test_corpus molweni -s natural2 -t t0gemma2 -m 4b --lr 2e-5 --seed 27 -b
EOT
)


# ==========================================
# Evaluation
# ==========================================
# Dep depends on all inference jobs
# $JID_INF_270M_NAT2:$JID_INF_270M_FOCUS:
ALL_INF_JIDS="$JID_INF_1B_FOCUS:$JID_INF_4B_FOCUS:$JID_INF_1B_NAT2:$JID_INF_4B_NAT2"

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

echo "Evaluating t0gemma2-1b on molweni with focus structure"
python3 eval_gen.py --fted_model t0gemma2-1b --train_corpus molweni --test_corpus molweni -s focus --lr 2e-5 --seed 27 > eval/t0gemma2-1b_molweni_focus.txt
echo "----------------------------------------"
echo "Evaluating t0gemma2-4b on molweni with focus structure"
python3 eval_gen.py --fted_model t0gemma2-4b --train_corpus molweni --test_corpus molweni -s focus --lr 2e-5 --seed 27 > eval/t0gemma2-4b_molweni_focus.txt
echo "----------------------------------------"

echo "Evaluating t0gemma2-1b on molweni with natural2 structure"
python3 eval_gen.py --fted_model t0gemma2-1b --train_corpus molweni --test_corpus molweni -s natural2 --lr 2e-5 --seed 27 > eval/t0gemma2-1b_molweni_natural2.txt
echo "----------------------------------------"
echo "Evaluating t0gemma2-4b on molweni with natural2 structure"
python3 eval_gen.py --fted_model t0gemma2-4b --train_corpus molweni --test_corpus molweni -s natural2 --lr 2e-5 --seed 27 > eval/t0gemma2-4b_molweni_natural2.txt
echo "----------------------------------------"

EOT

# echo "Evaluating t0gemma2-270m on molweni with natural2 structure"
# python3 eval_gen.py --fted_model t0gemma2-270m --train_corpus molweni --test_corpus molweni -s natural2 --lr 2e-5 --seed 27 > eval/t0gemma2-270m_molweni_natural2.txt
# echo "----------------------------------------"

# echo "Evaluating t0gemma2-270m on molweni with focus structure"
# python3 eval_gen.py --fted_model t0gemma2-270m --train_corpus molweni --test_corpus molweni -s focus --lr 2e-5 --seed 27 > eval/t0gemma2-270m_molweni_focus.txt
# echo "----------------------------------------"
