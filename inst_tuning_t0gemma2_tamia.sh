#!/bin/bash

mkdir -p slurm/logs/train
mkdir -p slurm/logs/inference

JID_INST_4B=$(sbatch --parsable <<'EOT'
#!/bin/bash
#SBATCH --job-name=inst_tuning_t0gemma2_4b
#SBATCH --output=slurm/logs/train/inst_tuning_t0gemma2_4b_%j.out
#SBATCH --error=slurm/logs/train/inst_tuning_t0gemma2_4b_%j.err
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h200:8 --cpus-per-task=64 --mem=950G --time=24:00:00
#SBATCH --ntasks=1

cd $SCRATCH/Seq2seq-DDP
source slurm/init_hpc.sh
export OMP_NUM_THREADS=2
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

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

		if torchrun --nproc_per_node=8 inst_tuning_t0gemma2.py --model_size "$model_size" --batch_size "$bs" --gradient_accumulation_steps "$accum" --num_workers "$num_workers" --seed "$seed" --walltime_stop 23.25; then
			echo "[inst_tuning:$model_size] Succeeded with batch_size=${bs}, gradient_accumulation_steps=${accum}"
			return 0
		fi

		echo "[inst_tuning:$model_size] Failed with batch_size=${bs}, gradient_accumulation_steps=${accum}; retrying with smaller batch size..."
	done

	echo "[inst_tuning:$model_size] All retry configs failed while keeping effective batch ${target_effective_batch}."
	return 1
}

# 4b label = 8B actual (4B enc + 4B dec). Retry policy keeps effective batch fixed at 1024 on 8 GPUs.
run_inst_tuning_with_fallback "4b" 8 27
EOT
)


