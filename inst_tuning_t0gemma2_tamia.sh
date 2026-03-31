#!/bin/bash

mkdir -p slurm/logs/train
mkdir -p slurm/logs/inference

sbatch --parsable --dependency=afterok:181409?afternotok:181409 <<'EOT'
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

torchrun --nproc_per_node=8 inst_tuning_t0gemma2.py \
	--model_size "4b" \
	--batch_size 32 \
	--gradient_accumulation_steps 4 \
	--num_workers 8 \
	--seed 27 \
	--walltime_stop 23.25
EOT



sbatch --parsable --dependency=afterok:$JID_INST_4B_DAY_2?afternotok:$JID_INST_4B_DAY_2 <<'EOT'
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

torchrun --nproc_per_node=8 inst_tuning_t0gemma2.py \
	--model_size "4b" \
	--batch_size 32 \
	--gradient_accumulation_steps 4 \
	--num_workers 8 \
	--seed 27 \
	--walltime_stop 23.25
EOT


sbatch --parsable --dependency=afterok:$JID_INST_4B_DAY_3?afternotok:$JID_INST_4B_DAY_3 <<'EOT'
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

torchrun --nproc_per_node=8 inst_tuning_t0gemma2.py \
	--model_size "4b" \
	--batch_size 32 \
	--gradient_accumulation_steps 4 \
	--num_workers 8 \
	--seed 27 \
	--walltime_stop 23.25
EOT


sbatch --parsable --dependency=afterok:$JID_INST_4B_DAY_4?afternotok:$JID_INST_4B_DAY_4 <<'EOT'
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

torchrun --nproc_per_node=8 inst_tuning_t0gemma2.py \
	--model_size "4b" \
	--batch_size 32 \
	--gradient_accumulation_steps 4 \
	--num_workers 8 \
	--seed 27 \
	--walltime_stop 23.25
EOT


sbatch --parsable --dependency=afterok:$JID_INST_4B_DAY_5?afternotok:$JID_INST_4B_DAY_5 <<'EOT'
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

torchrun --nproc_per_node=8 inst_tuning_t0gemma2.py \
	--model_size "4b" \
	--batch_size 32 \
	--gradient_accumulation_steps 4 \
	--num_workers 8 \
	--seed 27 \
	--walltime_stop 23.25
EOT
