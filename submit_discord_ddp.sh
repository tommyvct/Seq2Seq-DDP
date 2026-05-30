SEED=${1:-27}
# Multi-GPU / multi-node topology.
#   arg2 = NODES, arg3 = GPUS_PER_NODE, arg4 = GPU_TYPE. Default 2 x 4 h100 = 8 GPUs.
#   Examples:
#     bash submit_discord_ddp.sh 27 2 4 h100   # 2 nodes x 4 H100 (H100 nodes have 4 GPUs each)
#     bash submit_discord_ddp.sh 27 1 8 h200   # single node x 8 H200 (no inter-node comm)
#     bash submit_discord_ddp.sh 27 1 4 h100   # single-node fallback
NODES=${2:-2}
GPUS_PER_NODE=${3:-4}
GPU_TYPE=${4:-h100}
NGPU=$((NODES * GPUS_PER_NODE))

# Preserve the training recipe: keep the effective batch size fixed at 16 regardless of how many
# GPUs we use. We pass this effective batch straight to train.py via --batchsize; under torchrun
# train.py divides it across ranks to get the per-device batch. The output dir is named "_b16_"
# (by the effective batch), so the downstream inference/eval paths below stay stable.
EFFECTIVE_BATCH=16
if (( EFFECTIVE_BATCH % NGPU != 0 )); then
  echo "ERROR: EFFECTIVE_BATCH ($EFFECTIVE_BATCH) must be divisible by total GPUs ($NGPU)." >&2
  echo "       Pick NODES x GPUS_PER_NODE whose product divides 16 (e.g. 1x4, 2x4, 2x2)." >&2
  exit 1
fi

# Per-node resources.
CPUS=$((GPUS_PER_NODE * 8))
MEMG=$((GPUS_PER_NODE * 32))

for DATASET in discord-unveiled-hintfull-molwenik1 discord-unveiled-hintfull-nomolweni discord-unveiled-hintswap-molwenik1 discord-unveiled-hintswap-nomolweni; do
sbatch << INNER_EOF
#!/bin/bash
#SBATCH --job-name=t5gemma2-4b_train_${DATASET}_natural2_seed${SEED}_5e-6_e5_b16_s2000_newprompt_FROM_t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt
#SBATCH --output=slurm/logs/%j_t5gemma2-4b_train_${DATASET}_natural2_seed${SEED}_5e-6_e5_b16_s2000_newprompt_FROM_t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt.out
#SBATCH --error=slurm/logs/%j_t5gemma2-4b_train_${DATASET}_natural2_seed${SEED}_5e-6_e5_b16_s2000_newprompt_FROM_t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt.err
#SBATCH --time=1-0:00:00
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=${GPU_TYPE}:${GPUS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEMG}G
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=wut2@unbc.ca

cd \$SCRATCH/Seq2Seq-DDP
source slurm/init_hpc.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Rendezvous: first allocated node hosts the c10d store; per-job port avoids collisions.
export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=\$((10000 + SLURM_JOB_ID % 50000))
# Uncomment on the first multi-node run to confirm NCCL uses the fast fabric (InfiniBand):
# export NCCL_DEBUG=INFO

# DDP training: one torchrun per node (srun --ntasks-per-node=1), GPUS_PER_NODE workers each.
srun torchrun \
  --nnodes=${NODES} \
  --nproc_per_node=${GPUS_PER_NODE} \
  --rdzv_id=\$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT \
  train.py --train_corpus ${DATASET} --do_train -s natural2 -t t5gemma2 -m 4b -l 5e-6 -e 5 --batchsize ${EFFECTIVE_BATCH} --step 2000 --new_prompt -b --custom_model_dir ft-models/t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt
echo "Training Complete"

# Inference + eval run once on the first node (no srun); single-process generation.
python3 transition_predict.py --train_corpus ${DATASET} --test_corpus ${DATASET} -s natural2 -t t5gemma2 -m 4b --lr 5e-6 -e 5 --batchsize 32 --step 2000 --seed ${SEED} --new_prompt -b --custom_model_dir ft-models/t5gemma2-4b_train_${DATASET}_natural2_seed${SEED}_5e-6_e5_b16_s2000_newprompt_FROM_t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt
echo "Inference Complete"

python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus ${DATASET} --test_corpus ${DATASET} -s natural2 --lr 5e-6 -e 5 --batchsize 32 --step 2000 --seed 27 --new_prompt > eval/t5gemma2-4b_5e-6_e5_b16_s2000_${DATASET}_natural2_newprompt_FROM_molweni.txt
INNER_EOF
done
