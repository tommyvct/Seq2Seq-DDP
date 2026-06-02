SEED=${1:-27}
# Multi-GPU / multi-node topology.
#   arg2 = NODES, arg3 = GPUS_PER_NODE, arg4 = GPU_TYPE. Default 1 x 8 h200 = 8 GPUs.
#   Examples:
#     bash submit_discord_ddp.sh 27            # default: single node x 8 H200 (no inter-node comm)
#     bash submit_discord_ddp.sh 27 2 4 h100   # 2 nodes x 4 H100 (H100 nodes have 4 GPUs each)
#     bash submit_discord_ddp.sh 27 1 4 h100   # single node x 4 H100
NODES=${2:-1}
GPUS_PER_NODE=${3:-8}
GPU_TYPE=${4:-h200}
NGPU=$((NODES * GPUS_PER_NODE))

# Training hyperparameters baked into the model/generation/eval filenames below. These are the single
# source of truth: they fill both the CLI flags (-e/--batchsize/--step) and the "_e{E}_b{B}_s{S}_"
# path tokens, so the train (write) path and the inference/eval (read) path can never drift apart.
# NOTE: the molweni base model these jobs continue FROM is a fixed, pre-existing directory
# (seed27_5e-6_e5_b16_s2000_newprompt) — its tokens stay hardcoded and must NOT use these variables.
EPOCH=5
STEP=2000
# Effective batch is kept fixed regardless of GPU count: passed straight to train.py via --batchsize;
# under torchrun train.py divides it across ranks for the per-device batch. The new model's output dir
# is named "_b${EFFECTIVE_BATCH}_" (by the effective batch).
EFFECTIVE_BATCH=32
if (( EFFECTIVE_BATCH % NGPU != 0 )); then
  echo "ERROR: EFFECTIVE_BATCH ($EFFECTIVE_BATCH) must be divisible by total GPUs ($NGPU)." >&2
  echo "       Pick NODES x GPUS_PER_NODE whose product divides $EFFECTIVE_BATCH (e.g. 1x8, 2x4, 1x4)." >&2
  exit 1
fi

# Per-node resources.
CPUS=$((GPUS_PER_NODE * 8))
MEMG=$((GPUS_PER_NODE * 32))

# Launch command. Single node runs torchrun directly with --standalone (a local rendezvous on
# localhost) — this matches the working inst_tuning setup and avoids the c10d *endpoint* backend,
# which mis-detected the host on this cluster and timed out connecting to its own node. Multi-node
# uses srun to start one torchrun per node with the static backend (per-node rank from SLURM).
# NOTE: \$SLURM_NODEID / \$MASTER_* are escaped so they expand at job time, not at submit time.
if (( NODES > 1 )); then
  LAUNCH="srun torchrun --nnodes=${NODES} --node_rank=\$SLURM_NODEID --master_addr=\$MASTER_ADDR --master_port=\$MASTER_PORT --nproc_per_node=${GPUS_PER_NODE}"
else
  LAUNCH="torchrun --standalone --nproc_per_node=${GPUS_PER_NODE}"
fi

for DATASET in discord-unveiled-hintfull-molwenik1 discord-unveiled-hintfull-nomolweni discord-unveiled-hintswap-molwenik1 discord-unveiled-hintswap-nomolweni; do
if [[ $DATASET == *nomolweni* ]]; then
  LR=7e-6
elif [[ $DATASET == *molwenik1* ]]; then
  LR=8e-6
else
  echo "ERROR: no LR rule for DATASET='$DATASET' (name must contain 'nomolweni' or 'molwenik1'); refusing to submit with a stale/empty LR." >&2
  exit 1
fi
sbatch << INNER_EOF
#!/bin/bash
#SBATCH --job-name=t5gemma2-4b_train_${DATASET}_natural2_seed${SEED}_${LR}_e${EPOCH}_b${EFFECTIVE_BATCH}_s${STEP}_newprompt_FROM_t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt
#SBATCH --output=slurm/logs/%j_t5gemma2-4b_train_${DATASET}_natural2_seed${SEED}_${LR}_e${EPOCH}_b${EFFECTIVE_BATCH}_s${STEP}_newprompt_FROM_t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt.out
#SBATCH --error=slurm/logs/%j_t5gemma2-4b_train_${DATASET}_natural2_seed${SEED}_${LR}_e${EPOCH}_b${EFFECTIVE_BATCH}_s${STEP}_newprompt_FROM_t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt.err
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
mkdir -p eval generation   # eval_gen.py output is shell-redirected (>) so its dir must pre-exist
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Static-rendezvous coordinates for the multi-node case (unused/harmless for single node, which
# uses --standalone). Master = first allocated node; per-job port avoids collisions.
export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=\$((10000 + SLURM_JOB_ID % 50000))
# Uncomment on the first multi-node run to confirm NCCL uses the fast fabric (InfiniBand):
# export NCCL_DEBUG=INFO

# DDP training. Abort the whole job if training fails so we don't run inference on a missing model.
if ! ${LAUNCH} train.py --train_corpus ${DATASET} --do_train -s natural2 -t t5gemma2 -m 4b -l ${LR} -e ${EPOCH} --batchsize ${EFFECTIVE_BATCH} --step ${STEP} --seed ${SEED} --new_prompt -b --custom_model_dir ft-models/t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt; then
  echo "Training failed; skipping inference/eval." >&2
  exit 1
fi
echo "Training Complete"

# Inference + eval run once on the first node (no srun); single-process generation.
python3 transition_predict.py --train_corpus ${DATASET} --test_corpus ${DATASET} -s natural2 -t t5gemma2 -m 4b --lr ${LR} -e ${EPOCH} --batchsize ${EFFECTIVE_BATCH} --step ${STEP} --seed ${SEED} --new_prompt -b --gen_tag FROM_molweni --custom_model_dir ft-models/t5gemma2-4b_train_${DATASET}_natural2_seed${SEED}_${LR}_e${EPOCH}_b${EFFECTIVE_BATCH}_s${STEP}_newprompt_FROM_t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt
echo "Inference Complete"

# --gen_tag FROM_molweni keeps these continued-from-molweni generations on a distinct path from the
# from-scratch run in submit_discord_ddp_2.sh (same train/test/decode params would otherwise collide).
python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus ${DATASET} --test_corpus ${DATASET} -s natural2 --lr ${LR} -e ${EPOCH} --batchsize ${EFFECTIVE_BATCH} --step ${STEP} --seed ${SEED} --new_prompt --gen_tag FROM_molweni > eval/t5gemma2-4b_${LR}_e${EPOCH}_b${EFFECTIVE_BATCH}_s${STEP}_${DATASET}_natural2_newprompt_FROM_molweni.txt
INNER_EOF
done
