#!/bin/bash
# One-off, 1-GPU-per-job variant of temp_inference_eval.sh.
#
# Submits 4 INDEPENDENT SLURM jobs, each requesting a single H100 and running ONE dataset's
# inference + eval against the already-trained _b32_ FROM_molweni model. Use this when you can't get
# a whole 4xH100 node but can get four separate 1-GPU slots (they'll queue/run independently).
# The sibling temp_inference_eval.sh instead packs all 4 onto one 4xH100 node.
#
# Run on the cluster login node from the repo root:  bash temp_inference_eval_1gpu.sh [SEED]

SEED=${1:-27}
EPOCH=5
STEP=2000
EFFECTIVE_BATCH=32
# Fixed, pre-existing molweni base these models continued FROM (stays _b16_, hardcoded).
BASE=t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt

# Created here, at submit time, so SLURM can open each job's --output/--error (it opens them when the
# job starts, before the job body's own mkdir would run).
mkdir -p slurm/logs/inf

for DATASET in discord-unveiled-hintfull-molwenik1 discord-unveiled-hintfull-nomolweni \
               discord-unveiled-hintswap-molwenik1 discord-unveiled-hintswap-nomolweni; do
  if [[ $DATASET == *nomolweni* ]]; then LR=7e-6
  elif [[ $DATASET == *molwenik1* ]]; then LR=8e-6
  else echo "ERROR: no LR rule for DATASET='$DATASET' (name must contain 'nomolweni' or 'molwenik1')." >&2; exit 1; fi

  MODELDIR=ft-models/t5gemma2-4b_train_${DATASET}_natural2_seed${SEED}_${LR}_e${EPOCH}_b${EFFECTIVE_BATCH}_s${STEP}_newprompt_FROM_${BASE}
  EVAL_TXT=eval/t5gemma2-4b_${LR}_e${EPOCH}_b${EFFECTIVE_BATCH}_s${STEP}_${DATASET}_natural2_newprompt_FROM_molweni.txt

sbatch << INNER_EOF
#!/bin/bash
#SBATCH --job-name=t5gemma2-4b_${DATASET}_FROM_molweni_inf_eval
#SBATCH --output=slurm/logs/inf/%j_t5gemma2-4b_${DATASET}_lr${LR}_e${EPOCH}_b${EFFECTIVE_BATCH}_s${STEP}_FROM_molweni_inf_eval.out
#SBATCH --error=slurm/logs/inf/%j_t5gemma2-4b_${DATASET}_lr${LR}_e${EPOCH}_b${EFFECTIVE_BATCH}_s${STEP}_FROM_molweni_inf_eval.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=wut2@unbc.ca

cd \$SCRATCH/Seq2Seq-DDP
source slurm/init_hpc.sh
mkdir -p eval generation slurm/logs/inf
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Single 1-GPU job: SLURM exposes the one allocated H100 as cuda:0, so --use_gpu_num defaults to 0.
if ! python3 transition_predict.py --train_corpus ${DATASET} --test_corpus ${DATASET} -s natural2 -t t5gemma2 -m 4b --lr ${LR} -e ${EPOCH} --batchsize ${EFFECTIVE_BATCH} --step ${STEP} --seed ${SEED} --new_prompt -b --gen_tag FROM_molweni --custom_model_dir ${MODELDIR}; then
  echo "Inference failed for ${DATASET}; skipping eval." >&2
  exit 1
fi
echo "Inference Complete: ${DATASET}"

python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus ${DATASET} --test_corpus ${DATASET} -s natural2 --lr ${LR} -e ${EPOCH} --batchsize ${EFFECTIVE_BATCH} --step ${STEP} --seed ${SEED} --new_prompt --gen_tag FROM_molweni > ${EVAL_TXT}
echo "Eval Complete: ${EVAL_TXT}"
INNER_EOF
done
