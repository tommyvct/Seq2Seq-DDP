#!/bin/bash
# One-off, 1-GPU-per-job variant of temp_inference_eval.sh.
#
# Re-runs inference + eval for the Discord T5Gemma2-4B arms AFTER the sliding-window fix
# (transition_predict.py now decodes with get_context_window(test_corpus) = 41 for discord,
# instead of the hard-coded 18 it used when the thesis Table tab:discord-results was produced).
# The model WEIGHTS are unchanged; only decoding is re-run -- this does NOT retrain.
# Requires the window fix (constant.py get_context_window + transition_predict.py) on the cluster checkout.
#
# Submits one INDEPENDENT SLURM job per (dataset, arm), each on a single H100 MIG 3g.40gb slice.
# Jobs queue/run independently.
#
# Covers exactly the FOUR arms reported in Table tab:discord-results:
#   hint-full, no replay, Molweni ckpt  (row 1, headline)  hintfull-nomolweni  from     7e-6  was 34.60
#   hint-swap, no replay, Molweni ckpt  (row 2)            hintswap-nomolweni  from     7e-6  was 32.80
#   hint-swap, +Molweni,  Molweni ckpt  (row 3)            hintswap-molwenik1  from     8e-6  was 33.96
#   hint-full, no replay, vanilla       (row 4)            hintfull-nomolweni  vanilla  7e-6  was 29.37
# Plus one OPTIONAL arm (commented out below): the "hint-full, +Molweni, from-checkpoint" cell the
# thesis omits as "did not complete in time" -- uncomment to fill it.
#
# Run on the cluster login node from the repo root:  bash temp_inference_eval_1gpu.sh [SEED]

SEED=${1:-27}
EPOCH=5
STEP=2000
EFFECTIVE_BATCH=32
# Fixed, pre-existing molweni base the FROM arms continued from (stays _b16_, hardcoded).
BASE=t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt

# (dataset:arm) pairs.  arm=from    -> continued from the Molweni checkpoint (--gen_tag FROM_molweni)
#                       arm=vanilla -> trained from vanilla T5Gemma2-4B weights (no gen_tag)
# NOTE: hintfull-nomolweni appears twice (from + vanilla); they share dataset+LR and are kept
# distinct only by the arm/tag, which flows into MODELDIR, --gen_tag, log names and EVAL_TXT.
ARMS=(
  "discord-unveiled-hintfull-nomolweni:from"      # Table row 1 (headline)
  "discord-unveiled-hintswap-nomolweni:from"      # Table row 2
  "discord-unveiled-hintswap-molwenik1:from"      # Table row 3
  "discord-unveiled-hintfull-nomolweni:vanilla"   # Table row 4
  # "discord-unveiled-hintfull-molwenik1:from"    # OPTIONAL: the omitted cell (uncomment to run)
)

# Created here, at submit time, so SLURM can open each job's --output/--error before the job body runs.
mkdir -p slurm/logs/inf

for ENTRY in "${ARMS[@]}"; do
  DATASET=${ENTRY%%:*}
  ARM=${ENTRY##*:}

  if [[ $DATASET == *nomolweni* ]]; then LR=7e-6
  elif [[ $DATASET == *molwenik1* ]]; then LR=8e-6
  else echo "ERROR: no LR rule for DATASET='$DATASET' (name must contain 'nomolweni' or 'molwenik1')." >&2; exit 1; fi

  if [[ $ARM == from ]]; then
    MODELDIR=ft-models/t5gemma2-4b_train_${DATASET}_natural2_seed${SEED}_${LR}_e${EPOCH}_b${EFFECTIVE_BATCH}_s${STEP}_newprompt_FROM_${BASE}
    GENTAG_ARG="--gen_tag FROM_molweni"
    TAG=_FROM_molweni
  elif [[ $ARM == vanilla ]]; then
    MODELDIR=ft-models/t5gemma2-4b_train_${DATASET}_natural2_seed${SEED}_${LR}_e${EPOCH}_b${EFFECTIVE_BATCH}_s${STEP}_newprompt
    GENTAG_ARG=""
    TAG=""
  else
    echo "ERROR: unknown ARM='$ARM' for '$DATASET' (use 'from' or 'vanilla')." >&2; exit 1
  fi

  EVAL_TXT=eval/t5gemma2-4b_${LR}_e${EPOCH}_b${EFFECTIVE_BATCH}_s${STEP}_${DATASET}_natural2_newprompt${TAG}.txt

sbatch << INNER_EOF
#!/bin/bash
#SBATCH --job-name=t5gemma2-4b_${DATASET}${TAG}_inf_eval
#SBATCH --output=slurm/logs/inf/%j_t5gemma2-4b_${DATASET}_lr${LR}_e${EPOCH}_b${EFFECTIVE_BATCH}_s${STEP}${TAG}_inf_eval.out
#SBATCH --error=slurm/logs/inf/%j_t5gemma2-4b_${DATASET}_lr${LR}_e${EPOCH}_b${EFFECTIVE_BATCH}_s${STEP}${TAG}_inf_eval.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
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

# Single MIG slice: SLURM exposes the one allocated instance as cuda:0, so --use_gpu_num defaults to 0.
if ! python3 transition_predict.py --train_corpus ${DATASET} --test_corpus ${DATASET} -s natural2 -t t5gemma2 -m 4b --lr ${LR} -e ${EPOCH} --batchsize ${EFFECTIVE_BATCH} --step ${STEP} --seed ${SEED} --new_prompt -b ${GENTAG_ARG} --custom_model_dir ${MODELDIR}; then
  echo "Inference failed for ${DATASET} (${ARM}); skipping eval." >&2
  exit 1
fi
echo "Inference Complete: ${DATASET} (${ARM})"

python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus ${DATASET} --test_corpus ${DATASET} -s natural2 --lr ${LR} -e ${EPOCH} --batchsize ${EFFECTIVE_BATCH} --step ${STEP} --seed ${SEED} --new_prompt ${GENTAG_ARG} > ${EVAL_TXT}
echo "Eval Complete: ${EVAL_TXT}"
INNER_EOF
done
