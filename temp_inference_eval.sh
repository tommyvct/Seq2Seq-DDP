#!/bin/bash
# One-off: re-run the 4 discord FROM_molweni inferences + evals that the buggy in-flight job will
# fail at (it looked for a _b16_ model dir while training saved _b32_). Training already produced the
# correct _b32_ models; this job just (re)does inference + eval against them.
#
# One SLURM job on a single 4xH100 node. Each of the 4 datasets gets its own GPU (--use_gpu_num),
# so all 4 inferences run concurrently. Each dataset's eval_gen runs as soon as ITS inference
# finishes (per-dataset gate), in parallel with the other still-running inferences. eval_gen is
# CPU-only, so it doesn't contend for the GPUs. Submit with:  sbatch temp_inference_eval.sh
#SBATCH --job-name=t5gemma2-4b_discord_FROM_molweni_inf_eval
#SBATCH --output=slurm/logs/%j_t5gemma2-4b_discord_FROM_molweni_inf_eval.out
#SBATCH --error=slurm/logs/%j_t5gemma2-4b_discord_FROM_molweni_inf_eval.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=wut2@unbc.ca

cd $SCRATCH/Seq2Seq-DDP
source slurm/init_hpc.sh
mkdir -p eval generation slurm/logs/inf   # inf/ holds the 4 per-dataset inference logs
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Hyperparameters baked into the model/generation/eval paths — must match the training run that
# produced the models (single source of truth, same convention as submit_discord_ddp.sh).
SEED=27
EPOCH=5
STEP=2000
EFFECTIVE_BATCH=32
# The fixed, pre-existing molweni base these models continued FROM (stays _b16_, hardcoded).
BASE=t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt

DATASETS=(discord-unveiled-hintfull-molwenik1 discord-unveiled-hintfull-nomolweni \
          discord-unveiled-hintswap-molwenik1 discord-unveiled-hintswap-nomolweni)

pids=()
gpu=0
for DATASET in "${DATASETS[@]}"; do
  if [[ $DATASET == *nomolweni* ]]; then LR=7e-6
  elif [[ $DATASET == *molwenik1* ]]; then LR=8e-6
  else echo "ERROR: no LR rule for DATASET='$DATASET' (name must contain 'nomolweni' or 'molwenik1')." >&2; exit 1; fi

  MODELDIR=ft-models/t5gemma2-4b_train_${DATASET}_natural2_seed${SEED}_${LR}_e${EPOCH}_b${EFFECTIVE_BATCH}_s${STEP}_newprompt_FROM_${BASE}
  INF_LOG=slurm/logs/inf/${DATASET}_lr${LR}_e${EPOCH}_b${EFFECTIVE_BATCH}_s${STEP}_FROM_molweni_inf.log
  EVAL_TXT=eval/t5gemma2-4b_${LR}_e${EPOCH}_b${EFFECTIVE_BATCH}_s${STEP}_${DATASET}_natural2_newprompt_FROM_molweni.txt

  # One pipeline per dataset, backgrounded. The subshell exits non-zero if inference OR eval fails,
  # so the wait-loop below can tally failures. All process logs -> INF_LOG; eval metrics -> EVAL_TXT.
  (
    echo "[gpu ${gpu}] inference START: ${DATASET} (lr${LR})  $(date)"
    python3 transition_predict.py --train_corpus ${DATASET} --test_corpus ${DATASET} -s natural2 -t t5gemma2 -m 4b \
      --lr ${LR} -e ${EPOCH} --batchsize ${EFFECTIVE_BATCH} --step ${STEP} --seed ${SEED} --new_prompt -b \
      --use_gpu_num ${gpu} --gen_tag FROM_molweni --custom_model_dir ${MODELDIR} \
      || { echo "[gpu ${gpu}] inference FAILED: ${DATASET}" >&2; exit 1; }

    echo "[gpu ${gpu}] inference DONE, eval START: ${DATASET}  $(date)"
    python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus ${DATASET} --test_corpus ${DATASET} -s natural2 \
      --lr ${LR} -e ${EPOCH} --batchsize ${EFFECTIVE_BATCH} --step ${STEP} --seed ${SEED} --new_prompt \
      --gen_tag FROM_molweni > ${EVAL_TXT} \
      || { echo "[gpu ${gpu}] eval FAILED: ${DATASET}" >&2; exit 1; }
    echo "[gpu ${gpu}] eval DONE -> ${EVAL_TXT}  $(date)"
  ) > ${INF_LOG} 2>&1 &
  pids+=($!)

  gpu=$((gpu + 1))
done

# Wait for all 4 pipelines; tally any failures so the job exit status reflects them.
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=$((fail + 1))
done

if (( fail > 0 )); then
  echo "WARNING: ${fail} of ${#DATASETS[@]} inference+eval pipelines failed; see slurm/logs/inf/*.log" >&2
  exit 1
fi
echo "All ${#DATASETS[@]} inference+eval pipelines finished OK."
