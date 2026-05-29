SEED=${1:-27}

for DATASET in discord-unveiled-hintfull-molwenik1 discord-unveiled-hintfull-nomolweni discord-unveiled-hintswap-molwenik1 discord-unveiled-hintswap-nomolweni; do
sbatch << INNER_EOF
#!/bin/bash
#SBATCH --job-name=t5gemma2-4b_train_${DATASET}_natural2_seed${SEED}_5e-6_e5_b16_s2000_newprompt_FROM_t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt
#SBATCH --output=slurm/logs/%j_t5gemma2-4b_train_${DATASET}_natural2_seed${SEED}_5e-6_e5_b16_s2000_newprompt_FROM_t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt.out
#SBATCH --error=slurm/logs/%j_t5gemma2-4b_train_${DATASET}_natural2_seed${SEED}_5e-6_e5_b16_s2000_newprompt_FROM_t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt.err
#SBATCH --time=3-0:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=wut2@unbc.ca

cd \$SCRATCH/Seq2Seq-DDP
source slurm/init_hpc.sh
export PYTHONUNBUFFERED=1

python3 train.py --train_corpus ${DATASET} --do_train -s natural2 -t t5gemma2 -m 4b -l 5e-6 -e 5 --batchsize 16 --step 2000 --new_prompt -b --custom_model_dir ft-models/t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt
echo "Training Complete"

python3 transition_predict.py --train_corpus ${DATASET} --test_corpus ${DATASET} -s natural2 -t t5gemma2 -m 4b --lr 5e-6 -e 5 --batchsize 32 --step 2000 --seed ${SEED} --new_prompt -b --custom_model_dir ft-models/t5gemma2-4b_train_${DATASET}_natural2_seed${SEED}_5e-6_e5_b16_s2000_newprompt_FROM_t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt
echo "Inference Complete"

python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus ${DATASET} --test_corpus ${DATASET} -s natural2 --lr 5e-6 -e 5 --batchsize 32 --step 2000 --seed 27 --new_prompt > eval/t5gemma2-4b_5e-6_e5_b16_s2000_${DATASET}_natural2_newprompt_FROM_molweni.txt
INNER_EOF
done

