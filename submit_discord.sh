sbatch << 'INNER_EOF'
#!/bin/bash
#SBATCH --job-name=t5gemma2-4b_train_discord-unveiled_natural2_seed27_5e-6_e5_b16_s2000_newprompt_FROM_t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt
#SBATCH --output=slurm/logs/%j_t5gemma2-4b_train_discord-unveiled_natural2_seed27_5e-6_e5_b16_s2000_newprompt_FROM_t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt.out
#SBATCH --error=slurm/logs/%j_t5gemma2-4b_train_discord-unveiled_natural2_seed27_5e-6_e5_b16_s2000_newprompt_FROM_t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt.err
#SBATCH --time=3-0:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=wut2@unbc.ca

cd $SCRATCH/Seq2Seq-DDP
source slurm/init_hpc.sh

python3 train.py --train_corpus discord-unveiled --do_train -s natural2 -t t5gemma2 -m 4b -l 5e-6 -e 5 --batchsize 16 --step 2000 --new_prompt -b --custom_model_dir ft-models/t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt
echo "Training Complete"

python3 transition_predict.py --train_corpus discord-unveiled --test_corpus discord-unveiled -s natural2 -t t5gemma2 -m 4b --lr 5e-6 -e 5 --batchsize 16 --step 2000 --seed 27 --new_prompt -b --custom_model_dir ft-models/t5gemma2-4b_train_discord-unveiled_natural2_seed27_5e-6_e5_b16_s2000_newprompt_FROM_t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt
echo "Inference Complete"

python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus discord-unveiled --test_corpus discord-unveiled -s natural2 --lr 5e-6 -e 5 --batchsize 16 --step 2000 --seed 27 --new_prompt > eval/t5gemma2-4b_5e-6_e5_b16_s2000_discord-unveiled_natural2_newprompt_FROM_molweni.txt
INNER_EOF

sbatch << 'INNER_EOF'
#!/bin/bash
#SBATCH --job-name=t5gemma2-4b_train_discord-unveiled-hintswap_natural2_seed27_5e-6_e5_b16_s2000_newprompt_FROM_t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt
#SBATCH --output=slurm/logs/%j_t5gemma2-4b_train_discord-unveiled-hintswap_natural2_seed27_5e-6_e5_b16_s2000_newprompt_FROM_t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt.out
#SBATCH --error=slurm/logs/%j_t5gemma2-4b_train_discord-unveiled-hintswap_natural2_seed27_5e-6_e5_b16_s2000_newprompt_FROM_t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt.err
#SBATCH --time=3-0:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=wut2@unbc.ca

cd $SCRATCH/Seq2Seq-DDP
source slurm/init_hpc.sh

python3 train.py --train_corpus discord-unveiled-hintswap --do_train -s natural2 -t t5gemma2 -m 4b -l 5e-6 -e 5 --batchsize 16 --step 2000 --new_prompt -b --custom_model_dir ft-models/t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt
echo "Training Complete"

python3 transition_predict.py --train_corpus discord-unveiled-hintswap --test_corpus discord-unveiled-hintswap -s natural2 -t t5gemma2 -m 4b --lr 5e-6 -e 5 --batchsize 16 --step 2000 --seed 27 --new_prompt -b --custom_model_dir ft-models/t5gemma2-4b_train_discord-unveiled-hintswap_natural2_seed27_5e-6_e5_b16_s2000_newprompt_FROM_t5gemma2-4b_train_molweni_natural2_seed27_5e-6_e5_b16_s2000_newprompt
echo "Inference Complete"

python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus discord-unveiled-hintswap --test_corpus discord-unveiled-hintswap -s natural2 --lr 5e-6 -e 5 --batchsize 16 --step 2000 --seed 27 --new_prompt > eval/t5gemma2-4b_5e-6_e5_b16_s2000_discord-unveiled-hintswap_natural2_newprompt_FROM_molweni.txt
INNER_EOF