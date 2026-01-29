#!/bin/bash
# Auto-generated submission script

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_natural_t5gemma2_270m
#SBATCH --output=molweni_natural_t5gemma2_270m_%j.out
#SBATCH --error=molweni_natural_t5gemma2_270m_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural -t t5gemma2 -m 270m -l 5e-5 -e 3 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_natural_t5gemma2_1b
#SBATCH --output=molweni_natural_t5gemma2_1b_%j.out
#SBATCH --error=molweni_natural_t5gemma2_1b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural -t t5gemma2 -m 1b -l 5e-5 -e 3 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_natural_t5gemma2_4b
#SBATCH --output=molweni_natural_t5gemma2_4b_%j.out
#SBATCH --error=molweni_natural_t5gemma2_4b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural -t t5gemma2 -m 4b -l 5e-5 -e 3 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_natural_t5_3b
#SBATCH --output=molweni_natural_t5_3b_%j.out
#SBATCH --error=molweni_natural_t5_3b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural -t t5 -m 3b -l 5e-5 -e 3 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_natural_t5_large
#SBATCH --output=molweni_natural_t5_large_%j.out
#SBATCH --error=molweni_natural_t5_large_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural -t t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_natural_flan-t5_base
#SBATCH --output=molweni_natural_flan-t5_base_%j.out
#SBATCH --error=molweni_natural_flan-t5_base_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural -t flan-t5 -m base -l 5e-5 -e 3 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_natural_flan-t5_large
#SBATCH --output=molweni_natural_flan-t5_large_%j.out
#SBATCH --error=molweni_natural_flan-t5_large_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural -t flan-t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_natural_flan-t5_xl
#SBATCH --output=molweni_natural_flan-t5_xl_%j.out
#SBATCH --error=molweni_natural_flan-t5_xl_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural -t flan-t5 -m xl -l 5e-5 -e 3 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_natural_flan-t5_xxl
#SBATCH --output=molweni_natural_flan-t5_xxl_%j.out
#SBATCH --error=molweni_natural_flan-t5_xxl_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:2 --cpus-per-task=4 --mem=32G --time=24:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural -t flan-t5 -m xxl -l 2e-5 -e 3 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_natural_t0-3b
#SBATCH --output=molweni_natural_t0-3b_%j.out
#SBATCH --error=molweni_natural_t0-3b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural -t t0-3b -l 5e-5 -e 3 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_augmented_t5gemma2_270m
#SBATCH --output=molweni_augmented_t5gemma2_270m_%j.out
#SBATCH --error=molweni_augmented_t5gemma2_270m_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s augmented -t t5gemma2 -m 270m -l 5e-5 -e 3 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_augmented_t5gemma2_1b
#SBATCH --output=molweni_augmented_t5gemma2_1b_%j.out
#SBATCH --error=molweni_augmented_t5gemma2_1b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s augmented -t t5gemma2 -m 1b -l 5e-5 -e 3 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_augmented_t5gemma2_4b
#SBATCH --output=molweni_augmented_t5gemma2_4b_%j.out
#SBATCH --error=molweni_augmented_t5gemma2_4b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s augmented -t t5gemma2 -m 4b -l 5e-5 -e 3 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_augmented_t5_3b
#SBATCH --output=molweni_augmented_t5_3b_%j.out
#SBATCH --error=molweni_augmented_t5_3b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s augmented -t t5 -m 3b -l 5e-5 -e 3 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_augmented_t5_large
#SBATCH --output=molweni_augmented_t5_large_%j.out
#SBATCH --error=molweni_augmented_t5_large_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s augmented -t t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_augmented_flan-t5_base
#SBATCH --output=molweni_augmented_flan-t5_base_%j.out
#SBATCH --error=molweni_augmented_flan-t5_base_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s augmented -t flan-t5 -m base -l 5e-5 -e 3 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_augmented_flan-t5_large
#SBATCH --output=molweni_augmented_flan-t5_large_%j.out
#SBATCH --error=molweni_augmented_flan-t5_large_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s augmented -t flan-t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_augmented_flan-t5_xl
#SBATCH --output=molweni_augmented_flan-t5_xl_%j.out
#SBATCH --error=molweni_augmented_flan-t5_xl_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s augmented -t flan-t5 -m xl -l 5e-5 -e 3 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_augmented_flan-t5_xxl
#SBATCH --output=molweni_augmented_flan-t5_xxl_%j.out
#SBATCH --error=molweni_augmented_flan-t5_xxl_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:2 --cpus-per-task=4 --mem=32G --time=24:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s augmented -t flan-t5 -m xxl -l 2e-5 -e 3 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_augmented_t0-3b
#SBATCH --output=molweni_augmented_t0-3b_%j.out
#SBATCH --error=molweni_augmented_t0-3b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s augmented -t t0-3b -l 5e-5 -e 3 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_focus_t5gemma2_270m
#SBATCH --output=molweni_focus_t5gemma2_270m_%j.out
#SBATCH --error=molweni_focus_t5gemma2_270m_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t5gemma2 -m 270m -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_focus_t5gemma2_1b
#SBATCH --output=molweni_focus_t5gemma2_1b_%j.out
#SBATCH --error=molweni_focus_t5gemma2_1b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t5gemma2 -m 1b -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_focus_t5gemma2_4b
#SBATCH --output=molweni_focus_t5gemma2_4b_%j.out
#SBATCH --error=molweni_focus_t5gemma2_4b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t5gemma2 -m 4b -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_focus_t5_3b
#SBATCH --output=molweni_focus_t5_3b_%j.out
#SBATCH --error=molweni_focus_t5_3b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t5 -m 3b -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_focus_t5_large
#SBATCH --output=molweni_focus_t5_large_%j.out
#SBATCH --error=molweni_focus_t5_large_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_focus_flan-t5_base
#SBATCH --output=molweni_focus_flan-t5_base_%j.out
#SBATCH --error=molweni_focus_flan-t5_base_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t flan-t5 -m base -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_focus_flan-t5_large
#SBATCH --output=molweni_focus_flan-t5_large_%j.out
#SBATCH --error=molweni_focus_flan-t5_large_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t flan-t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_focus_flan-t5_xl
#SBATCH --output=molweni_focus_flan-t5_xl_%j.out
#SBATCH --error=molweni_focus_flan-t5_xl_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t flan-t5 -m xl -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_focus_flan-t5_xxl
#SBATCH --output=molweni_focus_flan-t5_xxl_%j.out
#SBATCH --error=molweni_focus_flan-t5_xxl_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:2 --cpus-per-task=4 --mem=32G --time=24:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t flan-t5 -m xxl -l 2e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_focus_t0-3b
#SBATCH --output=molweni_focus_t0-3b_%j.out
#SBATCH --error=molweni_focus_t0-3b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t0-3b -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_natural2_t5gemma2_270m
#SBATCH --output=molweni_natural2_t5gemma2_270m_%j.out
#SBATCH --error=molweni_natural2_t5gemma2_270m_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 270m -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_natural2_t5gemma2_1b
#SBATCH --output=molweni_natural2_t5gemma2_1b_%j.out
#SBATCH --error=molweni_natural2_t5gemma2_1b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 1b -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_natural2_t5gemma2_4b
#SBATCH --output=molweni_natural2_t5gemma2_4b_%j.out
#SBATCH --error=molweni_natural2_t5gemma2_4b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 4b -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_natural2_t5_3b
#SBATCH --output=molweni_natural2_t5_3b_%j.out
#SBATCH --error=molweni_natural2_t5_3b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t5 -m 3b -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_natural2_t5_large
#SBATCH --output=molweni_natural2_t5_large_%j.out
#SBATCH --error=molweni_natural2_t5_large_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_natural2_flan-t5_base
#SBATCH --output=molweni_natural2_flan-t5_base_%j.out
#SBATCH --error=molweni_natural2_flan-t5_base_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t flan-t5 -m base -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_natural2_flan-t5_large
#SBATCH --output=molweni_natural2_flan-t5_large_%j.out
#SBATCH --error=molweni_natural2_flan-t5_large_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t flan-t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_natural2_flan-t5_xl
#SBATCH --output=molweni_natural2_flan-t5_xl_%j.out
#SBATCH --error=molweni_natural2_flan-t5_xl_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t flan-t5 -m xl -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_natural2_flan-t5_xxl
#SBATCH --output=molweni_natural2_flan-t5_xxl_%j.out
#SBATCH --error=molweni_natural2_flan-t5_xxl_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:2 --cpus-per-task=4 --mem=32G --time=24:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t flan-t5 -m xxl -l 2e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=molweni_natural2_t0-3b
#SBATCH --output=molweni_natural2_t0-3b_%j.out
#SBATCH --error=molweni_natural2_t0-3b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t0-3b -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_natural_t5gemma2_270m
#SBATCH --output=stac_natural_t5gemma2_270m_%j.out
#SBATCH --error=stac_natural_t5gemma2_270m_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural -t t5gemma2 -m 270m -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_natural_t5gemma2_1b
#SBATCH --output=stac_natural_t5gemma2_1b_%j.out
#SBATCH --error=stac_natural_t5gemma2_1b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural -t t5gemma2 -m 1b -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_natural_t5gemma2_4b
#SBATCH --output=stac_natural_t5gemma2_4b_%j.out
#SBATCH --error=stac_natural_t5gemma2_4b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural -t t5gemma2 -m 4b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_natural_t5_3b
#SBATCH --output=stac_natural_t5_3b_%j.out
#SBATCH --error=stac_natural_t5_3b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural -t t5 -m 3b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_natural_t5_large
#SBATCH --output=stac_natural_t5_large_%j.out
#SBATCH --error=stac_natural_t5_large_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural -t t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_natural_flan-t5_base
#SBATCH --output=stac_natural_flan-t5_base_%j.out
#SBATCH --error=stac_natural_flan-t5_base_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural -t flan-t5 -m base -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_natural_flan-t5_large
#SBATCH --output=stac_natural_flan-t5_large_%j.out
#SBATCH --error=stac_natural_flan-t5_large_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural -t flan-t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_natural_flan-t5_xl
#SBATCH --output=stac_natural_flan-t5_xl_%j.out
#SBATCH --error=stac_natural_flan-t5_xl_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural -t flan-t5 -m xl -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_natural_flan-t5_xxl
#SBATCH --output=stac_natural_flan-t5_xxl_%j.out
#SBATCH --error=stac_natural_flan-t5_xxl_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:2 --cpus-per-task=4 --mem=32G --time=24:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural -t flan-t5 -m xxl -l 2e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_natural_t0-3b
#SBATCH --output=stac_natural_t0-3b_%j.out
#SBATCH --error=stac_natural_t0-3b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural -t t0-3b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_augmented_t5gemma2_270m
#SBATCH --output=stac_augmented_t5gemma2_270m_%j.out
#SBATCH --error=stac_augmented_t5gemma2_270m_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s augmented -t t5gemma2 -m 270m -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_augmented_t5gemma2_1b
#SBATCH --output=stac_augmented_t5gemma2_1b_%j.out
#SBATCH --error=stac_augmented_t5gemma2_1b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s augmented -t t5gemma2 -m 1b -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_augmented_t5gemma2_4b
#SBATCH --output=stac_augmented_t5gemma2_4b_%j.out
#SBATCH --error=stac_augmented_t5gemma2_4b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s augmented -t t5gemma2 -m 4b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_augmented_t5_3b
#SBATCH --output=stac_augmented_t5_3b_%j.out
#SBATCH --error=stac_augmented_t5_3b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s augmented -t t5 -m 3b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_augmented_t5_large
#SBATCH --output=stac_augmented_t5_large_%j.out
#SBATCH --error=stac_augmented_t5_large_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s augmented -t t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_augmented_flan-t5_base
#SBATCH --output=stac_augmented_flan-t5_base_%j.out
#SBATCH --error=stac_augmented_flan-t5_base_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s augmented -t flan-t5 -m base -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_augmented_flan-t5_large
#SBATCH --output=stac_augmented_flan-t5_large_%j.out
#SBATCH --error=stac_augmented_flan-t5_large_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s augmented -t flan-t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_augmented_flan-t5_xl
#SBATCH --output=stac_augmented_flan-t5_xl_%j.out
#SBATCH --error=stac_augmented_flan-t5_xl_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s augmented -t flan-t5 -m xl -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_augmented_flan-t5_xxl
#SBATCH --output=stac_augmented_flan-t5_xxl_%j.out
#SBATCH --error=stac_augmented_flan-t5_xxl_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:2 --cpus-per-task=4 --mem=32G --time=24:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s augmented -t flan-t5 -m xxl -l 2e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_augmented_t0-3b
#SBATCH --output=stac_augmented_t0-3b_%j.out
#SBATCH --error=stac_augmented_t0-3b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s augmented -t t0-3b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_focus_t5gemma2_270m
#SBATCH --output=stac_focus_t5gemma2_270m_%j.out
#SBATCH --error=stac_focus_t5gemma2_270m_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s focus -t t5gemma2 -m 270m -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_focus_t5gemma2_1b
#SBATCH --output=stac_focus_t5gemma2_1b_%j.out
#SBATCH --error=stac_focus_t5gemma2_1b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s focus -t t5gemma2 -m 1b -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_focus_t5gemma2_4b
#SBATCH --output=stac_focus_t5gemma2_4b_%j.out
#SBATCH --error=stac_focus_t5gemma2_4b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s focus -t t5gemma2 -m 4b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_focus_t5_3b
#SBATCH --output=stac_focus_t5_3b_%j.out
#SBATCH --error=stac_focus_t5_3b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s focus -t t5 -m 3b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_focus_t5_large
#SBATCH --output=stac_focus_t5_large_%j.out
#SBATCH --error=stac_focus_t5_large_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s focus -t t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_focus_flan-t5_base
#SBATCH --output=stac_focus_flan-t5_base_%j.out
#SBATCH --error=stac_focus_flan-t5_base_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s focus -t flan-t5 -m base -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_focus_flan-t5_large
#SBATCH --output=stac_focus_flan-t5_large_%j.out
#SBATCH --error=stac_focus_flan-t5_large_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s focus -t flan-t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_focus_flan-t5_xl
#SBATCH --output=stac_focus_flan-t5_xl_%j.out
#SBATCH --error=stac_focus_flan-t5_xl_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s focus -t flan-t5 -m xl -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_focus_flan-t5_xxl
#SBATCH --output=stac_focus_flan-t5_xxl_%j.out
#SBATCH --error=stac_focus_flan-t5_xxl_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:2 --cpus-per-task=4 --mem=32G --time=24:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s focus -t flan-t5 -m xxl -l 2e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_focus_t0-3b
#SBATCH --output=stac_focus_t0-3b_%j.out
#SBATCH --error=stac_focus_t0-3b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s focus -t t0-3b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_natural2_t5gemma2_270m
#SBATCH --output=stac_natural2_t5gemma2_270m_%j.out
#SBATCH --error=stac_natural2_t5gemma2_270m_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural2 -t t5gemma2 -m 270m -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_natural2_t5gemma2_1b
#SBATCH --output=stac_natural2_t5gemma2_1b_%j.out
#SBATCH --error=stac_natural2_t5gemma2_1b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural2 -t t5gemma2 -m 1b -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_natural2_t5gemma2_4b
#SBATCH --output=stac_natural2_t5gemma2_4b_%j.out
#SBATCH --error=stac_natural2_t5gemma2_4b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural2 -t t5gemma2 -m 4b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_natural2_t5_3b
#SBATCH --output=stac_natural2_t5_3b_%j.out
#SBATCH --error=stac_natural2_t5_3b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural2 -t t5 -m 3b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_natural2_t5_large
#SBATCH --output=stac_natural2_t5_large_%j.out
#SBATCH --error=stac_natural2_t5_large_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural2 -t t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_natural2_flan-t5_base
#SBATCH --output=stac_natural2_flan-t5_base_%j.out
#SBATCH --error=stac_natural2_flan-t5_base_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural2 -t flan-t5 -m base -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_natural2_flan-t5_large
#SBATCH --output=stac_natural2_flan-t5_large_%j.out
#SBATCH --error=stac_natural2_flan-t5_large_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural2 -t flan-t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_natural2_flan-t5_xl
#SBATCH --output=stac_natural2_flan-t5_xl_%j.out
#SBATCH --error=stac_natural2_flan-t5_xl_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural2 -t flan-t5 -m xl -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_natural2_flan-t5_xxl
#SBATCH --output=stac_natural2_flan-t5_xxl_%j.out
#SBATCH --error=stac_natural2_flan-t5_xxl_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:2 --cpus-per-task=4 --mem=32G --time=24:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural2 -t flan-t5 -m xxl -l 2e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=stac_natural2_t0-3b
#SBATCH --output=stac_natural2_t0-3b_%j.out
#SBATCH --error=stac_natural2_t0-3b_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural2 -t t0-3b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

