#!/bin/bash
# Auto-generated submission script

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_270m_molweni_natural
#SBATCH --output=t5gemma2_270m_molweni_natural_%j.out
#SBATCH --error=t5gemma2_270m_molweni_natural_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural -t t5gemma2 -m 270m -l 5e-5 -e 3 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_1b_molweni_natural
#SBATCH --output=t5gemma2_1b_molweni_natural_%j.out
#SBATCH --error=t5gemma2_1b_molweni_natural_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural -t t5gemma2 -m 1b -l 5e-5 -e 3 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_4b_molweni_natural
#SBATCH --output=t5gemma2_4b_molweni_natural_%j.out
#SBATCH --error=t5gemma2_4b_molweni_natural_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=32G --time=08:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural -t t5gemma2 -m 4b -l 5e-5 -e 3 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5_3b_molweni_natural
#SBATCH --output=t5_3b_molweni_natural_%j.out
#SBATCH --error=t5_3b_molweni_natural_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural -t t5 -m 3b -l 5e-5 -e 3 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5_large_molweni_natural
#SBATCH --output=t5_large_molweni_natural_%j.out
#SBATCH --error=t5_large_molweni_natural_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural -t t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_base_molweni_natural
#SBATCH --output=flan-t5_base_molweni_natural_%j.out
#SBATCH --error=flan-t5_base_molweni_natural_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural -t flan-t5 -m base -l 5e-5 -e 3 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_large_molweni_natural
#SBATCH --output=flan-t5_large_molweni_natural_%j.out
#SBATCH --error=flan-t5_large_molweni_natural_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural -t flan-t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_xl_molweni_natural
#SBATCH --output=flan-t5_xl_molweni_natural_%j.out
#SBATCH --error=flan-t5_xl_molweni_natural_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural -t flan-t5 -m xl -l 5e-5 -e 3 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t0-3b_molweni_natural
#SBATCH --output=t0-3b_molweni_natural_%j.out
#SBATCH --error=t0-3b_molweni_natural_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural -t t0-3b -l 5e-5 -e 3 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_270m_molweni_augmented
#SBATCH --output=t5gemma2_270m_molweni_augmented_%j.out
#SBATCH --error=t5gemma2_270m_molweni_augmented_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s augmented -t t5gemma2 -m 270m -l 5e-5 -e 3 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_1b_molweni_augmented
#SBATCH --output=t5gemma2_1b_molweni_augmented_%j.out
#SBATCH --error=t5gemma2_1b_molweni_augmented_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s augmented -t t5gemma2 -m 1b -l 5e-5 -e 3 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_4b_molweni_augmented
#SBATCH --output=t5gemma2_4b_molweni_augmented_%j.out
#SBATCH --error=t5gemma2_4b_molweni_augmented_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=32G --time=08:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s augmented -t t5gemma2 -m 4b -l 5e-5 -e 3 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5_3b_molweni_augmented
#SBATCH --output=t5_3b_molweni_augmented_%j.out
#SBATCH --error=t5_3b_molweni_augmented_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s augmented -t t5 -m 3b -l 5e-5 -e 3 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5_large_molweni_augmented
#SBATCH --output=t5_large_molweni_augmented_%j.out
#SBATCH --error=t5_large_molweni_augmented_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s augmented -t t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_base_molweni_augmented
#SBATCH --output=flan-t5_base_molweni_augmented_%j.out
#SBATCH --error=flan-t5_base_molweni_augmented_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s augmented -t flan-t5 -m base -l 5e-5 -e 3 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_large_molweni_augmented
#SBATCH --output=flan-t5_large_molweni_augmented_%j.out
#SBATCH --error=flan-t5_large_molweni_augmented_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s augmented -t flan-t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_xl_molweni_augmented
#SBATCH --output=flan-t5_xl_molweni_augmented_%j.out
#SBATCH --error=flan-t5_xl_molweni_augmented_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s augmented -t flan-t5 -m xl -l 5e-5 -e 3 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t0-3b_molweni_augmented
#SBATCH --output=t0-3b_molweni_augmented_%j.out
#SBATCH --error=t0-3b_molweni_augmented_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s augmented -t t0-3b -l 5e-5 -e 3 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_270m_molweni_focus
#SBATCH --output=t5gemma2_270m_molweni_focus_%j.out
#SBATCH --error=t5gemma2_270m_molweni_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t5gemma2 -m 270m -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_1b_molweni_focus
#SBATCH --output=t5gemma2_1b_molweni_focus_%j.out
#SBATCH --error=t5gemma2_1b_molweni_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t5gemma2 -m 1b -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_4b_molweni_focus
#SBATCH --output=t5gemma2_4b_molweni_focus_%j.out
#SBATCH --error=t5gemma2_4b_molweni_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=32G --time=08:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t5gemma2 -m 4b -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5_3b_molweni_focus
#SBATCH --output=t5_3b_molweni_focus_%j.out
#SBATCH --error=t5_3b_molweni_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t5 -m 3b -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5_large_molweni_focus
#SBATCH --output=t5_large_molweni_focus_%j.out
#SBATCH --error=t5_large_molweni_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_base_molweni_focus
#SBATCH --output=flan-t5_base_molweni_focus_%j.out
#SBATCH --error=flan-t5_base_molweni_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t flan-t5 -m base -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_large_molweni_focus
#SBATCH --output=flan-t5_large_molweni_focus_%j.out
#SBATCH --error=flan-t5_large_molweni_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t flan-t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_xl_molweni_focus
#SBATCH --output=flan-t5_xl_molweni_focus_%j.out
#SBATCH --error=flan-t5_xl_molweni_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t flan-t5 -m xl -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t0-3b_molweni_focus
#SBATCH --output=t0-3b_molweni_focus_%j.out
#SBATCH --error=t0-3b_molweni_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t0-3b -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_270m_molweni_natural2
#SBATCH --output=t5gemma2_270m_molweni_natural2_%j.out
#SBATCH --error=t5gemma2_270m_molweni_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 270m -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_1b_molweni_natural2
#SBATCH --output=t5gemma2_1b_molweni_natural2_%j.out
#SBATCH --error=t5gemma2_1b_molweni_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 1b -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_4b_molweni_natural2
#SBATCH --output=t5gemma2_4b_molweni_natural2_%j.out
#SBATCH --error=t5gemma2_4b_molweni_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=32G --time=08:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 4b -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5_3b_molweni_natural2
#SBATCH --output=t5_3b_molweni_natural2_%j.out
#SBATCH --error=t5_3b_molweni_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t5 -m 3b -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5_large_molweni_natural2
#SBATCH --output=t5_large_molweni_natural2_%j.out
#SBATCH --error=t5_large_molweni_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_base_molweni_natural2
#SBATCH --output=flan-t5_base_molweni_natural2_%j.out
#SBATCH --error=flan-t5_base_molweni_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t flan-t5 -m base -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_large_molweni_natural2
#SBATCH --output=flan-t5_large_molweni_natural2_%j.out
#SBATCH --error=flan-t5_large_molweni_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t flan-t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_xl_molweni_natural2
#SBATCH --output=flan-t5_xl_molweni_natural2_%j.out
#SBATCH --error=flan-t5_xl_molweni_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t flan-t5 -m xl -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t0-3b_molweni_natural2
#SBATCH --output=t0-3b_molweni_natural2_%j.out
#SBATCH --error=t0-3b_molweni_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t0-3b -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_270m_stac_natural
#SBATCH --output=t5gemma2_270m_stac_natural_%j.out
#SBATCH --error=t5gemma2_270m_stac_natural_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural -t t5gemma2 -m 270m -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_1b_stac_natural
#SBATCH --output=t5gemma2_1b_stac_natural_%j.out
#SBATCH --error=t5gemma2_1b_stac_natural_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural -t t5gemma2 -m 1b -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_4b_stac_natural
#SBATCH --output=t5gemma2_4b_stac_natural_%j.out
#SBATCH --error=t5gemma2_4b_stac_natural_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=32G --time=08:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural -t t5gemma2 -m 4b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5_3b_stac_natural
#SBATCH --output=t5_3b_stac_natural_%j.out
#SBATCH --error=t5_3b_stac_natural_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural -t t5 -m 3b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5_large_stac_natural
#SBATCH --output=t5_large_stac_natural_%j.out
#SBATCH --error=t5_large_stac_natural_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural -t t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_base_stac_natural
#SBATCH --output=flan-t5_base_stac_natural_%j.out
#SBATCH --error=flan-t5_base_stac_natural_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural -t flan-t5 -m base -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_large_stac_natural
#SBATCH --output=flan-t5_large_stac_natural_%j.out
#SBATCH --error=flan-t5_large_stac_natural_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural -t flan-t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_xl_stac_natural
#SBATCH --output=flan-t5_xl_stac_natural_%j.out
#SBATCH --error=flan-t5_xl_stac_natural_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural -t flan-t5 -m xl -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t0-3b_stac_natural
#SBATCH --output=t0-3b_stac_natural_%j.out
#SBATCH --error=t0-3b_stac_natural_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural -t t0-3b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_270m_stac_augmented
#SBATCH --output=t5gemma2_270m_stac_augmented_%j.out
#SBATCH --error=t5gemma2_270m_stac_augmented_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s augmented -t t5gemma2 -m 270m -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_1b_stac_augmented
#SBATCH --output=t5gemma2_1b_stac_augmented_%j.out
#SBATCH --error=t5gemma2_1b_stac_augmented_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s augmented -t t5gemma2 -m 1b -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_4b_stac_augmented
#SBATCH --output=t5gemma2_4b_stac_augmented_%j.out
#SBATCH --error=t5gemma2_4b_stac_augmented_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=32G --time=08:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s augmented -t t5gemma2 -m 4b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5_3b_stac_augmented
#SBATCH --output=t5_3b_stac_augmented_%j.out
#SBATCH --error=t5_3b_stac_augmented_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s augmented -t t5 -m 3b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5_large_stac_augmented
#SBATCH --output=t5_large_stac_augmented_%j.out
#SBATCH --error=t5_large_stac_augmented_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s augmented -t t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_base_stac_augmented
#SBATCH --output=flan-t5_base_stac_augmented_%j.out
#SBATCH --error=flan-t5_base_stac_augmented_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s augmented -t flan-t5 -m base -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_large_stac_augmented
#SBATCH --output=flan-t5_large_stac_augmented_%j.out
#SBATCH --error=flan-t5_large_stac_augmented_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s augmented -t flan-t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_xl_stac_augmented
#SBATCH --output=flan-t5_xl_stac_augmented_%j.out
#SBATCH --error=flan-t5_xl_stac_augmented_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s augmented -t flan-t5 -m xl -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t0-3b_stac_augmented
#SBATCH --output=t0-3b_stac_augmented_%j.out
#SBATCH --error=t0-3b_stac_augmented_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s augmented -t t0-3b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_270m_stac_focus
#SBATCH --output=t5gemma2_270m_stac_focus_%j.out
#SBATCH --error=t5gemma2_270m_stac_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s focus -t t5gemma2 -m 270m -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_1b_stac_focus
#SBATCH --output=t5gemma2_1b_stac_focus_%j.out
#SBATCH --error=t5gemma2_1b_stac_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s focus -t t5gemma2 -m 1b -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_4b_stac_focus
#SBATCH --output=t5gemma2_4b_stac_focus_%j.out
#SBATCH --error=t5gemma2_4b_stac_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=32G --time=08:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s focus -t t5gemma2 -m 4b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5_3b_stac_focus
#SBATCH --output=t5_3b_stac_focus_%j.out
#SBATCH --error=t5_3b_stac_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s focus -t t5 -m 3b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5_large_stac_focus
#SBATCH --output=t5_large_stac_focus_%j.out
#SBATCH --error=t5_large_stac_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s focus -t t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_base_stac_focus
#SBATCH --output=flan-t5_base_stac_focus_%j.out
#SBATCH --error=flan-t5_base_stac_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s focus -t flan-t5 -m base -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_large_stac_focus
#SBATCH --output=flan-t5_large_stac_focus_%j.out
#SBATCH --error=flan-t5_large_stac_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s focus -t flan-t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_xl_stac_focus
#SBATCH --output=flan-t5_xl_stac_focus_%j.out
#SBATCH --error=flan-t5_xl_stac_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s focus -t flan-t5 -m xl -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t0-3b_stac_focus
#SBATCH --output=t0-3b_stac_focus_%j.out
#SBATCH --error=t0-3b_stac_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s focus -t t0-3b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_270m_stac_natural2
#SBATCH --output=t5gemma2_270m_stac_natural2_%j.out
#SBATCH --error=t5gemma2_270m_stac_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural2 -t t5gemma2 -m 270m -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_1b_stac_natural2
#SBATCH --output=t5gemma2_1b_stac_natural2_%j.out
#SBATCH --error=t5gemma2_1b_stac_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural2 -t t5gemma2 -m 1b -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_4b_stac_natural2
#SBATCH --output=t5gemma2_4b_stac_natural2_%j.out
#SBATCH --error=t5gemma2_4b_stac_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=32G --time=08:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural2 -t t5gemma2 -m 4b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5_3b_stac_natural2
#SBATCH --output=t5_3b_stac_natural2_%j.out
#SBATCH --error=t5_3b_stac_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural2 -t t5 -m 3b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5_large_stac_natural2
#SBATCH --output=t5_large_stac_natural2_%j.out
#SBATCH --error=t5_large_stac_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural2 -t t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_base_stac_natural2
#SBATCH --output=flan-t5_base_stac_natural2_%j.out
#SBATCH --error=flan-t5_base_stac_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural2 -t flan-t5 -m base -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_large_stac_natural2
#SBATCH --output=flan-t5_large_stac_natural2_%j.out
#SBATCH --error=flan-t5_large_stac_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural2 -t flan-t5 -m large -l 5e-5 -e 10 --batchsize 8 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_xl_stac_natural2
#SBATCH --output=flan-t5_xl_stac_natural2_%j.out
#SBATCH --error=flan-t5_xl_stac_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural2 -t flan-t5 -m xl -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t0-3b_stac_natural2
#SBATCH --output=t0-3b_stac_natural2_%j.out
#SBATCH --error=t0-3b_stac_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00

source init_hpc.sh
python3 train.py --train_corpus stac --do_train -s natural2 -t t0-3b -l 5e-5 -e 10 --batchsize 4 --step 500 -b
EOT

